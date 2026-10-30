from __future__ import annotations

import time
import os
import json
import ssl
import logging
import certifi
import gernas
from gernas import Ctx
import httpx
from typing import Dict, List, Optional, Tuple, Union
from openai import AzureOpenAI
from dataclasses import dataclass
from inference.compliance_agent import prompt as _prompt_module
from inference.compliance_agent.prompt import SYSTEM_PROMPT
from logger import Logger
from settings import settings
logger = Logger(__name__).get_logger()

AZURE_OPENAI_ENDPOINT = settings.AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY = settings.AZURE_OPENAI_API_KEY
AZURE_OPENAI_API_VERSION = settings.AZURE_API_VERSION
AZURE_OPENAI_MODEL = settings.AZURE_DEPLOYMENT_NAME

# --- GERNAS telemetry (SOW §2 AI / Prompt domains) -----------------------------------------
# AZURE_DEPLOYMENT_NAME is the Azure *deployment*, not the model. model_id must be the
# canonical model name so dashboards and the rate card (GERNAS_RATE_CARD_PATH) match.
GERNAS_MODEL_ID = getattr(settings, "AZURE_OPENAI_MODEL_NAME", None) or "gpt-4.1"
PROMPT_VERSION = str(getattr(_prompt_module, "PROMPT_VERSION", "1"))  # bump when prompt.py changes


@dataclass
class RetryContextPolicy:
    """
    Controls how much context we pass on retries to avoid token bloat.
    """
    include_previous_output: bool = True
    max_previous_output_chars: int = 25000

    include_schema: bool = True
    ## schema_mode values:
    # "failing_fields": embed only the schema for fields that failed validation (if known)
    # "full": embed the entire model schema
    schema_mode: str = "failing_fields"
    max_schema_chars: int = 2500


class LLMClient:
    """
    Lightweight wrapper around autogen that provides:
      - deterministic gen defaults (temperature=0.0, top_p=0.1)
      - structured output via Pydantic validation
      - fresh-context, self-healing retries
      - token-aware retry context (clipped schema & previous output)
      - custom error mapping
      - PII-safe, structured logging
    """

    def __init__(
        self,
        temperature: float = 0.0,
        top_p: float = 0.1,
        retry_policy: RetryContextPolicy = RetryContextPolicy(),
    ) -> None:
        # self.config_list = config_list
        self.temperature = temperature
        self.top_p = top_p
        self.retry_policy = retry_policy

    # One llm.call span per attempt (each retry is visible with its own error / latency).
    # Tokens, cached tokens, stop reason and model version are read by the SDK from the
    # returned dict keys (input_token_count, output_token_count, cached_tokens,
    # finish_reason, model). Prompt and completion text are never recorded.
    @gernas.trace_llm(
        model_id=GERNAS_MODEL_ID,
        provider="azure_openai",
        deployment_id=AZURE_OPENAI_MODEL,
    )
    def invoke_gpt_4(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        event: Dict[str, str],
        temp: float = 0.6,
        model: str = "gpt-4.1",
        prompt_template_id: Optional[str] = None,
    ) -> Dict[str, Union[str, int]]:
        """
        Invokes Azure OpenAI GPT model with the given prompt and event metadata.
        """
        if prompt_template_id:
            gernas.annotate({Ctx.PROMPT_TEMPLATE_ID: prompt_template_id, Ctx.PROMPT_VERSION: PROMPT_VERSION})

        # NOTE (review F11): certificate / hostname verification disabled - unchanged.
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Gateway attribution headers - unchanged (see review notes on endUserId / "(Prod)").
        headers = {
            "Content-Type": "application/json",
            "endUserId": f'{event.get("user_id", "")}',
            "sessionId": f'{event.get("chat_id", "")}',
            "appId": f'{event.get("selectedDepartment", "")}(Prod)',
            "transactionId": f'{event.get("conversation_id", "")}',
            "agentId": f'{event.get("CurrentAgent", "LC Compliance Agent")}| (Prod)',
            "agentName": f'{event.get("CurrentAgent", "LC Compliance Agent")}',
            "valueStream": "Trade & Supply Chain"
        }

        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            http_client=httpx.Client(verify=ssl_context),
            default_headers=headers
        )
        #print(headers)
        messages = (
            prompt if isinstance(prompt, list) and all(isinstance(item, dict) and 'role' in item for item in prompt)
            else [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": prompt}
            ]
        )

        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_MODEL,
                messages=messages,
                temperature=temp,
                max_tokens=16000
            )

            content = response.choices[0].message.content
            usage = response.usage

            return {
                "response": content,
                "input_token_count": usage.prompt_tokens,
                "output_token_count": usage.completion_tokens,
                # telemetry-only keys (callers read the three keys above, as before)
                "cached_tokens": _cached_tokens(usage),
                "finish_reason": _finish_reason(response),
                "model": getattr(response, "model", None),
            }

        except Exception as e:
            logger.error(f"GPT invocation error: {type(e).__name__}: {e}")
            raise

    def invoke_model(
        self,
        messages: Union[str, List[Dict[str, str]]],
        event: Dict[str, str],
        temp: float = 0.1,
        prompt_template_id: Optional[str] = None,
    ) -> Tuple[str, int, int]:
        """
        Wrapper to invoke GPT model and return response and token usage.
        """
        result = self.invoke_gpt_4(prompt=messages, event=event, temp=temp, prompt_template_id=prompt_template_id)

        return result["response"], result["input_token_count"], result["output_token_count"]

    def generate(self, payload, event, retries=50, backoff_factor=10,
                 prompt_template_id: Optional[str] = None) -> Tuple[str, int, int]:
        ''' Invoking GPT Model '''
        for attempt in range(retries):
            try:
                response, input_token, output_token = self.invoke_model(payload, event, prompt_template_id=prompt_template_id)
                return response, input_token, output_token
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = backoff_factor * (2 ** attempt)
                    gernas.record_retry()  # SOW §5 Retry Rate (no-op outside an agent run)
                    logger.warning(f"LLM call failed ({type(e).__name__}). Retrying {attempt + 1}/{retries} in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM call failed ({type(e).__name__}): Maximum retries reached. Exiting.")
                    # The caller still receives (None, 0, 0) as before; the run is marked as
                    # ended in error so it is not counted as a successful task (SOW §5).
                    gernas.set_termination_reason("error")
                    return None, 0, 0


def _cached_tokens(usage) -> Optional[int]:
    try:
        return usage.prompt_tokens_details.cached_tokens
    except Exception:
        return None


def _finish_reason(response) -> Optional[str]:
    try:
        return response.choices[0].finish_reason
    except Exception:
        return None
