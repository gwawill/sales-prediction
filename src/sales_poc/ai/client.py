"""One place to create the OpenRouter client.

OpenRouter speaks the same API as OpenAI, so we use the official `openai` library and only
change the address (base_url) and the key.
"""

from __future__ import annotations

from openai import OpenAI

from sales_poc.config import LlmSettings, get_llm_settings


def openrouter_client(settings: LlmSettings | None = None) -> OpenAI:
    s = settings or get_llm_settings()
    return OpenAI(
        base_url=s.openrouter_base_url,
        api_key=s.openrouter_api_key.get_secret_value(),
        timeout=60,
        max_retries=2,
    )
