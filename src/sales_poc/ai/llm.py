"""One small client for the language model on OpenRouter (task T11.1).

Used by the RAG answer (T11.2) and the agent (T12), so the rules live in one place:

    llm = get_llm()
    reply = llm.chat([{"role": "user", "content": "Hello"}])
    reply.text, reply.tool_calls, reply.finish_reason, reply.usage
    llm.usage            # tokens used so far by this client, to keep an eye on cost

Settings (LlmSettings, from .env): LLM_MODEL, LLM_MAX_TOKENS (answer length cap),
LLM_TEMPERATURE (0 = most consistent), LLM_TIMEOUT_SECONDS.

Problems become an LlmError with a plain-English message (wrong key, unknown model, no
credits, no network), so callers never have to know the openai library's error types.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import openai

from sales_poc.ai.client import openrouter_client
from sales_poc.config import LlmSettings, get_llm_settings


class LlmError(RuntimeError):
    """The language model could not be used. The message says what to fix."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict            # decoded from the model's JSON
    raw_arguments: str         # exactly what the model sent
    bad_json: bool = False     # True if the arguments were not valid JSON


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls


@dataclass(frozen=True)
class Reply:
    text: str
    tool_calls: list[ToolCall]
    finish_reason: str         # "stop", "tool_calls", "length" (hit LLM_MAX_TOKENS), ...
    usage: Usage
    model: str
    message: dict = field(default_factory=dict)  # the reply as a chat message, to send back later


def _tool_call(raw) -> ToolCall:
    text = raw.function.arguments or "{}"
    try:
        arguments, bad = json.loads(text), False
        if not isinstance(arguments, dict):
            arguments, bad = {}, True
    except json.JSONDecodeError:
        arguments, bad = {}, True
    return ToolCall(id=raw.id, name=raw.function.name, arguments=arguments, raw_arguments=text, bad_json=bad)


class Llm:
    def __init__(self, settings: LlmSettings, client=None):
        self.settings = settings
        self.model = settings.llm_model
        self._client = client
        self.usage = Usage()

    @property
    def client(self):
        if self._client is None:
            self._client = openrouter_client(self.settings)
        return self._client

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Reply:
        """Send messages (and optional tool definitions); get one reply back."""
        request = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
        }
        if tools:
            request["tools"] = tools
        try:
            response = self.client.chat.completions.create(**request)
        except openai.AuthenticationError:
            raise LlmError("OpenRouter refused the key. Check OPENROUTER_API_KEY in .env.") from None
        except openai.NotFoundError:
            raise LlmError(f"OpenRouter does not know the model '{self.model}'. Copy an exact id "
                           "from openrouter.ai/models into LLM_MODEL.") from None
        except openai.APIStatusError as exc:
            hint = " Your OpenRouter account may be out of credits." if exc.status_code == 402 else ""
            raise LlmError(f"OpenRouter error {exc.status_code}: {exc.message}.{hint}") from None
        except openai.APIConnectionError as exc:
            raise LlmError(f"Could not reach OpenRouter at {self.settings.openrouter_base_url}: {exc}") from None

        if not response.choices:
            raise LlmError("OpenRouter sent back an empty reply.")
        choice = response.choices[0]
        message = choice.message
        raw_calls = message.tool_calls or []
        usage = Usage(
            prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
            calls=1,
        )
        self.usage.add(usage)

        as_message = {"role": "assistant", "content": message.content or ""}
        if raw_calls:
            as_message["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                for c in raw_calls
            ]
        return Reply(
            text=(message.content or "").strip(),
            tool_calls=[_tool_call(c) for c in raw_calls],
            finish_reason=choice.finish_reason or "",
            usage=usage,
            model=getattr(response, "model", self.model) or self.model,
            message=as_message,
        )


@lru_cache(maxsize=1)
def get_llm() -> Llm:
    return Llm(get_llm_settings())
