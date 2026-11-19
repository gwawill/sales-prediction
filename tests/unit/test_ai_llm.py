from types import SimpleNamespace as NS

import httpx
import openai
import pytest

from sales_poc.ai.llm import Llm, LlmError
from sales_poc.config import LlmSettings


def settings(**changes):
    values = {"openrouter_api_key": "key", "llm_model": "some/model", **changes}
    return LlmSettings(_env_file=None, **values)


def response(content="hello", tool_calls=None, finish="stop", prompt=10, completion=5):
    message = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(message=message, finish_reason=finish)],
              usage=NS(prompt_tokens=prompt, completion_tokens=completion), model="some/model")


def call(name, arguments, id="call_1"):
    return NS(id=id, function=NS(name=name, arguments=arguments))


class FakeOpenRouter:
    """Records requests; returns queued responses or raises queued errors."""

    def __init__(self, *outcomes):
        self.outcomes, self.requests = list(outcomes), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_a_plain_reply():
    fake = FakeOpenRouter(response("  The answer.  "))
    reply = Llm(settings(), fake).chat([{"role": "user", "content": "q"}])

    assert reply.text == "The answer."
    assert reply.tool_calls == [] and reply.finish_reason == "stop"
    assert reply.message == {"role": "assistant", "content": "  The answer.  "}


def test_settings_go_into_the_request():
    fake = FakeOpenRouter(response())
    Llm(settings(llm_max_tokens=123, llm_temperature=0.3), fake).chat([{"role": "user", "content": "q"}])

    request = fake.requests[0]
    assert (request["model"], request["max_tokens"], request["temperature"]) == ("some/model", 123, 0.3)
    assert "tools" not in request


def test_per_call_limits_override_settings():
    fake = FakeOpenRouter(response())
    Llm(settings(), fake).chat([{"role": "user", "content": "q"}], max_tokens=10, temperature=1.0)
    assert (fake.requests[0]["max_tokens"], fake.requests[0]["temperature"]) == (10, 1.0)


def test_tool_calls_are_decoded_and_kept_for_the_next_message():
    fake = FakeOpenRouter(response(None, [call("get_sales", '{"region": "EMEA"}')], finish="tool_calls"))
    tools = [{"type": "function", "function": {"name": "get_sales", "parameters": {}}}]

    reply = Llm(settings(), fake).chat([{"role": "user", "content": "q"}], tools=tools)

    assert fake.requests[0]["tools"] == tools
    [tc] = reply.tool_calls
    assert (tc.name, tc.arguments, tc.bad_json) == ("get_sales", {"region": "EMEA"}, False)
    assert reply.message["tool_calls"][0]["function"]["arguments"] == '{"region": "EMEA"}'
    assert reply.text == ""


@pytest.mark.parametrize("arguments", ["{not json", "[1, 2]"])
def test_bad_tool_arguments_are_flagged_not_fatal(arguments):
    fake = FakeOpenRouter(response(None, [call("get_sales", arguments)], finish="tool_calls"))
    [tc] = Llm(settings(), fake).chat([{"role": "user", "content": "q"}]).tool_calls

    assert tc.bad_json and tc.arguments == {} and tc.raw_arguments == arguments


def test_usage_adds_up_across_calls():
    llm = Llm(settings(), FakeOpenRouter(response(prompt=10, completion=5), response(prompt=20, completion=7)))
    llm.chat([{"role": "user", "content": "a"}])
    reply = llm.chat([{"role": "user", "content": "b"}])

    assert reply.usage.total_tokens == 27
    assert (llm.usage.prompt_tokens, llm.usage.completion_tokens, llm.usage.calls) == (30, 12, 2)


def _status_error(cls, status):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (_status_error(openai.AuthenticationError, 401), "OPENROUTER_API_KEY"),
        (_status_error(openai.NotFoundError, 404), "LLM_MODEL"),
        (_status_error(openai.APIStatusError, 402), "out of credits"),
        (openai.APIConnectionError(request=httpx.Request("POST", "https://x")), "Could not reach"),
    ],
)
def test_problems_become_plain_english_errors(error, message):
    with pytest.raises(LlmError, match=message):
        Llm(settings(), FakeOpenRouter(error)).chat([{"role": "user", "content": "q"}])


def test_an_empty_reply_is_an_error():
    with pytest.raises(LlmError, match="empty"):
        Llm(settings(), FakeOpenRouter(NS(choices=[], usage=None))).chat([{"role": "user", "content": "q"}])
