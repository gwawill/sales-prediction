"""The LLM client against the real OpenRouter model (costs a fraction of a cent)."""

from sales_poc.ai.llm import get_llm


def test_a_real_reply_and_usage_count():
    llm = get_llm()
    before = llm.usage.total_tokens

    reply = llm.chat([{"role": "user", "content": "Reply with exactly one word: pong"}], max_tokens=10)

    assert "pong" in reply.text.lower()
    assert reply.finish_reason == "stop"
    assert reply.usage.total_tokens > 0
    assert llm.usage.total_tokens == before + reply.usage.total_tokens
