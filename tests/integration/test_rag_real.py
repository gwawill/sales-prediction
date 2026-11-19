"""RAG against the real S3 Vectors index and OpenRouter model (costs a fraction of a cent)."""

from sales_poc.ai.rag import answer


def test_a_sample_question_names_its_sources():
    result = answer("How did laptop sales in EMEA change in March 2026?")

    assert result.sources, result.text
    assert "catreg:Laptop:EMEA:2026-03" in result.source_ids
    assert "2.40M" in result.text or "2.4M" in result.text or "11%" in result.text
