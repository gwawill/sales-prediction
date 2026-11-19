from types import SimpleNamespace as NS

import pytest

from sales_poc.ai import rag
from sales_poc.ai.llm import Llm
from sales_poc.config import LlmSettings
from sales_poc.vectors import store
from sales_poc.vectors.store import SearchResult

RESULTS = [
    SearchResult("catreg:Laptop:EMEA:2026-03", 0.77, "Laptop sales in EMEA, March 2026: $2.40M revenue.", {}),
    SearchResult("region:EMEA:2026-03", 0.78, "All sales in EMEA, March 2026: $12.88M revenue.", {}),
    SearchResult("catreg:Laptop:EMEA:2026-06", 0.76, "Laptop sales in EMEA, June 2026: $2.12M revenue.", {}),
]


class FakeOpenRouter:
    def __init__(self, text):
        self.text, self.requests = text, []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        message = NS(content=self.text, tool_calls=None)
        return NS(choices=[NS(message=message, finish_reason="stop")],
                  usage=NS(prompt_tokens=100, completion_tokens=20), model="some/model")


def fake_llm(text):
    settings = LlmSettings(_env_file=None, openrouter_api_key="key", llm_model="some/model")
    fake = FakeOpenRouter(text)
    return Llm(settings, fake), fake


@pytest.fixture
def searched(monkeypatch):
    """Replace the real vector search; record what it was asked."""
    asked = {}

    def fake_search(index, embedder, question, top_k, **labels):
        asked.update(question=question, top_k=top_k, labels=labels)
        return RESULTS[:top_k]

    monkeypatch.setattr(store, "search", fake_search)
    return asked


def ask(text, **kwargs):
    llm, fake = fake_llm(text)
    result = rag.answer("How did EMEA laptops do in March 2026?", llm=llm, index=object(), embedder=object(),
                        **kwargs)
    return result, fake


def test_the_sources_are_the_summaries_the_answer_cites(searched):
    result, _ = ask("Laptop revenue was $2.40M [S1], out of $12.88M for the whole region [S2].")

    assert result.source_ids == ["catreg:Laptop:EMEA:2026-03", "region:EMEA:2026-03"]
    assert len(result.retrieved) == 3
    assert result.usage.total_tokens == 120


def test_the_model_sees_every_summary_with_its_tag_and_the_rules(searched):
    _, fake = ask("x [S1]")
    system, user = fake.requests[0]["messages"]

    assert "Use only facts from the summaries" in system["content"]
    for i, r in enumerate(RESULTS, start=1):
        assert f"[S{i}] ({r.summary_id})\n{r.text}" in user["content"]
    assert user["content"].endswith("Question: How did EMEA laptops do in March 2026?")


def test_search_options_are_passed_on(searched):
    ask("x [S1]", top_k=2, region="EMEA", month="2026-03")
    assert searched["top_k"] == 2
    assert searched["labels"] == {"region": "EMEA", "month": "2026-03"}


def test_tags_the_model_invented_are_dropped(searched):
    result, _ = ask("Something [S3]. Something else [S9].")
    assert result.source_ids == ["catreg:Laptop:EMEA:2026-06"]


def test_an_answer_without_tags_has_no_sources(searched):
    result, _ = ask("The summaries do not say.")
    assert result.sources == [] and len(result.retrieved) == 3


def test_no_search_results_means_no_model_call(monkeypatch):
    monkeypatch.setattr(store, "search", lambda *a, **k: [])
    result, fake = ask("should not be used")

    assert fake.requests == []
    assert result.sources == [] and "could not find" in result.text


@pytest.mark.parametrize(
    ("text", "tags"),
    [
        ("a [S1] b [S2]", ["S1", "S2"]),
        ("a [S2, S1] b [S1]", ["S2", "S1"]),
        ("a [S1][S3]", ["S1", "S3"]),
        ("a [S1; S03]", ["S1", "S3"]),
        ("no tags S1 here", []),
        ("[see S2]", ["S2"]),
        ("[XS1] [S1x]", []),
    ],
)
def test_cited_tags(text, tags):
    assert rag.cited_tags(text) == tags
