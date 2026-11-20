"""Search against the real S3 Vectors index (load it first with vector-load)."""

import pytest

from sales_poc.vectors.embeddings import get_embedder
from sales_poc.vectors.store import default_index, search


@pytest.fixture(scope="module")
def index():
    return default_index()


@pytest.mark.parametrize(
    ("question", "want"),
    [
        ("laptop sales in EMEA", {"category": "Laptop", "region_id": "EMEA"}),
        ("How did servers sell in North America?", {"category": "Server", "region_id": "NA"}),
        ("Why did desktop sales drop in APJ in March 2026?",
         {"category": "Desktop", "region_id": "APJ", "month": "2026-03"}),
        ("monitor revenue in Latin America in June 2026",
         {"category": "Monitor", "region_id": "LATAM", "month": "2026-06"}),
        ("storage sales in Europe", {"category": "Storage", "region_id": "EMEA"}),
    ],
)
def test_the_top_result_matches_what_the_question_asks_for(index, question, want):
    [best, *_] = search(index, get_embedder(), question, top_k=3)
    for key, value in want.items():
        assert best.metadata.get(key) == value, (question, best.summary_id)
    assert 0 < best.score <= 1


def test_laptop_sales_in_emea_returns_emea_laptop_summaries_first(index):
    results = search(index, get_embedder(), "laptop sales in EMEA", top_k=5)
    assert all(r.metadata.get("category") == "Laptop" and r.metadata.get("region_id") == "EMEA"
               for r in results)


def test_the_index_holds_one_vector_per_summary(index):
    keys = index.keys()
    assert len(keys) == 864
    assert len({k for k in keys if k.startswith("customer:")}) == 500


def test_the_question_set_mostly_finds_the_right_summary(index):
    """The fixed question set in vectors/evaluate.py. Measured on 2026-11-20: 20 of 20 in the
    top 3, 20 ranked first. One miss is allowed for data changes (sync-stores rewrites
    summaries); more means a change made the search worse."""
    from sales_poc.vectors.evaluate import evaluate, report

    embedder = get_embedder()
    outcomes = evaluate(lambda q, k: search(index, embedder, q, top_k=k))
    found = sum(o.rank is not None for o in outcomes)
    assert found >= 19, report(outcomes)
