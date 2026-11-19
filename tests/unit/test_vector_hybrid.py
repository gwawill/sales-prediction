import pytest

from sales_poc.vectors.hybrid import LABEL_BONUS, label_matches, labels_in, rerank
from sales_poc.vectors.store import SearchResult


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("laptop sales in EMEA", {"category": ["Laptop"], "region_id": ["EMEA"]}),
        ("How did laptops sell in Europe?", {"category": ["Laptop"], "region_id": ["EMEA"]}),
        ("servers in North America", {"category": ["Server"], "region_id": ["NA"]}),
        ("Server revenue in NA", {"category": ["Server"], "region_id": ["NA"]}),
        ("Why did desktop sales drop in APJ in March 2026?",
         {"category": ["Desktop"], "region_id": ["APJ"], "month": ["2026-03"]}),
        ("monitors in Latin America in June", {"category": ["Monitor"], "region_id": ["LATAM"], "month": ["*-06"]}),
        ("accessories sold to consumers", {"category": ["Accessory"], "segment": ["Consumer"]}),
        ("what did C00013 buy?", {"customer_id": ["C00013"]}),
        ("sales for 2026-08", {"month": ["2026-08"]}),
        ("storage and servers", {"category": ["Server", "Storage"]}),
        ("how are we doing overall?", {}),
    ],
)
def test_labels_found_in_questions(question, expected):
    found = labels_in(question)
    assert {k: sorted(v) for k, v in found.items()} == {k: sorted(v) for k, v in expected.items()}


def test_na_inside_other_words_is_not_north_america():
    for question in ("financial analysis", "national sales", "banana"):
        assert "region_id" not in labels_in(question)


def test_may_as_an_ordinary_word_is_not_a_month():
    assert "month" not in labels_in("how may I see laptop sales?")
    assert labels_in("laptop sales in May")["month"] == ["*-05"]
    assert labels_in("laptop sales in may 2026")["month"] == ["2026-05"]


def test_a_month_without_a_year_matches_that_month_in_any_year():
    wanted = labels_in("March")
    assert label_matches({"month": "2026-03"}, wanted) == 1
    assert label_matches({"month": "2025-03"}, wanted) == 1
    assert label_matches({"month": "2026-05"}, wanted) == 0


def result(key, score, **meta):
    return SearchResult(key, score, "", meta)


def test_rerank_moves_matching_labels_up():
    plain = [
        result("acc-emea", 0.847, category="Accessory", region_id="EMEA"),
        result("lap-apj", 0.846, category="Laptop", region_id="APJ"),
        result("lap-emea", 0.845, category="Laptop", region_id="EMEA"),
    ]

    ranked = rerank("laptop sales in EMEA", plain, top_k=3)

    # lap-emea matches both labels. The other two match one each, so meaning decides.
    assert [r.summary_id for r in ranked] == ["lap-emea", "acc-emea", "lap-apj"]
    assert [r.matched_labels for r in ranked] == [2, 1, 1]
    assert ranked[0].score == pytest.approx(0.845)  # the meaning score is kept as it was
    assert LABEL_BONUS > 0.847 - 0.845  # one label is worth more than a tiny meaning gap


def test_meaning_still_decides_between_equal_matches():
    plain = [result("a", 0.70, category="Laptop"), result("b", 0.80, category="Laptop")]
    assert [r.summary_id for r in rerank("laptops", plain, 2)] == ["b", "a"]


def test_no_labels_in_the_question_means_plain_order():
    plain = [result("a", 0.9), result("b", 0.8), result("c", 0.7)]
    assert rerank("how are we doing?", plain, 2) == plain[:2]
