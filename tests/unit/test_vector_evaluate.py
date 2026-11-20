from types import SimpleNamespace as NS

from sales_poc.vectors.evaluate import QUESTIONS, Outcome, evaluate, report


def test_rank_matches_an_id_or_the_start_of_one():
    assert Outcome("q", "catreg:Laptop:EMEA:", ["catreg:Laptop:EMEA:2026-03", "x"]).rank == 1
    assert Outcome("q", "region:NA:2026-01", ["a", "b", "region:NA:2026-01"]).rank == 3
    assert Outcome("q", "customer:C00012", ["customer:C00120"]).rank is None


def test_evaluate_and_report_count_hits_and_list_the_misses():
    answers = {"first": ["good:1"], "second": ["bad", "good:2"], "missed": ["bad"]}
    outcomes = evaluate(lambda q, k: [NS(summary_id=i) for i in answers[q]][:k],
                        [("first", "good:"), ("second", "good:"), ("missed", "good:")])

    text = report(outcomes)

    assert text.splitlines()[0] == "Found in the top 3: 2 of 3. Ranked first: 1 of 3."
    assert "ranked 2" in text and "MISSED" in text and "'first'" not in text


def test_the_question_set_is_well_formed():
    assert len(QUESTIONS) == 20
    assert len({q for q, _ in QUESTIONS}) == 20
    assert all(expected.split(":")[0] in ("catreg", "region", "customer") for _, expected in QUESTIONS)
