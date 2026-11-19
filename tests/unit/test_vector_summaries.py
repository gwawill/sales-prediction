from collections import Counter
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from sales_poc.datagen.history import generate_history
from sales_poc.datagen.problems import add_problems
from sales_poc.datagen.reference import build_reference_data
from sales_poc.etl.pipeline import run_etl
from sales_poc.vectors.summaries import build_summaries, money


@pytest.fixture(scope="module")
def clean():
    """1 Jan to 15 Apr 2026: three full months and one partial one."""
    ref = build_reference_data(seed=42)
    messy = add_problems(generate_history(ref, seed=42, start=date(2026, 1, 1), days=105), seed=42)
    raw = pd.concat([messy.files[d] for d in sorted(messy.files)], ignore_index=True)
    return run_etl(raw, ref, "batch", ingested_at=datetime(2026, 11, 19, tzinfo=timezone.utc)).clean


@pytest.fixture(scope="module")
def summaries(clean):
    return build_summaries(clean)


def by_id(summaries):
    return {s.summary_id: s for s in summaries}


def test_money_format():
    assert money(1_234_567) == "$1.23M"
    assert money(45_600) == "$45.6K"
    assert money(812.4) == "$812"


def test_three_kinds_with_unique_stable_ids(summaries, clean):
    kinds = Counter(s.metadata["kind"] for s in summaries)
    assert set(kinds) == {"catreg", "region", "customer"}
    assert kinds["region"] == 4 * 4  # 4 regions x 4 months
    assert kinds["customer"] == clean["customer_id"].nunique()
    ids = [s.summary_id for s in summaries]
    assert len(ids) == len(set(ids))


def test_same_data_gives_identical_summaries(clean, summaries):
    assert build_summaries(clean) == summaries


def test_category_region_month_numbers_match_the_data(summaries, clean):
    s = by_id(summaries)["catreg:Laptop:EMEA:2026-03"]
    month = clean[(clean["category"] == "Laptop") & (clean["region_id"] == "EMEA")
                  & (pd.to_datetime(clean["event_date"]).dt.to_period("M") == "2026-03")]

    assert s.text.startswith("Laptop sales in Europe, Middle East and Africa (EMEA), March 2026:")
    assert f"{len(month)} orders" in s.text
    assert money(month["total_amount"].sum()) in s.text
    assert "than February 2026" in s.text
    assert s.metadata == {"kind": "catreg", "category": "Laptop", "region_id": "EMEA", "month": "2026-03"}


def test_the_first_month_has_nothing_to_compare_with(summaries):
    assert "No data for December 2025" in by_id(summaries)["region:NA:2026-01"].text


def test_partial_months_say_so_and_compare_per_day(summaries):
    april = by_id(summaries)["region:APJ:2026-04"].text
    march = by_id(summaries)["region:APJ:2026-03"].text
    assert "(partial month: data for 15 of 30 days)" in april
    assert "Average revenue per day with data was" in april and "than in March 2026" in april
    assert "partial month" not in march
    assert "Revenue was" in march  # full months compare totals


def test_a_month_with_gaps_counts_only_days_with_data(clean):
    # Keep 1 March and 17-19 March only, like a month where most days are missing.
    days = pd.to_datetime(clean["event_date"])
    gappy = clean[(days < "2026-03-01") | days.isin(pd.to_datetime(["2026-03-01", "2026-03-17",
                                                                   "2026-03-18", "2026-03-19"]))]
    text = by_id(build_summaries(gappy))["region:EMEA:2026-03"].text

    assert "(partial month: data for 4 of 31 days)" in text
    # The comparison is per day with data, worked out here by hand.
    emea, d = gappy[gappy["region_id"] == "EMEA"], pd.to_datetime(gappy[gappy["region_id"] == "EMEA"]["event_date"])
    feb_per_day = emea[(d >= "2026-02-01") & (d < "2026-03-01")]["total_amount"].sum() / 28
    mar_per_day = emea[d >= "2026-03-01"]["total_amount"].sum() / 4
    expected = round(abs(mar_per_day - feb_per_day) / feb_per_day * 100)
    direction = "higher" if mar_per_day >= feb_per_day else "lower"
    assert f"Average revenue per day with data was {expected}% {direction}" in text
    # Comparing totals would have reported a fake collapse; that wording must not appear.
    assert "Revenue was" not in text


def test_the_march_apj_desktop_drop_is_visible(summaries):
    text = by_id(summaries)["catreg:Desktop:APJ:2026-03"].text
    assert "% less than February 2026" in text


def test_customer_summary(summaries, clean):
    customer_id = clean["customer_id"].iloc[0]
    s = by_id(summaries)[f"customer:{customer_id}"]
    rows = clean[clean["customer_id"] == customer_id]

    assert s.text.startswith(f"Customer {customer_id}, ")
    assert f"{len(rows)} orders" in s.text
    assert "Preferred channel:" in s.text and "Last order:" in s.text
    assert s.metadata["segment"] == rows["segment"].iloc[0]


def test_summaries_are_short(summaries):
    assert max(len(s.text) for s in summaries) < 600


def test_no_data_means_no_summaries(clean):
    assert build_summaries(clean.head(0)) == []
