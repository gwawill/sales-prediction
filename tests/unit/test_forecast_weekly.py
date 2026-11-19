from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from sales_poc.datagen.history import generate_history
from sales_poc.datagen.problems import add_problems
from sales_poc.datagen.reference import build_reference_data
from sales_poc.etl.pipeline import run_etl
from sales_poc.forecast.weekly import week_start, weekly_sales


@pytest.fixture(scope="module")
def clean():
    """Saturday 7 March 2026 for 8 weeks and 2 days: partial weeks at both ends."""
    ref = build_reference_data(seed=42)
    messy = add_problems(generate_history(ref, seed=42, start=date(2026, 3, 7), days=58), seed=42)
    raw = pd.concat([messy.files[d] for d in sorted(messy.files)], ignore_index=True)
    return run_etl(raw, ref, "batch", ingested_at=datetime(2026, 11, 19, tzinfo=timezone.utc)).clean


@pytest.fixture(scope="module")
def weekly(clean):
    return weekly_sales(clean)


def test_weeks_start_on_monday():
    days = pd.Series([date(2026, 3, 9), date(2026, 3, 12), date(2026, 3, 15)])  # Mon, Thu, Sun
    assert list(week_start(days)) == [date(2026, 3, 9)] * 3


def test_only_complete_weeks_are_used_and_the_rest_are_listed(weekly):
    # 7 Mar (Sat) to 3 May (Sun): first week has 2 days, then 8 complete weeks.
    assert weekly.weeks_used[0] == date(2026, 3, 9)
    assert weekly.weeks_used[-1] == date(2026, 4, 27)
    assert len(weekly.weeks_used) == 8
    assert weekly.weeks_left_out == {date(2026, 3, 2): 2}
    assert all(w.weekday() == 0 for w in weekly.weeks_used)


def test_every_category_and_region_has_every_week(weekly, clean):
    n_series = clean["category"].nunique() * clean["region_id"].nunique()
    assert len(weekly.table) == n_series * len(weekly.weeks_used)
    assert not weekly.table.isna().any().any()


def test_totals_match_the_rows_they_came_from(weekly, clean):
    in_weeks = clean[week_start(clean["event_date"]).isin(weekly.weeks_used)]
    assert weekly.table["revenue"].sum() == pytest.approx(in_weeks["total_amount"].sum(), abs=0.05)
    assert weekly.table["orders"].sum() == len(in_weeks)
    assert weekly.table["units"].sum() == in_weeks["quantity"].sum()


def test_one_series_by_hand(weekly, clean):
    rows = clean[(clean["category"] == "Laptop") & (clean["region_id"] == "EMEA")
                 & (week_start(clean["event_date"]) == date(2026, 3, 16))]
    series = weekly.series("Laptop", "EMEA")

    assert series[date(2026, 3, 16)] == pytest.approx(rows["total_amount"].sum(), abs=0.01)
    assert weekly.series("Laptop", "EMEA", "orders")[date(2026, 3, 16)] == len(rows)


def test_a_week_with_no_sales_for_a_series_is_zero(clean):
    # Remove every APJ server sale in one complete week.
    week = week_start(clean["event_date"])
    gone = (clean["category"] == "Server") & (clean["region_id"] == "APJ") & (week == date(2026, 3, 23))
    weekly = weekly_sales(clean[~gone])

    assert weekly.series("Server", "APJ")[date(2026, 3, 23)] == 0
    assert weekly.series("Server", "APJ", "orders")[date(2026, 3, 23)] == 0


def test_a_gap_makes_weeks_partial(clean):
    days = pd.to_datetime(clean["event_date"])
    gappy = clean[~((days >= "2026-04-01") & (days <= "2026-04-03"))]  # Wed-Fri missing

    weekly = weekly_sales(gappy)

    assert date(2026, 3, 30) not in weekly.weeks_used
    assert weekly.weeks_left_out[date(2026, 3, 30)] == 4


def test_partial_weeks_can_be_included_on_request(clean):
    weekly = weekly_sales(clean, full_weeks_only=False)
    assert date(2026, 3, 2) in weekly.weeks_used
    assert weekly.weeks_left_out == {}


def test_no_rows(clean):
    weekly = weekly_sales(clean.head(0))
    assert weekly.weeks_used == [] and len(weekly.table) == 0
