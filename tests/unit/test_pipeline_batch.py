from datetime import date, datetime, timezone

import pandas as pd
import pytest

from sales_poc.datagen.cli import generate
from sales_poc.datagen.reference import load_reference_data
from sales_poc.lake.clean import list_clean_days, read_clean_day
from sales_poc.lake.runs import latest_report, read_quarantine
from sales_poc.lake.store import LocalStore
from sales_poc.pipelines.batch import find_daily_files, read_daily_file, run_batch

NOW = datetime(2026, 11, 18, 9, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """A real generate-data output: 15 days starting 1 March 2026."""
    out = tmp_path_factory.mktemp("data")
    summary = generate(out, seed=42, start=date(2026, 3, 1), days=15)
    return out, summary


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "bucket")


def run(store, data, now=NOW):
    out, _ = data
    return run_batch(store, find_daily_files(out / "incoming"), load_reference_data(out / "reference"), now=now)


def test_files_are_found_oldest_first(data):
    out, _ = data
    names = [p.name for p in find_daily_files(out / "incoming")]
    assert names[0] == "sales_2026-03-01.csv" and names[-1] == "sales_2026-03-15.csv"
    assert names == sorted(names) and len(names) == 15


def test_reading_keeps_text_exactly(data):
    out, _ = data
    rows = read_daily_file(find_daily_files(out / "incoming")[0])
    assert all(str(t) in ("str", "object") for t in rows.dtypes)
    assert rows["quantity"].dropna().map(lambda v: isinstance(v, str)).all()


def test_one_run_loads_everything_and_the_report_adds_up(store, data):
    _, summary = data
    result = run(store, data)

    report = result.report
    c = report["counts"]
    assert c["rows_in"] == summary["rows_in_files"]
    assert c["clean"] + c["duplicates_removed"] + c["quarantined"] == c["rows_in"]
    assert c["duplicates_removed"] == summary["problems_by_kind"]["duplicate"]
    assert report["lake"]["rows_added"] == c["clean"]
    assert len(report["inputs"]) == 15
    assert latest_report(store) == report


def test_raw_clean_quarantine_and_report_all_land_in_the_store(store, data):
    result = run(store, data)

    assert len(store.list("raw/batch/")) == 15
    days = list_clean_days(store)
    assert days[0] == date(2026, 3, 1)
    stored = pd.concat([read_clean_day(store, d) for d in days])
    assert len(stored) == result.report["counts"]["clean"]
    assert stored["transaction_id"].is_unique
    assert len(read_quarantine(store, result.run_id)) == result.report["counts"]["quarantined"]


def test_north_america_customers_keep_their_region(store, data):
    run(store, data)
    stored = pd.concat([read_clean_day(store, d) for d in list_clean_days(store)])
    assert "NA" in set(stored["region_id"])
    assert stored["region_id"].notna().all()


def test_a_second_run_changes_no_clean_data(store, data):
    run(store, data)
    before = {k: store.get(k) for k in store.list("clean/")}

    second = run(store, data, now=datetime(2026, 11, 18, 10, 0, tzinfo=timezone.utc))

    assert {k: store.get(k) for k in store.list("clean/")} == before
    assert second.report["lake"]["rows_added"] == 0
    assert second.report["lake"]["days_written"] == []


def test_no_files_means_no_run(store, data):
    out, _ = data
    assert run_batch(store, [], load_reference_data(out / "reference"), now=NOW) is None
    assert store.list() == []
