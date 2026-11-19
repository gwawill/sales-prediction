import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from sales_poc.datagen.history import COLUMNS, generate_history
from sales_poc.datagen.problems import add_problems
from sales_poc.datagen.reference import build_reference_data
from sales_poc.etl.pipeline import run_etl
from sales_poc.lake.clean import write_clean
from sales_poc.lake.runs import (
    build_report,
    counts_from_report,
    latest_report,
    list_run_ids,
    new_run_id,
    quarantine_key,
    read_quarantine,
    read_report,
    report_key,
    write_quarantine,
    write_report,
)
from sales_poc.lake.store import LocalStore

STARTED = datetime(2026, 11, 18, 9, 30, 0, tzinfo=timezone.utc)
FINISHED = datetime(2026, 11, 18, 9, 30, 4, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def etl():
    ref = build_reference_data(seed=42)
    messy = add_problems(generate_history(ref, seed=42, start=date(2026, 3, 1), days=20), seed=42)
    raw = pd.concat([messy.files[d] for d in sorted(messy.files)], ignore_index=True)
    return run_etl(raw, ref, "batch", ingested_at=STARTED)


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "bucket")


def test_run_ids_sort_by_time_and_name_their_source():
    run_id = new_run_id("batch", STARTED)
    assert run_id.startswith("20261118T093000Z-batch-")
    assert new_run_id("stream", datetime(2026, 11, 18, 9, 31, tzinfo=timezone.utc)) > run_id


def test_key_layout():
    run_id = "20261118T093000Z-batch-a1b2c3"
    assert quarantine_key(run_id) == "quarantine/run=20261118T093000Z-batch-a1b2c3/rows.parquet"
    assert report_key(run_id) == "reports/run=20261118T093000Z-batch-a1b2c3.json"
    with pytest.raises(ValueError):
        report_key("../oops")


def test_quarantine_round_trip_keeps_raw_text_and_reasons(store, etl):
    run_id = new_run_id("batch", STARTED)

    key = write_quarantine(store, run_id, etl.quarantine)
    back = read_quarantine(store, run_id)

    assert key == quarantine_key(run_id)
    assert len(back) == etl.counts.quarantined
    assert set(COLUMNS) | {"reject_reasons", "source", "ingested_at"} <= set(back.columns)
    negative = back[back["reject_reasons"] == "invalid:quantity"]
    assert len(negative) and negative["quantity"].str.startswith("-").all()


def test_no_rejected_rows_means_no_quarantine_file(store, etl):
    assert write_quarantine(store, new_run_id("batch", STARTED), etl.quarantine.head(0)) is None
    assert store.list("quarantine/") == []


def test_report_has_counts_lake_changes_and_inputs(store, etl):
    run_id = new_run_id("batch", STARTED)
    merges = write_clean(store, etl.clean)
    q_key = write_quarantine(store, run_id, etl.quarantine)

    report = build_report(run_id, "batch", STARTED, FINISHED, ["raw/batch/a.csv"], etl.counts, merges, q_key)
    key = write_report(store, report)

    stored = json.loads(store.get(key))
    assert stored == read_report(store, run_id) == report
    assert stored["counts"]["rows_in"] == etl.counts.rows_in
    assert stored["lake"]["rows_added"] == etl.counts.clean
    assert stored["lake"]["days_touched"] == len(merges) == len(stored["lake"]["days_written"])
    assert stored["quarantine_key"] == q_key
    assert stored["started_at"] == "2026-11-18T09:30:00+00:00"
    assert counts_from_report(stored) == etl.counts


def test_latest_report_is_the_newest_one(store, etl):
    ids = [
        new_run_id("batch", datetime(2026, 11, 18, 9, tzinfo=timezone.utc)),
        new_run_id("stream", datetime(2026, 11, 18, 10, tzinfo=timezone.utc)),
        new_run_id("batch", datetime(2026, 11, 18, 8, tzinfo=timezone.utc)),
    ]
    for run_id in ids:
        source = "stream" if "-stream-" in run_id else "batch"
        write_report(store, build_report(run_id, source, STARTED, FINISHED, [], etl.counts, [], None))

    assert list_run_ids(store) == sorted(ids)
    assert latest_report(store)["run_id"] == ids[1]
    assert latest_report(store, source="batch")["run_id"] == ids[0]


def test_no_reports_yet(store):
    assert latest_report(store) is None
