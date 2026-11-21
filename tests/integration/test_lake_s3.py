"""The lake code against the real S3 bucket, inside a throwaway test folder."""

import uuid
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from sales_poc.datagen.history import generate_history
from sales_poc.datagen.problems import add_problems
from sales_poc.datagen.reference import build_reference_data
from sales_poc.etl.pipeline import run_etl
from sales_poc.lake.clean import list_clean_days, read_clean_day, write_clean
from sales_poc.lake.raw import RawConflictError, save_raw_batch_file, save_raw_stream_batch
from sales_poc.lake.runs import (
    build_report,
    latest_report,
    new_run_id,
    read_quarantine,
    write_quarantine,
    write_report,
)
from sales_poc.lake.store import s3_store


@pytest.fixture
def store():
    """Everything under _test/<random>/ in the real bucket, deleted afterwards."""
    s = s3_store(prefix=f"_test/{uuid.uuid4().hex}/")
    yield s
    for key in s.list():
        s.delete(key)
    assert s.list() == []


def test_s3_store_basics(store):
    store.put("a/b.txt", b"hello")
    assert store.exists("a/b.txt")
    assert not store.exists("a/missing.txt")
    assert store.get("a/b.txt") == b"hello"
    assert store.list("a/") == ["a/b.txt"]
    with pytest.raises(KeyError):
        store.get("a/missing.txt")


def test_s3_conditional_write(store):
    """S3 really refuses a save when the file changed since it was read (no lost updates)."""
    assert store.get_with_tag("c.txt") == (None, None)
    assert store.put_if_unchanged("c.txt", b"one", None)
    assert not store.put_if_unchanged("c.txt", b"two", None)        # exists now

    data, tag = store.get_with_tag("c.txt")
    assert data == b"one"
    store.put("c.txt", b"someone else")
    assert not store.put_if_unchanged("c.txt", b"two", tag)         # stale version refused
    assert store.get("c.txt") == b"someone else"

    _, tag = store.get_with_tag("c.txt")
    assert store.put_if_unchanged("c.txt", b"two", tag)
    assert store.get("c.txt") == b"two"


def test_raw_batch_file_round_trip_on_s3(store, tmp_path):
    path = tmp_path / "sales_2026-03-01.csv"
    path.write_bytes(b"transaction_id\nTX-1\n")

    key, written = save_raw_batch_file(store, path)
    assert written and store.get(key) == path.read_bytes()
    assert save_raw_batch_file(store, path) == (key, False)

    path.write_bytes(b"changed\n")
    with pytest.raises(RawConflictError):
        save_raw_batch_file(store, path)


def test_raw_stream_batch_on_s3(store):
    key, written = save_raw_stream_batch(store, [b'{"a":1}', b'{"a":2}'], date(2026, 11, 18), "b1")
    assert written
    assert store.get(key) == b'{"a":1}\n{"a":2}\n'


def test_clean_rows_written_twice_leave_the_files_unchanged(store):
    ref = build_reference_data(seed=42)
    messy = add_problems(generate_history(ref, seed=42, start=date(2026, 3, 1), days=3), seed=42)
    raw = pd.concat([messy.files[d] for d in sorted(messy.files)], ignore_index=True)
    first = run_etl(raw, ref, "batch", ingested_at=datetime(2026, 11, 18, 9, tzinfo=timezone.utc))
    again = run_etl(raw, ref, "batch", ingested_at=datetime(2026, 11, 18, 10, tzinfo=timezone.utc))

    write_clean(store, first.clean)
    before = {k: store.get(k) for k in store.list("clean/")}
    results = write_clean(store, again.clean)
    after = {k: store.get(k) for k in store.list("clean/")}

    assert len(before) >= 3
    assert after == before
    assert not any(r.written for r in results)
    stored = pd.concat([read_clean_day(store, d) for d in list_clean_days(store)])
    assert len(stored) == len(first.clean)
    assert stored["transaction_id"].is_unique


def test_quarantine_and_report_on_s3(store):
    ref = build_reference_data(seed=42)
    messy = add_problems(generate_history(ref, seed=42, start=date(2026, 3, 1), days=20), seed=42)
    raw = pd.concat([messy.files[d] for d in sorted(messy.files)], ignore_index=True)
    started = datetime(2026, 11, 18, 9, tzinfo=timezone.utc)
    result = run_etl(raw, ref, "batch", ingested_at=started)
    run_id = new_run_id("batch", started)

    q_key = write_quarantine(store, run_id, result.quarantine)
    report = build_report(run_id, "batch", started, started, [], result.counts, [], q_key)
    write_report(store, report)

    assert len(read_quarantine(store, run_id)) == result.counts.quarantined > 0
    assert latest_report(store) == report
