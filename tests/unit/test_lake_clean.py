from datetime import date, datetime, timezone

import pandas as pd
import pytest

from sales_poc.datagen.history import generate_history
from sales_poc.datagen.problems import add_problems
from sales_poc.datagen.reference import build_reference_data
from sales_poc.etl.pipeline import OUTPUT_COLUMNS, run_etl
from sales_poc.lake.clean import (
    LAKE_SCHEMA,
    clean_key,
    from_parquet_bytes,
    list_clean_days,
    read_clean_day,
    to_parquet_bytes,
    write_clean,
)
from sales_poc.lake.store import LocalStore

FIRST = datetime(2026, 11, 18, 9, 0, tzinfo=timezone.utc)
LATER = datetime(2026, 11, 18, 10, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def ref():
    return build_reference_data(seed=42)


@pytest.fixture(scope="module")
def raw(ref):
    messy = add_problems(generate_history(ref, seed=42, start=date(2026, 3, 1), days=10), seed=42)
    return pd.concat([messy.files[d] for d in sorted(messy.files)], ignore_index=True)


@pytest.fixture(scope="module")
def clean(ref, raw):
    return run_etl(raw, ref, source="batch", ingested_at=FIRST).clean


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "bucket")


def file_bytes(store):
    return {k: store.get(k) for k in store.list("clean/")}


def test_key_layout():
    assert clean_key(date(2026, 3, 1)) == "clean/transactions/event_date=2026-03-01/data.parquet"


def test_parquet_round_trip_keeps_every_type(clean):
    back = from_parquet_bytes(to_parquet_bytes(clean))

    assert dict(back.dtypes.astype(str)) == LAKE_SCHEMA
    pd.testing.assert_frame_equal(back, clean[OUTPUT_COLUMNS].astype(LAKE_SCHEMA).reset_index(drop=True))


def test_first_write_makes_one_file_per_day(store, clean):
    results = write_clean(store, clean)

    days = sorted(clean["event_date"].unique())
    assert list_clean_days(store) == days
    assert [r.day for r in results] == days
    assert sum(r.added for r in results) == len(clean)
    assert all(r.written for r in results)


def test_writing_the_same_rows_again_changes_nothing(store, ref, raw, clean):
    write_clean(store, clean)
    before = file_bytes(store)

    # A real re-run: same input, new ingested_at.
    rerun = run_etl(raw, ref, source="batch", ingested_at=LATER).clean
    results = write_clean(store, rerun)

    assert file_bytes(store) == before
    assert not any(r.written for r in results)
    assert sum(r.unchanged for r in results) == len(clean)


def test_same_order_from_batch_and_stream_is_stored_once(store, clean):
    write_clean(store, clean)
    as_stream = clean.head(20).copy()
    as_stream["source"] = "stream"
    as_stream["ingested_at"] = pd.Timestamp(LATER)

    results = write_clean(store, as_stream)

    assert sum(r.unchanged for r in results) == 20
    stored = pd.concat([read_clean_day(store, d) for d in list_clean_days(store)])
    assert len(stored) == len(clean)
    assert stored["transaction_id"].is_unique
    assert (stored["source"] == "batch").all()  # the original copy is kept


def test_a_changed_copy_replaces_the_old_one(store, clean):
    write_clean(store, clean)
    corrected = clean.head(1).copy()
    corrected["quantity"] = corrected["quantity"] + 1
    day = corrected["event_date"].iloc[0]

    [result] = write_clean(store, corrected)

    assert (result.replaced, result.added, result.written) == (1, 0, True)
    stored = read_clean_day(store, day).set_index("transaction_id")
    tx = corrected["transaction_id"].iloc[0]
    assert stored.loc[tx, "quantity"] == corrected["quantity"].iloc[0]
    assert len(stored) == (clean["event_date"] == day).sum()


def test_new_rows_for_an_existing_day_are_added_in_time_order(store, clean):
    day = clean["event_date"].iloc[0]
    day_rows = clean[clean["event_date"] == day]
    write_clean(store, day_rows.iloc[::2])

    [result] = write_clean(store, day_rows.iloc[1::2])

    assert result.added == len(day_rows.iloc[1::2])
    stored = read_clean_day(store, day)
    assert len(stored) == len(day_rows)
    assert stored["event_time"].is_monotonic_increasing


def test_late_rows_go_into_the_older_day(store, clean):
    """A row whose event_date is older than the others lands in that older day's file."""
    write_clean(store, clean)
    old_day = min(clean["event_date"])
    late = clean[clean["event_date"] == old_day].head(1).copy()
    late["transaction_id"] = "TX-LATE-1"
    late["ingested_at"] = pd.Timestamp(LATER)

    [result] = write_clean(store, late)

    assert result.day == old_day and result.added == 1
    assert "TX-LATE-1" in set(read_clean_day(store, old_day)["transaction_id"])


def test_parallel_writing_gives_the_same_result(tmp_path, clean):
    one, many = LocalStore(tmp_path / "one"), LocalStore(tmp_path / "many")

    serial = write_clean(one, clean)
    parallel = write_clean(many, clean, workers=8)

    assert parallel == serial  # same summaries, oldest day first
    assert file_bytes(many) == file_bytes(one)


def test_reading_a_missing_day_gives_an_empty_typed_table(store):
    empty = read_clean_day(store, date(2020, 1, 1))
    assert len(empty) == 0
    assert list(empty.columns) == OUTPUT_COLUMNS


def test_duplicate_ids_in_the_input_are_refused(store, clean):
    with pytest.raises(ValueError, match="unique"):
        write_clean(store, pd.concat([clean.head(2), clean.head(1)]))


# --- Two writers on the same day (batch-load and stream-consume at once) ---------------------


class OtherWriterInBetween:
    """A store where another writer saves the day file right after we read it, `clashes` times."""

    def __init__(self, inner, other_rows, clashes=1):
        self.inner, self.other_rows, self.clashes = inner, other_rows, clashes
        self.reads = 0

    def get_with_tag(self, key):
        found = self.inner.get_with_tag(key)
        self.reads += 1
        if self.reads <= self.clashes:
            write_clean(self.inner, self.other_rows.assign(quantity=self.other_rows["quantity"] + self.reads))
        return found

    def put_if_unchanged(self, key, data, tag):
        return self.inner.put_if_unchanged(key, data, tag)


@pytest.fixture
def no_wait(monkeypatch):
    import sales_poc.lake.clean as lake_clean
    monkeypatch.setattr(lake_clean, "RETRY_SECONDS", 0)


def one_day(clean):
    day = clean["event_date"].iloc[0]
    rows = clean[clean["event_date"] == day]
    return day, rows.iloc[:5], rows.iloc[5:10]


def test_a_writer_that_saves_in_between_does_not_lose_its_rows(store, clean, no_wait):
    day, ours, theirs = one_day(clean)
    clashing = OtherWriterInBetween(store, theirs)

    [result] = write_clean(clashing, ours)

    stored = set(read_clean_day(store, day)["transaction_id"])
    assert set(ours["transaction_id"]) | set(theirs["transaction_id"]) <= stored
    assert result.added == len(ours) and result.total == 10
    assert clashing.reads == 2          # the clash made it read the day again and redo the merge


def test_it_gives_up_after_repeated_clashes_without_overwriting(store, clean, no_wait):
    from sales_poc.lake.clean import MERGE_ATTEMPTS, ConcurrentWriteError

    day, ours, theirs = one_day(clean)
    clashing = OtherWriterInBetween(store, theirs, clashes=MERGE_ATTEMPTS)

    with pytest.raises(ConcurrentWriteError, match="run again"):
        write_clean(clashing, ours)

    stored = read_clean_day(store, day)
    assert set(stored["transaction_id"]) == set(theirs["transaction_id"])   # theirs kept, intact
    assert clashing.reads == MERGE_ATTEMPTS


def test_several_writers_at_once_keep_every_row(store, clean):
    """4 writers (the project has 2: batch and stream), each adding 3 orders to the same day,
    with the real wait between tries."""
    from concurrent.futures import ThreadPoolExecutor

    day = clean["event_date"].iloc[0]
    rows = clean[clean["event_date"] == day].head(12)

    def writer(w):
        for i in range(w, 12, 4):
            write_clean(store, rows.iloc[[i]])

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(writer, range(4)))

    assert set(read_clean_day(store, day)["transaction_id"]) == set(rows["transaction_id"])
