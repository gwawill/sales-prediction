from collections import Counter
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sales_poc.datagen.live import LiveOrderMaker
from sales_poc.datagen.reference import build_reference_data
from sales_poc.lake.clean import list_clean_days, read_clean_day
from sales_poc.lake.runs import latest_report, read_quarantine
from sales_poc.lake.store import LocalStore
from sales_poc.pipelines.consumer import StreamCountError, build_batch, process_batch
from sales_poc.pipelines.producer import PROBLEM_HEADER, encode

NOW = datetime(2026, 11, 19, 3, 0, tzinfo=timezone.utc)


class FakeMessage:
    def __init__(self, partition, offset, key, value, headers):
        self._p, self._o, self._k, self._v, self._h = partition, offset, key, value, headers

    def partition(self):
        return self._p

    def offset(self):
        return self._o

    def key(self):
        return self._k

    def value(self):
        return self._v

    def headers(self):
        return self._h

    def error(self):
        return None


@pytest.fixture(scope="module")
def ref():
    return build_reference_data(seed=42)


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "bucket")


def messages(ref, n, seed=7, rates=None, start_offset=0):
    """n orders as Kafka messages, with the standard problem mix unless rates={}."""
    maker = LiveOrderMaker(ref, seed=seed, rates=rates, started_at=NOW)
    out = []
    for i in range(n):
        key, value, headers = encode(maker.next_order(NOW + timedelta(seconds=i)))
        out.append(FakeMessage(0, start_offset + i, key, value, headers))
    return out


def labels(msgs):
    """The check-only labels. Only the TEST reads them; the consumer never does."""
    return Counter(dict(m.headers())[PROBLEM_HEADER].decode() for m in msgs)


def all_clean(store):
    days = list_clean_days(store)
    return pd.concat([read_clean_day(store, d) for d in days]) if days else pd.DataFrame()


def test_a_batch_is_saved_cleaned_loaded_and_reported(store, ref):
    msgs = messages(ref, 2000)
    batch = build_batch(msgs)

    report = process_batch(store, batch, ref, now=NOW)

    seen = labels(msgs)
    c = report["counts"]
    assert c["rows_in"] == 2000
    assert c["duplicates_removed"] == seen["duplicate"]
    assert c["quarantined"] == seen["missing_required"] + seen["unknown_id"] + seen["negative_quantity"]
    assert c["clean_flagged"] == seen["missing_fillable"]
    assert c["rows_with_fixes"] == seen["bad_format"]
    assert report["stream"] == {"batch_id": "p0o0-1999", "messages": 2000, "order_rows": 2000,
                                "healthchecks": 0, "unreadable": 0}
    assert report["inputs"] == ["raw/stream/date=2026-11-19/p0o0-1999.jsonl"]
    assert store.get(report["inputs"][0]).count(b"\n") == 2000
    assert latest_report(store, source="stream") == report
    assert len(read_quarantine(store, report["run_id"])) == c["quarantined"]


def test_clean_rows_are_marked_stream_and_late_orders_land_on_their_real_day(store, ref):
    msgs = messages(ref, 2000)
    process_batch(store, build_batch(msgs), ref, now=NOW)

    stored = all_clean(store)
    assert (stored["source"] == "stream").all()
    assert stored["transaction_id"].is_unique
    late = labels(msgs)["late"]
    assert late > 0
    older = stored[stored["event_date"] < NOW.date()]
    assert len(older) == late  # every other order is dated today; each order has one problem


def test_the_same_batch_twice_changes_nothing(store, ref):
    batch = build_batch(messages(ref, 300))
    process_batch(store, batch, ref, now=NOW)
    clean_before = {k: store.get(k) for k in store.list("clean/")}
    raw_before = {k: store.get(k) for k in store.list("raw/")}

    again = process_batch(store, batch, ref, now=NOW + timedelta(minutes=5))

    assert {k: store.get(k) for k in store.list("clean/")} == clean_before
    assert {k: store.get(k) for k in store.list("raw/")} == raw_before
    assert again["lake"]["rows_added"] == 0


def test_batch_and_stream_copies_of_one_order_are_stored_once(store, ref):
    msgs = messages(ref, 50, rates={})
    process_batch(store, build_batch(msgs), ref, now=NOW)
    # The same 50 orders arrive again as a later batch (different Kafka positions).
    replay = [FakeMessage(1, i, m.key(), m.value(), m.headers()) for i, m in enumerate(msgs)]

    report = process_batch(store, build_batch(replay), ref, now=NOW)

    assert report["lake"]["rows_unchanged"] == 50
    assert len(all_clean(store)) == 50


def test_healthchecks_and_unreadable_messages_are_counted_in_the_report(store, ref):
    msgs = [
        FakeMessage(1, 0, b"healthcheck", b"sales-poc-check-1", []),
        FakeMessage(1, 1, b"x", b"not json", []),
        *messages(ref, 3, rates={}, start_offset=2),
    ]

    report = process_batch(store, build_batch(msgs), ref, now=NOW)

    assert report["stream"]["healthchecks"] == 1
    assert report["stream"]["unreadable"] == 1
    assert report["counts"]["rows_in"] == 3
    assert store.get(report["inputs"][0]).count(b"\n") == 5  # raw keeps all of them


def test_an_empty_batch_writes_nothing(store, ref):
    assert process_batch(store, build_batch([]), ref, now=NOW) is None
    assert store.list() == []


def test_counts_that_do_not_add_up_are_refused(store, ref):
    batch = build_batch(messages(ref, 3, rates={}))
    broken = type(batch)(batch.batch_id, batch.values, batch.rows.head(2), 0, 0, batch.next_offsets)

    with pytest.raises(StreamCountError):
        process_batch(store, broken, ref, now=NOW)
