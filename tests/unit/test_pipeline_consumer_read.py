import json
from datetime import datetime, timezone

import pytest

from sales_poc.datagen.history import COLUMNS
from sales_poc.datagen.live import LiveOrderMaker
from sales_poc.datagen.reference import build_reference_data
from sales_poc.etl.pipeline import run_etl
from sales_poc.pipelines.consumer import build_batch, collect_messages
from sales_poc.pipelines.producer import encode


class FakeMessage:
    def __init__(self, partition, offset, key, value, error=None):
        self._p, self._o, self._k, self._v, self._e = partition, offset, key, value, error

    def partition(self):
        return self._p

    def offset(self):
        return self._o

    def key(self):
        return self._k

    def value(self):
        return self._v

    def error(self):
        return self._e


class FakeConsumer:
    """Hands out queued messages; each poll moves a fake clock forward."""

    def __init__(self, messages, clock):
        self.queue, self.clock = list(messages), clock

    def poll(self, timeout):
        self.clock.now += 0.01 if self.queue else timeout
        return self.queue.pop(0) if self.queue else None


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def order_messages(n, partition=0, start=0):
    ref = build_reference_data(seed=42)
    maker = LiveOrderMaker(ref, seed=3, rates={}, started_at=datetime(2026, 11, 18, tzinfo=timezone.utc))
    out = []
    for i in range(n):
        key, value, _ = encode(maker.next_order())
        out.append(FakeMessage(partition, start + i, key, value))
    return out


def test_batch_ends_at_the_size_limit():
    clock = Clock()
    consumer = FakeConsumer(order_messages(30), clock)

    got = collect_messages(consumer, max_messages=10, max_seconds=10, clock=clock)

    assert len(got) == 10
    assert len(consumer.queue) == 20


def test_batch_ends_at_the_time_limit_when_messages_stop():
    clock = Clock()
    consumer = FakeConsumer(order_messages(3), clock)

    got = collect_messages(consumer, max_messages=100, max_seconds=5, clock=clock)

    assert len(got) == 3
    assert clock.now == pytest.approx(5, abs=1.1)


def test_empty_topic_gives_an_empty_batch():
    clock = Clock()
    batch = build_batch(collect_messages(FakeConsumer([], clock), 100, 2, clock=clock))

    assert batch.size == 0 and batch.batch_id == ""
    assert list(batch.rows.columns) == COLUMNS and len(batch.rows) == 0


def test_kafka_errors_are_raised():
    clock = Clock()
    consumer = FakeConsumer([FakeMessage(0, 0, None, b"", error="broker down")], clock)

    with pytest.raises(RuntimeError, match="broker down"):
        collect_messages(consumer, 10, 5, clock=clock)


def test_orders_become_rows_with_the_daily_file_columns():
    messages = order_messages(5)
    batch = build_batch(messages)

    assert list(batch.rows.columns) == COLUMNS
    assert len(batch.rows) == 5
    assert list(batch.rows["transaction_id"]) == [json.loads(m.value())["transaction_id"] for m in messages]
    assert batch.values == [m.value() for m in messages]


def test_healthchecks_are_skipped_and_counted():
    messages = [FakeMessage(1, 0, b"healthcheck", b"sales-poc-check-abc")] + order_messages(2, partition=1, start=1)
    batch = build_batch(messages)

    assert batch.healthchecks == 1
    assert len(batch.rows) == 2
    assert batch.size == 3  # still kept for the raw area


def test_unreadable_messages_are_counted_not_fatal():
    messages = [
        FakeMessage(0, 0, b"x", b"not json"),
        FakeMessage(0, 1, b"x", b"[1, 2]"),
        FakeMessage(0, 2, b"x", b"\xff\xfe"),
        *order_messages(1, start=3),
    ]
    batch = build_batch(messages)

    assert batch.unreadable == 3
    assert len(batch.rows) == 1


def test_missing_and_number_fields_become_text_or_none():
    value = json.dumps({"transaction_id": "LV-1", "quantity": 3, "unit_price": None, "extra": "ignored"}).encode()
    batch = build_batch([FakeMessage(0, 0, b"LV-1", value)])
    row = batch.rows.iloc[0]

    assert row["quantity"] == "3"
    assert row["unit_price"] is None and row["customer_id"] is None
    assert "extra" not in batch.rows.columns


def test_batch_id_and_commit_positions_come_from_kafka_positions():
    messages = order_messages(3, partition=0, start=10) + order_messages(2, partition=2, start=4)
    batch = build_batch(messages)

    assert batch.batch_id == "p0o10-12_p2o4-5"
    assert batch.next_offsets == {0: 13, 2: 6}
    assert [(tp.partition, tp.offset) for tp in batch.commit_positions("orders")] == [(0, 13), (2, 6)]


def test_same_messages_give_the_same_batch():
    messages = order_messages(4)
    a, b = build_batch(messages), build_batch(messages)
    assert a.batch_id == b.batch_id and a.values == b.values


def test_rows_go_straight_into_the_shared_etl():
    batch = build_batch(order_messages(20))
    result = run_etl(batch.rows, build_reference_data(seed=42), source="stream")

    assert result.counts.clean == 20
    assert (result.clean["source"] == "stream").all()
