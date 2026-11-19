from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sales_poc.datagen.live import LiveOrderMaker
from sales_poc.datagen.reference import build_reference_data
from sales_poc.lake.clean import list_clean_days, read_clean_day
from sales_poc.lake.store import LocalStore
from sales_poc.pipelines.consumer import TooManyFailures, consume_loop, process_batch
from sales_poc.pipelines.producer import encode

NOW = datetime(2026, 11, 19, 3, 0, tzinfo=timezone.utc)
TOPIC = "orders"


class Message:
    def __init__(self, offset, key, value):
        self._o, self._k, self._v = offset, key, value

    def partition(self):
        return 0

    def offset(self):
        return self._o

    def key(self):
        return self._k

    def value(self):
        return self._v

    def error(self):
        return None


class FakeKafka:
    """One partition. Remembers the committed position like a real consumer group."""

    def __init__(self, messages):
        self.messages = messages
        self.committed = 0          # what the group has confirmed
        self.position = 0
        self.commits = []

    def new_consumer(self):
        """A restarted consumer starts at the committed position."""
        self.position = self.committed
        return self

    def poll(self, timeout):
        if self.position < len(self.messages):
            self.position += 1
            return self.messages[self.position - 1]
        return None

    def seek(self, tp):
        self.position = tp.offset

    def commit(self, offsets, asynchronous):
        assert asynchronous is False  # we wait for Kafka to agree
        [tp] = offsets
        self.committed = tp.offset
        self.commits.append(tp.offset)


@pytest.fixture(scope="module")
def ref():
    return build_reference_data(seed=42)


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "bucket")


def orders(ref, n):
    maker = LiveOrderMaker(ref, seed=11, started_at=NOW)  # standard problem mix
    return [Message(i, *encode(maker.next_order(NOW + timedelta(seconds=i)))[:2]) for i in range(n)]


def loop(kafka, process, **kw):
    kw.setdefault("idle_batches", 1)
    return consume_loop(kafka, process, TOPIC, batch_size=50, batch_seconds=0.05, sleep=lambda s: None, **kw)


def lake(store):
    days = list_clean_days(store)
    return pd.concat([read_clean_day(store, d) for d in days]) if days else pd.DataFrame()


def test_commits_happen_only_after_each_batch_is_stored(store, ref):
    kafka = FakeKafka(orders(ref, 120))
    stored_before_commit = []

    def process(batch):
        process_batch(store, batch, ref, now=NOW)
        stored_before_commit.append(kafka.committed)  # nothing for this batch committed yet

    stats = loop(kafka.new_consumer(), process)

    assert stats.batches == 3 and stats.messages == 120
    assert kafka.commits == [50, 100, 120]
    assert stored_before_commit == [0, 50, 100]


def test_a_failed_store_is_not_committed_and_is_retried(store, ref):
    kafka = FakeKafka(orders(ref, 100))
    attempts = []

    def flaky(batch):
        attempts.append(batch.batch_id)
        if len(attempts) == 2:
            raise ConnectionError("S3 went away")
        process_batch(store, batch, ref, now=NOW)

    stats = loop(kafka.new_consumer(), flaky)

    assert attempts == ["p0o0-49", "p0o50-99", "p0o50-99"]  # same batch read again after a rewind
    assert kafka.commits == [50, 100]
    assert stats.failures == 1 and stats.batches == 2


def test_it_gives_up_after_too_many_failures_in_a_row(ref):
    kafka = FakeKafka(orders(ref, 10))

    def always_broken(batch):
        raise ConnectionError("down")

    with pytest.raises(TooManyFailures):
        loop(kafka.new_consumer(), always_broken, max_failures_in_a_row=3)
    assert kafka.commits == []


def test_killed_after_storing_but_before_committing_changes_nothing_on_restart(store, ref):
    """The risky moment: the batch is in the lake, but Kafka was never told."""
    kafka = FakeKafka(orders(ref, 100))

    def store_then_die(batch):
        process_batch(store, batch, ref, now=NOW)
        raise KeyboardInterrupt  # killed before the commit line

    loop(kafka.new_consumer(), store_then_die)
    assert kafka.committed == 0
    ids_after_crash = set(lake(store)["transaction_id"])
    assert ids_after_crash

    # Restart: the same 50 messages come back first, and add nothing new.
    reports = []
    loop(kafka.new_consumer(), lambda b: reports.append(process_batch(store, b, ref, now=NOW)))

    assert reports[0]["stream"]["batch_id"] == "p0o0-49"
    assert reports[0]["lake"]["rows_added"] == 0
    assert kafka.committed == 100
    stored = lake(store)
    assert stored["transaction_id"].is_unique
    assert ids_after_crash <= set(stored["transaction_id"])


def test_stop_and_restart_in_the_middle_gives_each_transaction_once(store, ref):
    msgs = orders(ref, 400)
    kafka = FakeKafka(msgs)

    loop(kafka.new_consumer(), lambda b: process_batch(store, b, ref, now=NOW), max_batches=3)
    assert kafka.committed == 150
    loop(kafka.new_consumer(), lambda b: process_batch(store, b, ref, now=NOW))
    assert kafka.committed == 400

    # Same messages in one uninterrupted run, into a second store.
    other = LocalStore(store.root.parent / "other")
    once = FakeKafka(msgs)
    loop(once.new_consumer(), lambda b: process_batch(other, b, ref, now=NOW))

    a, b = lake(store), lake(other)
    assert a["transaction_id"].is_unique
    assert sorted(a["transaction_id"]) == sorted(b["transaction_id"])


def test_empty_batches_while_joining_the_group_are_not_idle(ref):
    kafka = FakeKafka(orders(ref, 20))
    joins = iter([[], [], ["partition 0"]])  # no partitions for the first two polls of assignment
    kafka.assignment = lambda: next(joins, ["partition 0"])
    real_poll = kafka.poll
    calls = {"n": 0}

    def slow_start_poll(timeout):
        calls["n"] += 1
        return None if calls["n"] <= 2 else real_poll(timeout)  # nothing arrives while joining

    kafka.poll = slow_start_poll
    stats = loop(kafka.new_consumer(), lambda b: None, idle_batches=1)

    assert stats.batches == 1 and kafka.committed == 20


def test_idle_and_max_batch_limits(ref):
    kafka = FakeKafka(orders(ref, 200))
    stats = loop(kafka.new_consumer(), lambda b: None, max_batches=2, idle_batches=None)
    assert stats.batches == 2 and kafka.committed == 100

    empty = FakeKafka([])
    assert loop(empty.new_consumer(), lambda b: None, idle_batches=3).batches == 0
