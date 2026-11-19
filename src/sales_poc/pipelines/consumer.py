"""Stream consumer: read orders from Kafka in micro-batches (FR-7).

A micro-batch ends at STREAM_BATCH_SIZE messages or after STREAM_BATCH_SECONDS, whichever
comes first. For each batch:
  - every message value is kept as-is, for the raw area (FR-8),
  - order messages become rows with the daily-file columns, ready for the shared ETL,
  - health-check messages (key "healthcheck", left by the T1.3 cloud check) are skipped,
  - messages that are not a JSON object are counted as unreadable instead of crashing,
  - the batch gets an ID from its Kafka positions, e.g. "p0o10-14_p2o3-9", so reading the
    same messages again (after a crash) gives the same ID and the same raw file,
  - the positions to confirm to Kafka afterwards are worked out (used in task T7.3).

The "generator-problem" header is never read here: a real source would not have it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
from confluent_kafka import Consumer, TopicPartition

from sales_poc.config import KafkaSettings, get_kafka_settings
from sales_poc.datagen.history import COLUMNS
from sales_poc.kafka import connection_config

HEALTHCHECK_KEY = b"healthcheck"


@dataclass(frozen=True)
class MicroBatch:
    batch_id: str                       # from the Kafka positions; "" if empty
    values: list[bytes]                 # every message value, unchanged, in arrival order
    rows: pd.DataFrame                  # order rows (COLUMNS, text or None)
    healthchecks: int = 0               # skipped health-check messages
    unreadable: int = 0                 # messages that were not a JSON object
    next_offsets: dict[int, int] = field(default_factory=dict)  # partition -> offset to commit

    @property
    def size(self) -> int:
        return len(self.values)

    def commit_positions(self, topic: str) -> list[TopicPartition]:
        return [TopicPartition(topic, p, o) for p, o in sorted(self.next_offsets.items())]


def make_consumer(settings: KafkaSettings | None = None, group_id: str | None = None) -> Consumer:
    """A consumer in the ETL's consumer group, subscribed to the orders topic.

    Positions are never confirmed automatically: the pipeline confirms them itself, only
    after a batch is safely stored (task T7.3). A brand-new group starts at the oldest
    message, so nothing already in the topic is missed.
    """
    s = settings or get_kafka_settings()
    consumer = Consumer({
        **connection_config(s),
        "group.id": group_id or s.kafka_consumer_group,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "log_level": 0,
    })
    consumer.subscribe([s.kafka_topic])
    return consumer


def collect_messages(
    consumer,
    max_messages: int,
    max_seconds: float,
    clock: Callable[[], float] = time.monotonic,
) -> list:
    """Poll until `max_messages` arrive or `max_seconds` pass. Errors are raised."""
    messages = []
    deadline = clock() + max_seconds
    while len(messages) < max_messages:
        remaining = deadline - clock()
        if remaining <= 0:
            break
        msg = consumer.poll(min(remaining, 1.0))
        if msg is None:
            continue
        if msg.error():
            raise RuntimeError(f"Kafka error while reading: {msg.error()}")
        messages.append(msg)
    return messages


def _to_row(value: bytes) -> dict | None:
    """The order fields from one message, or None if it is not a JSON object."""
    try:
        data = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    # Missing fields become None; numbers are turned into text like a CSV value would be.
    return {c: (None if data.get(c) is None else str(data.get(c))) for c in COLUMNS}


def batch_id_for(messages: list) -> str:
    ranges = {}
    for m in messages:
        low, high = ranges.get(m.partition(), (m.offset(), m.offset()))
        ranges[m.partition()] = (min(low, m.offset()), max(high, m.offset()))
    return "_".join(f"p{p}o{low}-{high}" for p, (low, high) in sorted(ranges.items()))


def build_batch(messages: list) -> MicroBatch:
    """Turn polled messages into a MicroBatch. Pure: no Kafka calls."""
    rows, healthchecks, unreadable = [], 0, 0
    for m in messages:
        if m.key() == HEALTHCHECK_KEY:
            healthchecks += 1
            continue
        row = _to_row(m.value())
        if row is None:
            unreadable += 1
        else:
            rows.append(row)

    next_offsets: dict[int, int] = {}
    for m in messages:
        next_offsets[m.partition()] = max(next_offsets.get(m.partition(), 0), m.offset() + 1)

    return MicroBatch(
        batch_id=batch_id_for(messages),
        values=[m.value() for m in messages],
        rows=pd.DataFrame(rows, columns=COLUMNS),
        healthchecks=healthchecks,
        unreadable=unreadable,
        next_offsets=next_offsets,
    )


def read_batch(consumer, max_messages: int, max_seconds: float) -> MicroBatch:
    return build_batch(collect_messages(consumer, max_messages, max_seconds))
