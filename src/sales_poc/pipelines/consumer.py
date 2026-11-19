"""Stream consumer: read orders from Kafka in micro-batches and load them (FR-7).

    stream-consume                          run until Ctrl+C
    stream-consume --idle-batches 3         stop after 3 empty batches in a row
    stream-consume --max-batches 5          stop after 5 batches

Delivery (task T7.3): Kafka is told "done" (the offsets are committed) only AFTER a batch
is safely stored. If storing fails, the consumer moves back to the start of the batch and
tries again. If the consumer is killed between storing and committing, the batch is read
again after a restart, and loading it again changes nothing. At-least-once delivery plus
dedupe gives each transaction exactly once in the lake.

Reading (task T7.1). A micro-batch ends at STREAM_BATCH_SIZE messages or after
STREAM_BATCH_SECONDS, whichever comes first. For each batch:
  - every message value is kept as-is, for the raw area (FR-8),
  - order messages become rows with the daily-file columns, ready for the shared ETL,
  - health-check messages (key "healthcheck", left by the T1.3 cloud check) are skipped,
  - messages that are not a JSON object are counted as unreadable instead of crashing,
  - the batch gets an ID from its Kafka positions, e.g. "p0o10-14_p2o3-9", so reading the
    same messages again (after a crash) gives the same ID and the same raw file,
  - the positions to confirm to Kafka afterwards are worked out (used in task T7.3).

The "generator-problem" header is never read here: a real source would not have it.

Loading (task T7.2), `process_batch`, for every non-empty batch:
    1. save the message values unchanged to raw/stream/date=<received>/<batch id>.jsonl
    2. run the shared ETL with source="stream"                          (FR-9)
    3. merge clean rows into the lake; late orders land on their real day (FR-15)
    4. write this batch's quarantine file and report, with a "stream" section
An empty batch writes nothing, so an idle consumer does not fill the bucket.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
from confluent_kafka import Consumer, TopicPartition

from sales_poc.config import KafkaSettings, get_kafka_settings
from sales_poc.datagen.history import COLUMNS
from sales_poc.datagen.reference import ReferenceData
from sales_poc.etl.pipeline import run_etl
from sales_poc.kafka import connection_config
from sales_poc.lake.clean import write_clean
from sales_poc.lake.raw import save_raw_stream_batch
from sales_poc.lake.runs import build_report, new_run_id, write_quarantine, write_report
from sales_poc.lake.store import ObjectStore
from sales_poc.log import get_logger

log = get_logger(__name__)

LOAD_WORKERS = 8

HEALTHCHECK_KEY = b"healthcheck"


@dataclass(frozen=True)
class MicroBatch:
    batch_id: str                       # from the Kafka positions; "" if empty
    values: list[bytes]                 # every message value, unchanged, in arrival order
    rows: pd.DataFrame                  # order rows (COLUMNS, text or None)
    healthchecks: int = 0               # skipped health-check messages
    unreadable: int = 0                 # messages that were not a JSON object
    next_offsets: dict[int, int] = field(default_factory=dict)   # partition -> offset to commit
    start_offsets: dict[int, int] = field(default_factory=dict)  # partition -> first offset read

    @property
    def size(self) -> int:
        return len(self.values)

    def commit_positions(self, topic: str) -> list[TopicPartition]:
        return [TopicPartition(topic, p, o) for p, o in sorted(self.next_offsets.items())]

    def rewind_positions(self, topic: str) -> list[TopicPartition]:
        return [TopicPartition(topic, p, o) for p, o in sorted(self.start_offsets.items())]


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
    start_offsets: dict[int, int] = {}
    for m in messages:
        next_offsets[m.partition()] = max(next_offsets.get(m.partition(), 0), m.offset() + 1)
        start_offsets[m.partition()] = min(start_offsets.get(m.partition(), m.offset()), m.offset())

    return MicroBatch(
        batch_id=batch_id_for(messages),
        values=[m.value() for m in messages],
        rows=pd.DataFrame(rows, columns=COLUMNS),
        healthchecks=healthchecks,
        unreadable=unreadable,
        next_offsets=next_offsets,
        start_offsets=start_offsets,
    )


def read_batch(consumer, max_messages: int, max_seconds: float) -> MicroBatch:
    return build_batch(collect_messages(consumer, max_messages, max_seconds))


class StreamCountError(RuntimeError):
    """Messages in a batch do not add up to order rows + health checks + unreadable."""


def process_batch(
    store: ObjectStore,
    batch: MicroBatch,
    ref: ReferenceData,
    now: datetime | None = None,
) -> dict | None:
    """Save, clean and load one micro-batch. Returns its report, or None if it was empty."""
    if batch.size == 0:
        return None
    if len(batch.rows) + batch.healthchecks + batch.unreadable != batch.size:
        raise StreamCountError(
            f"Batch {batch.batch_id}: {batch.size} messages, but {len(batch.rows)} rows + "
            f"{batch.healthchecks} health checks + {batch.unreadable} unreadable."
        )

    received = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run_id = new_run_id("stream", received)

    raw_key, _ = save_raw_stream_batch(store, batch.values, received.date(), batch.batch_id)
    result = run_etl(batch.rows, ref, source="stream", ingested_at=received)
    merges = write_clean(store, result.clean, workers=LOAD_WORKERS)
    q_key = write_quarantine(store, run_id, result.quarantine)

    report = build_report(
        run_id, "stream", received, datetime.now(timezone.utc), [raw_key], result.counts, merges, q_key
    )
    report["stream"] = {
        "batch_id": batch.batch_id,
        "messages": batch.size,
        "order_rows": len(batch.rows),
        "healthchecks": batch.healthchecks,
        "unreadable": batch.unreadable,
    }
    write_report(store, report)
    log.info("stream_batch_loaded", extra={"run_id": run_id, "batch_id": batch.batch_id,
                                           "messages": batch.size, "clean": result.counts.clean,
                                           "quarantined": result.counts.quarantined})
    return report


@dataclass
class LoopStats:
    batches: int = 0      # non-empty batches stored and committed
    messages: int = 0
    failures: int = 0     # attempts that failed and were retried


class TooManyFailures(RuntimeError):
    """Storing kept failing; the consumer stops instead of retrying forever."""


def _has_partitions(consumer) -> bool:
    """True once Kafka has assigned partitions (consumers without `assignment` count as ready)."""
    assignment = getattr(consumer, "assignment", None)
    return True if assignment is None else bool(assignment())


def consume_loop(
    consumer,
    process: Callable[[MicroBatch], object],
    topic: str,
    batch_size: int,
    batch_seconds: float,
    max_batches: int | None = None,
    idle_batches: int | None = None,
    retry_seconds: float = 5.0,
    max_failures_in_a_row: int = 10,
    reader: Callable = read_batch,
    sleep: Callable[[float], None] = time.sleep,
    on_batch: Callable[[MicroBatch, object], None] | None = None,
) -> LoopStats:
    """Read, store, then commit, batch after batch.

    Stops at Ctrl+C, after `max_batches` stored batches, or after `idle_batches` empty
    batches in a row. `reader` and `sleep` can be replaced in tests.
    """
    stats, idle, failed_in_a_row = LoopStats(), 0, 0
    try:
        while max_batches is None or stats.batches < max_batches:
            batch = reader(consumer, batch_size, batch_seconds)
            if batch.size == 0:
                # While Kafka is still handing out partitions (joining the group takes a few
                # seconds), an empty batch says nothing about the topic, so it is not "idle".
                if not _has_partitions(consumer):
                    continue
                idle += 1
                if idle_batches is not None and idle >= idle_batches:
                    break
                continue
            idle = 0

            try:
                result = process(batch)
            except Exception as exc:  # noqa: BLE001 - nothing is committed; retry the same batch
                stats.failures += 1
                failed_in_a_row += 1
                log.exception("stream_batch_failed", extra={"batch_id": batch.batch_id})
                if failed_in_a_row >= max_failures_in_a_row:
                    raise TooManyFailures(f"Batch {batch.batch_id} failed {failed_in_a_row} times: {exc}") from exc
                for position in batch.rewind_positions(topic):
                    consumer.seek(position)
                sleep(retry_seconds)
                continue

            # Stored safely: only now tell Kafka these messages are done.
            consumer.commit(offsets=batch.commit_positions(topic), asynchronous=False)
            failed_in_a_row = 0
            stats.batches += 1
            stats.messages += batch.size
            if on_batch:
                on_batch(batch, result)
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C.")
    return stats


def _print_batch(batch: MicroBatch, report: dict | None) -> None:
    c, lake = report["counts"], report["lake"]
    print(f"[{datetime.now():%H:%M:%S}] {batch.size} messages ({batch.healthchecks} health checks, "
          f"{batch.unreadable} unreadable) -> clean {c['clean']}, duplicates {c['duplicates_removed']}, "
          f"quarantined {c['quarantined']} | lake +{lake['rows_added']} new, "
          f"{lake['rows_unchanged']} already there | {batch.batch_id}")


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from sales_poc.config import get_app_settings
    from sales_poc.datagen.reference import load_reference_data
    from sales_poc.lake.store import s3_store
    from sales_poc.log import setup_logging

    app, kafka = get_app_settings(), get_kafka_settings()
    parser = argparse.ArgumentParser(prog="stream-consume", description="Load orders from Kafka into the lake.")
    parser.add_argument("--max-batches", type=int, default=None, help="stop after this many stored batches")
    parser.add_argument("--idle-batches", type=int, default=None,
                        help="stop after this many empty batches in a row")
    parser.add_argument("--batch-size", type=int, default=app.stream_batch_size)
    parser.add_argument("--batch-seconds", type=float, default=app.stream_batch_seconds)
    parser.add_argument("--reference", default=str(app.data_dir / "reference"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    sys.stdout.reconfigure(line_buffering=True)
    setup_logging()
    store, ref = s3_store(), load_reference_data(args.reference)
    consumer = make_consumer(kafka)
    print(f"Reading '{kafka.kafka_topic}' as group '{kafka.kafka_consumer_group}' in batches of "
          f"{args.batch_size} messages or {args.batch_seconds:g}s. Press Ctrl+C to stop.")
    try:
        stats = consume_loop(
            consumer, lambda b: process_batch(store, b, ref), kafka.kafka_topic,
            args.batch_size, args.batch_seconds, args.max_batches, args.idle_batches,
            on_batch=_print_batch,
        )
    except TooManyFailures as exc:
        print(f"Stopped: {exc}")
        return 1
    finally:
        consumer.close()
    print(f"Stored and committed {stats.batches} batches, {stats.messages} messages; "
          f"{stats.failures} failed attempts were retried.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
