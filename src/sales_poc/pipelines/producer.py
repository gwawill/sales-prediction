"""Stream producer: send live orders to Kafka (FR-3).

    stream-produce --rate 2 --count 100       2 orders a second, 100 in total
    stream-produce --rate 5 --seconds 60      5 a second for one minute
    stream-produce                            1 a second until Ctrl+C

Each message:
    key    the transaction_id, so a resent copy lands in the same partition as the original
    value  the order as JSON (same fields as a daily file row; missing values are null)

Orders come from the live order maker (datagen/live.py). In this version they are sent
clean; deliberate problems are switched on in task T6.2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from confluent_kafka import Producer

from sales_poc.config import get_app_settings, get_kafka_settings
from sales_poc.datagen.live import LiveOrder, LiveOrderMaker
from sales_poc.datagen.reference import load_reference_data
from sales_poc.kafka import connection_config
from sales_poc.log import get_logger, setup_logging

log = get_logger(__name__)

# Called once per order: (key, value). Real runs send to Kafka; tests collect them.
Send = Callable[[bytes | None, bytes], None]


@dataclass
class ProduceStats:
    sent: int = 0
    by_problem: dict[str, int] = field(default_factory=dict)


def encode(order: LiveOrder) -> tuple[bytes | None, bytes]:
    """(key, value) for one order."""
    tx = order.payload.get("transaction_id")
    key = tx.encode() if tx else None
    return key, json.dumps(order.payload).encode()


def produce_orders(
    send: Send,
    maker: LiveOrderMaker,
    rate: float,
    count: int | None = None,
    seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ProduceStats:
    """Send orders at `rate` per second until `count` orders or `seconds` have passed.

    Each order is scheduled at start + n / rate, so small delays don't add up over time.
    With neither limit it runs until Ctrl+C. `clock` and `sleep` can be faked in tests.
    """
    stats = ProduceStats()
    start = clock()
    try:
        while count is None or stats.sent < count:
            due = start + stats.sent / rate
            if seconds is not None and due - start >= seconds:
                break
            wait = due - clock()
            if wait > 0:
                sleep(wait)
            order = maker.next_order()
            send(*encode(order))
            stats.sent += 1
            label = order.problem or "clean"
            stats.by_problem[label] = stats.by_problem.get(label, 0) + 1
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C.")
    return stats


class KafkaSender:
    """Sends to the real topic and counts deliveries confirmed or failed by Kafka."""

    def __init__(self, topic: str | None = None):
        settings = get_kafka_settings()
        self.topic = topic or settings.kafka_topic
        self.producer = Producer({
            **connection_config(settings),
            "client.id": "sales-poc-producer",
            "acks": "all",                 # wait until Kafka has safely stored it
            "enable.idempotence": True,    # Kafka's own retries never create extra copies
            "linger.ms": 50,               # group messages briefly for efficiency
        })
        self.delivered = 0
        self.failed = 0

    def _on_delivery(self, err, msg) -> None:
        if err is None:
            self.delivered += 1
        else:
            self.failed += 1
            log.error("delivery_failed", extra={"error": str(err)})

    def __call__(self, key: bytes | None, value: bytes) -> None:
        while True:
            try:
                self.producer.produce(self.topic, key=key, value=value, on_delivery=self._on_delivery)
                break
            except BufferError:
                self.producer.poll(0.5)  # local queue full: let it drain, then retry
        self.producer.poll(0)

    def close(self, timeout: float = 30) -> int:
        """Wait for outstanding messages. Returns how many were still not confirmed."""
        return self.producer.flush(timeout)


def parse_args(argv: list[str]) -> argparse.Namespace:
    data_dir = get_app_settings().data_dir
    parser = argparse.ArgumentParser(prog="stream-produce", description="Send live orders to Kafka.")
    parser.add_argument("--rate", type=float, default=1.0, help="orders per second (default 1)")
    parser.add_argument("--count", type=int, default=None, help="stop after this many orders")
    parser.add_argument("--seconds", type=float, default=None, help="stop after this many seconds")
    parser.add_argument("--seed", type=int, default=None, help="repeatable orders (default: different each run)")
    parser.add_argument("--reference", type=Path, default=data_dir / "reference")
    args = parser.parse_args(argv)
    if args.rate <= 0:
        parser.error("--rate must be more than 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    sys.stdout.reconfigure(line_buffering=True)
    setup_logging()
    maker = LiveOrderMaker(load_reference_data(args.reference), seed=args.seed, rates={})
    sender = KafkaSender()

    limit = f"{args.count} orders" if args.count else f"{args.seconds:g} seconds" if args.seconds else "Ctrl+C"
    print(f"Sending {args.rate:g} orders/second to '{sender.topic}' until {limit}.")
    stats = produce_orders(sender, maker, args.rate, args.count, args.seconds)
    unconfirmed = sender.close()

    print(f"Sent {stats.sent}: confirmed {sender.delivered}, failed {sender.failed}, "
          f"unconfirmed {unconfirmed}. By kind: {stats.by_problem}")
    return 0 if sender.failed == 0 and unconfirmed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
