"""Show the most recent messages in the Kafka topic, without disturbing anything.

    .venv\\Scripts\\python.exe scripts\\kafka_peek.py          last 5 messages per partition
    .venv\\Scripts\\python.exe scripts\\kafka_peek.py 20       last 20 per partition

It reads by position and never saves a reading position, so the real consumer group is
not affected.
"""

import sys
import uuid

from confluent_kafka import Consumer, TopicPartition

from sales_poc.config import get_kafka_settings
from sales_poc.kafka import connection_config


def peek(per_partition: int = 5, timeout: float = 15.0) -> list:
    settings = get_kafka_settings()
    consumer = Consumer({
        **connection_config(settings),
        "group.id": f"sales-poc-peek-{uuid.uuid4().hex[:8]}",
        "enable.auto.commit": False,
        "log_level": 0,
    })
    try:
        partitions = consumer.list_topics(settings.kafka_topic, timeout=timeout).topics[settings.kafka_topic].partitions
        targets, expected = [], 0
        for p in sorted(partitions):
            low, high = consumer.get_watermark_offsets(TopicPartition(settings.kafka_topic, p), timeout=timeout)
            start = max(low, high - per_partition)
            if high > start:
                targets.append(TopicPartition(settings.kafka_topic, p, start))
                expected += high - start
        if not targets:
            return []
        consumer.assign(targets)
        messages = []
        while len(messages) < expected:
            msg = consumer.poll(timeout)
            if msg is None:
                break
            if not msg.error():
                messages.append(msg)
        return sorted(messages, key=lambda m: (m.timestamp()[1], m.partition(), m.offset()))
    finally:
        consumer.close()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    found = peek(n)
    for m in found:
        key = m.key().decode() if m.key() else None
        print(f"partition {m.partition()} offset {m.offset():>5}  key={key}  {m.value().decode()[:160]}")
    print(f"{len(found)} messages shown.")
