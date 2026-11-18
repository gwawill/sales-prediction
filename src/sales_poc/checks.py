"""Cloud connection checks (task T1).

Each check does one small real action against one service, cleans up after itself, and
returns a short message. If something is wrong it raises an error that says what to fix.

Run one or more checks (no name = all):

    python -m sales_poc.checks s3
    python -m sales_poc.checks s3 s3vectors
"""

from __future__ import annotations

import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, ProfileNotFound
from confluent_kafka import Consumer, KafkaError, KafkaException, Producer, TopicPartition
from neo4j import READ_ACCESS
from neo4j.exceptions import AuthError, ConfigurationError, Neo4jError, ServiceUnavailable

from sales_poc.aws import s3_client, s3vectors_client
from sales_poc.config import (
    ConfigError,
    get_aws_settings,
    get_kafka_settings,
    get_neo4j_settings,
    get_vector_store_settings,
)
from sales_poc.graph.connection import driver as neo4j_driver
from sales_poc.kafka import KafkaSetupError, connection_config, login_method


class CheckError(RuntimeError):
    """A check failed. The message says what went wrong and what to look at."""


# Plain-English hints for AWS error codes common to every AWS check.
_COMMON_AWS_HINTS = {
    "AccessDenied": "Your AWS user is not allowed to do this.",
    "AccessDeniedException": "Your AWS user is not allowed to do this.",
    "InvalidAccessKeyId": "The AWS access key is wrong.",
    "UnrecognizedClientException": "The AWS access key is wrong.",
    "SignatureDoesNotMatch": "The AWS secret key is wrong.",
    "InvalidSignatureException": "The AWS secret key is wrong.",
}


@contextmanager
def _aws_errors(service: str, hints: dict[str, str] | None = None) -> Iterator[None]:
    """Turn AWS library errors into a CheckError with a short hint."""
    all_hints = {**_COMMON_AWS_HINTS, **(hints or {})}
    try:
        yield
    except ProfileNotFound as exc:
        raise CheckError(f"{exc}. Check AWS_PROFILE in .env.") from None
    except NoCredentialsError:
        raise CheckError(
            "No AWS credentials found. Set AWS_PROFILE, or AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY, in .env."
        ) from None
    except ClientError as exc:
        error = exc.response.get("Error", {})
        code = error.get("Code", "Unknown")
        hint = all_hints.get(code)
        detail = error.get("Message") or str(exc)
        # Show the hint first, then AWS's own message, which often names the exact action.
        message = f"{service} error {code}. " + (f"{hint} AWS says: {detail}" if hint else detail)
        raise CheckError(message) from None
    except BotoCoreError as exc:
        raise CheckError(f"Could not reach {service}: {exc}") from None


def check_s3() -> str:
    """Write a small file to the bucket, read it back, compare, then delete it."""
    settings = get_aws_settings()
    bucket = settings.s3_bucket
    key = f"_healthcheck/{uuid.uuid4().hex}.txt"
    body = b"sales-poc s3 check"

    hints = {"NoSuchBucket": f"Bucket '{bucket}' does not exist. Check S3_BUCKET in .env."}
    with _aws_errors("S3", hints):
        client = s3_client(settings)
        # Confirm the bucket is in the region we think it is. A mismatch causes
        # confusing errors later, so catch it here.
        location = client.get_bucket_location(Bucket=bucket)["LocationConstraint"]
        bucket_region = location or "us-east-1"  # AWS reports us-east-1 as empty
        if bucket_region != settings.aws_region:
            raise CheckError(
                f"Bucket '{bucket}' is in {bucket_region}, but AWS_REGION is "
                f"{settings.aws_region}. Make them match in .env."
            )

        client.put_object(Bucket=bucket, Key=key, Body=body)
        try:
            read_back = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        finally:
            client.delete_object(Bucket=bucket, Key=key)

    if read_back != body:
        raise CheckError("S3 returned different content than was written.")
    return f"wrote, read and deleted s3://{bucket}/{key} in {settings.aws_region}"


def check_s3_vectors() -> str:
    """Create a tiny temporary index, add one vector, find it by search, delete the index.

    The real index (S3_VECTOR_INDEX) is not touched.
    """
    aws = get_aws_settings()
    vectors = get_vector_store_settings()
    bucket = vectors.s3_vector_bucket
    index = f"healthcheck-{uuid.uuid4().hex[:12]}"
    stored = [1.0, 0.0, 0.0, 0.0]

    hints = {
        "NotFoundException": (
            f"Vector bucket '{bucket}' was not found in {aws.aws_region}. "
            "Check S3_VECTOR_BUCKET and AWS_REGION in .env."
        ),
    }
    with _aws_errors("S3 Vectors", hints):
        client = s3vectors_client(aws)
        client.get_vector_bucket(vectorBucketName=bucket)

        client.create_index(
            vectorBucketName=bucket,
            indexName=index,
            dataType="float32",
            dimension=len(stored),
            distanceMetric="cosine",
        )
        try:
            client.put_vectors(
                vectorBucketName=bucket,
                indexName=index,
                vectors=[
                    {
                        "key": "check-1",
                        "data": {"float32": stored},
                        "metadata": {"purpose": "healthcheck"},
                    }
                ],
            )
            # Search with a vector that points almost the same way.
            result = client.query_vectors(
                vectorBucketName=bucket,
                indexName=index,
                queryVector={"float32": [0.9, 0.1, 0.0, 0.0]},
                topK=1,
                returnMetadata=True,
                returnDistance=True,
            )
        finally:
            client.delete_index(vectorBucketName=bucket, indexName=index)

    found = result.get("vectors", [])
    if not found or found[0]["key"] != "check-1":
        raise CheckError(f"Search did not return the vector that was just stored: {found}")
    if found[0].get("metadata", {}).get("purpose") != "healthcheck":
        raise CheckError("Search returned the vector but its metadata was lost.")
    return (
        f"created index '{index}' in vector bucket '{bucket}', stored and found one vector "
        f"(distance {found[0].get('distance', 0):.3f}), then deleted the index"
    )


KAFKA_TIMEOUT_SECONDS = 30


def check_kafka() -> str:
    """Send one message to the topic, then read that exact message back.

    The consumer reads only the one partition and offset the message landed on, and never
    commits, so the real consumer group (KAFKA_CONSUMER_GROUP) is not affected.
    """
    settings = get_kafka_settings()
    topic = settings.kafka_topic
    try:
        config = connection_config(settings)
    except KafkaSetupError as exc:
        raise CheckError(str(exc)) from None

    # Silence librdkafka's own console logging; errors still reach us through the API.
    quiet = {"log_level": 0}
    try:
        producer = Producer({**config, **quiet, "client.id": "sales-poc-check"})

        # 1. Is the server reachable, and does the topic exist?
        metadata = producer.list_topics(timeout=KAFKA_TIMEOUT_SECONDS)
        if topic not in metadata.topics:
            raise CheckError(
                f"Connected to Kafka, but topic '{topic}' does not exist. "
                "Create it in the Aiven console, or fix KAFKA_TOPIC in .env."
            )

        # 2. Send one message and learn where it landed.
        marker = f"sales-poc-check-{uuid.uuid4().hex}"
        delivered: dict[str, object] = {}

        def on_delivery(err, msg):
            delivered["error"] = err
            delivered["msg"] = msg

        producer.produce(topic, key=b"healthcheck", value=marker.encode(), on_delivery=on_delivery)
        producer.flush(KAFKA_TIMEOUT_SECONDS)
        if "msg" not in delivered:
            raise CheckError(f"Message was not delivered within {KAFKA_TIMEOUT_SECONDS}s.")
        if delivered["error"] is not None:
            raise CheckError(f"Kafka refused the message: {delivered['error']}")
        sent = delivered["msg"]
        partition, offset = sent.partition(), sent.offset()

        # 3. Read exactly that partition and offset back.
        consumer = Consumer(
            {
                **config,
                **quiet,
                "group.id": f"sales-poc-check-{uuid.uuid4().hex[:8]}",
                "enable.auto.commit": False,
            }
        )
        try:
            consumer.assign([TopicPartition(topic, partition, offset)])
            deadline = time.monotonic() + KAFKA_TIMEOUT_SECONDS
            received = None
            while time.monotonic() < deadline and received is None:
                msg = consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    raise CheckError(f"Error while reading: {msg.error()}")
                if msg.value() == marker.encode():
                    received = msg
        finally:
            consumer.close()
    except KafkaException as exc:
        raise CheckError(f"Kafka error (login: {login_method(settings)}): {exc}") from None

    if received is None:
        raise CheckError(
            f"Sent a message to partition {partition}, offset {offset}, but could not read it "
            f"back within {KAFKA_TIMEOUT_SECONDS}s."
        )
    return (
        f"sent and read back one message on topic '{topic}' "
        f"(partition {partition}, offset {offset}, login: {login_method(settings)})"
    )


def check_neo4j() -> str:
    """Connect, log in, and run `RETURN 1` in a read-only session. Nothing is written."""
    settings = get_neo4j_settings()
    try:
        with neo4j_driver(settings) as drv:
            drv.verify_connectivity()
            server = drv.get_server_info()
            with drv.session(default_access_mode=READ_ACCESS) as session:
                value = session.run("RETURN 1 AS ok").single()["ok"]
    except AuthError:
        raise CheckError(
            "Neo4j refused the login. Check NEO4J_USER and NEO4J_PASSWORD in .env."
        ) from None
    except ServiceUnavailable as exc:
        raise CheckError(
            f"Could not reach Neo4j at {settings.neo4j_uri}. Check NEO4J_URI, and in the Aura "
            f"console check the instance is running (free instances pause when idle). "
            f"Details: {exc}"
        ) from None
    except (ConfigurationError, ValueError) as exc:
        raise CheckError(f"NEO4J_URI looks wrong ({settings.neo4j_uri}): {exc}") from None
    except Neo4jError as exc:
        raise CheckError(f"Neo4j error: {exc}") from None

    if value != 1:
        raise CheckError(f"RETURN 1 gave back {value!r}.")
    return f"logged in as '{settings.neo4j_user}' and ran RETURN 1 on {server.agent}"


CHECKS: dict[str, Callable[[], str]] = {
    "s3": check_s3,
    "s3vectors": check_s3_vectors,
    "kafka": check_kafka,
    "neo4j": check_neo4j,
}


def main(argv: list[str]) -> int:
    names = argv or list(CHECKS)
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        print(f"Unknown check: {', '.join(unknown)}. Choose from: {', '.join(CHECKS)}")
        return 2

    failed = 0
    for name in names:
        try:
            print(f"{name}: OK - {CHECKS[name]()}")
        except (CheckError, ConfigError) as exc:
            failed += 1
            print(f"{name}: FAILED - {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
