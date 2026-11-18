"""Cloud connection checks (task T1).

Each check does one small real action against one service, cleans up after itself, and
returns a short message. If something is wrong it raises an error that says what to fix.

Run one or more checks (no name = all):

    python -m sales_poc.checks s3
    python -m sales_poc.checks s3 s3vectors
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, ProfileNotFound

from sales_poc.aws import s3_client, s3vectors_client
from sales_poc.config import ConfigError, get_aws_settings, get_vector_store_settings


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


CHECKS: dict[str, Callable[[], str]] = {
    "s3": check_s3,
    "s3vectors": check_s3_vectors,
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
