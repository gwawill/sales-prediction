"""Cloud connection checks (task T1).

Each check does one small real action against one service, cleans up after itself, and
returns a short message. If something is wrong it raises an error that says what to fix.

Run a single check:

    python -m sales_poc.checks s3
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Callable

from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, ProfileNotFound

from sales_poc.aws import s3_client
from sales_poc.config import ConfigError, get_aws_settings


class CheckError(RuntimeError):
    """A check failed. The message says what went wrong and what to look at."""


def check_s3() -> str:
    """Write a small file to the bucket, read it back, compare, then delete it."""
    settings = get_aws_settings()
    bucket = settings.s3_bucket
    key = f"_healthcheck/{uuid.uuid4().hex}.txt"
    body = b"sales-poc s3 check"

    try:
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
    except ProfileNotFound as exc:
        raise CheckError(f"{exc}. Check AWS_PROFILE in .env.") from None
    except NoCredentialsError:
        raise CheckError(
            "No AWS credentials found. Set AWS_PROFILE, or AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY, in .env."
        ) from None
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        hints = {
            "NoSuchBucket": f"Bucket '{bucket}' does not exist. Check S3_BUCKET in .env.",
            "AccessDenied": "Your AWS user is not allowed to do this on the bucket.",
            "InvalidAccessKeyId": "The AWS access key is wrong.",
            "SignatureDoesNotMatch": "The AWS secret key is wrong.",
        }
        raise CheckError(f"S3 error {code}. {hints.get(code, str(exc))}") from None
    except BotoCoreError as exc:
        raise CheckError(f"Could not reach S3: {exc}") from None

    if read_back != body:
        raise CheckError("S3 returned different content than was written.")
    return f"wrote, read and deleted s3://{bucket}/{key} in {settings.aws_region}"


CHECKS: dict[str, Callable[[], str]] = {
    "s3": check_s3,
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
