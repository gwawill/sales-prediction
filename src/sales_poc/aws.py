"""One place to create AWS clients, so every part of the project uses the same region and
credentials from `AwsSettings`."""

from __future__ import annotations

import boto3
from botocore.config import Config

from sales_poc.config import AwsSettings, get_aws_settings

# The lake is read with up to 16 parallel downloads (lake/clean.py), and boto3 keeps only 10
# connections by default, so threads queued for a connection. 32 leaves room to spare.
# "standard" retry mode retries throttling and brief network errors on each request, a little
# more patiently than boto3's default "legacy" mode. max_attempts=5 means 5 retries after the
# first try (boto3 reports it as total_max_attempts=6).
S3_CONFIG = Config(max_pool_connections=32, retries={"mode": "standard", "max_attempts": 5})


def session(settings: AwsSettings | None = None) -> boto3.Session:
    """An AWS session for the configured profile (or default credentials) and region."""
    s = settings or get_aws_settings()
    return boto3.Session(profile_name=s.aws_profile, region_name=s.aws_region)


def s3_client(settings: AwsSettings | None = None):
    return session(settings).client("s3", config=S3_CONFIG)


def s3vectors_client(settings: AwsSettings | None = None):
    """Amazon S3 Vectors, the vector store. Same region and credentials as S3."""
    return session(settings).client("s3vectors")
