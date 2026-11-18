"""One place to create AWS clients, so every part of the project uses the same region and
credentials from `AwsSettings`."""

from __future__ import annotations

import boto3

from sales_poc.config import AwsSettings, get_aws_settings


def session(settings: AwsSettings | None = None) -> boto3.Session:
    """An AWS session for the configured profile (or default credentials) and region."""
    s = settings or get_aws_settings()
    return boto3.Session(profile_name=s.aws_profile, region_name=s.aws_region)


def s3_client(settings: AwsSettings | None = None):
    return session(settings).client("s3")


def s3vectors_client(settings: AwsSettings | None = None):
    """Amazon S3 Vectors, the vector store. Same region and credentials as S3."""
    return session(settings).client("s3vectors")
