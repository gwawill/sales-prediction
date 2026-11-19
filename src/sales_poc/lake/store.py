"""Where the data lake's files live: S3 for real, or a local folder for tests.

Both stores have the same small set of operations, so the lake code does not care which one
it gets. Keys are S3-style paths like "raw/batch/sales_2026-03-01.csv".

    store = s3_store()                      # the real bucket from S3_BUCKET
    store = s3_store(prefix="_test/abc/")   # everything under a test folder
    store = LocalStore(tmp_path)            # unit tests: a normal folder
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from botocore.exceptions import ClientError

from sales_poc.aws import s3_client
from sales_poc.config import AwsSettings, get_aws_settings


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str = "") -> list[str]: ...
    def delete(self, key: str) -> None: ...


class LocalStore:
    """A folder on disk that behaves like a bucket. Used by unit tests."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(f"Bad key: {key!r}")
        return self.root.joinpath(*key.split("/"))

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise KeyError(key)
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str = "") -> list[str]:
        if not self.root.exists():
            return []
        keys = (p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file())
        return sorted(k for k in keys if k.startswith(prefix))

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class S3Store:
    """One S3 bucket, optionally limited to keys under `prefix`."""

    def __init__(self, client, bucket: str, prefix: str = ""):
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        self.client, self.bucket, self.prefix = client, bucket, prefix

    def _key(self, key: str) -> str:
        return self.prefix + key

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def get(self, key: str) -> bytes:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                raise KeyError(key) from None
            raise

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list(self, prefix: str = "") -> list[str]:
        keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
            keys += [item["Key"][len(self.prefix):] for item in page.get("Contents", [])]
        return sorted(keys)

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))


def s3_store(settings: AwsSettings | None = None, prefix: str = "") -> S3Store:
    s = settings or get_aws_settings()
    return S3Store(s3_client(s), s.s3_bucket, prefix)
