"""S3Store.get tries a broken download again; the S3 client has room for parallel downloads."""

import pytest
from botocore.exceptions import (ClientError, ConnectionClosedError, IncompleteReadError, ReadTimeoutError,
                                 ResponseStreamingError)

from sales_poc import aws
from sales_poc.lake import store as store_module
from sales_poc.lake.store import GET_ATTEMPTS, S3Store


class Body:
    def __init__(self, error=None):
        self.error = error

    def read(self):
        if self.error:
            raise self.error
        return b"data"


class FlakyS3:
    """get_object works, but the first `failures` downloads break while reading."""

    def __init__(self, failures=0, error=None, get_error=None):
        self.failures, self.error, self.get_error, self.calls = failures, error, get_error, 0

    def get_object(self, Bucket, Key):
        self.calls += 1
        if self.get_error:
            raise self.get_error
        if self.calls <= self.failures:
            return {"Body": Body(self.error)}
        return {"Body": Body()}


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    waits = []
    monkeypatch.setattr(store_module.time, "sleep", waits.append)
    return waits


BROKEN = [
    ReadTimeoutError(endpoint_url="https://s3"),
    ResponseStreamingError(error="reset"),
    IncompleteReadError(actual_bytes=1, expected_bytes=2),
    ConnectionClosedError(endpoint_url="https://s3"),
]


@pytest.mark.parametrize("error", BROKEN, ids=lambda e: type(e).__name__)
def test_a_broken_download_is_tried_again(error, no_waiting):
    client = FlakyS3(failures=GET_ATTEMPTS - 1, error=error)
    assert S3Store(client, "bucket").get("key") == b"data"
    assert client.calls == GET_ATTEMPTS
    assert no_waiting == [1.0, 2.0]          # a little longer each time


def test_it_gives_up_after_the_last_attempt():
    client = FlakyS3(failures=99, error=BROKEN[0])
    with pytest.raises(ReadTimeoutError):
        S3Store(client, "bucket").get("key")
    assert client.calls == GET_ATTEMPTS


def test_a_missing_file_is_not_retried():
    client = FlakyS3(get_error=ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject"))
    with pytest.raises(KeyError):
        S3Store(client, "bucket").get("key")
    assert client.calls == 1


def test_other_s3_errors_are_not_retried_here():
    client = FlakyS3(get_error=ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject"))
    with pytest.raises(ClientError):
        S3Store(client, "bucket").get("key")
    assert client.calls == 1


def test_the_s3_client_has_room_for_parallel_downloads_and_retries():
    from sales_poc.lake.clean import read_all_clean
    import inspect

    workers = inspect.signature(read_all_clean).parameters["workers"].default
    assert aws.S3_CONFIG.max_pool_connections >= workers
    assert aws.S3_CONFIG.retries == {"mode": "standard", "max_attempts": 5}
