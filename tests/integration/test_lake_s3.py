"""The lake code against the real S3 bucket, inside a throwaway test folder."""

import uuid
from datetime import date

import pytest

from sales_poc.lake.raw import RawConflictError, save_raw_batch_file, save_raw_stream_batch
from sales_poc.lake.store import s3_store


@pytest.fixture
def store():
    """Everything under _test/<random>/ in the real bucket, deleted afterwards."""
    s = s3_store(prefix=f"_test/{uuid.uuid4().hex}/")
    yield s
    for key in s.list():
        s.delete(key)
    assert s.list() == []


def test_s3_store_basics(store):
    store.put("a/b.txt", b"hello")
    assert store.exists("a/b.txt")
    assert not store.exists("a/missing.txt")
    assert store.get("a/b.txt") == b"hello"
    assert store.list("a/") == ["a/b.txt"]
    with pytest.raises(KeyError):
        store.get("a/missing.txt")


def test_raw_batch_file_round_trip_on_s3(store, tmp_path):
    path = tmp_path / "sales_2026-03-01.csv"
    path.write_bytes(b"transaction_id\nTX-1\n")

    key, written = save_raw_batch_file(store, path)
    assert written and store.get(key) == path.read_bytes()
    assert save_raw_batch_file(store, path) == (key, False)

    path.write_bytes(b"changed\n")
    with pytest.raises(RawConflictError):
        save_raw_batch_file(store, path)


def test_raw_stream_batch_on_s3(store):
    key, written = save_raw_stream_batch(store, [b'{"a":1}', b'{"a":2}'], date(2026, 11, 18), "b1")
    assert written
    assert store.get(key) == b'{"a":1}\n{"a":2}\n'
