import json
from datetime import date

import pytest

from sales_poc.lake.raw import (
    RawConflictError,
    raw_batch_key,
    raw_stream_key,
    save_raw_batch_file,
    save_raw_stream_batch,
)
from sales_poc.lake.store import LocalStore


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "bucket")


@pytest.fixture
def batch_file(tmp_path):
    path = tmp_path / "sales_2026-03-01.csv"
    path.write_bytes(b"transaction_id,quantity\r\nTX-1,$2\n")  # odd bytes on purpose
    return path


def test_keys_follow_the_plan_layout():
    assert raw_batch_key("sales_2026-03-01.csv") == "raw/batch/sales_2026-03-01.csv"
    assert raw_stream_key(date(2026, 11, 18), "p0-o100-o199") == "raw/stream/date=2026-11-18/p0-o100-o199.jsonl"


def test_bad_names_are_refused():
    with pytest.raises(ValueError):
        raw_batch_key("data/incoming/sales.csv")
    with pytest.raises(ValueError):
        raw_stream_key(date(2026, 11, 18), "a/b")


def test_batch_file_is_saved_byte_for_byte(store, batch_file):
    key, written = save_raw_batch_file(store, batch_file)

    assert key == "raw/batch/sales_2026-03-01.csv"
    assert written
    assert store.get(key) == batch_file.read_bytes()


def test_saving_the_same_file_again_does_nothing(store, batch_file):
    save_raw_batch_file(store, batch_file)

    key, written = save_raw_batch_file(store, batch_file)

    assert not written
    assert store.list("raw/") == [key]


def test_different_content_under_the_same_name_is_refused(store, batch_file):
    save_raw_batch_file(store, batch_file)
    batch_file.write_bytes(b"something else\n")

    with pytest.raises(RawConflictError, match="never overwritten"):
        save_raw_batch_file(store, batch_file)
    assert b"TX-1" in store.get("raw/batch/sales_2026-03-01.csv")  # original kept


def test_stream_batch_is_one_message_per_line(store):
    messages = [json.dumps({"transaction_id": f"LV-{i}"}).encode() for i in range(3)]

    key, written = save_raw_stream_batch(store, messages, date(2026, 11, 18), "p0-o0-o2")

    assert written
    lines = store.get(key).decode().splitlines()
    assert [json.loads(line)["transaction_id"] for line in lines] == ["LV-0", "LV-1", "LV-2"]


def test_stream_batch_is_also_written_only_once(store):
    messages = [b'{"a": 1}']
    save_raw_stream_batch(store, messages, date(2026, 11, 18), "b1")

    assert save_raw_stream_batch(store, messages, date(2026, 11, 18), "b1")[1] is False
    with pytest.raises(RawConflictError):
        save_raw_stream_batch(store, [b'{"a": 2}'], date(2026, 11, 18), "b1")


def test_message_with_a_line_break_is_refused(store):
    with pytest.raises(ValueError, match="line break"):
        save_raw_stream_batch(store, [b'{"a":\n1}'], date(2026, 11, 18), "b1")


def test_local_store_basics(store):
    store.put("x/y/z.txt", b"hi")
    assert store.exists("x/y/z.txt")
    assert store.list("x/") == ["x/y/z.txt"]
    assert store.list("nope/") == []
    store.delete("x/y/z.txt")
    assert not store.exists("x/y/z.txt")
    with pytest.raises(KeyError):
        store.get("x/y/z.txt")
    with pytest.raises(ValueError):
        store.put("../escape.txt", b"no")
