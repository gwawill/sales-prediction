"""The raw area: every input saved unchanged before cleaning (FR-8), so any run can be
replayed later.

    raw/batch/<file name>                         a batch file, byte for byte
    raw/stream/date=YYYY-MM-DD/<batch id>.jsonl   one Kafka micro-batch, one message per line

Raw data is a permanent record, so nothing here is ever overwritten:
  - saving the same content again does nothing (safe to re-run),
  - different content under an existing name raises RawConflictError.

Stream batches are grouped by the day they were *received*, not by the order's date:
raw means "exactly as it arrived".
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from sales_poc.lake.store import ObjectStore

RAW_BATCH_PREFIX = "raw/batch/"
RAW_STREAM_PREFIX = "raw/stream/"


class RawConflictError(RuntimeError):
    """A raw file already exists with different content. Raw data is never overwritten."""


def raw_batch_key(file_name: str) -> str:
    if "/" in file_name or "\\" in file_name:
        raise ValueError(f"Expected a file name, not a path: {file_name!r}")
    return RAW_BATCH_PREFIX + file_name


def raw_stream_key(received: date, batch_id: str) -> str:
    if not batch_id or "/" in batch_id:
        raise ValueError(f"Bad batch id: {batch_id!r}")
    return f"{RAW_STREAM_PREFIX}date={received:%Y-%m-%d}/{batch_id}.jsonl"


def _put_once(store: ObjectStore, key: str, data: bytes) -> bool:
    """Write `data` unless the key exists. True if written, False if already there."""
    if store.exists(key):
        if store.get(key) == data:
            return False
        raise RawConflictError(
            f"{key} already exists with different content. Raw data is never overwritten; "
            "give the new file a different name."
        )
    store.put(key, data)
    return True


def save_raw_batch_file(store: ObjectStore, path: Path) -> tuple[str, bool]:
    """Save a local batch file unchanged. Returns (key, written?)."""
    key = raw_batch_key(Path(path).name)
    return key, _put_once(store, key, Path(path).read_bytes())


def save_raw_stream_batch(
    store: ObjectStore, messages: list[bytes], received: date, batch_id: str
) -> tuple[str, bool]:
    """Save one micro-batch of Kafka message values, one per line. Returns (key, written?)."""
    for message in messages:
        if b"\n" in message:
            raise ValueError("A message contains a line break, so it cannot go in a JSONL file.")
    key = raw_stream_key(received, batch_id)
    data = b"".join(m + b"\n" for m in messages)
    return key, _put_once(store, key, data)
