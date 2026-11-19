"""The clean area: one Parquet file per day of clean transactions (FR-16, FR-17).

    clean/transactions/event_date=YYYY-MM-DD/data.parquet

`write_clean` merges new rows into each day they touch (read, merge, write back):
  - a new transaction_id is added,
  - the same ID with identical content is skipped, keeping the original row (so loading the
    same data twice changes nothing, even though the second run has a newer ingested_at),
  - the same ID with different content replaces the old row (the newest version wins).
`source` and `ingested_at` are not part of "content", so one order that arrives by both
batch and stream is stored once.

A day is only written if something changed, so an unchanged day's file stays byte for byte
the same.

Known limit, fine at PoC size: a corrected copy whose event_date changed would leave the old
copy in the old day's file. Handling that properly needs a table format with MERGE, such as
Apache Iceberg or Delta Lake (see plan.md).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date

import pandas as pd

from sales_poc.etl.join import JOINED_COLUMNS
from sales_poc.etl.pipeline import OUTPUT_COLUMNS
from sales_poc.etl.schema import CLEAN_SCHEMA
from sales_poc.lake.store import ObjectStore

CLEAN_PREFIX = "clean/transactions/"

# Exact pandas types of a clean row, restored after reading a Parquet file back.
LAKE_SCHEMA: dict[str, str] = {
    **CLEAN_SCHEMA,
    **{c: "string" for c in JOINED_COLUMNS},
    "event_date": "object",  # Python date objects; stored as a Parquet date
    "source": "string",
    "ingested_at": "datetime64[ns, UTC]",
    "quality_flags": "string",
}

# What decides whether two copies of a transaction are "the same".
CONTENT_COLUMNS = [c for c in OUTPUT_COLUMNS if c not in ("source", "ingested_at")]


@dataclass(frozen=True)
class DayMerge:
    day: date
    added: int        # new transaction IDs
    replaced: int     # existing IDs whose content changed
    unchanged: int    # existing IDs sent again with identical content
    total: int        # rows in the day's file afterwards
    written: bool     # False if nothing changed, so the file was not touched


def clean_key(day: date) -> str:
    return f"{CLEAN_PREFIX}event_date={day:%Y-%m-%d}/data.parquet"


def _typed(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[OUTPUT_COLUMNS].astype(LAKE_SCHEMA).reset_index(drop=True)


def to_parquet_bytes(rows: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    _typed(rows).to_parquet(buffer, index=False, engine="pyarrow")
    return buffer.getvalue()


def from_parquet_bytes(data: bytes) -> pd.DataFrame:
    rows = pd.read_parquet(io.BytesIO(data), engine="pyarrow")
    return _typed(rows)


def read_clean_day(store: ObjectStore, day: date) -> pd.DataFrame:
    """The clean rows stored for `day`, or an empty table with the right columns."""
    key = clean_key(day)
    if not store.exists(key):
        return _typed(pd.DataFrame(columns=OUTPUT_COLUMNS))
    return from_parquet_bytes(store.get(key))


def _same_content(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Row by row (same index): True where every content column is equal."""
    left = a[CONTENT_COLUMNS].astype("string").fillna("<NA>")
    right = b[CONTENT_COLUMNS].astype("string").fillna("<NA>")
    return (left == right).all(axis=1)


def merge_day(existing: pd.DataFrame, new: pd.DataFrame, day: date) -> tuple[pd.DataFrame, DayMerge]:
    """Merge `new` rows (all for `day`, IDs unique) into `existing`. Pure, no storage."""
    existing = existing.set_index("transaction_id", drop=False)
    new = new.set_index("transaction_id", drop=False)

    seen = new.index.isin(existing.index)
    fresh = new[~seen]
    again = new[seen]
    identical = _same_content(again, existing.loc[again.index]) if len(again) else pd.Series(dtype=bool)
    changed = again[~identical.reindex(again.index, fill_value=False)]

    merged = pd.concat([existing.drop(index=changed.index), changed, fresh])
    # The index is also called transaction_id; unname it so sorting by the column is clear.
    merged = merged.rename_axis(None).sort_values(["event_time", "transaction_id"], kind="stable")
    summary = DayMerge(
        day=day,
        added=len(fresh),
        replaced=len(changed),
        unchanged=len(again) - len(changed),
        total=len(merged),
        written=bool(len(fresh) or len(changed)),
    )
    return _typed(merged), summary


def write_clean(store: ObjectStore, rows: pd.DataFrame) -> list[DayMerge]:
    """Merge clean rows (output of run_etl) into the lake, one day at a time."""
    if rows["transaction_id"].duplicated().any():
        raise ValueError("Rows to write must have unique transaction IDs (run remove_duplicates first).")

    results = []
    for day, day_rows in rows.groupby("event_date", sort=True):
        existing = read_clean_day(store, day)
        merged, summary = merge_day(existing, day_rows, day)
        if summary.written:
            store.put(clean_key(day), to_parquet_bytes(merged))
        results.append(summary)
    return results


def list_clean_days(store: ObjectStore) -> list[date]:
    days = []
    for key in store.list(CLEAN_PREFIX):
        part = key[len(CLEAN_PREFIX):].split("/", 1)[0]  # "event_date=2026-03-01"
        if part.startswith("event_date="):
            days.append(date.fromisoformat(part.split("=", 1)[1]))
    return sorted(days)
