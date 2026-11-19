"""Per-run files: quarantined rows and the quality report (FR-14, FR-20).

Every ETL run gets a run ID that sorts by time, like 20261118T093000Z-batch-a1b2c3:

    quarantine/run=<run id>/rows.parquet    rejected rows: raw text + reject_reasons
    reports/run=<run id>.json               the quality report for that run

A run with no rejected rows writes no quarantine file; its report says so.
"""

from __future__ import annotations

import io
import json
import re
import uuid
from datetime import datetime, timezone

import pandas as pd

from sales_poc.etl.quality import QualityCounts
from sales_poc.lake.clean import DayMerge
from sales_poc.lake.store import ObjectStore

QUARANTINE_PREFIX = "quarantine/"
REPORTS_PREFIX = "reports/"
_RUN_ID = re.compile(r"^\d{8}T\d{6}Z-(batch|stream)-[0-9a-f]{6}$")


def new_run_id(source: str, started: datetime | None = None) -> str:
    started = (started or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"{started:%Y%m%dT%H%M%SZ}-{source}-{uuid.uuid4().hex[:6]}"


def _check(run_id: str) -> str:
    if not _RUN_ID.match(run_id):
        raise ValueError(f"Not a run ID: {run_id!r}")
    return run_id


def quarantine_key(run_id: str) -> str:
    return f"{QUARANTINE_PREFIX}run={_check(run_id)}/rows.parquet"


def report_key(run_id: str) -> str:
    return f"{REPORTS_PREFIX}run={_check(run_id)}.json"


def write_quarantine(store: ObjectStore, run_id: str, rows: pd.DataFrame) -> str | None:
    """Save the rejected rows as Parquet. Returns the key, or None if there were none."""
    if rows.empty:
        return None
    table = rows.copy()
    for column in table.columns:
        if column != "ingested_at":
            table[column] = table[column].astype("string")  # raw text stays text
    buffer = io.BytesIO()
    table.to_parquet(buffer, index=False, engine="pyarrow")
    key = quarantine_key(run_id)
    store.put(key, buffer.getvalue())
    return key


def read_quarantine(store: ObjectStore, run_id: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(store.get(quarantine_key(run_id))), engine="pyarrow")


def build_report(
    run_id: str,
    source: str,
    started_at: datetime,
    finished_at: datetime,
    inputs: list[str],
    counts: QualityCounts,
    merges: list[DayMerge],
    quarantine_key: str | None,
) -> dict:
    """The quality report for one run, as plain JSON-ready values."""
    return {
        "run_id": run_id,
        "source": source,
        "started_at": started_at.astimezone(timezone.utc).isoformat(),
        "finished_at": finished_at.astimezone(timezone.utc).isoformat(),
        "inputs": inputs,
        "counts": counts.as_dict(),
        "lake": {
            "days_touched": len(merges),
            "days_written": [m.day.isoformat() for m in merges if m.written],
            "rows_added": sum(m.added for m in merges),
            "rows_replaced": sum(m.replaced for m in merges),
            "rows_unchanged": sum(m.unchanged for m in merges),
        },
        "quarantine_key": quarantine_key,
    }


def write_report(store: ObjectStore, report: dict) -> str:
    key = report_key(report["run_id"])
    store.put(key, (json.dumps(report, indent=2) + "\n").encode("utf-8"))
    return key


def read_report(store: ObjectStore, run_id: str) -> dict:
    return json.loads(store.get(report_key(run_id)))


def list_run_ids(store: ObjectStore) -> list[str]:
    """All run IDs with a report, oldest first."""
    ids = []
    for key in store.list(REPORTS_PREFIX):
        name = key[len(REPORTS_PREFIX):]
        if name.startswith("run=") and name.endswith(".json"):
            ids.append(name[len("run="):-len(".json")])
    return sorted(ids)


def latest_report(store: ObjectStore, source: str | None = None) -> dict | None:
    """The newest report (optionally only for 'batch' or 'stream'), or None."""
    ids = [r for r in list_run_ids(store) if source is None or f"-{source}-" in r]
    return read_report(store, ids[-1]) if ids else None


def counts_from_report(report: dict) -> QualityCounts:
    return QualityCounts(**report["counts"])
