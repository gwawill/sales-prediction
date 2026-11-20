"""An in-memory copy of the clean lake that keeps up by re-reading only the days that changed.

    mirror = LakeMirror(s3_store())
    mirror.refresh()       # first call reads every day (about 30 seconds on S3)
    mirror.refresh()       # later calls read only new or changed days; [] if nothing changed
    rows = mirror.rows()   # every clean row, oldest day first

How it knows what changed: listing the clean folder gives each day file's fingerprint (its S3
ETag) without downloading anything. A day whose fingerprint differs from the one read last
time is read again. Days that disappear from the lake are dropped.

Used by sync-stores, the agent's SQL tool (so the web API sees new orders) and the dashboard.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd

from sales_poc.lake.clean import CLEAN_PREFIX, empty_clean, read_clean_day
from sales_poc.lake.store import ObjectStore

READ_WORKERS = 16


def day_stamps(store: ObjectStore) -> dict[date, str]:
    """{day: fingerprint} for every clean day file in the lake."""
    stamps = {}
    for key, stamp in store.stamps(CLEAN_PREFIX).items():
        part = key[len(CLEAN_PREFIX):].split("/", 1)[0]  # "event_date=2026-03-01"
        if part.startswith("event_date="):
            stamps[date.fromisoformat(part.split("=", 1)[1])] = stamp
    return stamps


class LakeMirror:
    def __init__(self, store: ObjectStore, workers: int = READ_WORKERS):
        self.store, self.workers = store, workers
        self._days: dict[date, pd.DataFrame] = {}
        self._stamps: dict[date, str] = {}

    @property
    def stamps(self) -> dict[date, str]:
        """{day: fingerprint} of the days held now."""
        return dict(self._stamps)

    def refresh(self) -> list[date]:
        """Read new or changed days and drop days that are gone. Returns the days that changed."""
        current = day_stamps(self.store)
        to_read = [day for day, stamp in current.items() if self._stamps.get(day) != stamp]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            tables = list(pool.map(lambda d: read_clean_day(self.store, d), to_read))
        for day, table in zip(to_read, tables):
            # A file rewritten between the listing and the read gives newer rows with the older
            # fingerprint, so the next refresh simply reads it again.
            self._days[day], self._stamps[day] = table, current[day]
        gone = sorted(set(self._days) - set(current))
        for day in gone:
            del self._days[day], self._stamps[day]
        return sorted(to_read + gone)

    def rows(self, days: list[date] | None = None) -> pd.DataFrame:
        """The rows of `days` (default: every day held), oldest day first."""
        wanted = sorted(self._days if days is None else [d for d in days if d in self._days])
        if not wanted:
            return empty_clean()
        return pd.concat([self._days[d] for d in wanted], ignore_index=True)
