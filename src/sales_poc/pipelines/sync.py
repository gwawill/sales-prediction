"""Keep the graph and the summaries up to date with the lake.

    sync-stores --once          bring Neo4j and S3 Vectors up to date with the lake, then stop
    sync-stores --every 5       do that now and then every 5 minutes, until Ctrl+C

The batch pipeline and the stream consumer only write to the lake. This program follows them,
like a separate service in a real platform. Each pass:

  1. lists the clean day files with their fingerprints (S3 ETags), without downloading them
  2. reads only the days that are new or changed into its in-memory copy of the lake
  3. loads those days' transactions into Neo4j (MERGE, so loading again is harmless), and
     remembers their fingerprints in state/graph_synced.json, so the next pass skips them
  4. rebuilds the summaries from the in-memory lake, which is quick, and embeds and stores
     only the ones whose text changed; summaries that no longer exist are removed

New orders on a known day change only a handful of summaries (their category in their
region, their region, their customer). A whole new day changes every customer summary,
because each one names the period's last day, so those are re-embedded about once a day.

The first pass reads the whole lake (about 30 seconds). If state/graph_synced.json does not
exist yet, the first pass also loads every day into Neo4j once, which takes a few minutes.
After that, a pass with nothing new takes a second or two.

Days are updated a few minutes after they reach the lake, not instantly: Neo4j and the
embedding model are far slower than writing a Parquet file, so they catch up in passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

import pandas as pd

from sales_poc.datagen.reference import ReferenceData
from sales_poc.graph.loader import load_reference, load_transactions
from sales_poc.lake.clean import CLEAN_PREFIX, read_clean_day
from sales_poc.lake.store import ObjectStore
from sales_poc.log import get_logger
from sales_poc.vectors.embeddings import Embedder
from sales_poc.vectors.store import TEXT_KEY, VectorIndex
from sales_poc.vectors.summaries import Summary, build_summaries

log = get_logger(__name__)

STATE_KEY = "state/graph_synced.json"
READ_WORKERS = 16


@dataclass(frozen=True)
class SyncResult:
    days_read: int              # days (re)read from the lake into memory
    graph_days: int             # days loaded into Neo4j
    graph_transactions: int     # transactions sent to Neo4j
    summaries_stored: int       # summaries embedded and stored (new or changed text)
    summaries_deleted: int      # summaries removed because they no longer exist
    summaries_total: int        # summaries in the index afterwards

    @property
    def changed(self) -> bool:
        return bool(self.graph_days or self.summaries_stored or self.summaries_deleted)


def day_stamps(store: ObjectStore) -> dict[date, str]:
    """{day: fingerprint} for every clean day file in the lake."""
    stamps = {}
    for key, stamp in store.stamps(CLEAN_PREFIX).items():
        part = key[len(CLEAN_PREFIX):].split("/", 1)[0]  # "event_date=2026-03-01"
        if part.startswith("event_date="):
            stamps[date.fromisoformat(part.split("=", 1)[1])] = stamp
    return stamps


def read_graph_state(store: ObjectStore) -> dict[date, str]:
    """{day: fingerprint} of the days already loaded into Neo4j. Empty if none yet."""
    if not store.exists(STATE_KEY):
        return {}
    days = json.loads(store.get(STATE_KEY))["days"]
    return {date.fromisoformat(day): stamp for day, stamp in days.items()}


def write_graph_state(store: ObjectStore, days: dict[date, str]) -> None:
    body = {"days": {day.isoformat(): stamp for day, stamp in sorted(days.items())}}
    store.put(STATE_KEY, (json.dumps(body, indent=2) + "\n").encode("utf-8"))


def changed_summaries(summaries: list[Summary], stored: dict[str, dict]) -> tuple[list[Summary], list[str]]:
    """(summaries to store because they are new or their text or labels changed,
    keys to delete because their summary no longer exists)."""
    to_store = [s for s in summaries if stored.get(s.summary_id) != {**s.metadata, TEXT_KEY: s.text}]
    stale = sorted(set(stored) - {s.summary_id for s in summaries})
    return to_store, stale


@dataclass
class StoreSync:
    """Brings Neo4j and the vector index up to date with the lake. Call `run_once` repeatedly;
    it keeps the lake and the stored summaries in memory between passes."""

    store: ObjectStore
    driver: object               # a neo4j Driver (anything with execute_query)
    index: VectorIndex
    embedder: Embedder
    ref: ReferenceData
    workers: int = READ_WORKERS
    _days: dict[date, pd.DataFrame] = field(default_factory=dict)   # the lake, one table per day
    _read_stamps: dict[date, str] = field(default_factory=dict)     # fingerprints of _days
    _stored: dict[str, dict] | None = None                           # the index's summaries
    _reference_loaded: bool = False

    def _refresh_lake(self, current: dict[date, str]) -> int:
        """Read new or changed days into memory and drop days that are gone. Returns days read."""
        to_read = [day for day, stamp in current.items() if self._read_stamps.get(day) != stamp]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            tables = list(pool.map(lambda d: read_clean_day(self.store, d), to_read))
        for day, table in zip(to_read, tables):
            self._days[day], self._read_stamps[day] = table, current[day]
        for day in set(self._days) - set(current):
            del self._days[day], self._read_stamps[day]
        return len(to_read)

    def _rows(self, days: list[date]) -> pd.DataFrame:
        """The in-memory rows of `days` (all of them already read), oldest day first."""
        return pd.concat([self._days[d] for d in sorted(days)], ignore_index=True)

    def _sync_graph(self, current: dict[date, str]) -> tuple[int, int]:
        """Load new or changed days into Neo4j. Returns (days, transactions)."""
        synced = read_graph_state(self.store)
        pending = sorted(day for day, stamp in current.items() if synced.get(day) != stamp)
        if not pending:
            return 0, 0
        if not self._reference_loaded:
            load_reference(self.driver, self.ref)   # constraints and reference nodes, once per start
            self._reference_loaded = True
        sent = load_transactions(self.driver, self._rows(pending))
        # Only now, after Neo4j accepted them, are the days marked as done. Every day in the
        # lake is now either newly loaded or was already loaded with the same fingerprint.
        write_graph_state(self.store, current)
        return len(pending), sent

    def _sync_vectors(self) -> tuple[int, int, int]:
        """Store changed summaries and delete stale ones. Returns (stored, deleted, total)."""
        if self._stored is None:
            self.index.ensure()
            self._stored = self.index.stored()
        if not self._days:
            # An empty lake would mean "delete every summary". That is far more likely a
            # mistake (wrong bucket or prefix) than real, so leave the index alone.
            return 0, 0, len(self._stored)
        summaries = build_summaries(self._rows(list(self._days)))
        to_store, stale = changed_summaries(summaries, self._stored)
        if to_store:
            self.index.put(to_store, self.embedder.embed_documents([s.text for s in to_store]))
            self._stored.update({s.summary_id: {**s.metadata, TEXT_KEY: s.text} for s in to_store})
        if stale:
            self.index.delete(stale)
            for key in stale:
                del self._stored[key]
        return len(to_store), len(stale), len(self._stored)

    def run_once(self) -> SyncResult:
        current = day_stamps(self.store)
        days_read = self._refresh_lake(current)
        graph_days, sent = self._sync_graph(current)
        stored, deleted, total = self._sync_vectors()
        result = SyncResult(days_read, graph_days, sent, stored, deleted, total)
        log.info("sync_pass_done", extra=asdict(result))
        return result


def describe(result: SyncResult) -> str:
    if not result.changed:
        return "Nothing new: the graph and the summaries already match the lake."
    return (f"Graph: {result.graph_transactions} transactions on {result.graph_days} days loaded. "
            f"Summaries: {result.summaries_stored} stored, {result.summaries_deleted} removed, "
            f"{result.summaries_total} in the index.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    from sales_poc.config import get_app_settings

    parser = argparse.ArgumentParser(prog="sync-stores",
                                     description="Keep Neo4j and the summaries up to date with the lake.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="bring both up to date once and stop")
    mode.add_argument("--every", type=float, metavar="MINUTES",
                      help="do that now and then every MINUTES (fractions allowed)")
    parser.add_argument("--max-passes", type=int, default=None, help="with --every: stop after this many passes")
    parser.add_argument("--reference", default=str(get_app_settings().data_dir / "reference"))
    args = parser.parse_args(argv)
    if args.every is not None and args.every <= 0:
        parser.error("--every must be more than 0 minutes")
    if args.max_passes is not None and args.every is None:
        parser.error("--max-passes only works with --every")
    return args


def main(argv: list[str] | None = None) -> int:
    """sync-stores: follow the lake and update Neo4j and S3 Vectors."""
    from sales_poc.datagen.reference import load_reference_data
    from sales_poc.graph.connection import driver as neo4j_driver
    from sales_poc.lake.store import s3_store
    from sales_poc.log import setup_logging
    from sales_poc.pipelines.batch import run_every
    from sales_poc.vectors.embeddings import get_embedder
    from sales_poc.vectors.store import default_index

    args = parse_args(sys.argv[1:] if argv is None else argv)
    sys.stdout.reconfigure(line_buffering=True)
    setup_logging()
    store = s3_store()
    if not store.exists(STATE_KEY):
        print("No record of an earlier sync, so the first pass loads every day into Neo4j once "
              "(a few minutes).")

    with neo4j_driver() as drv:
        sync = StoreSync(store, drv, default_index(), get_embedder(), load_reference_data(args.reference))

        def one_pass() -> None:
            started = time.monotonic()
            result = sync.run_once()
            print(f"[{datetime.now():%H:%M:%S}] {describe(result)} ({time.monotonic() - started:.0f}s)")

        if args.once:
            one_pass()
            return 0
        print(f"Updating the graph and the summaries every {args.every:g} minutes. Press Ctrl+C to stop.")
        failures = run_every(one_pass, args.every * 60, args.max_passes)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
