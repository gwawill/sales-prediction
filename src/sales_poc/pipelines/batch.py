"""Batch pipeline: load daily sales files into the lake (FR-6, FR-34).

    batch-load --once                 load every new file in data/incoming/ and stop
    batch-load --every 60             do that every 60 minutes until Ctrl+C
    batch-load --every 0.25 --max-passes 4    every 15 seconds, 4 times (for trying it out)

One command = one run = one quality report:
    1. find the daily files (sales_YYYY-MM-DD.csv), oldest first
    2. skip files already loaded (state/batch_loaded.json, see batch_state.py)
    3. save each new one unchanged to raw/batch/                    (FR-8)
    4. read them all, run the shared ETL once                       (FR-9)
    5. merge clean rows into clean/transactions/                    (FR-16, FR-17)
    6. write quarantine/ and reports/ for this run                  (FR-14, FR-20)
    7. add the files to the loaded list
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sales_poc.config import get_app_settings
from sales_poc.datagen.history import COLUMNS
from sales_poc.datagen.reference import ReferenceData, load_reference_data
from sales_poc.etl.pipeline import run_etl
from sales_poc.lake.clean import write_clean
from sales_poc.lake.raw import save_raw_batch_file
from sales_poc.lake.runs import build_report, new_run_id, write_quarantine, write_report
from sales_poc.lake.store import ObjectStore, s3_store
from sales_poc.log import get_logger, setup_logging
from sales_poc.pipelines.batch_state import ChangedFileError, mark_loaded, new_files

log = get_logger(__name__)

FILE_PATTERN = "sales_*.csv"


@dataclass(frozen=True)
class BatchRun:
    run_id: str
    report: dict
    files: list[Path]


def find_daily_files(incoming: Path) -> list[Path]:
    """The daily files in `incoming`, oldest first (the date is in the name)."""
    return sorted(Path(incoming).glob(FILE_PATTERN))


def read_daily_file(path: Path) -> pd.DataFrame:
    """Every value as text. Only truly empty cells are missing ("NA" stays "NA")."""
    rows = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    missing = [c for c in COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}")
    return rows[COLUMNS]


DEFAULT_WORKERS = 16


def run_batch(
    store: ObjectStore,
    files: list[Path],
    ref: ReferenceData,
    now: datetime | None = None,
    workers: int = DEFAULT_WORKERS,
) -> BatchRun | None:
    """Load `files` as one run. Returns None if there is nothing to load."""
    if not files:
        log.info("batch_nothing_to_load")
        return None

    started = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run_id = new_run_id("batch", started)
    log.info("batch_started", extra={"run_id": run_id, "files": len(files)})

    # Each file is independent, so upload several at once (S3 requests wait on the network).
    with ThreadPoolExecutor(max_workers=workers) as pool:
        saved = list(pool.map(lambda p: save_raw_batch_file(store, p), files))
    raw_keys = [key for key, _ in saved]
    log.info("raw_saved", extra={"run_id": run_id, "files": len(saved),
                                 "newly_written": sum(written for _, written in saved)})

    raw = pd.concat([read_daily_file(p) for p in files], ignore_index=True)
    result = run_etl(raw, ref, source="batch", ingested_at=started)
    log.info("etl_done", extra={"run_id": run_id, **{k: v for k, v in result.counts.as_dict().items()
                                                       if isinstance(v, int)}})

    merges = write_clean(store, result.clean, workers=workers)
    q_key = write_quarantine(store, run_id, result.quarantine)
    finished = datetime.now(timezone.utc)
    report = build_report(run_id, "batch", started, finished, raw_keys, result.counts, merges, q_key)
    report_key = write_report(store, report)
    log.info("batch_finished", extra={"run_id": run_id, "report": report_key,
                                      "seconds": round((finished - started).total_seconds(), 1)})
    return BatchRun(run_id=run_id, report=report, files=files)


def load_new_files(
    store: ObjectStore,
    files: list[Path],
    ref: ReferenceData,
    now: datetime | None = None,
    workers: int = DEFAULT_WORKERS,
) -> tuple[BatchRun | None, list[Path]]:
    """Load only the files not loaded before. Returns (the run or None, skipped files)."""
    pending, skipped = new_files(store, files)
    log.info("batch_files_checked", extra={"new": len(pending), "already_loaded": len(skipped)})
    run = run_batch(store, pending, ref, now=now, workers=workers)
    if run is not None:
        mark_loaded(store, pending, run.run_id)  # only after the run succeeded
    return run, skipped


def print_summary(run: BatchRun | None, skipped: list[Path] = ()) -> None:
    if skipped:
        print(f"Skipped {len(skipped)} files that were already loaded.")
    if run is None:
        print("Nothing new to load.")
        return
    c, lake = run.report["counts"], run.report["lake"]
    print(f"Run {run.run_id}: {len(run.files)} files, {c['rows_in']} rows in")
    print(f"  clean {c['clean']} (flagged {c['clean_flagged']}), duplicates removed "
          f"{c['duplicates_removed']}, quarantined {c['quarantined']}")
    print(f"  lake: {lake['rows_added']} added, {lake['rows_replaced']} replaced, "
          f"{lake['rows_unchanged']} unchanged, {len(lake['days_written'])} of "
          f"{lake['days_touched']} days written")
    print(f"  report: reports/run={run.run_id}.json")


def run_every(
    one_pass: Callable[[], None],
    every_seconds: float,
    max_passes: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Call `one_pass` now and then every `every_seconds`, until Ctrl+C or `max_passes`.

    A pass that fails is logged, and the next pass tries again. Returns the number of
    passes that failed. `sleep` can be replaced in tests so they don't really wait.
    """
    passes = failures = 0
    try:
        while True:
            passes += 1
            log.info("batch_pass_started", extra={"pass": passes})
            try:
                one_pass()
            except Exception as exc:  # noqa: BLE001 - keep the schedule alive
                failures += 1
                log.exception("batch_pass_failed", extra={"pass": passes})
                print(f"Pass {passes} failed: {exc}. Trying again next time.")
            if max_passes is not None and passes >= max_passes:
                break
            sleep(every_seconds)
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C.")
    return failures


def parse_args(argv: list[str]) -> argparse.Namespace:
    data_dir = get_app_settings().data_dir
    parser = argparse.ArgumentParser(prog="batch-load", description="Load daily sales files into the lake.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="load all new files once and stop")
    mode.add_argument("--every", type=float, metavar="MINUTES",
                      help="load new files now and then every MINUTES (fractions allowed)")
    parser.add_argument("--max-passes", type=int, default=None,
                        help="with --every: stop after this many passes")
    parser.add_argument("--incoming", type=Path, default=data_dir / "incoming")
    parser.add_argument("--reference", type=Path, default=data_dir / "reference")
    parser.add_argument("--limit", type=int, default=None, help="only the first N files (for trying it out)")
    args = parser.parse_args(argv)
    if args.every is not None and args.every <= 0:
        parser.error("--every must be more than 0 minutes")
    if args.max_passes is not None and args.every is None:
        parser.error("--max-passes only works with --every")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    # Write each line out at once, even when output goes to a file, so a long-running
    # schedule's log shows every pass as it happens.
    sys.stdout.reconfigure(line_buffering=True)
    setup_logging()
    store = s3_store()

    def one_pass() -> None:
        # Files and reference data are read fresh each pass, so new files are picked up.
        files = find_daily_files(args.incoming)[: args.limit]
        run, skipped = load_new_files(store, files, load_reference_data(args.reference))
        print(f"[{datetime.now():%H:%M:%S}]", end=" ")
        print_summary(run, skipped)

    if args.once:
        try:
            one_pass()
        except ChangedFileError as exc:
            print(f"Stopped: {exc}")
            return 1
        return 0

    print(f"Loading new files every {args.every:g} minutes. Press Ctrl+C to stop.")
    failures = run_every(one_pass, args.every * 60, args.max_passes)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
