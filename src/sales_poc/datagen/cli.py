"""The `generate-data` command: write all made-up data to the data folder (task T2.5).

    generate-data                      seed and folder from .env (RANDOM_SEED, DATA_DIR)
    generate-data --seed 7 --days 30
    generate-data --overwrite          replace files from an earlier run

Writes:
    <data>/reference/regions.csv, channels.csv, products.csv, customers.csv
    <data>/incoming/sales_YYYY-MM-DD.csv     one messy file per day (what the batch loads)
    <data>/problem_log.csv                   every problem added, for checking the ETL
    <data>/generation.json                   seed, dates and counts used
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from sales_poc.config import get_app_settings
from sales_poc.datagen.history import DEFAULT_DAYS, DEFAULT_START, generate_history, write_daily_files
from sales_poc.datagen.problems import add_problems
from sales_poc.datagen.reference import REFERENCE_TABLES, build_reference_data


def parse_args(argv: list[str]) -> argparse.Namespace:
    settings = get_app_settings()
    parser = argparse.ArgumentParser(
        prog="generate-data", description="Write all made-up sales data to the data folder."
    )
    parser.add_argument("--seed", type=int, default=settings.random_seed,
                        help="random seed; the same seed gives the same files (default: RANDOM_SEED)")
    parser.add_argument("--out", type=Path, default=settings.data_dir,
                        help="data folder (default: DATA_DIR)")
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START,
                        help=f"first day, YYYY-MM-DD (default: {DEFAULT_START})")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"number of days (default: {DEFAULT_DAYS})")
    parser.add_argument("--customers", type=int, default=500, help="number of customers (default: 500)")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete files from an earlier run first")
    args = parser.parse_args(argv)
    if args.days < 1:
        parser.error("--days must be at least 1")
    return args


def generate(
    out: Path,
    seed: int,
    start: date = DEFAULT_START,
    days: int = DEFAULT_DAYS,
    customers: int = 500,
    overwrite: bool = False,
) -> dict:
    """Write everything and return a summary. Refuses to mix with an earlier run's files."""
    incoming = out / "incoming"
    if incoming.exists() and any(incoming.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{incoming} already has files. Run with --overwrite to replace them."
            )
        shutil.rmtree(incoming)

    ref = build_reference_data(seed=seed, n_customers=customers)
    reference_dir = out / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    for name in REFERENCE_TABLES:
        getattr(ref, name).to_csv(reference_dir / f"{name}.csv", index=False, lineterminator="\n")

    clean = generate_history(ref, seed=seed, start=start, days=days)
    messy = add_problems(clean, seed=seed)
    paths = write_daily_files(messy.files, incoming)
    messy.log.to_csv(out / "problem_log.csv", index=False, lineterminator="\n")

    summary = {
        "seed": seed,
        "start": start.isoformat(),
        "end": (start + timedelta(days=days - 1)).isoformat(),
        "days": days,
        "customers": customers,
        "products": len(ref.products),
        "clean_transactions": int(sum(len(rows) for rows in clean.values())),
        "rows_in_files": int(sum(len(rows) for rows in messy.files.values())),
        "problems": int(len(messy.log)),
        "problems_by_kind": {k: int(v) for k, v in messy.log["problem"].value_counts().sort_index().items()},
        "daily_files": len(paths),
    }
    (out / "generation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        summary = generate(args.out, args.seed, args.start, args.days, args.customers, args.overwrite)
    except FileExistsError as exc:
        print(f"Stopped: {exc}")
        return 1

    print(f"Wrote data for {summary['start']} to {summary['end']} (seed {summary['seed']}) into {args.out}/")
    print(f"  reference/   {len(REFERENCE_TABLES)} tables, {summary['customers']} customers, "
          f"{summary['products']} products")
    print(f"  incoming/    {summary['daily_files']} daily files, {summary['rows_in_files']} rows "
          f"({summary['clean_transactions']} real transactions)")
    print(f"  problem_log.csv  {summary['problems']} problems: "
          + ", ".join(f"{k} {v}" for k, v in summary["problems_by_kind"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
