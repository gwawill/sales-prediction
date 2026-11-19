"""Make the forecast and save it to the lake (FR-23, FR-25).

    forecast-run          lake -> weekly totals -> backtest -> best model per series
                          -> forecast the next 4 weeks -> forecasts/run=<id>/

Saved per run:
    forecasts/run=<id>/forecast.parquet   series x future week: forecast revenue, the model
                                          used, and that model's backtest error (WAPE)
    forecasts/run=<id>/scores.parquet     every series x model backtest score
    forecasts/run=<id>/summary.json       weeks used, test weeks, forecast weeks, overall
                                          error and wins per model

Each forecast row carries its model's measured error, so whoever reads the forecast (the AI
layer) can also say how reliable it is.
"""

from __future__ import annotations

import io
import json
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from sales_poc.forecast.evaluate import HORIZON, Evaluation, evaluate
from sales_poc.forecast.models import forecast
from sales_poc.forecast.weekly import SERIES_KEYS, WeeklySales, weekly_sales
from sales_poc.lake.store import ObjectStore

FORECASTS_PREFIX = "forecasts/"
_RUN_ID = re.compile(r"^\d{8}T\d{6}Z-forecast-[0-9a-f]{6}$")


@dataclass(frozen=True)
class ForecastRun:
    run_id: str
    table: pd.DataFrame       # category, region_id, week_start, forecast_revenue, model, backtest_wape, baseline_wape
    evaluation: Evaluation
    summary: dict


def new_forecast_run_id(now: datetime | None = None) -> str:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"{now:%Y%m%dT%H%M%SZ}-forecast-{uuid.uuid4().hex[:6]}"


def _key(run_id: str, name: str) -> str:
    if not _RUN_ID.match(run_id):
        raise ValueError(f"Not a forecast run ID: {run_id!r}")
    return f"{FORECASTS_PREFIX}run={run_id}/{name}"


def future_weeks(last_week: date, horizon: int) -> list[date]:
    return [last_week + timedelta(weeks=i) for i in range(1, horizon + 1)]


def make_forecast(weekly: WeeklySales, horizon: int = HORIZON, run_id: str | None = None) -> ForecastRun:
    """Pick the best model per series by backtest, then forecast with it on all weeks."""
    ev = evaluate(weekly, horizon=horizon)
    weeks = future_weeks(weekly.weeks_used[-1], horizon)
    best = ev.best.set_index(SERIES_KEYS)

    rows = []
    for (category, region), group in weekly.table.groupby(SERIES_KEYS):
        history = group.sort_values("week_start")["revenue"].to_numpy(dtype=float)
        choice = best.loc[(category, region)]
        result = forecast(history, choice["best_model"], horizon)
        if not result.ok:  # very unlikely: it fitted fine in the backtest
            result = forecast(history, "baseline", horizon)
        for week, value in zip(weeks, result.values):
            rows.append({
                "category": category, "region_id": region, "week_start": week,
                "forecast_revenue": round(float(value), 2), "model": result.model,
                "backtest_wape": round(float(choice["best_wape"]), 4),
                "baseline_wape": round(float(choice["baseline_wape"]), 4),
            })
    table = pd.DataFrame(rows)

    summary_models = ev.summary.copy()
    summary = {
        "run_id": run_id or new_forecast_run_id(),
        "value": "revenue",
        "weeks_used": [str(weekly.weeks_used[0]), str(weekly.weeks_used[-1])],
        "weeks_left_out": {str(k): v for k, v in weekly.weeks_left_out.items()},
        "test_weeks": [str(ev.test_weeks[0]), str(ev.test_weeks[-1])],
        "backtest": {"windows": ev.folds, "weeks_per_window": ev.horizon, "measure": "WAPE"},
        "forecast_weeks": [str(w) for w in weeks],
        "series": int(len(best)),
        "best_beats_baseline": int(ev.best["beats_baseline"].sum()),
        "models": [
            {"model": r.model, "overall_wape": round(float(r.overall_wape), 4), "series_won": int(r.series_won)}
            for r in summary_models.itertuples()
        ],
    }
    return ForecastRun(summary["run_id"], table, ev, summary)


def _parquet(table: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    table.to_parquet(buffer, index=False, engine="pyarrow")
    return buffer.getvalue()


def save_forecast(store: ObjectStore, run: ForecastRun) -> list[str]:
    keys = [_key(run.run_id, n) for n in ("forecast.parquet", "scores.parquet", "summary.json")]
    store.put(keys[0], _parquet(run.table))
    store.put(keys[1], _parquet(run.evaluation.scores))
    store.put(keys[2], (json.dumps(run.summary, indent=2) + "\n").encode("utf-8"))
    return keys


def list_forecast_runs(store: ObjectStore) -> list[str]:
    ids = set()
    for key in store.list(FORECASTS_PREFIX):
        part = key[len(FORECASTS_PREFIX):].split("/", 1)[0]
        if part.startswith("run=") and _RUN_ID.match(part[4:]):
            ids.add(part[4:])
    return sorted(ids)


def latest_forecast(store: ObjectStore) -> tuple[pd.DataFrame, dict] | None:
    """(forecast table, summary) of the newest complete run, or None.

    summary.json is saved last, so a run without it did not finish saving and is skipped.
    """
    for run_id in reversed(list_forecast_runs(store)):
        if not store.exists(_key(run_id, "summary.json")):
            continue  # a run that did not finish saving
        summary = json.loads(store.get(_key(run_id, "summary.json")))
        table = pd.read_parquet(io.BytesIO(store.get(_key(run_id, "forecast.parquet"))))
        table["week_start"] = pd.to_datetime(table["week_start"]).dt.date
        return table, summary
    return None


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from sales_poc.lake.clean import read_all_clean
    from sales_poc.lake.store import s3_store

    argparse.ArgumentParser(prog="forecast-run", description="Forecast the next 4 weeks and save it.").parse_args(
        sys.argv[1:] if argv is None else argv)
    sys.stdout.reconfigure(line_buffering=True)
    store = s3_store()
    weekly = weekly_sales(read_all_clean(store))
    run = make_forecast(weekly)
    keys = save_forecast(store, run)

    s = run.summary
    print(f"Forecast {run.run_id}: {s['series']} series x {len(s['forecast_weeks'])} weeks "
          f"({s['forecast_weeks'][0]} to {s['forecast_weeks'][-1]}), learned from {s['weeks_used'][0]} to {s['weeks_used'][1]}")
    for m in s["models"]:
        print(f"  {m['model']:13s} overall error {m['overall_wape']:.1%}, best for {m['series_won']} series")
    print(f"  best model beats the baseline in {s['best_beats_baseline']} of {s['series']} series")
    print(f"  saved: {', '.join(keys)}")
    return 0
