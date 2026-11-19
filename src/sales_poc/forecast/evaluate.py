"""How good is each forecast model? Measured on weeks the model did not see (FR-24).

Rolling backtest: for each of the last FOLDS windows of HORIZON weeks, every model learns
from all the weeks before the window, then predicts the window. Errors are averaged over the
windows, so one lucky window cannot decide the winner.

Error measures:
  WAPE  total |error| / total actual sales. "Off by X% overall". Works with zero weeks.
        Used to choose the best model.
  MAPE  average of |error| / actual per week. The plan asked for it, so it is reported, but
        it cannot be calculated when an actual week is zero, so it is NaN then.

    ev = evaluate(weekly)
    ev.scores    one row per series x model: wape, mape, windows
    ev.best      one row per series: the winning model and the baseline's WAPE for comparison
    ev.summary   one row per model: overall WAPE across all series, and how often it won
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sales_poc.forecast.models import MODELS, forecast
from sales_poc.forecast.weekly import SERIES_KEYS, WeeklySales

HORIZON = 4
FOLDS = 3


@dataclass(frozen=True)
class Evaluation:
    scores: pd.DataFrame
    best: pd.DataFrame
    summary: pd.DataFrame
    horizon: int
    folds: int
    test_weeks: list


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    total = np.abs(actual).sum()
    if total == 0:
        return 0.0 if np.abs(predicted).sum() == 0 else float("inf")
    return float(np.abs(actual - predicted).sum() / total)


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    if np.any(actual == 0):
        return float("nan")
    return float(np.mean(np.abs(actual - predicted) / np.abs(actual)))


def backtest(series: np.ndarray, model: str, horizon: int = HORIZON, folds: int = FOLDS) -> dict:
    """Errors for one series and one model over the last `folds` windows."""
    actual_all, predicted_all = [], []
    for k in range(folds, 0, -1):
        cut = len(series) - k * horizon
        result = forecast(series[:cut], model, horizon)
        if not result.ok:
            return {"wape": np.nan, "mape": np.nan, "windows": 0, "note": result.note}
        actual_all.append(series[cut:cut + horizon])
        predicted_all.append(result.values)
    actual, predicted = np.concatenate(actual_all), np.concatenate(predicted_all)
    return {"wape": wape(actual, predicted), "mape": mape(actual, predicted), "windows": folds,
            "abs_error": float(np.abs(actual - predicted).sum()), "actual": float(np.abs(actual).sum()),
            "note": ""}


def evaluate(weekly: WeeklySales, value: str = "revenue", horizon: int = HORIZON, folds: int = FOLDS) -> Evaluation:
    if len(weekly.weeks_used) < folds * horizon + horizon:
        raise ValueError(f"Need at least {folds * horizon + horizon} weeks; have {len(weekly.weeks_used)}.")

    rows = []
    for (category, region), group in weekly.table.groupby(SERIES_KEYS):
        series = group.sort_values("week_start")[value].to_numpy(dtype=float)
        for model in MODELS:
            rows.append({"category": category, "region_id": region, "model": model,
                         **backtest(series, model, horizon, folds)})
    scores = pd.DataFrame(rows)

    usable = scores.dropna(subset=["wape"])
    winners = usable.loc[usable.groupby(SERIES_KEYS)["wape"].idxmin()]
    baseline = scores[scores["model"] == "baseline"].set_index(SERIES_KEYS)["wape"]
    best = winners[[*SERIES_KEYS, "model", "wape", "mape"]].rename(
        columns={"model": "best_model", "wape": "best_wape", "mape": "best_mape"}).reset_index(drop=True)
    best["baseline_wape"] = [baseline[(c, r)] for c, r in zip(best["category"], best["region_id"])]
    best["beats_baseline"] = best["best_wape"] < best["baseline_wape"]

    # Overall WAPE per model across every series: total error / total sales.
    totals = usable.groupby("model")[["abs_error", "actual"]].sum()
    summary = pd.DataFrame({
        "overall_wape": totals["abs_error"] / totals["actual"],
        "series_won": best["best_model"].value_counts().reindex(totals.index, fill_value=0),
    }).reindex(list(MODELS)).reset_index().rename(columns={"index": "model"})

    test_weeks = weekly.weeks_used[-folds * horizon:]
    return Evaluation(scores, best, summary, horizon, folds, test_weeks)
