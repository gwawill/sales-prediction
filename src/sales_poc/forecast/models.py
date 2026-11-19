"""Four forecasting models for one weekly series (FR-23).

All four come from the exponential smoothing family, except the baseline:

    baseline      average of the last 4 weeks, as a flat line. The bar to beat.
    simple        simple exponential smoothing: a smoothed level, as a flat line.
    holt          Holt's method: level + a damped trend (the slope flattens over time).
    holt_winters  Holt-Winters: level + damped trend + a 13-week season. 13 weeks is one
                  quarter, so it can learn the quarter-end rush. It needs to see the season
                  at least twice, so it is only used with 26 or more weeks of data.

    result = forecast(series, "holt_winters", horizon=4)
    result.ok, result.values, result.note

Forecasts are never below zero (negative sales make no sense). A model that cannot be fitted
says so in `note` instead of raising, so one awkward series does not stop the others.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

MODELS = ("baseline", "simple", "holt", "holt_winters")
SEASON_WEEKS = 13
BASELINE_WEEKS = 4


@dataclass(frozen=True)
class Forecast:
    model: str
    values: np.ndarray        # one value per future week; empty if not ok
    ok: bool = True
    note: str = ""


def _baseline(history: np.ndarray, horizon: int) -> np.ndarray:
    return np.full(horizon, history[-BASELINE_WEEKS:].mean())


def _smoothing(history: np.ndarray, horizon: int, trend: bool, season: bool) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    model = ExponentialSmoothing(
        history,
        trend="add" if trend else None,
        damped_trend=trend,
        seasonal="add" if season else None,
        seasonal_periods=SEASON_WEEKS if season else None,
        initialization_method="estimated",
    )
    with warnings.catch_warnings():
        # statsmodels warns when its optimiser stops early; the fit is still usable, and
        # the evaluation in T10.3 will show if a model does badly.
        warnings.simplefilter("ignore")
        fitted = model.fit(optimized=True)
    return np.asarray(fitted.forecast(horizon), dtype=float)


def forecast(series: pd.Series | np.ndarray, model: str, horizon: int = 4) -> Forecast:
    """Forecast the next `horizon` weeks of one weekly series with one model."""
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; choose from {MODELS}")
    history = np.asarray(series, dtype=float)

    if len(history) < BASELINE_WEEKS:
        return Forecast(model, np.array([]), ok=False, note=f"needs at least {BASELINE_WEEKS} weeks")
    if model == "holt_winters" and len(history) < 2 * SEASON_WEEKS:
        return Forecast(model, np.array([]), ok=False,
                        note=f"needs at least {2 * SEASON_WEEKS} weeks to learn a {SEASON_WEEKS}-week season")

    try:
        if model == "baseline":
            values = _baseline(history, horizon)
        elif np.all(history == history[0]):
            # A series that never changes (often all zeros) forecasts itself. statsmodels
            # would fit it perfectly and then divide by zero in its quality scores.
            values = np.full(horizon, history[0])
        else:
            values = _smoothing(history, horizon, trend=model != "simple", season=model == "holt_winters")
    except Exception as exc:  # noqa: BLE001 - one awkward series must not stop the others
        return Forecast(model, np.array([]), ok=False, note=f"could not be fitted: {exc}")

    if not np.all(np.isfinite(values)):
        return Forecast(model, np.array([]), ok=False, note="produced non-numeric values")
    return Forecast(model, np.clip(values, 0, None))
