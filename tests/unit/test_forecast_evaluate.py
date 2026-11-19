from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from sales_poc.forecast.evaluate import backtest, evaluate, mape, wape
from sales_poc.forecast.models import MODELS, SEASON_WEEKS
from sales_poc.forecast.weekly import WeeklySales

RNG = np.random.default_rng(1)
WEEKS = [date(2025, 11, 3) + timedelta(weeks=i) for i in range(52)]


def weekly_from(series_by_key: dict) -> WeeklySales:
    rows = []
    for (category, region), values in series_by_key.items():
        for week, v in zip(WEEKS, values):
            rows.append({"week_start": week, "category": category, "region_id": region,
                         "revenue": float(v), "orders": 1, "units": 1})
    return WeeklySales(pd.DataFrame(rows), WEEKS)


def seasonal():
    weeks = np.arange(52)
    return 100 + 80 * (weeks % SEASON_WEEKS >= SEASON_WEEKS - 2) + RNG.normal(0, 2, 52)


def steady():
    return 100 + RNG.normal(0, 3, 52)


def test_wape_and_mape():
    actual, predicted = np.array([100.0, 200.0]), np.array([110.0, 180.0])
    assert wape(actual, predicted) == pytest.approx(30 / 300)
    assert mape(actual, predicted) == pytest.approx((0.10 + 0.10) / 2)


def test_zero_weeks_break_mape_but_not_wape():
    actual, predicted = np.array([0.0, 100.0]), np.array([5.0, 95.0])
    assert np.isnan(mape(actual, predicted))
    assert wape(actual, predicted) == pytest.approx(10 / 100)
    assert wape(np.zeros(3), np.zeros(3)) == 0.0


def test_backtest_uses_only_weeks_before_each_window(monkeypatch):
    import sales_poc.forecast.evaluate as ev
    from sales_poc.forecast.models import Forecast

    seen = []

    def spy(history, model, horizon):
        seen.append(len(history))
        return Forecast(model, np.zeros(horizon))

    monkeypatch.setattr(ev, "forecast", spy)
    backtest(np.arange(52, dtype=float), "baseline", horizon=4, folds=3)

    assert seen == [40, 44, 48]  # never sees the window it predicts


def test_holt_winters_wins_a_seasonal_series_and_beats_the_baseline():
    ev = evaluate(weekly_from({("Server", "NA"): seasonal()}))
    [row] = ev.best.itertuples()

    assert row.best_model == "holt_winters"
    assert row.beats_baseline


def test_scores_best_and_summary_shapes():
    ev = evaluate(weekly_from({("Server", "NA"): seasonal(), ("Laptop", "EMEA"): steady()}))

    assert len(ev.scores) == 2 * len(MODELS)
    assert len(ev.best) == 2
    assert list(ev.summary["model"]) == list(MODELS)
    assert ev.summary["series_won"].sum() == 2
    assert ev.test_weeks == WEEKS[-12:]
    assert (ev.scores["windows"] == 3).all()


def test_overall_wape_is_total_error_over_total_sales():
    ev = evaluate(weekly_from({("A", "X"): steady(), ("B", "Y"): seasonal()}))
    s = ev.scores[ev.scores["model"] == "baseline"]
    expected = s["abs_error"].sum() / s["actual"].sum()
    got = ev.summary.set_index("model").loc["baseline", "overall_wape"]
    assert got == pytest.approx(expected)


def test_too_few_weeks_is_refused():
    short = WeeklySales(pd.DataFrame(columns=["week_start", "category", "region_id", "revenue"]), WEEKS[:10])
    with pytest.raises(ValueError, match="at least 16 weeks"):
        evaluate(short)
