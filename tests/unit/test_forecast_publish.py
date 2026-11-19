from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from sales_poc.forecast.models import SEASON_WEEKS
from sales_poc.forecast.publish import (
    future_weeks,
    latest_forecast,
    list_forecast_runs,
    make_forecast,
    new_forecast_run_id,
    save_forecast,
)
from sales_poc.forecast.weekly import WeeklySales
from sales_poc.lake.store import LocalStore

RNG = np.random.default_rng(3)
WEEKS = [date(2025, 11, 3) + timedelta(weeks=i) for i in range(52)]
SERIES = [("Laptop", "EMEA"), ("Server", "NA"), ("Monitor", "APJ")]


@pytest.fixture(scope="module")
def weekly():
    rows = []
    for i, (category, region) in enumerate(SERIES):
        n = np.arange(52)
        values = 100 + 80 * (n % SEASON_WEEKS >= 11) * (i == 1) + RNG.normal(0, 5, 52)
        for week, v in zip(WEEKS, values):
            rows.append({"week_start": week, "category": category, "region_id": region,
                         "revenue": float(v), "orders": 1, "units": 1})
    return WeeklySales(pd.DataFrame(rows), WEEKS)


@pytest.fixture(scope="module")
def run(weekly):
    return make_forecast(weekly)


def test_every_series_gets_the_next_four_weeks(run):
    assert len(run.table) == len(SERIES) * 4
    assert sorted(set(run.table["week_start"])) == future_weeks(WEEKS[-1], 4)
    assert future_weeks(WEEKS[-1], 4)[0] == WEEKS[-1] + timedelta(weeks=1)
    assert (run.table["forecast_revenue"] >= 0).all()


def test_each_series_uses_its_backtest_winner_and_carries_its_error(run):
    best = run.evaluation.best.set_index(["category", "region_id"])
    for (category, region), rows in run.table.groupby(["category", "region_id"]):
        assert set(rows["model"]) == {best.loc[(category, region), "best_model"]}
        assert set(rows["backtest_wape"]) == {round(best.loc[(category, region), "best_wape"], 4)}


def test_the_seasonal_series_is_forecast_by_holt_winters(run):
    assert set(run.table.loc[run.table["category"] == "Server", "model"]) == {"holt_winters"}


def test_summary_explains_the_run(run):
    s = run.summary
    assert s["series"] == 3
    assert s["weeks_used"] == ["2025-11-03", "2026-10-26"]
    assert s["forecast_weeks"][0] == "2026-11-02"
    assert s["backtest"] == {"windows": 3, "weeks_per_window": 4, "measure": "WAPE"}
    assert [m["model"] for m in s["models"]] == ["baseline", "simple", "holt", "holt_winters"]


def test_save_and_read_back_the_latest(tmp_path, weekly):
    store = LocalStore(tmp_path / "bucket")
    older = make_forecast(weekly, run_id="20261119T090000Z-forecast-aaaaaa")
    newer = make_forecast(weekly, run_id="20261119T100000Z-forecast-bbbbbb")
    save_forecast(store, older)
    keys = save_forecast(store, newer)

    assert keys == [
        "forecasts/run=20261119T100000Z-forecast-bbbbbb/forecast.parquet",
        "forecasts/run=20261119T100000Z-forecast-bbbbbb/scores.parquet",
        "forecasts/run=20261119T100000Z-forecast-bbbbbb/summary.json",
    ]
    assert list_forecast_runs(store) == ["20261119T090000Z-forecast-aaaaaa", "20261119T100000Z-forecast-bbbbbb"]
    table, summary = latest_forecast(store)
    assert summary["run_id"] == "20261119T100000Z-forecast-bbbbbb"
    pd.testing.assert_frame_equal(table, newer.table)


def test_a_half_saved_run_is_skipped(tmp_path, weekly):
    store = LocalStore(tmp_path / "bucket")
    good = make_forecast(weekly, run_id="20261119T090000Z-forecast-aaaaaa")
    save_forecast(store, good)
    store.put("forecasts/run=20261119T100000Z-forecast-cccccc/forecast.parquet", b"half")

    _, summary = latest_forecast(store)
    assert summary["run_id"] == good.run_id


def test_no_forecast_yet(tmp_path):
    assert latest_forecast(LocalStore(tmp_path / "empty")) is None


def test_run_ids_sort_by_time():
    assert new_forecast_run_id().split("-forecast-")[0].endswith("Z")
