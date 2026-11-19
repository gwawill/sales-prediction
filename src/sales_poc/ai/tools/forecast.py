"""The `get_forecast` tool: read the latest saved sales forecast (task T12.3, FR-27).

Reads the newest complete run that forecast-run saved in the lake (forecast/publish.py).
The forecast is read once, on first use; call `refresh()` to read it again. Only reads.

    no filter            one row per category and region: the 4 weeks added up (24 rows)
    category and/or      one row per week for the matching series
    region given

Every row carries the model's backtest error (WAPE, "weighted absolute percentage error"):
how far off the model was, on average, when it forecast weeks we already knew.
"""

from __future__ import annotations

import pandas as pd

from sales_poc.ai.tools import ToolError
from sales_poc.ai.tools.search import CATEGORIES, REGIONS
from sales_poc.ai.tools.table import Table

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "get_forecast",
        "description": (
            "Get the latest revenue forecast for the next 4 weeks, per product category and region. "
            "Without filters you get each category and region's 4-week total; with a category or region "
            "you get week-by-week figures. Each row shows the forecasting model used and its backtest "
            "error (how far off it was on past weeks, as a percent), so you can say how reliable it is."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": CATEGORIES, "description": "only this category"},
                "region": {"type": "string", "enum": REGIONS, "description": "only this region"},
            },
            "required": [],
        },
    },
}


class ForecastToolError(ToolError):
    """No forecast to read, or a bad filter."""


def _percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def forecast_table(table: pd.DataFrame, summary: dict, category=None, region=None) -> Table:
    """The tool's answer from a saved forecast `table` and its `summary`."""
    if category and category not in CATEGORIES:
        raise ForecastToolError(f"Unknown category {category!r}. Use one of: {', '.join(CATEGORIES)}.")
    if region and region not in REGIONS:
        raise ForecastToolError(f"Unknown region {region!r}. Use one of: {', '.join(REGIONS)}.")

    rows = table
    if category:
        rows = rows[rows["category"] == category]
    if region:
        rows = rows[rows["region_id"] == region]

    weeks = summary.get("forecast_weeks", [])
    note = (f"Forecast run {summary.get('run_id', '?')}, weeks starting {', '.join(map(str, weeks))}. "
            f"Revenue in USD. backtest_error = the model's average miss on past weeks (lower is better); "
            f"baseline_error = the same for a naive 'repeat recent weeks' forecast.")

    if category or region:
        rows = rows.sort_values(["category", "region_id", "week_start"])
        columns = ["category", "region", "week_start", "forecast_revenue", "model", "backtest_error",
                   "baseline_error"]
        out = [(r.category, r.region_id, str(r.week_start), round(float(r.forecast_revenue), 2), r.model,
                _percent(r.backtest_wape), _percent(r.baseline_wape)) for r in rows.itertuples()]
    else:
        totals = (rows.groupby(["category", "region_id"], as_index=False)
                  .agg(forecast_revenue=("forecast_revenue", "sum"), model=("model", "first"),
                       backtest_wape=("backtest_wape", "first"), baseline_wape=("baseline_wape", "first"))
                  .sort_values("forecast_revenue", ascending=False))
        columns = ["category", "region", "next_4_weeks_revenue", "model", "backtest_error", "baseline_error"]
        out = [(r.category, r.region_id, round(float(r.forecast_revenue), 2), r.model,
                _percent(r.backtest_wape), _percent(r.baseline_wape)) for r in totals.itertuples()]

    label = ", ".join(f"{k}={v}" for k, v in (("category", category), ("region", region)) if v) or "all"
    return Table(f"forecast {summary.get('run_id', '?')} ({label})", columns, out, truncated=False, note=note)


class ForecastTool:
    def __init__(self, store=None, saved: tuple[pd.DataFrame, dict] | None = None):
        """Give the `saved` (table, summary) directly (tests), or a lake `store` (default: S3)."""
        self._store, self._saved = store, saved

    def refresh(self) -> None:
        self._saved = None

    def _load(self) -> tuple[pd.DataFrame, dict]:
        if self._saved is None:
            from sales_poc.forecast.publish import latest_forecast
            from sales_poc.lake.store import s3_store

            found = latest_forecast(self._store or s3_store())
            if found is None:
                raise ForecastToolError("No forecast has been saved yet. Run forecast-run first.")
            self._saved = found
        return self._saved

    def run(self, category=None, region=None) -> Table:
        table, summary = self._load()
        return forecast_table(table, summary, category or None, region or None)
