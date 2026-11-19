"""Weekly sales per category and region, ready for forecasting (FR-23).

    weekly = weekly_sales(clean_rows)
    weekly.table        week_start, category, region_id, revenue, orders, units
    weekly.weeks_used   the complete weeks in the table, oldest first
    weekly.weeks_left_out   {week_start: days with data} for weeks that were not complete

Rules:
  - Weeks run Monday to Sunday and are labelled by their Monday (week_start).
  - Only complete weeks are used: every one of the 7 days must have data. A partial week
    (the start or end of the data, or a gap) would look like a sudden drop or spike and
    mislead the forecast.
  - Every category x region gets a row for every week used. No orders that week means a
    real zero, not a missing value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

SERIES_KEYS = ["category", "region_id"]
VALUE_COLUMNS = ["revenue", "orders", "units"]


@dataclass(frozen=True)
class WeeklySales:
    table: pd.DataFrame
    weeks_used: list[date]
    weeks_left_out: dict[date, int] = field(default_factory=dict)

    def series(self, category: str, region_id: str, value: str = "revenue") -> pd.Series:
        """One weekly series, indexed by week_start."""
        rows = self.table[(self.table["category"] == category) & (self.table["region_id"] == region_id)]
        return rows.set_index("week_start")[value]


def week_start(days: pd.Series) -> pd.Series:
    """The Monday of each day's week."""
    stamps = pd.to_datetime(days)
    return (stamps - pd.to_timedelta(stamps.dt.day_of_week, unit="D")).dt.date


def weekly_sales(clean: pd.DataFrame, full_weeks_only: bool = True) -> WeeklySales:
    """Weekly totals per category and region from clean lake rows."""
    if clean.empty:
        empty = pd.DataFrame(columns=["week_start", *SERIES_KEYS, *VALUE_COLUMNS])
        return WeeklySales(empty, [])

    rows = clean[["event_date", *SERIES_KEYS, "total_amount", "quantity"]].copy()
    rows["week_start"] = week_start(rows["event_date"])

    # How many different days of each week have any data at all.
    days_per_week = rows.groupby("week_start")["event_date"].nunique()
    complete = sorted(w for w, n in days_per_week.items() if n == 7)
    left_out = {w: int(n) for w, n in days_per_week.items() if n < 7}
    weeks = complete if full_weeks_only else sorted(days_per_week.index)

    totals = (
        rows[rows["week_start"].isin(weeks)]
        .groupby(["week_start", *SERIES_KEYS])
        .agg(revenue=("total_amount", "sum"), orders=("total_amount", "size"), units=("quantity", "sum"))
    )

    # Every week x category x region, with zeros where nothing was sold.
    categories = sorted(clean["category"].dropna().unique())
    regions = sorted(clean["region_id"].dropna().unique())
    full_index = pd.MultiIndex.from_product([weeks, categories, regions], names=["week_start", *SERIES_KEYS])
    table = totals.reindex(full_index, fill_value=0).reset_index()
    table["revenue"] = table["revenue"].astype(float).round(2)
    table["orders"] = table["orders"].astype(int)
    table["units"] = table["units"].astype(int)

    return WeeklySales(table=table, weeks_used=list(weeks),
                       weeks_left_out={} if not full_weeks_only else dict(sorted(left_out.items())))
