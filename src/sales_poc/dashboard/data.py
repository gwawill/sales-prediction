"""The numbers behind the dashboard (task T14, FR-32). No Streamlit here, so it can be tested.

Sales history comes from the clean lake through the read-only SQL tool (DuckDB), the
forecast from the latest saved forecast run, and answers from the web API's POST /ask.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import httpx
import pandas as pd

REGIONS = ["NA", "EMEA", "APJ", "LATAM"]          # fixed order = fixed colour per region
CATEGORIES = ["Laptop", "Desktop", "Server", "Storage", "Monitor", "Accessory"]
ACTUAL_WEEKS_SHOWN = 16


def first_of_month(day: date) -> date:
    return day.replace(day=1)


def monthly_revenue(sql, today: date, category: str | None = None) -> pd.DataFrame:
    """Revenue and orders per complete month and region. The current month is left out,
    because a part-month would look like a sudden drop."""
    where = [f"event_date < DATE '{first_of_month(today).isoformat()}'"]
    if category:
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category {category!r}")
        where.append(f"category = '{category}'")
    result = sql.run(
        "SELECT date_trunc('month', event_date) AS month, region_id, "
        "SUM(total_amount) AS revenue, COUNT(*) AS orders "
        f"FROM sales WHERE {' AND '.join(where)} GROUP BY 1, 2 ORDER BY 1, 2"
    )
    frame = pd.DataFrame(result.rows, columns=result.columns)
    frame["month"] = pd.to_datetime(frame["month"])
    return frame


@dataclass(frozen=True)
class MonthSummary:
    month: date | None
    revenue: float
    orders: int
    previous_revenue: float | None

    @property
    def change(self) -> float | None:
        if not self.previous_revenue:
            return None
        return self.revenue / self.previous_revenue - 1


def last_month_summary(monthly: pd.DataFrame) -> MonthSummary:
    """Totals for the latest complete month in `monthly`, compared with the month before."""
    if monthly.empty:
        return MonthSummary(None, 0.0, 0, None)
    totals = monthly.groupby("month").agg(revenue=("revenue", "sum"), orders=("orders", "sum")).sort_index()
    last = totals.iloc[-1]
    previous = totals.iloc[-2]["revenue"] if len(totals) > 1 else None
    return MonthSummary(totals.index[-1].date(), float(last["revenue"]), int(last["orders"]),
                        float(previous) if previous is not None else None)


def weekly_actuals(sql, category: str, region: str, last_week: date, weeks: int = ACTUAL_WEEKS_SHOWN) -> pd.DataFrame:
    """Actual revenue per Monday week for one series, ending with the forecast's last known week.
    Weeks without orders count as 0, as they do in the forecast."""
    if category not in CATEGORIES or region not in REGIONS:
        raise ValueError(f"Unknown series {category}/{region}")
    first_week = last_week - timedelta(weeks=weeks - 1)
    result = sql.run(
        "SELECT date_trunc('week', event_date) AS week_start, SUM(total_amount) AS revenue FROM sales "
        f"WHERE category = '{category}' AND region_id = '{region}' "
        f"AND event_date >= DATE '{first_week.isoformat()}' AND event_date < DATE '{(last_week + timedelta(days=7)).isoformat()}' "
        "GROUP BY 1"
    )
    found = {pd.Timestamp(w).date(): float(r) for w, r in result.rows}
    all_weeks = [first_week + timedelta(weeks=i) for i in range(weeks)]
    return pd.DataFrame({"week_start": pd.to_datetime(all_weeks),
                         "revenue": [found.get(w, 0.0) for w in all_weeks]})


@dataclass(frozen=True)
class SeriesForecast:
    chart: pd.DataFrame          # week_start, revenue, kind ("Actual" or "Forecast")
    model: str
    backtest_wape: float
    baseline_wape: float
    total: float                 # forecast revenue over the forecast weeks


def series_forecast(actuals: pd.DataFrame, table: pd.DataFrame, category: str, region: str) -> SeriesForecast:
    """Actual weeks followed by forecast weeks, for one category and region. The forecast line
    starts at the last actual point, so the two lines join up on the chart."""
    rows = table[(table["category"] == category) & (table["region_id"] == region)].sort_values("week_start")
    if rows.empty:
        raise ValueError(f"No forecast for {category}/{region}")
    future = pd.DataFrame({"week_start": pd.to_datetime(rows["week_start"]),
                           "revenue": rows["forecast_revenue"].astype(float).to_numpy(), "kind": "Forecast"})
    past = actuals.assign(kind="Actual")
    bridge = past.tail(1).assign(kind="Forecast")
    chart = pd.concat([past, bridge, future], ignore_index=True)
    first = rows.iloc[0]
    return SeriesForecast(chart, str(first["model"]), float(first["backtest_wape"]),
                          float(first["baseline_wape"]), float(rows["forecast_revenue"].sum()))


@dataclass(frozen=True)
class AskResult:
    ok: bool
    answer: str
    sources: list[dict]
    detail: str = ""             # tokens and time when ok, the problem when not


def ask_api(api_url: str, question: str, mode: str = "agent", client: httpx.Client | None = None,
            timeout: float = 180.0) -> AskResult:
    """POST the question to the web API's /ask and return a result the page can show."""
    url = api_url.rstrip("/") + "/ask"
    own = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        response = client.post(url, json={"question": question, "mode": mode})
    except httpx.HTTPError as exc:
        return AskResult(False, "", [], f"Could not reach the API at {api_url}. Is api-serve running? ({exc})")
    finally:
        if own:
            client.close()
    if response.status_code != 200:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text
        if isinstance(detail, list):   # FastAPI validation errors
            detail = "; ".join(d.get("msg", str(d)) for d in detail)
        return AskResult(False, "", [], f"The API said {response.status_code}: {detail}")
    body = response.json()
    return AskResult(True, body["answer"], body["sources"],
                     f"{len(body.get('steps', []))} tool calls, {body['tokens']} tokens, {body['seconds']}s")
