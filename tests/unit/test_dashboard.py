from datetime import date

import httpx
import pandas as pd
import pytest

from sales_poc.ai.tools.sql import SqlTool
from sales_poc.dashboard import charts
from sales_poc.dashboard.data import (REGIONS, ask_api, last_month_summary, monthly_revenue, series_forecast,
                                      weekly_actuals)


def lake():
    rows = [
        ("T1", date(2026, 9, 3), "EMEA", "Laptop", 100.0),
        ("T2", date(2026, 9, 20), "NA", "Server", 300.0),
        ("T3", date(2026, 10, 5), "EMEA", "Laptop", 150.0),     # Monday of week 2026-10-05
        ("T4", date(2026, 10, 7), "EMEA", "Laptop", 50.0),      # same week
        ("T5", date(2026, 10, 21), "EMEA", "Laptop", 80.0),     # week 2026-10-19
        ("T6", date(2026, 10, 30), "NA", "Server", 120.0),
        ("T7", date(2026, 11, 2), "EMEA", "Laptop", 999.0),     # current month: left out
    ]
    frame = pd.DataFrame(rows, columns=["transaction_id", "event_date", "region_id", "category", "total_amount"])
    return SqlTool(rows=frame)


TODAY = date(2026, 11, 19)


# --- data ----------------------------------------------------------------------------------

def test_monthly_revenue_leaves_out_the_current_month():
    monthly = monthly_revenue(lake(), TODAY)
    assert list(monthly.columns) == ["month", "region_id", "revenue", "orders"]
    assert [(m.strftime("%Y-%m"), r, v, n) for m, r, v, n in monthly.itertuples(index=False)] == [
        ("2026-09", "EMEA", 100.0, 1), ("2026-09", "NA", 300.0, 1),
        ("2026-10", "EMEA", 280.0, 3), ("2026-10", "NA", 120.0, 1)]


def test_monthly_revenue_for_one_category():
    monthly = monthly_revenue(lake(), TODAY, "Server")
    assert set(monthly["region_id"]) == {"NA"} and monthly["revenue"].sum() == 420.0


def test_monthly_revenue_refuses_an_unknown_category():
    with pytest.raises(ValueError):
        monthly_revenue(lake(), TODAY, "Laptop' OR '1'='1")


def test_last_month_summary_compares_with_the_month_before():
    s = last_month_summary(monthly_revenue(lake(), TODAY))
    assert (s.month, s.revenue, s.orders, s.previous_revenue) == (date(2026, 10, 1), 400.0, 4, 400.0)
    assert s.change == 0.0


def test_last_month_summary_of_nothing():
    s = last_month_summary(pd.DataFrame(columns=["month", "region_id", "revenue", "orders"]))
    assert s.month is None and s.change is None


def test_weekly_actuals_fill_empty_weeks_with_zero():
    weekly = weekly_actuals(lake(), "Laptop", "EMEA", last_week=date(2026, 10, 19), weeks=3)
    assert [d.date() for d in weekly["week_start"]] == [date(2026, 10, 5), date(2026, 10, 12), date(2026, 10, 19)]
    assert list(weekly["revenue"]) == [200.0, 0.0, 80.0]


def test_weekly_actuals_refuse_unknown_series():
    with pytest.raises(ValueError):
        weekly_actuals(lake(), "Laptop", "Europe", last_week=date(2026, 10, 19))


def forecast_table():
    return pd.DataFrame({
        "category": ["Laptop"] * 2 + ["Server"] * 2, "region_id": ["EMEA"] * 2 + ["NA"] * 2,
        "week_start": [date(2026, 11, 2), date(2026, 10, 26), date(2026, 10, 26), date(2026, 11, 2)],
        "forecast_revenue": [90.0, 85.0, 1.0, 2.0], "model": ["holt"] * 2 + ["simple"] * 2,
        "backtest_wape": [0.34] * 2 + [0.5] * 2, "baseline_wape": [0.43] * 2 + [0.6] * 2})


def test_series_forecast_joins_actual_and_forecast_lines():
    actuals = weekly_actuals(lake(), "Laptop", "EMEA", last_week=date(2026, 10, 19), weeks=3)
    result = series_forecast(actuals, forecast_table(), "Laptop", "EMEA")

    chart = result.chart
    assert list(chart["kind"]) == ["Actual"] * 3 + ["Forecast"] * 3
    # the forecast line starts at the last actual point, then the forecast weeks in order
    forecast = chart[chart["kind"] == "Forecast"]
    assert [d.date() for d in forecast["week_start"]] == [date(2026, 10, 19), date(2026, 10, 26), date(2026, 11, 2)]
    assert list(forecast["revenue"]) == [80.0, 85.0, 90.0]
    assert (result.model, result.backtest_wape, result.baseline_wape, result.total) == ("holt", 0.34, 0.43, 175.0)


def test_series_forecast_without_that_series():
    with pytest.raises(ValueError):
        series_forecast(pd.DataFrame({"week_start": [], "revenue": []}), forecast_table(), "Monitor", "APJ")


# --- the question box's call to the API ----------------------------------------------------

def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ask_api_success():
    seen = {}

    def handler(request):
        seen["url"], seen["body"] = str(request.url), request.read()
        return httpx.Response(200, json={"answer": "68.", "sources": [{"kind": "graph_query", "detail": "MATCH"}],
                                         "steps": [{}, {}], "tokens": 900, "seconds": 4.2})

    result = ask_api("http://api:8000/", "How many?", "rag", client=client(handler))
    assert seen["url"] == "http://api:8000/ask"
    assert b'"mode":"rag"' in seen["body"].replace(b" ", b"")
    assert result.ok and result.answer == "68." and result.sources[0]["kind"] == "graph_query"
    assert result.detail == "2 tool calls, 900 tokens, 4.2s"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(502, json={"detail": "OpenRouter refused the key."}), "502: OpenRouter refused the key."),
        (httpx.Response(422, json={"detail": [{"msg": "too long"}]}), "422: too long"),
        (httpx.Response(500, text="boom"), "500: boom"),
    ],
)
def test_ask_api_errors_are_explained(response, expected):
    result = ask_api("http://api", "q", client=client(lambda request: response))
    assert not result.ok and expected in result.detail


def test_ask_api_when_the_api_is_not_running():
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    result = ask_api("http://127.0.0.1:1", "q", client=client(refuse))
    assert not result.ok and "Is api-serve running?" in result.detail


# --- charts --------------------------------------------------------------------------------

def layers(spec):
    return spec.get("layer", [])


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_monthly_chart_keeps_region_colours_fixed(theme):
    only_emea = monthly_revenue(lake(), TODAY, "Laptop")      # one region left after filtering
    spec = charts.monthly_chart(only_emea, theme).to_dict()
    scale = layers(spec)[0]["encoding"]["color"]["scale"]
    assert scale["domain"] == REGIONS and scale["range"] == charts.THEMES[theme]["series"]


def test_the_monthly_chart_has_hover_tooltips_and_end_labels():
    spec = charts.monthly_chart(monthly_revenue(lake(), TODAY), "light").to_dict()
    marks = [layer["mark"]["type"] for layer in layers(spec)]
    assert marks == ["line", "rule", "point", "text"]
    assert any("tooltip" in layer["encoding"] for layer in layers(spec))
    text_layer = layers(spec)[3]
    assert text_layer["mark"]["color"] == charts.THEMES["light"]["text"]   # labels use text colour


def test_the_forecast_chart_dashes_the_forecast():
    actuals = weekly_actuals(lake(), "Laptop", "EMEA", last_week=date(2026, 10, 19), weeks=3)
    spec = charts.forecast_chart(series_forecast(actuals, forecast_table(), "Laptop", "EMEA").chart, "dark").to_dict()
    dash = layers(spec)[0]["encoding"]["strokeDash"]["scale"]
    assert dash == {"domain": ["Actual", "Forecast"], "range": [[1, 0], [6, 4]]}
    assert layers(spec)[0]["mark"]["color"] == charts.THEMES["dark"]["series"][0]


def test_charts_use_their_own_dark_colours():
    assert charts._theme("dark") is charts.THEMES["dark"]
    assert charts._theme("anything else") is charts.THEMES["light"]


def test_monthly_lines_are_straight():
    spec = charts.monthly_chart(monthly_revenue(lake(), TODAY), "light").to_dict()
    assert "interpolate" not in layers(spec)[0]["mark"]


def test_forecast_ticks_are_real_week_starts_ending_with_the_last_week():
    actuals = weekly_actuals(lake(), "Laptop", "EMEA", last_week=date(2026, 10, 19), weeks=3)
    data = series_forecast(actuals, forecast_table(), "Laptop", "EMEA").chart   # weeks 5 Oct .. 2 Nov
    values = charts.forecast_chart(data, "light").to_dict()["layer"][0]["encoding"]["x"]["axis"]["values"]
    days = [date(v["year"], v["month"], v["date"]) for v in values]
    assert days == [date(2026, 10, 5), date(2026, 10, 19), date(2026, 11, 2)]
    assert all(d.weekday() == 0 for d in days)   # Mondays
