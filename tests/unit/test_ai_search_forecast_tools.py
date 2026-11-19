import pandas as pd
import pytest

from sales_poc.ai.tools import ToolError
from sales_poc.ai.tools.forecast import TOOL_SPEC as FORECAST_SPEC
from sales_poc.ai.tools.forecast import ForecastTool, ForecastToolError
from sales_poc.ai.tools.graph import GraphToolError
from sales_poc.ai.tools.search import TOOL_SPEC as SEARCH_SPEC
from sales_poc.ai.tools.search import SearchToolError, VectorSearchTool, check_arguments
from sales_poc.ai.tools.sql import SqlToolError
from sales_poc.vectors import store
from sales_poc.vectors.store import SearchResult


# --- vector_search -------------------------------------------------------------------------

@pytest.fixture
def searched(monkeypatch):
    asked = {}

    def fake_search(index, embedder, question, top_k, **labels):
        asked.update(question=question, top_k=top_k, labels=labels)
        return [SearchResult("catreg:Laptop:EMEA:2026-03", 0.7912, "Laptop sales ...", {}),
                SearchResult("region:EMEA:2026-03", 0.8, "All sales ...", {})][:top_k]

    monkeypatch.setattr(store, "search", fake_search)
    return asked


def tool():
    return VectorSearchTool(index=object(), embedder=object())


def test_search_returns_ids_scores_and_text(searched):
    result = tool().run("  laptops in EMEA  ")
    assert result.columns == ["summary_id", "match", "text"]
    assert result.rows[0] == ("catreg:Laptop:EMEA:2026-03", 0.79, "Laptop sales ...")
    assert searched == {"question": "laptops in EMEA", "top_k": 5,
                        "labels": {"region": None, "category": None, "month": None}}


def test_search_filters_are_passed_on(searched):
    tool().run("x", region="EMEA", category="Laptop", month="2026-03", top_k=1)
    assert searched["top_k"] == 1
    assert searched["labels"] == {"region": "EMEA", "category": "Laptop", "month": "2026-03"}


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"query": " "}, "empty"),
        ({"query": "x", "region": "Europe"}, "Unknown region"),
        ({"query": "x", "category": "Phone"}, "Unknown category"),
        ({"query": "x", "month": "2026-13"}, "YYYY-MM"),
        ({"query": "x", "month": "March"}, "YYYY-MM"),
        ({"query": "x", "top_k": 0}, "top_k"),
        ({"query": "x", "top_k": 11}, "top_k"),
        ({"query": "x", "top_k": "5"}, "top_k"),
        ({"query": "x", "top_k": True}, "top_k"),
    ],
)
def test_bad_search_arguments_are_explained(arguments, message):
    with pytest.raises(SearchToolError, match=message):
        check_arguments(**arguments)


def test_search_failures_become_tool_errors(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(store, "search", broken)
    with pytest.raises(SearchToolError, match="index unavailable"):
        tool().run("x")


# --- get_forecast --------------------------------------------------------------------------

def saved():
    rows = []
    for category, region, base, model, wape in [("Laptop", "EMEA", 100.0, "holt", 0.34),
                                                ("Server", "NA", 1000.0, "simple", 0.42)]:
        for i, week in enumerate(["2026-11-02", "2026-11-09"]):
            rows.append({"category": category, "region_id": region, "week_start": week,
                         "forecast_revenue": base + i, "model": model,
                         "backtest_wape": wape, "baseline_wape": 0.5})
    summary = {"run_id": "20261119T154821Z-forecast-6a6bef", "forecast_weeks": ["2026-11-02", "2026-11-09"]}
    return pd.DataFrame(rows), summary


def test_without_filters_each_series_is_one_total_biggest_first():
    result = ForecastTool(saved=saved()).run()
    assert result.columns[2] == "next_4_weeks_revenue"
    assert result.rows == [("Server", "NA", 2001.0, "simple", "42%", "50%"),
                           ("Laptop", "EMEA", 201.0, "holt", "34%", "50%")]


def test_with_a_filter_rows_are_weekly():
    result = ForecastTool(saved=saved()).run(category="Laptop", region="EMEA")
    assert [r[2] for r in result.rows] == ["2026-11-02", "2026-11-09"]
    assert result.rows[0] == ("Laptop", "EMEA", "2026-11-02", 100.0, "holt", "34%", "50%")


def test_a_filter_with_no_match_gives_no_rows():
    assert ForecastTool(saved=saved()).run(region="APJ").rows == []


def test_the_note_says_which_run_and_how_to_read_the_error():
    text = ForecastTool(saved=saved()).run().to_text()
    first = text.splitlines()[0]
    assert "20261119T154821Z-forecast-6a6bef" in first and "2026-11-02, 2026-11-09" in first
    assert "backtest_error" in first


def test_bad_forecast_filters_are_explained():
    with pytest.raises(ForecastToolError, match="Unknown category"):
        ForecastTool(saved=saved()).run(category="Phone")
    with pytest.raises(ForecastToolError, match="Unknown region"):
        ForecastTool(saved=saved()).run(region="Europe")


def test_no_saved_forecast_is_explained(monkeypatch):
    import sales_poc.forecast.publish as publish

    monkeypatch.setattr(publish, "latest_forecast", lambda store: None)
    with pytest.raises(ForecastToolError, match="forecast-run"):
        ForecastTool(store=object()).run()


def test_the_forecast_is_read_once_until_refreshed(monkeypatch):
    import sales_poc.forecast.publish as publish

    reads = []
    monkeypatch.setattr(publish, "latest_forecast", lambda store: reads.append(1) or saved())
    tool = ForecastTool(store=object())
    tool.run()
    tool.run(category="Laptop")
    assert len(reads) == 1
    tool.refresh()
    tool.run()
    assert len(reads) == 2


# --- shared --------------------------------------------------------------------------------

def test_every_tool_error_is_a_tool_error():
    for cls in (SqlToolError, GraphToolError, SearchToolError, ForecastToolError):
        assert issubclass(cls, ToolError)


def test_the_tool_specs():
    assert SEARCH_SPEC["function"]["name"] == "vector_search"
    assert SEARCH_SPEC["function"]["parameters"]["required"] == ["query"]
    assert FORECAST_SPEC["function"]["name"] == "get_forecast"
    assert FORECAST_SPEC["function"]["parameters"]["required"] == []
