"""vector_search and get_forecast against the real S3 Vectors index and the real lake."""

from sales_poc.ai.tools.forecast import ForecastTool
from sales_poc.ai.tools.search import VectorSearchTool


def test_vector_search_finds_the_named_summary():
    result = VectorSearchTool().run("laptop sales in EMEA in March 2026", top_k=3)
    assert result.rows[0][0] == "catreg:Laptop:EMEA:2026-03"


def test_vector_search_filters_narrow_the_results():
    result = VectorSearchTool().run("how did sales go", region="APJ", category="Server", month="2026-06")
    assert [row[0] for row in result.rows] == ["catreg:Server:APJ:2026-06"]


def test_get_forecast_covers_every_series_for_four_weeks():
    tool = ForecastTool()
    totals = tool.run()
    assert len(totals.rows) == 24  # 6 categories x 4 regions
    weekly = tool.run(category="Laptop", region="EMEA")
    assert len(weekly.rows) == 4
    assert totals.note.startswith("Forecast run ")
