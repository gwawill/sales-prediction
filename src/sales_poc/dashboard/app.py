"""The dashboard page (task T14, FR-32): sales over time, the forecast, and a question box.

    dashboard                       opens http://127.0.0.1:8501 (needs api-serve for questions)
    dashboard --port 8600 --api http://127.0.0.1:8077

Streamlit re-runs this script from the top whenever someone changes a control. The slow parts
are cached: the lake is read into DuckDB once, and the forecast for five minutes.
Everything on the page only reads.

The page keeps up by itself: every LAKE_CHECK_SECONDS (default 60) a small background part of
the page asks the SQL tool to check the lake for changed days. Only if the data changed does
the whole page redraw, with the new numbers. The last answer to a question stays on screen.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

API_URL_ENV = "SALES_DASHBOARD_API_URL"   # set by the `dashboard` command for the page to read


def page() -> None:
    import streamlit as st

    from sales_poc.ai.tools.sql import SqlTool
    from sales_poc.config import get_app_settings
    from sales_poc.dashboard import charts
    from sales_poc.dashboard.data import (CATEGORIES, REGIONS, ask_api, last_month_summary, monthly_revenue,
                                          series_forecast, weekly_actuals)

    st.set_page_config(page_title="Sales dashboard", page_icon=":bar_chart:", layout="wide")

    @st.cache_resource(show_spinner="Reading the data lake into memory (about 30 seconds, only once)...")
    def sql_tool() -> SqlTool:
        tool = SqlTool()
        tool.run("SELECT 1")   # load now, behind the spinner
        return tool

    @st.cache_data(ttl=300, show_spinner="Reading the latest forecast...")
    def saved_forecast():
        from sales_poc.forecast.publish import latest_forecast
        from sales_poc.lake.store import s3_store
        return latest_forecast(s3_store())

    # `version` is the lake data's version: when new data arrives it changes, so these
    # cached results are worked out again instead of being reused.
    @st.cache_data(ttl=300, max_entries=100, show_spinner=False)
    def monthly(category: str | None, today: date, version: int):
        return monthly_revenue(sql_tool(), today, category)

    @st.cache_data(ttl=300, max_entries=100, show_spinner=False)
    def actuals(category: str, region: str, last_week: date, version: int):
        return weekly_actuals(sql_tool(), category, region, last_week)

    theme = getattr(getattr(st.context, "theme", None), "type", None) or "light"
    today = date.today()
    version = sql_tool().check_lake()
    st.session_state["lake_version"] = version

    @st.fragment(run_every=get_app_settings().lake_check_seconds)
    def watch_lake():
        """Runs on its own timer. Redraws the whole page only when the data has changed."""
        if sql_tool().check_lake() != st.session_state.get("lake_version"):
            st.rerun(scope="app")

    with st.sidebar:
        st.header("Settings")
        api_url = st.text_input("Web API address", os.environ.get(API_URL_ENV, "http://127.0.0.1:8000"),
                                help="Where api-serve is running. Only the question box uses it.")
        if st.button("Reload data"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    st.title("Sales dashboard")
    st.caption("A read-only view of the clean data lake, the latest forecast, and the AI agent. "
               "New data appears by itself within about a minute.")
    watch_lake()

    # --- headline numbers ------------------------------------------------------------------
    all_months = monthly(None, today, version)
    summary = last_month_summary(all_months)
    found = saved_forecast()
    cols = st.columns(3)
    if summary.month:
        change = f"{summary.change:+.1%} vs month before" if summary.change is not None else None
        cols[0].metric(f"Revenue, {summary.month:%B %Y}", f"${summary.revenue / 1e6:,.2f}M", change)
        cols[1].metric(f"Orders, {summary.month:%B %Y}", f"{summary.orders:,}")
    if found is not None:
        table, info = found
        cols[2].metric("Forecast, next 4 weeks (all series)", f"${table['forecast_revenue'].sum() / 1e6:,.2f}M",
                       help=f"Forecast run {info['run_id']}")

    # --- sales over time -------------------------------------------------------------------
    st.subheader("Monthly revenue by region")
    choice = st.selectbox("Category", ["All categories", *CATEGORIES], key="monthly_category")
    category = None if choice == "All categories" else choice
    data = all_months if category is None else monthly(category, today, version)
    st.altair_chart(charts.monthly_chart(data, theme), width="stretch")
    st.caption(f"Complete months only; {today:%B %Y} is still in progress and is left out.")
    with st.expander("Show as table"):
        st.dataframe(data.assign(month=data["month"].dt.strftime("%Y-%m")), hide_index=True,
                     column_config={"revenue": st.column_config.NumberColumn("revenue", format="$%,.0f")})

    # --- forecast --------------------------------------------------------------------------
    st.subheader("Weekly revenue and 4-week forecast")
    if found is None:
        st.info("No forecast has been saved yet. Run forecast-run.")
    else:
        table, info = found
        left, right = st.columns(2)
        f_category = left.selectbox("Category", CATEGORIES, key="forecast_category")
        f_region = right.selectbox("Region", REGIONS, key="forecast_region")
        last_known = date.fromisoformat(info["weeks_used"][1])
        result = series_forecast(actuals(f_category, f_region, last_known, version), table, f_category, f_region)
        st.altair_chart(charts.forecast_chart(result.chart, theme), width="stretch")
        verdict = "better than" if result.backtest_wape < result.baseline_wape else "no better than"
        st.caption(
            f"Forecast for the next 4 weeks: ${result.total:,.0f}, by the {result.model} model. "
            f"On past weeks it was off by {result.backtest_wape:.0%} on average, {verdict} the simple "
            f"baseline ({result.baseline_wape:.0%}). Actual weeks end {last_known:%d %b %Y}, the last "
            f"complete week the forecast run used.")
        with st.expander("Show as table"):
            rows = result.chart.drop_duplicates(subset=["week_start"], keep="first")
            st.dataframe(rows.assign(week_start=rows["week_start"].dt.strftime("%Y-%m-%d")), hide_index=True,
                         column_config={"revenue": st.column_config.NumberColumn("revenue", format="$%,.0f")})

    # --- question box ----------------------------------------------------------------------
    st.subheader("Ask a question")
    st.caption("Answered by the AI agent, which can query the orders, the graph, the monthly summaries "
               "and the forecast. It lists the sources it used.")
    with st.form("ask"):
        question = st.text_area("Question", placeholder="Which enterprise customers who bought servers "
                                "also bought storage?", max_chars=1000)
        submitted = st.form_submit_button("Ask")
    if submitted and question.strip():
        with st.spinner("Thinking... the agent may run a few queries."):
            # the agent (the API's default mode). Kept, so it survives the page redrawing.
            st.session_state["last_answer"] = ask_api(api_url, question.strip())
    answer = st.session_state.get("last_answer")
    if answer is not None:
        if not answer.ok:
            st.error(answer.detail)
        else:
            st.markdown(answer.answer)
            with st.expander(f"Sources ({len(answer.sources)})", expanded=True):
                for source in answer.sources:
                    st.markdown(f"**{source['kind']}**")
                    st.code(source["detail"], language="sql" if source["kind"] == "sql_query" else None,
                            wrap_lines=True)
            st.caption(answer.detail)


def main(argv: list[str] | None = None) -> int:
    """dashboard: start the Streamlit page."""
    from streamlit.web import cli as streamlit_cli

    from sales_poc.config import get_app_settings

    settings = get_app_settings()
    parser = argparse.ArgumentParser(prog="dashboard", description="Run the sales dashboard (Streamlit).")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--api", default=f"http://{settings.api_host}:{settings.api_port}",
                        help="the web API address the question box calls")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    os.environ[API_URL_ENV] = args.api
    sys.argv = ["streamlit", "run", str(Path(__file__)), "--server.address", settings.api_host,
                "--server.port", str(args.port), "--browser.gatherUsageStats", "false",
                # headless: no first-run email question in the terminal; open the printed address yourself
                "--server.headless", "true",
                # viewer: hide Streamlit's own Deploy button and developer menu items
                "--client.toolbarMode", "viewer"]
    return streamlit_cli.main()


if __name__ == "__main__":
    page()
