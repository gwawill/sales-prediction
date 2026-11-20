from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from sales_poc.ai.tools import sql
from sales_poc.ai.tools.sql import COLUMNS, TOOL_SPEC, SqlTool, SqlToolError, check_select


def rows():
    return pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3"],
        "event_date": [date(2026, 3, 1), date(2026, 3, 2), date(2026, 4, 1)],
        "customer_id": ["C1", "C2", "C1"],
        "region_id": ["EMEA", "NA", "EMEA"],
        "category": ["Laptop", "Server", "Laptop"],
        "total_amount": [100.0, 250.5, 1 / 3],
        "not_a_lake_column": [1, 2, 3],
    })


@pytest.fixture
def tool():
    return SqlTool(rows=rows())


def test_a_select_returns_columns_and_rows(tool):
    result = tool.run("SELECT region_id, SUM(total_amount) AS revenue FROM sales GROUP BY 1 ORDER BY 1;")
    assert result.columns == ["region_id", "revenue"]
    assert result.rows == [("EMEA", pytest.approx(100 + 1 / 3)), ("NA", 250.5)]
    assert not result.truncated


def test_dates_are_real_dates_and_months_work(tool):
    result = tool.run("SELECT strftime(event_date, '%Y-%m') AS month, COUNT(*) FROM sales GROUP BY 1 ORDER BY 1")
    assert result.rows == [("2026-03", 2), ("2026-04", 1)]


def test_the_region_code_na_is_text_not_missing(tool):
    assert tool.run("SELECT COUNT(*) FROM sales WHERE region_id = 'NA'").rows == [(1,)]


def test_only_known_lake_columns_are_loaded(tool):
    with pytest.raises(SqlToolError, match="not_a_lake_column"):
        tool.run("SELECT not_a_lake_column FROM sales")


def test_text_for_the_model_rounds_money(tool):
    text = tool.run("SELECT transaction_id, total_amount FROM sales ORDER BY 1").to_text()
    assert text.splitlines() == ["transaction_id | total_amount", "T1 | 100.00", "T2 | 250.50", "T3 | 0.33"]


def test_empty_results_say_so(tool):
    assert tool.run("SELECT * FROM sales WHERE 1 = 0").to_text().endswith("(no rows)")


def test_large_results_are_cut_and_say_so(tool, monkeypatch):
    monkeypatch.setattr(sql, "MAX_ROWS", 2)
    result = tool.run("SELECT * FROM range(10)")
    assert len(result.rows) == 2 and result.truncated
    assert "only the first 2 rows" in result.to_text()


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("", "empty"),
        ("   ;  ", "empty"),
        ("DELETE FROM sales", "DELETE"),
        ("DROP TABLE sales", "DROP"),
        ("UPDATE sales SET total_amount = 0", "UPDATE"),
        ("INSERT INTO sales (transaction_id) VALUES ('X')", "INSERT"),
        ("CREATE TABLE x AS SELECT 1", "CREATE"),
        ("COPY sales TO 'leak.csv'", "COPY"),
        ("ATTACH 'other.db'", "ATTACH"),
        ("SET enable_external_access = true", "SET"),
        ("SELECT 1; DELETE FROM sales", "exactly one statement, not 2"),
        ("SELEC 1", "could not be read"),
    ],
)
def test_anything_but_one_select_is_refused(query, message):
    with pytest.raises(SqlToolError, match=message):
        check_select(query)


def test_selects_and_with_queries_pass_the_check():
    assert check_select("  SELECT 1 ;") == "SELECT 1"
    assert check_select("WITH x AS (SELECT 1 AS a) SELECT a FROM x") == "WITH x AS (SELECT 1 AS a) SELECT a FROM x"


def test_a_select_cannot_read_files(tool, tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text("a\n1\n")
    for query in (f"SELECT * FROM read_csv('{secret.as_posix()}')", f"SELECT * FROM '{secret.as_posix()}'"):
        with pytest.raises(SqlToolError, match="(?i)permission|disabled"):
            tool.run(query)


def test_the_connection_is_sealed(tool):
    settings = ("python_enable_replacements", "enable_external_access", "lock_configuration")
    query = "SELECT " + ", ".join(f"current_setting('{s}')" for s in settings)
    assert tool.run(query).rows == [(False, False, True)]


def test_the_data_is_unchanged_after_refused_writes(tool):
    for query in ("DELETE FROM sales", "DROP TABLE sales"):
        with pytest.raises(SqlToolError):
            tool.run(query)
    assert tool.run("SELECT COUNT(*) FROM sales").rows == [(3,)]


def test_a_slow_query_is_stopped(tool, monkeypatch):
    monkeypatch.setattr(sql, "TIMEOUT_SECONDS", 0.2)
    with pytest.raises(SqlToolError, match="longer than"):
        tool.run("SELECT COUNT(*) FROM range(100000000000) a, range(1000) b")


@pytest.fixture(scope="module")
def lake_rows():
    """Clean lake rows for 1 to 3 March 2026."""
    from datetime import date, datetime, timezone

    from sales_poc.datagen.history import generate_history
    from sales_poc.datagen.problems import add_problems
    from sales_poc.datagen.reference import build_reference_data
    from sales_poc.etl.pipeline import run_etl

    ref = build_reference_data(seed=42)
    messy = add_problems(generate_history(ref, seed=42, start=date(2026, 3, 1), days=3), seed=42)
    raw = pd.concat([messy.files[d] for d in sorted(messy.files)], ignore_index=True)
    return run_etl(raw, ref, "batch", ingested_at=datetime(2026, 11, 20, tzinfo=timezone.utc)).clean


class CountingStore:
    """A LocalStore that counts listings (each lake check is one listing)."""

    def __init__(self, root):
        from sales_poc.lake.store import LocalStore
        self.inner, self.listings = LocalStore(root), 0

    def stamps(self, prefix=""):
        self.listings += 1
        return self.inner.stamps(prefix)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def count(tool):
    return tool.run("SELECT COUNT(*) FROM sales").rows[0][0]


def test_new_lake_data_shows_up_after_the_check_interval(tmp_path, lake_rows):
    from sales_poc.lake.clean import write_clean

    first_two, third = lake_rows[lake_rows["event_date"].astype(str) < "2026-03-03"], \
        lake_rows[lake_rows["event_date"].astype(str) == "2026-03-03"]
    store, now = CountingStore(tmp_path / "bucket"), [0.0]
    write_clean(store.inner, first_two)
    tool = SqlTool(store=store, check_seconds=60, clock=lambda: now[0])

    assert count(tool) == len(first_two) and tool.check_lake() == 1
    write_clean(store.inner, third)                  # new orders reach the lake

    now[0] = 30.0                                    # before the interval: no check, old data
    assert count(tool) == len(first_two) and store.listings == 1

    now[0] = 61.0                                    # after it: one check, new data
    assert count(tool) == len(lake_rows) and store.listings == 2
    assert tool.check_lake() == 2                    # the version went up


def test_a_check_that_finds_nothing_keeps_the_version(tmp_path, lake_rows):
    from sales_poc.lake.clean import write_clean

    store, now = CountingStore(tmp_path / "bucket"), [0.0]
    write_clean(store.inner, lake_rows)
    tool = SqlTool(store=store, check_seconds=60, clock=lambda: now[0])
    assert tool.check_lake() == 1

    now[0] = 120.0
    assert tool.check_lake() == 1 and store.listings == 2


def test_refresh_checks_on_the_next_query_without_waiting(tmp_path, lake_rows):
    from sales_poc.lake.clean import write_clean

    store = CountingStore(tmp_path / "bucket")
    write_clean(store.inner, lake_rows)
    tool = SqlTool(store=store, check_seconds=3600, clock=lambda: 0.0)
    count(tool)

    tool.refresh()
    count(tool)
    assert store.listings == 2


def test_rows_given_directly_are_never_re_read():
    tool = SqlTool(rows=rows())
    assert tool.check_lake() == 1
    tool.refresh()
    assert count(tool) == len(rows()) and tool.check_lake() == 1


def test_the_tool_spec_names_every_column():
    spec = TOOL_SPEC["function"]
    assert spec["name"] == "sql_query" and spec["parameters"]["required"] == ["query"]
    for column in COLUMNS:
        assert column in spec["description"]


def test_the_columns_match_the_lake():
    from sales_poc.etl.pipeline import OUTPUT_COLUMNS
    assert set(COLUMNS) == set(OUTPUT_COLUMNS)


def test_no_files_were_left_behind():
    assert not Path("leak.csv").exists() and not Path("other.db").exists()


def test_times_are_shown_in_utc():
    frame = pd.DataFrame({"transaction_id": ["T1"],
                          "ingested_at": pd.to_datetime(["2026-11-19 02:29:01"], utc=True)})
    text = SqlTool(rows=frame).run("SELECT ingested_at FROM sales").to_text()
    assert "2026-11-19 02:29:01+00:00" in text
