"""The `sql_query` tool: read-only SQL on the clean data lake (task T12.1, FR-27 and FR-29).

The agent writes one SQL SELECT; this code runs it with DuckDB and returns a small table.

    sql-query "SELECT region_id, SUM(total_amount) AS revenue FROM sales GROUP BY 1 ORDER BY 2 DESC"

DuckDB is a small SQL engine that runs inside this Python process, with no server.

How the data gets in: the clean Parquet files are read from the lake the first time a query
runs, and copied into an in-memory DuckDB table called `sales`. Later queries reuse it. At
most every LAKE_CHECK_SECONDS (default 60), a query first asks the lake which days changed
(one quick listing, see lake/mirror.py). If any did, only those days are read again and the
table is rebuilt, so a long-running web API sees new orders without a restart.

Why it cannot change anything (FR-29), from the outside in:
    1. The query must parse as exactly one SELECT statement (DuckDB's own parser decides,
       not a text search). DELETE, COPY, SET, ATTACH, two statements and so on are refused.
    2. The DuckDB connection then has file and network access switched off, Python variable
       lookups switched off, and its settings locked, so a SELECT cannot read other files
       or turn the switches back on.
    3. Even if all that failed, the table is only an in-memory copy. The lake itself is never
       written by this code.

Limits: at most MAX_ROWS rows come back (the result says when it was cut), and a query that
runs longer than TIMEOUT_SECONDS is stopped.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections.abc import Callable

import duckdb
import pandas as pd

from sales_poc.ai.tools import ToolError
from sales_poc.ai.tools.table import Table

SqlResult = Table   # the name used before the table was shared with graph_query

TABLE = "sales"
MAX_ROWS = 50
TIMEOUT_SECONDS = 10.0

# Shown to the model so it can write correct SQL. Kept next to the code that builds the table.
COLUMNS = {
    "transaction_id": "unique order id",
    "event_time": "when the order happened (UTC timestamp)",
    "event_date": "the order's date (DATE)",
    "customer_id": "e.g. C00088",
    "customer_name": "company name",
    "segment": "customer segment: Enterprise, SMB or Consumer",
    "region_id": "NA (North America), EMEA, APJ or LATAM",
    "region_name": "full region name",
    "country": "customer's country",
    "product_id": "e.g. P0005",
    "product_name": "e.g. Atlas Laptop 577",
    "category": "Laptop, Desktop, Server, Storage, Monitor or Accessory",
    "channel_id": "ONLINE, RETAIL or DIRECT; UNKNOWN if it was missing",
    "channel_name": "Online, Retail Partner or Direct Sales",
    "quantity": "units ordered",
    "unit_price": "price per unit in USD",
    "discount_pct": "discount in percent, 0 to 100",
    "total_amount": "order value in USD after discount (use this for revenue)",
    "currency": "always USD",
    "source": "how it arrived: batch or stream",
    "ingested_at": "when the pipeline stored it (UTC timestamp)",
    "quality_flags": "notes on values the pipeline filled in; empty if none",
}

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "sql_query",
        "description": (
            "Run one read-only SQL SELECT (DuckDB dialect) on the clean sales data and get the rows back. "
            f"There is one table, `{TABLE}`, with one row per order. Columns: "
            + "; ".join(f"{name} ({about})" for name, about in COLUMNS.items())
            + f". At most {MAX_ROWS} rows are returned, so aggregate (SUM, COUNT, GROUP BY) instead of "
            "listing raw orders. For a month use strftime(event_date, '%Y-%m')."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "a single SQL SELECT statement"}},
            "required": ["query"],
        },
    },
}


class SqlToolError(ToolError):
    """The query was refused or failed. The message is meant to be shown to the model."""


def check_select(query: str) -> str:
    """The query, stripped, if it is exactly one SELECT. Otherwise SqlToolError."""
    text = (query or "").strip().rstrip(";").strip()
    if not text:
        raise SqlToolError("The query is empty.")
    try:
        statements = duckdb.extract_statements(text)
    except duckdb.Error as exc:
        raise SqlToolError(f"The SQL could not be read: {exc}") from None
    if len(statements) != 1:
        raise SqlToolError(f"Send exactly one statement, not {len(statements)}.")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise SqlToolError(f"Only SELECT is allowed, but this statement is {statements[0].type.name}.")
    return text


def locked_connection(rows: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB with `rows` as the table `sales`, then sealed off."""
    con = duckdb.connect(":memory:")
    frame = rows[[c for c in COLUMNS if c in rows.columns]].copy()
    if "event_date" in frame:
        frame["event_date"] = pd.to_datetime(frame["event_date"]).dt.date
    con.register("incoming", frame)
    con.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM incoming")
    con.unregister("incoming")
    if "event_date" in frame:
        con.execute(f"ALTER TABLE {TABLE} ALTER event_date TYPE DATE")
    con.execute("SET TimeZone = 'UTC'")    # show times in UTC, not this PC's local time
    con.execute("SET python_enable_replacements = false")
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


class SqlTool:
    def __init__(self, store=None, rows: pd.DataFrame | None = None, check_seconds: float | None = None,
                 clock: Callable[[], float] = time.monotonic):
        """Give `rows` directly (tests; never re-read), or a lake `store` (default: the S3 lake),
        checked for changed days at most every `check_seconds` (default: LAKE_CHECK_SECONDS)."""
        self._store, self._rows, self._con = store, rows, None
        self._check_seconds, self._clock = check_seconds, clock
        self._mirror = None
        self._checked_at: float | None = None
        self._version = 0
        self._lock = threading.Lock()

    def _connection(self) -> duckdb.DuckDBPyConnection:
        """The sealed connection, loading or updating the data first if needed.
        Call with self._lock held."""
        if self._rows is not None:
            if self._con is None:
                self._con, self._version = locked_connection(self._rows), 1
            return self._con
        now = self._clock()
        if self._con is None or self._checked_at is None or now - self._checked_at >= self._interval():
            changed = self._lake().refresh()
            self._checked_at = now
            if changed or self._con is None:
                self._con = locked_connection(self._lake().rows())
                self._version += 1
        return self._con

    def _interval(self) -> float:
        if self._check_seconds is None:
            from sales_poc.config import get_app_settings
            self._check_seconds = get_app_settings().lake_check_seconds
        return self._check_seconds

    def _lake(self):
        if self._mirror is None:
            from sales_poc.lake.mirror import LakeMirror
            from sales_poc.lake.store import s3_store
            self._mirror = LakeMirror(self._store or s3_store())
        return self._mirror

    def check_lake(self) -> int:
        """Load or update the data if a check is due, and return the data's version: a number
        that goes up every time the table is rebuilt with changed data."""
        with self._lock:
            self._connection()
            return self._version

    def refresh(self) -> None:
        """Check the lake for changed days on the next query, without waiting for the timer."""
        with self._lock:
            self._checked_at = None

    def run(self, query: str) -> SqlResult:
        text = check_select(query)
        # Queries take turns on the one sealed connection. (A DuckDB cursor would start with
        # default settings, and the locked configuration could not seal it again.)
        with self._lock:
            con = self._connection()
            timer = threading.Timer(TIMEOUT_SECONDS, con.interrupt)
            timer.start()
            try:
                result = con.execute(text)
                columns = [d[0] for d in result.description]
                rows = result.fetchmany(MAX_ROWS + 1)
            except duckdb.InterruptException:
                raise SqlToolError(
                    f"The query took longer than {TIMEOUT_SECONDS:.0f} seconds and was stopped.") from None
            except duckdb.Error as exc:
                raise SqlToolError(f"The query failed: {exc}") from None
            finally:
                timer.cancel()
        return SqlResult(text, columns, rows[:MAX_ROWS], len(rows) > MAX_ROWS)


def main(argv: list[str] | None = None) -> int:
    """sql-query: run one read-only SELECT on the lake, the same way the agent does."""
    parser = argparse.ArgumentParser(prog="sql-query", description="Read-only SQL on the clean sales data.")
    parser.add_argument("query")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        print(SqlTool().run(args.query).to_text())
    except SqlToolError as exc:
        print(f"Refused or failed: {exc}")
        return 1
    return 0
