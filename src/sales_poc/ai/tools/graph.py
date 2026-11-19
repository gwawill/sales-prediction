"""The `graph_query` tool: read-only Cypher on Neo4j (task T12.2, FR-27 and FR-29).

The agent writes one Cypher query; this code runs it on the graph and returns a small table.
Cypher is Neo4j's query language: it describes a pattern of linked nodes to find.

    graph-query "MATCH (c:Customer)-[:LOCATED_IN]->(r:Region) RETURN r.region_id, count(c) ORDER BY 2 DESC"

Why it cannot change anything (FR-29), from the outside in:
    1. LOAD CSV is refused by name. It only reads, but it can fetch any web address.
    2. Neo4j plans the query with EXPLAIN (plans it without running it) and reports its type.
       Only "r" (read only) is allowed; "w", "rw" (writes) and "s" (schema changes) are refused.
    3. The query then runs inside a read-only transaction, and the Neo4j server itself
       rejects any write in one ("Writing in read access mode not allowed").

Limits: at most MAX_ROWS rows come back (the result says when it was cut), and Neo4j stops a
query that runs longer than TIMEOUT_SECONDS.
"""

from __future__ import annotations

import argparse
import re
import sys

import neo4j
from neo4j import Query, unit_of_work
from neo4j.exceptions import ClientError, CypherSyntaxError, Neo4jError
from neo4j.graph import Node, Path, Relationship

from sales_poc.ai.tools.table import Table

MAX_ROWS = 50
TIMEOUT_SECONDS = 10.0
LOAD_CSV = re.compile(r"\bLOAD\s+CSV\b", re.IGNORECASE)

# Shown to the model so it can write correct Cypher. Matches graph/loader.py.
GRAPH_SCHEMA = (
    "Nodes: "
    "(:Customer {customer_id, name, segment: Enterprise|SMB|Consumer, country}); "
    "(:Region {region_id: NA|EMEA|APJ|LATAM, name}); "
    "(:Product {product_id, name, category: Laptop|Desktop|Server|Storage|Monitor|Accessory, list_price}); "
    "(:Channel {channel_id: ONLINE|RETAIL|DIRECT|UNKNOWN, name}); "
    "(:Transaction {transaction_id, event_time (datetime), event_date (date), quantity, unit_price, "
    "discount_pct, total_amount (USD after discount), source, quality_flags}). "
    "Links: (Customer)-[:LOCATED_IN]->(Region), (Customer)-[:MADE]->(Transaction), "
    "(Transaction)-[:FOR]->(Product), (Transaction)-[:VIA]->(Channel)."
)

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "graph_query",
        "description": (
            "Run one read-only Cypher query on the Neo4j sales graph and get the rows back. "
            "Best for questions about connections, such as which customers bought both X and Y. "
            + GRAPH_SCHEMA
            + f" At most {MAX_ROWS} rows are returned, so return properties and counts, "
            "not whole nodes, and use count() or LIMIT for long lists."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "a single read-only Cypher query"}},
            "required": ["query"],
        },
    },
}


class GraphToolError(ValueError):
    """The query was refused or failed. The message is meant to be shown to the model."""


def plain(value):
    """Neo4j values as plain Python values that print readably."""
    if isinstance(value, Node):
        return {"labels": sorted(value.labels), **{k: plain(v) for k, v in value.items()}}
    if isinstance(value, Relationship):
        return {"type": value.type, **{k: plain(v) for k, v in value.items()}}
    if isinstance(value, Path):
        return [plain(n) for n in value.nodes]
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if hasattr(value, "iso_format"):          # neo4j.time Date, DateTime, Duration, ...
        return value.iso_format()
    return value


def clean_query(query: str) -> str:
    text = (query or "").strip().rstrip(";").strip()
    if not text:
        raise GraphToolError("The query is empty.")
    if LOAD_CSV.search(text):
        raise GraphToolError("LOAD CSV is not allowed.")
    return text


def _message(exc: Neo4jError) -> str:
    return (getattr(exc, "message", None) or str(exc)).strip()


class GraphTool:
    def __init__(self, driver=None):
        """Give a Neo4j `driver` (tests), or one is opened from the settings on first use."""
        self._driver, self._owns_driver = driver, driver is None

    @property
    def driver(self):
        if self._driver is None:
            from sales_poc.graph.connection import driver as neo4j_driver
            self._driver = neo4j_driver()
        return self._driver

    def close(self) -> None:
        if self._owns_driver and self._driver is not None:
            self._driver.close()
            self._driver = None

    def _session(self):
        # Read access; and no "label does not exist" style warnings printed to the console.
        return self.driver.session(default_access_mode=neo4j.READ_ACCESS, notifications_min_severity="OFF")

    def query_type(self, text: str) -> str:
        """Neo4j's own verdict, without running the query: r, w, rw or s."""
        try:
            with self._session() as session:
                return session.run(Query("EXPLAIN " + text, timeout=TIMEOUT_SECONDS)).consume().query_type
        except CypherSyntaxError as exc:
            raise GraphToolError(f"The Cypher could not be read: {_message(exc)}") from None
        except Neo4jError as exc:
            raise GraphToolError(f"The query failed: {_message(exc)}") from None

    def run(self, query: str) -> Table:
        text = clean_query(query)
        kind = self.query_type(text)
        if kind != "r":
            what = {"w": "writes", "rw": "writes", "s": "changes the schema"}.get(kind, f"is of type {kind!r}")
            raise GraphToolError(f"Only read-only queries are allowed, but this one {what}.")

        @unit_of_work(timeout=TIMEOUT_SECONDS)
        def work(tx):
            result = tx.run(text)
            columns, rows = list(result.keys()), []
            for record in result:
                rows.append(tuple(plain(v) for v in record.values()))
                if len(rows) > MAX_ROWS:
                    break
            return columns, rows

        try:
            with self._session() as session:
                columns, rows = session.execute_read(work)
        except ClientError as exc:
            if "TimedOut" in (exc.code or "") or "Terminated" in (exc.code or ""):
                raise GraphToolError(
                    f"The query took longer than {TIMEOUT_SECONDS:.0f} seconds and was stopped.") from None
            raise GraphToolError(f"The query failed: {_message(exc)}") from None
        except Neo4jError as exc:
            raise GraphToolError(f"The query failed: {_message(exc)}") from None
        return Table(text, columns, rows[:MAX_ROWS], len(rows) > MAX_ROWS)


def main(argv: list[str] | None = None) -> int:
    """graph-query: run one read-only Cypher query, the same way the agent does."""
    parser = argparse.ArgumentParser(prog="graph-query", description="Read-only Cypher on the sales graph.")
    parser.add_argument("query")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    tool = GraphTool()
    try:
        print(tool.run(args.query).to_text())
    except GraphToolError as exc:
        print(f"Refused or failed: {exc}")
        return 1
    finally:
        tool.close()
    return 0
