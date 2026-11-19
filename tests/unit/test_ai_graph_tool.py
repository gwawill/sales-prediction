from types import SimpleNamespace as NS

import neo4j
import pytest
from neo4j.exceptions import ClientError, CypherSyntaxError
from neo4j.time import Date, DateTime

from sales_poc.ai.tools import graph
from sales_poc.ai.tools.graph import TOOL_SPEC, GraphTool, GraphToolError, clean_query, plain
from sales_poc.ai.tools.table import Table


def neo4j_error(cls, code, message):
    exc = cls(message)
    exc._neo4j_code, exc._message = code, message  # how the driver fills these in
    return exc


class FakeResult:
    def __init__(self, columns, rows):
        self._columns, self._rows = columns, rows

    def keys(self):
        return self._columns

    def __iter__(self):
        return (NS(values=lambda r=row: list(r)) for row in self._rows)


class FakeDriver:
    """Neo4j's answers, chosen per test: the EXPLAIN query type, and the rows (or an error)."""

    def __init__(self, query_type="r", columns=("n",), rows=((1,),), error=None, explain_error=None):
        self.query_type, self.columns, self.rows = query_type, list(columns), list(rows)
        self.error, self.explain_error = error, explain_error
        self.sessions, self.ran, self.closed = [], [], False

    def session(self, **config):
        self.sessions.append(config)
        driver = self

        class Session:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def run(self, query):
                assert query.text.startswith("EXPLAIN ")
                if driver.explain_error:
                    raise driver.explain_error
                return NS(consume=lambda: NS(query_type=driver.query_type))

            def execute_read(self, work):
                tx = NS(run=self._tx_run)
                return work(tx)

            def _tx_run(self, text):
                driver.ran.append(text)
                if driver.error:
                    raise driver.error
                return FakeResult(driver.columns, driver.rows)

        return Session()

    def close(self):
        self.closed = True


def test_a_read_query_returns_a_table():
    fake = FakeDriver(columns=["region", "customers"], rows=[("NA", 201), ("EMEA", 144)])
    result = GraphTool(fake).run("MATCH (c:Customer) RETURN c ;")

    assert isinstance(result, Table)
    assert result.columns == ["region", "customers"] and result.rows == [("NA", 201), ("EMEA", 144)]
    assert fake.ran == ["MATCH (c:Customer) RETURN c"]


def test_every_session_is_read_only_and_quiet():
    fake = FakeDriver()
    GraphTool(fake).run("MATCH (n) RETURN n")
    assert fake.sessions and all(s["default_access_mode"] == neo4j.READ_ACCESS for s in fake.sessions)
    assert all(s["notifications_min_severity"] == "OFF" for s in fake.sessions)


@pytest.mark.parametrize(("kind", "words"), [("w", "writes"), ("rw", "writes"), ("s", "schema")])
def test_queries_neo4j_calls_writes_are_refused_without_running(kind, words):
    fake = FakeDriver(query_type=kind)
    with pytest.raises(GraphToolError, match=words):
        GraphTool(fake).run("CREATE (:X)")
    assert fake.ran == []


@pytest.mark.parametrize("query", ["LOAD CSV FROM 'https://x' AS r RETURN r", "load\n csv from 'x' as r return r"])
def test_load_csv_is_refused_before_asking_neo4j(query):
    fake = FakeDriver()
    with pytest.raises(GraphToolError, match="LOAD CSV"):
        GraphTool(fake).run(query)
    assert fake.sessions == []


@pytest.mark.parametrize("query", ["", "  ; "])
def test_an_empty_query_is_refused(query):
    with pytest.raises(GraphToolError, match="empty"):
        clean_query(query)


def test_unreadable_cypher_is_explained():
    error = neo4j_error(CypherSyntaxError, "Neo.ClientError.Statement.SyntaxError", "Invalid input 'MATC'")
    with pytest.raises(GraphToolError, match="could not be read: Invalid input 'MATC'"):
        GraphTool(FakeDriver(explain_error=error)).run("MATC (n) RETURN n")


def test_a_slow_query_is_reported_as_stopped():
    error = neo4j_error(ClientError, "Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration", "timed out")
    with pytest.raises(GraphToolError, match="longer than"):
        GraphTool(FakeDriver(error=error)).run("MATCH (n) RETURN n")


def test_other_failures_are_explained():
    error = neo4j_error(ClientError, "Neo.ClientError.Statement.ArgumentError", "bad argument")
    with pytest.raises(GraphToolError, match="failed: bad argument"):
        GraphTool(FakeDriver(error=error)).run("MATCH (n) RETURN n")


def test_large_results_are_cut_and_say_so(monkeypatch):
    monkeypatch.setattr(graph, "MAX_ROWS", 2)
    result = GraphTool(FakeDriver(rows=[(i,) for i in range(10)])).run("MATCH (n) RETURN n")
    assert result.rows == [(0,), (1,)] and result.truncated
    assert "only the first 2 rows" in result.to_text()


def test_neo4j_values_become_plain_values():
    assert plain(Date(2026, 3, 1)) == "2026-03-01"
    assert plain(DateTime(2026, 3, 1, 12, 0, 0)).startswith("2026-03-01T12:00:00")
    assert plain([Date(2026, 3, 1), {"d": Date(2026, 3, 2)}]) == ["2026-03-01", {"d": "2026-03-02"}]
    assert plain(13.5) == 13.5 and plain("x") == "x" and plain(None) is None


def test_a_driver_given_by_the_caller_is_not_closed():
    fake = FakeDriver()
    GraphTool(fake).close()
    assert not fake.closed


def test_the_tool_spec_describes_the_graph():
    spec = TOOL_SPEC["function"]
    assert spec["name"] == "graph_query" and spec["parameters"]["required"] == ["query"]
    for word in ("Customer", "Transaction", "Product", "Region", "Channel", "MADE", "FOR", "VIA", "LOCATED_IN"):
        assert word in spec["description"]


def test_the_tool_spec_matches_the_loader():
    """Every property the loader sets is named in the description the model sees."""
    import re

    from sales_poc.graph import loader

    cypher = " ".join(v for k, v in vars(loader).items() if k.endswith("_CYPHER") and isinstance(v, str))
    for prop in set(re.findall(r"\b[a-z]{1,3}\.([a-z_]+) =", cypher)):
        assert prop in TOOL_SPEC["function"]["description"], prop
