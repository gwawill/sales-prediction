"""The graph_query tool against the real Neo4j AuraDB instance."""

import pytest

from sales_poc.ai.tools.graph import GraphTool, GraphToolError


@pytest.fixture(scope="module")
def tool():
    t = GraphTool()
    yield t
    t.close()


def node_count(tool):
    return tool.run("MATCH (n) RETURN count(n) AS n").rows[0][0]


def test_a_real_read(tool):
    result = tool.run("MATCH (r:Region) RETURN r.region_id AS region ORDER BY region")
    assert [row[0] for row in result.rows] == ["APJ", "EMEA", "LATAM", "NA"]


@pytest.mark.parametrize("query", [
    "CREATE (:ToolTest {x: 1})",
    "MERGE (r:Region {region_id: '_TOOL_TEST'}) RETURN r",
    "MATCH (r:Region) SET r.tool_test = 1 RETURN count(r)",
    "CREATE INDEX tool_test IF NOT EXISTS FOR (n:ToolTest) ON (n.x)",
])
def test_writes_are_refused_and_change_nothing(tool, query):
    before = node_count(tool)
    with pytest.raises(GraphToolError, match="read-only"):
        tool.run(query)
    assert node_count(tool) == before
    assert tool.run("MATCH (r:Region) WHERE r.tool_test IS NOT NULL RETURN count(r)").rows == [(0,)]
