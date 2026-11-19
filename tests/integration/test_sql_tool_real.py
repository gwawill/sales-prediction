"""The sql_query tool on the real S3 lake, and acceptance check 6 against the real Neo4j.

Check 6: Neo4j answers "which enterprise customers who bought servers also bought storage"
the same way a SQL query on the lake does. Assumes graph-load has run after the last lake load.
"""

import pytest

from sales_poc.ai.tools.sql import SqlTool
from sales_poc.graph.connection import driver as neo4j_driver

SQL_CHECK_6 = """
SELECT DISTINCT customer_id FROM sales
WHERE segment = 'Enterprise' AND category = 'Server'
INTERSECT
SELECT DISTINCT customer_id FROM sales
WHERE segment = 'Enterprise' AND category = 'Storage'
ORDER BY customer_id
"""

CYPHER_CHECK_6 = """
MATCH (c:Customer {segment: 'Enterprise'})-[:MADE]->(:Transaction)-[:FOR]->(:Product {category: 'Server'})
WHERE EXISTS { (c)-[:MADE]->(:Transaction)-[:FOR]->(:Product {category: 'Storage'}) }
RETURN DISTINCT c.customer_id AS customer_id ORDER BY customer_id
"""


@pytest.fixture(scope="module")
def tool():
    return SqlTool()  # reads the real lake once (about 30 seconds)


@pytest.fixture(scope="module")
def drv():
    with neo4j_driver() as d:
        yield d


def test_the_whole_lake_is_loaded_and_matches_the_graph(tool, drv):
    [(lake_rows,)] = tool.run("SELECT COUNT(*) FROM sales").rows
    records, _, _ = drv.execute_query("MATCH (t:Transaction) RETURN count(t) AS n")
    assert lake_rows > 0
    assert lake_rows == records[0]["n"]


def test_acceptance_check_6_sql_and_graph_agree(tool, drv):
    # One row holding the whole list, so the tool's row limit cannot hide a difference.
    [(lake_list,)] = tool.run(f"SELECT string_agg(customer_id, ',' ORDER BY customer_id) FROM ({SQL_CHECK_6})").rows
    records, _, _ = drv.execute_query(CYPHER_CHECK_6)
    graph = [r["customer_id"] for r in records]

    print(f"\nCheck 6: {len(graph)} enterprise customers bought both servers and storage")
    assert graph, "the graph found no such customers"
    assert lake_list.split(",") == graph
