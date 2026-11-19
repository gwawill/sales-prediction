"""The graph code against the real Neo4j AuraDB instance."""

import pytest
from neo4j.exceptions import ConstraintError

from sales_poc.graph.connection import driver as neo4j_driver
from sales_poc.graph.schema import NODE_KEYS, ensure_constraints, existing_constraints, missing_constraints


@pytest.fixture(scope="module")
def drv():
    with neo4j_driver() as d:
        yield d


def test_constraints_are_created_and_running_again_is_harmless(drv):
    ensure_constraints(drv)
    first = existing_constraints(drv)
    ensure_constraints(drv)

    assert missing_constraints(drv) == []
    assert existing_constraints(drv) == first


def test_a_second_node_with_the_same_id_is_refused(drv):
    ensure_constraints(drv)
    drv.execute_query("CREATE (:Region {region_id: '_TEST_REGION'})")
    try:
        with pytest.raises(ConstraintError):
            drv.execute_query("CREATE (:Region {region_id: '_TEST_REGION'})")
    finally:
        drv.execute_query("MATCH (r:Region {region_id: '_TEST_REGION'}) DELETE r")
    assert NODE_KEYS["Region"] == "region_id"
