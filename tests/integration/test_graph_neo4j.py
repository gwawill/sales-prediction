"""The graph code against the real Neo4j AuraDB instance."""

import pytest
from neo4j.exceptions import ConstraintError

from sales_poc.config import get_app_settings
from sales_poc.datagen.reference import load_reference_data
from sales_poc.graph.connection import driver as neo4j_driver
from sales_poc.graph.loader import graph_counts, load_reference
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


def test_reference_nodes_load_and_loading_again_changes_nothing(drv):
    """Loads the REAL reference data and leaves it there: it is what the graph should hold."""
    ref = load_reference_data(get_app_settings().data_dir / "reference")

    sent = load_reference(drv, ref)
    first = graph_counts(drv)
    load_reference(drv, ref)
    second = graph_counts(drv)

    assert sent == {"regions": 4, "channels": 4, "products": 40, "customers": 500}
    assert second == first
    for label, n in [("Region", 4), ("Channel", 4), ("Product", 40), ("Customer", 500)]:
        assert first[label] == n
    assert first["LOCATED_IN"] == 500


def test_every_customer_has_exactly_one_region_matching_the_reference(drv):
    ref = load_reference_data(get_app_settings().data_dir / "reference")
    load_reference(drv, ref)

    records, _, _ = drv.execute_query(
        "MATCH (c:Customer) OPTIONAL MATCH (c)-[:LOCATED_IN]->(r:Region) "
        "RETURN c.customer_id AS id, collect(r.region_id) AS regions"
    )
    expected = dict(zip(ref.customers["customer_id"], ref.customers["region_id"]))
    assert len(records) == 500
    for r in records:
        assert r["regions"] == [expected[r["id"]]]


def test_a_customer_who_moves_region_keeps_only_the_new_link(drv):
    from sales_poc.graph.loader import CUSTOMERS_CYPHER

    customer = {"customer_id": "_TEST_C", "customer_name": "Test", "segment": "SMB", "country": "x"}
    try:
        drv.execute_query(CUSTOMERS_CYPHER, rows=[{**customer, "region_id": "NA"}])
        drv.execute_query(CUSTOMERS_CYPHER, rows=[{**customer, "region_id": "EMEA"}])
        records, _, _ = drv.execute_query(
            "MATCH (:Customer {customer_id: '_TEST_C'})-[:LOCATED_IN]->(r:Region) RETURN r.region_id AS r"
        )
        assert [r["r"] for r in records] == ["EMEA"]
    finally:
        drv.execute_query("MATCH (c:Customer {customer_id: '_TEST_C'}) DETACH DELETE c")


def _test_tx(**changes):
    row = {
        "transaction_id": "_TEST_TX", "customer_id": "C00001", "product_id": "P0001",
        "channel_id": "ONLINE", "event_time": "2026-03-01T10:00:00+00:00", "event_date": "2026-03-01",
        "quantity": 1, "unit_price": 10.0, "discount_pct": 0.0, "total_amount": 10.0,
        "source": "batch", "quality_flags": "",
    }
    return {**row, **changes}


def _links(drv):
    records, _, _ = drv.execute_query("""
        MATCH (t:Transaction {transaction_id: '_TEST_TX'})
        RETURN [(c:Customer)-[:MADE]->(t) | c.customer_id] AS customers,
               [(t)-[:FOR]->(p:Product) | p.product_id] AS products,
               [(t)-[:VIA]->(ch:Channel) | ch.channel_id] AS channels,
               t.quantity AS quantity, toString(t.event_date) AS day
    """)
    return dict(records[0])


def test_a_transaction_gets_one_link_of_each_kind_and_corrections_move_them(drv):
    from sales_poc.graph.loader import TRANSACTIONS_CYPHER

    load_reference(drv, load_reference_data(get_app_settings().data_dir / "reference"))
    try:
        drv.execute_query(TRANSACTIONS_CYPHER, rows=[_test_tx()])
        drv.execute_query(TRANSACTIONS_CYPHER, rows=[_test_tx()])  # again: no change
        assert _links(drv) == {"customers": ["C00001"], "products": ["P0001"], "channels": ["ONLINE"],
                               "quantity": 1, "day": "2026-03-01"}

        # A corrected copy: different customer, product, channel and quantity.
        drv.execute_query(TRANSACTIONS_CYPHER, rows=[_test_tx(
            customer_id="C00002", product_id="P0002", channel_id="UNKNOWN", quantity=3)])
        assert _links(drv) == {"customers": ["C00002"], "products": ["P0002"], "channels": ["UNKNOWN"],
                               "quantity": 3, "day": "2026-03-01"}
    finally:
        drv.execute_query("MATCH (t:Transaction {transaction_id: '_TEST_TX'}) DETACH DELETE t")
