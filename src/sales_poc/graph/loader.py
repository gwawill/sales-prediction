"""Load the sales graph into Neo4j (FR-18).

    (:Customer)-[:LOCATED_IN]->(:Region)
    (:Customer)-[:MADE]->(:Transaction)-[:FOR]->(:Product)      (transactions: task T8.3)
    (:Transaction)-[:VIA]->(:Channel)

Every node is written with MERGE ("find it, or create it if missing") and then its details
are set, so loading the same data again changes nothing. The uniqueness constraints from
schema.py back this up. Rows are sent in groups (UNWIND), so a few requests carry thousands
of rows instead of one request per row.
"""

from __future__ import annotations

import pandas as pd
from neo4j import Driver

from sales_poc.datagen.reference import ReferenceData
from sales_poc.etl.clean import UNKNOWN_CHANNEL
from sales_poc.etl.join import UNKNOWN_CHANNEL_NAME
from sales_poc.graph.schema import ensure_constraints

BATCH_ROWS = 500

REGIONS_CYPHER = """
UNWIND $rows AS row
MERGE (r:Region {region_id: row.region_id})
SET r.name = row.region_name
"""

CHANNELS_CYPHER = """
UNWIND $rows AS row
MERGE (c:Channel {channel_id: row.channel_id})
SET c.name = row.channel_name
"""

PRODUCTS_CYPHER = """
UNWIND $rows AS row
MERGE (p:Product {product_id: row.product_id})
SET p.name = row.product_name, p.category = row.category, p.list_price = row.list_price
"""

# A customer belongs to exactly one region: any old LOCATED_IN link is removed first.
CUSTOMERS_CYPHER = """
UNWIND $rows AS row
MERGE (c:Customer {customer_id: row.customer_id})
SET c.name = row.customer_name, c.segment = row.segment, c.country = row.country
WITH c, row
OPTIONAL MATCH (c)-[old:LOCATED_IN]->(other:Region)
WHERE other.region_id <> row.region_id
DELETE old
WITH DISTINCT c, row
MATCH (r:Region {region_id: row.region_id})
MERGE (c)-[:LOCATED_IN]->(r)
"""


def _records(table: pd.DataFrame) -> list[dict]:
    """Plain Python values (no numpy types), which the Neo4j driver can send."""
    return [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        for row in table.to_dict("records")
    ]


def channel_rows(ref: ReferenceData) -> list[dict]:
    """The real channels plus the UNKNOWN channel used when a channel was missing (FR-33)."""
    rows = _records(ref.channels[["channel_id", "channel_name"]])
    if UNKNOWN_CHANNEL not in {r["channel_id"] for r in rows}:
        rows.append({"channel_id": UNKNOWN_CHANNEL, "channel_name": UNKNOWN_CHANNEL_NAME})
    return rows


def reference_rows(ref: ReferenceData) -> dict[str, list[dict]]:
    """The rows for each reference query, in load order (regions before customers)."""
    return {
        "regions": _records(ref.regions[["region_id", "region_name"]]),
        "channels": channel_rows(ref),
        "products": _records(ref.products[["product_id", "product_name", "category", "list_price"]]),
        "customers": _records(
            ref.customers[["customer_id", "customer_name", "segment", "country", "region_id"]]
        ),
    }


def _run_in_batches(driver: Driver, cypher: str, rows: list[dict]) -> None:
    for start in range(0, len(rows), BATCH_ROWS):
        driver.execute_query(cypher, rows=rows[start:start + BATCH_ROWS])


def load_reference(driver: Driver, ref: ReferenceData) -> dict[str, int]:
    """Load regions, channels, products and customers. Returns rows sent per table."""
    ensure_constraints(driver)
    queries = {
        "regions": REGIONS_CYPHER,
        "channels": CHANNELS_CYPHER,
        "products": PRODUCTS_CYPHER,
        "customers": CUSTOMERS_CYPHER,
    }
    rows = reference_rows(ref)
    for table, cypher in queries.items():
        _run_in_batches(driver, cypher, rows[table])
    return {table: len(table_rows) for table, table_rows in rows.items()}


def graph_counts(driver: Driver) -> dict[str, int]:
    """How many nodes of each type and links of each type the graph holds."""
    nodes, _, _ = driver.execute_query(
        "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n"
    )
    links, _, _ = driver.execute_query("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n")
    counts = {r["label"]: r["n"] for r in nodes}
    counts.update({r["type"]: r["n"] for r in links})
    return dict(sorted(counts.items()))
