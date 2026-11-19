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


# A transaction links to exactly one customer, product and channel. If a corrected copy
# changes one of them, the old link is removed before the new one is made.
TRANSACTIONS_CYPHER = """
UNWIND $rows AS row
MERGE (t:Transaction {transaction_id: row.transaction_id})
SET t.event_time = datetime(row.event_time),
    t.event_date = date(row.event_date),
    t.quantity = row.quantity,
    t.unit_price = row.unit_price,
    t.discount_pct = row.discount_pct,
    t.total_amount = row.total_amount,
    t.source = row.source,
    t.quality_flags = row.quality_flags
WITH t, row
OPTIONAL MATCH (t)-[old_for:FOR]->(p0:Product) WHERE p0.product_id <> row.product_id
DELETE old_for
WITH DISTINCT t, row
OPTIONAL MATCH (t)-[old_via:VIA]->(ch0:Channel) WHERE ch0.channel_id <> row.channel_id
DELETE old_via
WITH DISTINCT t, row
OPTIONAL MATCH (c0:Customer)-[old_made:MADE]->(t) WHERE c0.customer_id <> row.customer_id
DELETE old_made
WITH DISTINCT t, row
MATCH (c:Customer {customer_id: row.customer_id})
MATCH (p:Product {product_id: row.product_id})
MATCH (ch:Channel {channel_id: row.channel_id})
MERGE (c)-[:MADE]->(t)
MERGE (t)-[:FOR]->(p)
MERGE (t)-[:VIA]->(ch)
"""

TRANSACTION_BATCH_ROWS = 1000


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


def transaction_rows(clean: pd.DataFrame) -> list[dict]:
    """Clean lake rows as plain values for TRANSACTIONS_CYPHER."""
    table = pd.DataFrame({
        "transaction_id": clean["transaction_id"].astype(str),
        "customer_id": clean["customer_id"].astype(str),
        "product_id": clean["product_id"].astype(str),
        "channel_id": clean["channel_id"].astype(str),
        # ISO text, which Cypher's datetime() and date() read directly.
        "event_time": clean["event_time"].map(lambda t: t.isoformat()),
        "event_date": clean["event_date"].map(lambda d: d.isoformat()),
        "quantity": clean["quantity"].astype(int),
        "unit_price": clean["unit_price"].astype(float),
        "discount_pct": clean["discount_pct"].astype(float),
        "total_amount": clean["total_amount"].astype(float),
        "source": clean["source"].astype(str),
        "quality_flags": clean["quality_flags"].fillna("").astype(str),
    })
    return _records(table)


def load_transactions(driver: Driver, clean: pd.DataFrame, batch_rows: int = TRANSACTION_BATCH_ROWS) -> int:
    """Load clean transactions and their links. Reference nodes must be loaded first."""
    rows = transaction_rows(clean)
    for start in range(0, len(rows), batch_rows):
        driver.execute_query(TRANSACTIONS_CYPHER, rows=rows[start:start + batch_rows])
    return len(rows)


def link_problems(driver: Driver) -> dict[str, int]:
    """Transactions that do not have exactly one customer, product and channel (should be 0)."""
    records, _, _ = driver.execute_query("""
        MATCH (t:Transaction)
        WITH t,
             COUNT { (:Customer)-[:MADE]->(t) } AS made,
             COUNT { (t)-[:FOR]->(:Product) } AS for_,
             COUNT { (t)-[:VIA]->(:Channel) } AS via
        RETURN sum(CASE WHEN made <> 1 THEN 1 ELSE 0 END) AS bad_made,
               sum(CASE WHEN for_ <> 1 THEN 1 ELSE 0 END) AS bad_for,
               sum(CASE WHEN via <> 1 THEN 1 ELSE 0 END) AS bad_via
    """)
    return dict(records[0])


def graph_counts(driver: Driver) -> dict[str, int]:
    """How many nodes of each type and links of each type the graph holds."""
    nodes, _, _ = driver.execute_query(
        "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n"
    )
    links, _, _ = driver.execute_query("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n")
    counts = {r["label"]: r["n"] for r in nodes}
    counts.update({r["type"]: r["n"] for r in links})
    return dict(sorted(counts.items()))


def main(argv: list[str] | None = None) -> int:
    """graph-load: load the reference data and every clean transaction into Neo4j."""
    import argparse
    import sys
    import time

    from sales_poc.config import get_app_settings
    from sales_poc.datagen.reference import load_reference_data
    from sales_poc.graph.connection import driver as neo4j_driver
    from sales_poc.lake.clean import read_all_clean
    from sales_poc.lake.store import s3_store

    parser = argparse.ArgumentParser(prog="graph-load", description="Load the sales graph into Neo4j.")
    parser.add_argument("--reference", default=str(get_app_settings().data_dir / "reference"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    sys.stdout.reconfigure(line_buffering=True)

    started = time.monotonic()
    ref = load_reference_data(args.reference)
    print("Reading the clean lake from S3...")
    clean = read_all_clean(s3_store())
    print(f"  {len(clean)} transactions on {clean['event_date'].nunique()} days")

    with neo4j_driver() as drv:
        sent = load_reference(drv, ref)
        print(f"Loaded reference nodes: {sent}")
        n = load_transactions(drv, clean)
        print(f"Loaded {n} transactions")
        counts = graph_counts(drv)
        problems = link_problems(drv)

    print(f"Graph now: {counts}")
    print(f"Transactions without exactly one customer/product/channel: {problems}")
    print(f"Done in {time.monotonic() - started:.0f}s")
    return 0 if not any(problems.values()) else 1
