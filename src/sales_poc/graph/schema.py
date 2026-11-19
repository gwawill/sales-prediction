"""Neo4j uniqueness rules (constraints) for the sales graph (FR-18).

Graph shape (plan.md):

    (:Customer)-[:LOCATED_IN]->(:Region)
    (:Customer)-[:MADE]->(:Transaction)-[:FOR]->(:Product)
    (:Transaction)-[:VIA]->(:Channel)

One uniqueness constraint per node type means two nodes of the same type can never share an
ID. Together with MERGE in the loader, that makes loading twice harmless. Neo4j also builds
an index for each constraint, so finding a node by its ID is fast.

Every statement uses IF NOT EXISTS, so `ensure_constraints` can run any number of times.
"""

from __future__ import annotations

from neo4j import Driver

# node label -> the property that identifies it
NODE_KEYS: dict[str, str] = {
    "Customer": "customer_id",
    "Product": "product_id",
    "Region": "region_id",
    "Channel": "channel_id",
    "Transaction": "transaction_id",
}


def constraint_name(label: str) -> str:
    return f"{label.lower()}_{NODE_KEYS[label]}_unique"


def constraint_statements() -> list[str]:
    return [
        f"CREATE CONSTRAINT {constraint_name(label)} IF NOT EXISTS "
        f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
        for label, key in NODE_KEYS.items()
    ]


def ensure_constraints(driver: Driver) -> None:
    """Create any missing constraints. Safe to run again."""
    for statement in constraint_statements():
        driver.execute_query(statement)


# Neo4j renamed this type during version 5; accept both names.
UNIQUENESS_TYPES = ["UNIQUENESS", "NODE_PROPERTY_UNIQUENESS"]


def existing_constraints(driver: Driver) -> dict[str, tuple[str, str]]:
    """{constraint name: (label, property)} for the uniqueness constraints in the database."""
    records, _, _ = driver.execute_query(
        "SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties "
        "WHERE type IN $types RETURN name, labelsOrTypes, properties",
        types=UNIQUENESS_TYPES,
    )
    return {r["name"]: (r["labelsOrTypes"][0], r["properties"][0]) for r in records}


def missing_constraints(driver: Driver) -> list[str]:
    """Node types that do not yet have their uniqueness constraint."""
    have = set(existing_constraints(driver).values())
    return [label for label, key in NODE_KEYS.items() if (label, key) not in have]
