from sales_poc.graph.schema import NODE_KEYS, constraint_name, constraint_statements


def test_one_constraint_per_node_type_from_the_plan():
    assert set(NODE_KEYS) == {"Customer", "Product", "Region", "Channel", "Transaction"}
    assert len(constraint_statements()) == 5


def test_statements_are_safe_to_run_again_and_name_the_right_property():
    for statement, (label, key) in zip(constraint_statements(), NODE_KEYS.items()):
        assert "IF NOT EXISTS" in statement
        assert f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE" in statement
        assert constraint_name(label) in statement


def test_constraint_names_are_unique_and_readable():
    names = [constraint_name(label) for label in NODE_KEYS]
    assert len(set(names)) == len(names)
    assert constraint_name("Transaction") == "transaction_transaction_id_unique"
