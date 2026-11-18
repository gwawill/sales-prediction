import pytest

from sales_poc.graph.connection import driver_arguments


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("neo4j+s://abc.databases.neo4j.io", "neo4j://abc.databases.neo4j.io"),
        ("bolt+s://abc.databases.neo4j.io:7687", "bolt://abc.databases.neo4j.io:7687"),
    ],
)
def test_secure_schemes_become_plain_scheme_with_encryption_on(uri, expected):
    new_uri, options = driver_arguments(uri)

    assert new_uri == expected
    assert options["encrypted"] is True
    assert "trusted_certificates" in options


def test_other_uris_are_left_unchanged():
    assert driver_arguments("neo4j://localhost:7687") == ("neo4j://localhost:7687", {})
