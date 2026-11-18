from datetime import datetime, timezone

import pandas as pd
import pytest

from sales_poc.etl.schema import (
    CLEAN_COLUMNS,
    CLEAN_SCHEMA,
    check_schema,
    empty_clean_frame,
    validate,
)

NOW = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)

GOOD = {
    "transaction_id": "TX-20261118-00001",
    "event_time": pd.Timestamp("2026-11-18T10:00:00Z"),
    "customer_id": "C00001",
    "product_id": "P0001",
    "channel_id": "ONLINE",
    "quantity": 2,
    "unit_price": 100.0,
    "discount_pct": 5.0,
    "total_amount": 190.0,
    "currency": "USD",
}


def rows(*changes: dict) -> pd.DataFrame:
    """A typed table with one row per dict of changes applied to GOOD."""
    records = [{**GOOD, **change} for change in changes]
    table = pd.DataFrame(records, columns=CLEAN_COLUMNS)
    return table.astype(CLEAN_SCHEMA)


def reasons_for(change: dict) -> str:
    return validate(rows(change), now=NOW).iloc[0]


def test_a_good_row_has_no_reasons():
    assert reasons_for({}) == ""


@pytest.mark.parametrize("column", CLEAN_COLUMNS)
def test_each_missing_field_is_reported(column):
    assert reasons_for({column: None}) == f"missing:{column}"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"customer_id": "C99"}, "invalid:customer_id"),
        ({"customer_id": "X00001"}, "invalid:customer_id"),
        ({"product_id": "P12345"}, "invalid:product_id"),
        ({"quantity": 0}, "invalid:quantity"),
        ({"quantity": -3}, "invalid:quantity"),
        ({"unit_price": 0.0}, "invalid:unit_price"),
        ({"discount_pct": -1.0}, "invalid:discount_pct"),
        ({"discount_pct": 100.5}, "invalid:discount_pct"),
        ({"total_amount": -0.01}, "invalid:total_amount"),
        ({"currency": "usd"}, "invalid:currency"),
        ({"event_time": pd.Timestamp("2026-11-20T12:00:01Z")}, "invalid:event_time_future"),
    ],
)
def test_each_rule(change, reason):
    assert reasons_for(change) == reason


def test_edges_that_are_allowed():
    assert reasons_for({"discount_pct": 0.0}) == ""
    assert reasons_for({"discount_pct": 100.0}) == ""
    assert reasons_for({"total_amount": 0.0}) == ""
    # Up to one day ahead is allowed (clocks and time zones are not perfect).
    assert reasons_for({"event_time": pd.Timestamp("2026-11-19T11:59:00Z")}) == ""


def test_well_formed_but_unknown_ids_pass_here():
    # Whether C99999 exists is checked against reference data in the join step (T3.5).
    assert reasons_for({"customer_id": "C99999", "product_id": "P9999"}) == ""


def test_several_problems_in_one_row_are_all_listed():
    assert reasons_for({"customer_id": None, "quantity": -1}) == "missing:customer_id; invalid:quantity"


def test_one_reason_per_row_and_index_is_kept():
    table = rows({}, {"quantity": -1}, {"currency": None})
    table.index = [10, 20, 30]

    result = validate(table, now=NOW)

    assert list(result.index) == [10, 20, 30]
    assert list(result) == ["", "invalid:quantity", "missing:currency"]


def test_empty_table_is_fine():
    assert len(validate(empty_clean_frame(), now=NOW)) == 0


def test_wrong_shape_is_a_programming_error():
    with pytest.raises(ValueError, match="Missing columns"):
        validate(rows({}).drop(columns="currency"), now=NOW)
    with pytest.raises(TypeError, match="Wrong column types"):
        check_schema(rows({}).astype({"quantity": "float64"}))
