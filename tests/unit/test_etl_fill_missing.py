import pandas as pd
import pytest

from sales_poc.datagen.history import COLUMNS, generate_history
from sales_poc.datagen.problems import add_problems
from sales_poc.datagen.reference import build_reference_data
from sales_poc.etl.clean import UNKNOWN_CHANNEL, clean_formats, fill_missing
from sales_poc.etl.schema import CLEAN_SCHEMA, validate

PRODUCTS = pd.DataFrame(
    {"product_id": ["P0001", "P0002"], "list_price": [999.0, 25.5]}
)

RAW_GOOD = {
    "transaction_id": "TX-20260301-00001",
    "event_time": "2026-03-01T14:05:09Z",
    "customer_id": "C00001",
    "product_id": "P0002",
    "channel_id": "ONLINE",
    "quantity": "2",
    "unit_price": "26.00",
    "discount_pct": "5",
    "total_amount": "49.40",
    "currency": "USD",
}


def typed(*changes):
    raw = pd.DataFrame([{**RAW_GOOD, **c} for c in changes], columns=COLUMNS)
    return clean_formats(raw).rows


def fill_one(**changes):
    result = fill_missing(typed(changes), PRODUCTS)
    return result.rows.iloc[0], result.quality_flags.iloc[0]


def test_nothing_missing_means_no_change_and_no_flags():
    rows = typed({})
    result = fill_missing(rows, PRODUCTS)

    pd.testing.assert_frame_equal(result.rows, rows)
    assert result.quality_flags.iloc[0] == ""


def test_missing_discount_becomes_zero():
    row, flags = fill_one(discount_pct=None)
    assert row["discount_pct"] == 0.0
    assert flags == "discount_filled"


def test_missing_channel_becomes_unknown():
    row, flags = fill_one(channel_id=None)
    assert row["channel_id"] == UNKNOWN_CHANNEL
    assert flags == "channel_unknown"


def test_missing_price_uses_that_products_list_price():
    row, flags = fill_one(unit_price=None)
    assert row["unit_price"] == 25.5  # P0002, not P0001
    assert flags == "unit_price_filled"


def test_price_cannot_be_filled_for_an_unknown_or_missing_product():
    for product in ("P9999", None):
        row, flags = fill_one(unit_price=None, product_id=product)
        assert pd.isna(row["unit_price"])
        assert flags == ""


def test_several_fills_in_one_row_are_all_flagged():
    row, flags = fill_one(discount_pct=None, channel_id=None, unit_price=None)
    assert flags == "discount_filled; channel_unknown; unit_price_filled"


@pytest.mark.parametrize("column", ["transaction_id", "customer_id", "product_id", "event_time", "quantity"])
def test_required_values_are_never_guessed(column):
    row, flags = fill_one(**{column: None})
    assert pd.isna(row[column])
    assert flags == ""


def test_types_and_index_are_kept_and_input_is_not_changed():
    rows = typed({}, {"discount_pct": None})
    rows.index = [4, 8]
    before = rows.copy()

    result = fill_missing(rows, PRODUCTS)

    assert dict(result.rows.dtypes.astype(str)) == CLEAN_SCHEMA
    assert list(result.rows.index) == list(result.quality_flags.index) == [4, 8]
    pd.testing.assert_frame_equal(rows, before)


def test_generated_fillable_rows_are_filled_flagged_and_then_pass_validation():
    """End to end with the real generator: every 'missing_fillable' problem gets fixed."""
    ref = build_reference_data(seed=42)
    messy = add_problems(generate_history(ref, seed=42, days=90), seed=42)
    raw = pd.concat(messy.files.values(), ignore_index=True)
    log = messy.log[messy.log["problem"] == "missing_fillable"]
    assert len(log) > 30

    target = raw[raw["transaction_id"].isin(log["transaction_id"])]
    formats = clean_formats(target)
    result = fill_missing(formats.rows, ref.products)

    expected_flag = {"discount_pct": "discount_filled", "channel_id": "channel_unknown",
                     "unit_price": "unit_price_filled"}
    flags = result.quality_flags.set_axis(result.rows["transaction_id"])
    for entry in log.itertuples():
        assert flags[entry.transaction_id] == expected_flag[entry.detail]
    assert (validate(result.rows) == "").all()
