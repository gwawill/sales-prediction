import pandas as pd
import pytest

from sales_poc.datagen.history import COLUMNS, generate_history
from sales_poc.datagen.problems import add_problems
from sales_poc.datagen.reference import build_reference_data
from sales_poc.etl.clean import clean_formats
from sales_poc.etl.schema import CLEAN_SCHEMA, check_schema

RAW_GOOD = {
    "transaction_id": "TX-20260301-00001",
    "event_time": "2026-03-01T14:05:09Z",
    "customer_id": "C00001",
    "product_id": "P0001",
    "channel_id": "ONLINE",
    "quantity": "2",
    "unit_price": "100.00",
    "discount_pct": "5",
    "total_amount": "190.00",
    "currency": "USD",
}


def clean_one(**changes):
    raw = pd.DataFrame([{**RAW_GOOD, **changes}], columns=COLUMNS)
    result = clean_formats(raw)
    return result.rows.iloc[0], result.fixes.iloc[0], result.unparseable.iloc[0]


def test_a_good_row_is_typed_and_needs_no_fixes():
    row, fixes, unparseable = clean_one()

    assert row["event_time"] == pd.Timestamp("2026-03-01T14:05:09Z")
    assert row["quantity"] == 2
    assert row["unit_price"] == 100.0
    assert row["discount_pct"] == 5.0
    assert fixes == ""
    assert unparseable == ""


def test_output_matches_the_clean_schema():
    raw = pd.DataFrame([RAW_GOOD] * 3, columns=COLUMNS)
    check_schema(clean_formats(raw).rows)


@pytest.mark.parametrize(
    "raw_time", ["2026-03-01 14:05:09", "03/01/2026 14:05:09", " 2026-03-01T14:05:09Z "]
)
def test_other_time_formats_are_understood_as_utc(raw_time):
    row, fixes, _ = clean_one(event_time=raw_time)

    assert row["event_time"] == pd.Timestamp("2026-03-01T14:05:09Z")
    assert ("fixed:event_time" in fixes) == (raw_time.strip() != RAW_GOOD["event_time"])


def test_money_with_dollar_and_commas():
    row, fixes, _ = clean_one(unit_price="$4,032.12", total_amount="$8,064.24")

    assert row["unit_price"] == 4032.12
    assert row["total_amount"] == 8064.24
    assert fixes == "fixed:unit_price; fixed:total_amount"


def test_lowercase_codes_are_uppercased():
    row, fixes, _ = clean_one(channel_id="online", currency="usd", customer_id="c00001")

    assert (row["channel_id"], row["currency"], row["customer_id"]) == ("ONLINE", "USD", "C00001")
    assert fixes == "fixed:customer_id; fixed:channel_id; fixed:currency"


def test_whole_number_quantity_written_as_decimal_is_fine():
    row, _, unparseable = clean_one(quantity="2.0")
    assert row["quantity"] == 2
    assert unparseable == ""


def test_negative_quantity_is_kept_for_validation_to_reject():
    row, _, unparseable = clean_one(quantity="-3")
    assert row["quantity"] == -3
    assert unparseable == ""


@pytest.mark.parametrize(
    ("change", "note"),
    [
        ({"event_time": "yesterday"}, "unparseable:event_time"),
        ({"event_time": "31/12/2026 10:00:00"}, "unparseable:event_time"),
        ({"quantity": "two"}, "unparseable:quantity"),
        ({"quantity": "2.5"}, "unparseable:quantity"),
        ({"unit_price": "12,34.5x"}, "unparseable:unit_price"),
    ],
)
def test_values_that_cannot_be_understood_become_empty_and_are_noted(change, note):
    [(column, _)] = change.items()
    row, _, unparseable = clean_one(**change)

    assert pd.isna(row[column])
    assert unparseable == note


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_empty_values_stay_missing_without_an_unparseable_note(empty):
    row, _, unparseable = clean_one(channel_id=empty, discount_pct=empty)

    assert pd.isna(row["channel_id"]) and pd.isna(row["discount_pct"])
    assert unparseable == ""


def test_input_index_is_kept():
    raw = pd.DataFrame([RAW_GOOD, RAW_GOOD], columns=COLUMNS, index=[7, 9])
    result = clean_formats(raw)

    assert list(result.rows.index) == list(result.fixes.index) == list(result.unparseable.index) == [7, 9]


def test_missing_column_is_a_programming_error():
    with pytest.raises(ValueError, match="missing columns"):
        clean_formats(pd.DataFrame([RAW_GOOD]).drop(columns="currency"))


def test_every_generated_bad_format_is_repaired_to_the_original_value():
    """End to end with the real generator: break formats, clean, compare with the clean data."""
    ref = build_reference_data(seed=42)
    clean = generate_history(ref, seed=42, days=60)
    messy = add_problems(clean, seed=42)

    original = pd.concat(clean.values()).set_index("transaction_id")
    raw = pd.concat(messy.files.values(), ignore_index=True)
    bad_ids = messy.log.loc[messy.log["problem"] == "bad_format", "transaction_id"]
    assert len(bad_ids) > 20

    result = clean_formats(raw[raw["transaction_id"].isin(bad_ids)])
    repaired = result.rows.set_index("transaction_id")

    expected_time = pd.to_datetime(original.loc[repaired.index, "event_time"], utc=True)
    assert (repaired["event_time"] == expected_time).all()
    assert (repaired["channel_id"] == original.loc[repaired.index, "channel_id"]).all()
    for column in ("unit_price", "total_amount"):
        assert (repaired[column] - original.loc[repaired.index, column]).abs().max() < 0.005
    assert (result.unparseable == "").all()
    assert (result.fixes != "").all()


def test_whole_generated_year_parses_with_types_intact():
    ref = build_reference_data(seed=42)
    messy = add_problems(generate_history(ref, seed=42, days=30), seed=42)
    raw = pd.concat(messy.files.values(), ignore_index=True)

    result = clean_formats(raw)

    assert dict(result.rows.dtypes.astype(str)) == CLEAN_SCHEMA
    # The generator never makes unreadable values: only missing or fixable ones.
    assert (result.unparseable == "").all()
