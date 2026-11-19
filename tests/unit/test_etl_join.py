import pandas as pd
import pytest

from sales_poc.datagen.history import COLUMNS, generate_history
from sales_poc.datagen.problems import add_problems
from sales_poc.datagen.reference import build_reference_data
from sales_poc.etl.clean import clean_formats, fill_missing
from sales_poc.etl.join import JOINED_COLUMNS, join_reference
from sales_poc.etl.schema import CLEAN_COLUMNS


@pytest.fixture(scope="module")
def ref():
    return build_reference_data(seed=42)


def typed(ref, *changes):
    customer = ref.customers.iloc[0]
    product = ref.products.iloc[0]
    good = {
        "transaction_id": "TX-1", "event_time": "2026-03-01T14:05:09Z",
        "customer_id": customer["customer_id"], "product_id": product["product_id"],
        "channel_id": "ONLINE", "quantity": "1", "unit_price": "10.00",
        "discount_pct": "0", "total_amount": "10.00", "currency": "USD",
    }
    raw = pd.DataFrame([{**good, **c} for c in changes], columns=COLUMNS)
    return clean_formats(raw).rows


def test_details_are_attached(ref):
    result = join_reference(typed(ref, {}), ref)
    row = result.rows.iloc[0]
    customer = ref.customers.iloc[0]
    product = ref.products.iloc[0]

    assert row["customer_name"] == customer["customer_name"]
    assert row["segment"] == customer["segment"]
    assert row["region_id"] == customer["region_id"]
    assert row["country"] == customer["country"]
    assert row["region_name"] == ref.regions.set_index("region_id").loc[customer["region_id"], "region_name"]
    assert row["product_name"] == product["product_name"]
    assert row["category"] == product["category"]
    assert row["channel_name"] == "Online"
    assert result.unknown.iloc[0] == ""


def test_columns_are_the_clean_ones_plus_the_joined_ones(ref):
    result = join_reference(typed(ref, {}), ref)

    assert list(result.rows.columns) == CLEAN_COLUMNS + JOINED_COLUMNS
    for column in JOINED_COLUMNS:
        assert str(result.rows[column].dtype) == "string"


def test_the_unknown_channel_code_gets_a_name_and_is_not_an_error(ref):
    rows = fill_missing(typed(ref, {"channel_id": None}), ref.products).rows
    result = join_reference(rows, ref)

    assert result.rows["channel_name"].iloc[0] == "Unknown"
    assert result.unknown.iloc[0] == ""


@pytest.mark.parametrize(
    ("change", "note", "empty_columns"),
    [
        ({"customer_id": "C99999"}, "unknown:customer_id",
         ["customer_name", "segment", "region_id", "region_name", "country"]),
        ({"product_id": "P9999"}, "unknown:product_id", ["product_name", "category"]),
        ({"channel_id": "PHONE"}, "unknown:channel_id", ["channel_name"]),
    ],
)
def test_ids_not_in_the_reference_data_are_noted(ref, change, note, empty_columns):
    result = join_reference(typed(ref, change), ref)

    assert result.unknown.iloc[0] == note
    for column in empty_columns:
        assert pd.isna(result.rows[column].iloc[0])


def test_several_unknowns_are_all_noted(ref):
    result = join_reference(typed(ref, {"customer_id": "C99999", "product_id": "P9999"}), ref)
    assert result.unknown.iloc[0] == "unknown:customer_id; unknown:product_id"


def test_missing_ids_are_left_to_validation(ref):
    result = join_reference(typed(ref, {"customer_id": None, "product_id": None}), ref)
    assert result.unknown.iloc[0] == ""


def test_rows_index_and_order_are_kept(ref):
    rows = typed(ref, {}, {"customer_id": "C99999"}, {})
    rows.index = [5, 3, 9]

    result = join_reference(rows, ref)

    assert list(result.rows.index) == list(result.unknown.index) == [5, 3, 9]
    assert list(result.unknown) == ["", "unknown:customer_id", ""]


def test_generated_unknown_ids_are_exactly_the_ones_noted(ref):
    """End to end with the real generator: every C99999/P9999 row is caught, nothing else."""
    messy = add_problems(generate_history(ref, seed=42, days=120), seed=42)
    raw = pd.concat(messy.files.values(), ignore_index=True)
    rows = fill_missing(clean_formats(raw).rows, ref.products).rows

    result = join_reference(rows, ref)

    log = messy.log[messy.log["problem"] == "unknown_id"]
    assert len(log) > 10
    noted = result.rows.loc[result.unknown != "", "transaction_id"]
    assert sorted(noted) == sorted(log["transaction_id"])
    # Rows with all three IDs present and known have every detail. (Rows with a missing ID
    # have none, and that's fine: validation reports them as missing:<column>.)
    ids_present = rows[["customer_id", "product_id", "channel_id"]].notna().all(axis=1)
    known = result.rows[(result.unknown == "") & ids_present]
    assert known[JOINED_COLUMNS].notna().all().all()
