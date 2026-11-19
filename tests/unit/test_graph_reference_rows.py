import json

from sales_poc.datagen.reference import build_reference_data
from sales_poc.graph.loader import reference_rows


def test_rows_for_each_reference_table():
    rows = reference_rows(build_reference_data(seed=42))

    assert list(rows) == ["regions", "channels", "products", "customers"]  # regions first
    assert len(rows["regions"]) == 4
    assert len(rows["products"]) == 40
    assert len(rows["customers"]) == 500


def test_the_unknown_channel_is_added_once():
    rows = reference_rows(build_reference_data(seed=42))["channels"]

    ids = [r["channel_id"] for r in rows]
    assert ids == ["ONLINE", "RETAIL", "DIRECT", "UNKNOWN"]
    assert rows[-1]["channel_name"] == "Unknown"


def test_values_are_plain_python_so_the_driver_can_send_them():
    rows = reference_rows(build_reference_data(seed=42))

    json.dumps(rows)  # numpy numbers would fail here
    assert type(rows["products"][0]["list_price"]) is float


def test_customers_carry_their_region_for_the_located_in_link():
    rows = reference_rows(build_reference_data(seed=42))["customers"]
    assert {r["region_id"] for r in rows} == {"NA", "EMEA", "APJ", "LATAM"}
