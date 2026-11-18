import pandas as pd
import pytest

from sales_poc.datagen.reference import (
    COUNTRIES_BY_REGION,
    PRODUCT_CATALOG,
    SEGMENTS,
    build_reference_data,
)


@pytest.fixture(scope="module")
def ref():
    return build_reference_data(seed=42)


def test_same_seed_gives_identical_tables():
    a = build_reference_data(seed=7)
    b = build_reference_data(seed=7)

    for table in ("regions", "channels", "products", "customers"):
        pd.testing.assert_frame_equal(getattr(a, table), getattr(b, table))


def test_different_seed_gives_different_customers():
    a = build_reference_data(seed=1)
    b = build_reference_data(seed=2)

    assert not a.customers.equals(b.customers)


def test_table_sizes(ref):
    assert len(ref.regions) == 4
    assert len(ref.channels) == 3
    assert len(ref.products) == 40
    assert len(ref.customers) == 500


def test_customer_count_can_be_changed():
    assert len(build_reference_data(seed=42, n_customers=25).customers) == 25


def test_ids_are_unique_and_well_formed(ref):
    for table, column, pattern in [
        (ref.products, "product_id", r"^P\d{4}$"),
        (ref.customers, "customer_id", r"^C\d{5}$"),
    ]:
        assert table[column].is_unique
        assert table[column].str.match(pattern).all()


def test_every_customer_points_at_a_real_region_and_a_country_in_it(ref):
    assert set(ref.customers["region_id"]) <= set(ref.regions["region_id"])
    for region_id, country in ref.customers[["region_id", "country"]].itertuples(index=False):
        assert country in COUNTRIES_BY_REGION[region_id]


def test_all_segments_and_regions_are_used(ref):
    assert set(ref.customers["segment"]) == set(SEGMENTS)
    assert set(ref.customers["region_id"]) == set(ref.regions["region_id"])


def test_products_cover_every_category_within_its_price_range(ref):
    assert set(ref.products["category"]) == set(PRODUCT_CATALOG)
    for category, (count, low, high, _) in PRODUCT_CATALOG.items():
        prices = ref.products.loc[ref.products["category"] == category, "list_price"]
        assert len(prices) == count
        assert prices.between(low, high).all()


def test_no_missing_values(ref):
    for table in (ref.regions, ref.channels, ref.products, ref.customers):
        assert not table.isna().any().any()
