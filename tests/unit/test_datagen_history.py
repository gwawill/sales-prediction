from datetime import date, timedelta

import pandas as pd
import pytest

from sales_poc.datagen.history import (
    COLUMNS,
    DEFAULT_START,
    generate_history,
    write_daily_files,
)
from sales_poc.datagen.reference import build_reference_data


@pytest.fixture(scope="module")
def ref():
    return build_reference_data(seed=42)


@pytest.fixture(scope="module")
def year(ref):
    """The full default year, generated once for the whole module."""
    history = generate_history(ref, seed=42)
    return pd.concat(history.values(), ignore_index=True), history


def test_one_entry_per_day_for_the_whole_period(year):
    _, history = year
    assert len(history) == 365
    assert min(history) == DEFAULT_START
    assert max(history) == DEFAULT_START + timedelta(days=364)


def test_one_file_per_day_named_by_date(ref, tmp_path):
    history = generate_history(ref, seed=42, start=date(2026, 3, 1), days=5)

    paths = write_daily_files(history, tmp_path)

    assert [p.name for p in paths] == [
        "sales_2026-03-01.csv",
        "sales_2026-03-02.csv",
        "sales_2026-03-03.csv",
        "sales_2026-03-04.csv",
        "sales_2026-03-05.csv",
    ]
    assert sorted(tmp_path.iterdir()) == paths


def test_same_seed_gives_byte_identical_files(ref, tmp_path):
    a = write_daily_files(generate_history(ref, seed=9, days=10), tmp_path / "a")
    b = write_daily_files(generate_history(ref, seed=9, days=10), tmp_path / "b")

    assert [p.read_bytes() for p in a] == [p.read_bytes() for p in b]


def test_different_seed_gives_different_data(ref):
    a = generate_history(ref, seed=1, days=3)
    b = generate_history(ref, seed=2, days=3)

    assert not a[DEFAULT_START].equals(b[DEFAULT_START])


def test_file_has_the_expected_columns(ref, tmp_path):
    [path] = write_daily_files(generate_history(ref, seed=42, days=1), tmp_path)

    assert list(pd.read_csv(path).columns) == COLUMNS


def test_year_size_is_roughly_as_planned(year):
    rows, _ = year
    assert 20_000 <= len(rows) <= 35_000


def test_rows_are_clean(year, ref):
    rows, _ = year
    assert rows["transaction_id"].is_unique
    assert not rows.isna().any().any()
    assert (rows["quantity"] > 0).all()
    assert rows["customer_id"].isin(ref.customers["customer_id"]).all()
    assert rows["product_id"].isin(ref.products["product_id"]).all()
    assert rows["channel_id"].isin(ref.channels["channel_id"]).all()
    expected_total = (
        rows["quantity"] * rows["unit_price"] * (1 - rows["discount_pct"] / 100)
    ).round(2)
    # At most one cent apart: Python and pandas round an exact half cent differently.
    assert (rows["total_amount"] - expected_total).abs().max() <= 0.0101


def test_event_time_matches_the_file_day(year):
    _, history = year
    for day, rows in list(history.items())[:30]:
        assert rows["event_time"].str.startswith(day.isoformat()).all()
        assert rows["event_time"].str.endswith("Z").all()


def test_consumers_never_use_direct_sales(year, ref):
    rows, _ = year
    segment = rows["customer_id"].map(ref.customers.set_index("customer_id")["segment"])
    assert not ((segment == "Consumer") & (rows["channel_id"] == "DIRECT")).any()


def test_quarter_end_rush_raises_business_orders(year, ref):
    rows, _ = year
    segment = rows["customer_id"].map(ref.customers.set_index("customer_id")["segment"])
    business = rows[segment != "Consumer"].copy()
    day = pd.to_datetime(business["event_time"].str[:10])
    per_day = business.groupby(day).size()
    weekdays = per_day[per_day.index.day_of_week < 5]

    march_rush = weekdays["2026-03-18":"2026-03-31"].mean()
    march_before = weekdays["2026-03-01":"2026-03-14"].mean()
    assert march_rush > march_before * 1.4


def test_apj_desktop_sales_drop_in_march(year, ref):
    rows, _ = year
    region = rows["customer_id"].map(ref.customers.set_index("customer_id")["region_id"])
    category = rows["product_id"].map(ref.products.set_index("product_id")["category"])
    month = rows["event_time"].str[:7]
    apj_desktop = rows[(region == "APJ") & (category == "Desktop")].groupby(month.rename("m")).size()

    assert apj_desktop["2026-03"] < 0.5 * apj_desktop[["2026-01", "2026-02"]].mean()
