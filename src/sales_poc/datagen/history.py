"""Clean daily sales history: 12 months of transactions, one CSV file per day (FR-2).

This is the *clean* version. Task T2.3 adds the deliberate problems (duplicates, missing
values, bad formats, ...) on top of it.

Patterns built in, so forecasting and the AI layer have something real to find:
  - slow growth over the year,
  - business customers buy more in the last two weeks of each quarter, less at weekends,
  - consumers buy more in November and December,
  - a deliberate story: APJ desktop sales drop sharply in March.

The dates are fixed by the arguments, not by today's date, so the same seed always gives
the same files (FR-5).
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sales_poc.datagen.reference import ReferenceData

DEFAULT_START = date(2025, 11, 1)
DEFAULT_DAYS = 365

# Columns of a daily export file, in order.
COLUMNS = [
    "transaction_id",
    "event_time",
    "customer_id",
    "product_id",
    "channel_id",
    "quantity",
    "unit_price",
    "discount_pct",
    "total_amount",
    "currency",
]

# Average orders per day at the start of the period, before seasonal effects.
BASE_DAILY_ORDERS = {"Consumer": 40.0, "Business": 30.0}
YEARLY_GROWTH = 0.10

# How likely each segment is to buy each category (each row adds up to 1).
CATEGORY_MIX = {
    "Consumer": {"Laptop": 0.40, "Desktop": 0.10, "Monitor": 0.20, "Accessory": 0.30},
    "SMB": {"Laptop": 0.35, "Desktop": 0.20, "Monitor": 0.20, "Server": 0.05,
            "Storage": 0.05, "Accessory": 0.15},
    "Enterprise": {"Laptop": 0.30, "Desktop": 0.15, "Monitor": 0.15, "Server": 0.15,
                   "Storage": 0.15, "Accessory": 0.10},
}

# Which channel each segment uses (each row adds up to 1).
CHANNEL_MIX = {
    "Consumer": {"ONLINE": 0.70, "RETAIL": 0.30},
    "SMB": {"ONLINE": 0.40, "RETAIL": 0.30, "DIRECT": 0.30},
    "Enterprise": {"ONLINE": 0.20, "DIRECT": 0.80},
}

# Largest quantity per order line, by segment.
MAX_QUANTITY = {"Consumer": 2, "SMB": 10, "Enterprise": 50}

# Discount range in percent, by segment.
DISCOUNT_RANGE = {"Consumer": (0.0, 5.0), "SMB": (0.0, 10.0), "Enterprise": (5.0, 25.0)}

# The March story: share of APJ desktop orders that disappear in March.
APJ_DESKTOP_MARCH_DROP = 0.70


def _is_quarter_end_rush(day: date) -> bool:
    """The last 14 days of March, June, September and December."""
    if day.month not in (3, 6, 9, 12):
        return False
    next_month_first = date(day.year + (day.month == 12), day.month % 12 + 1, 1)
    return (next_month_first - day).days <= 14


def expected_orders(day: date, start: date) -> dict[str, float]:
    """Average number of consumer and business orders expected on `day`."""
    growth = 1 + YEARLY_GROWTH * (day - start).days / 365
    weekend = day.weekday() >= 5

    consumer = BASE_DAILY_ORDERS["Consumer"] * growth
    if day.month in (11, 12):
        consumer *= 1.6  # holiday season
    if weekend:
        consumer *= 1.2  # people shop at weekends

    business = BASE_DAILY_ORDERS["Business"] * growth
    if _is_quarter_end_rush(day):
        business *= 1.8  # sales teams push to hit quarterly targets
    if weekend:
        business *= 0.3  # offices are closed

    return {"Consumer": consumer, "Business": business}


def _pick(rng: np.random.Generator, mix: dict[str, float], size: int) -> np.ndarray:
    return rng.choice(list(mix), size=size, p=list(mix.values()))


def pick_category(rng: np.random.Generator, segment: str) -> str:
    return str(_pick(rng, CATEGORY_MIX[segment], 1)[0])


def make_order(
    rng: np.random.Generator,
    segment: str,
    customer_id: str,
    category: str,
    products_by_category: dict[str, pd.DataFrame],
) -> dict:
    """One clean order line, without its ID and time (the caller adds those).

    Shared by the daily history and the live order maker, so both follow the same rules.
    """
    options = products_by_category[category]
    product = options.iloc[int(rng.integers(0, len(options)))]
    quantity = int(rng.integers(1, MAX_QUANTITY[segment] + 1))
    if category == "Accessory":
        quantity *= 2  # accessories are bought in bigger numbers
    unit_price = round(float(product["list_price"]) * float(rng.uniform(0.97, 1.03)), 2)
    low, high = DISCOUNT_RANGE[segment]
    discount = round(float(rng.uniform(low, high)) * 2) / 2  # steps of 0.5%
    total = round(quantity * unit_price * (1 - discount / 100), 2)
    return {
        "customer_id": customer_id,
        "product_id": product["product_id"],
        "channel_id": str(_pick(rng, CHANNEL_MIX[segment], 1)[0]),
        "quantity": quantity,
        "unit_price": unit_price,
        "discount_pct": discount,
        "total_amount": total,
        "currency": "USD",
    }


def format_event_time(t: datetime) -> str:
    """The standard time format in clean data: UTC, like 2026-03-01T14:05:09Z."""
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _orders_for_day(
    rng: np.random.Generator, day: date, start: date, ref: ReferenceData
) -> pd.DataFrame:
    customers = ref.customers
    consumers = customers[customers["segment"] == "Consumer"]
    businesses = customers[customers["segment"] != "Consumer"]
    # Enterprises place orders more often than small businesses.
    business_weights = np.where(businesses["segment"] == "Enterprise", 3.0, 1.0)
    business_weights /= business_weights.sum()

    expected = expected_orders(day, start)
    n_consumer = int(rng.poisson(expected["Consumer"]))
    n_business = int(rng.poisson(expected["Business"]))

    picked = pd.concat(
        [
            consumers.iloc[rng.integers(0, len(consumers), size=n_consumer)],
            businesses.iloc[rng.choice(len(businesses), size=n_business, p=business_weights)],
        ],
        ignore_index=True,
    )
    if picked.empty:
        return pd.DataFrame(columns=COLUMNS)

    products_by_category = {c: g for c, g in ref.products.groupby("category")}
    rows = []
    for customer in picked.itertuples(index=False):
        category = pick_category(rng, customer.segment)

        if (
            day.month == 3
            and customer.region_id == "APJ"
            and category == "Desktop"
            and rng.random() < APJ_DESKTOP_MARCH_DROP
        ):
            continue  # the March story

        order = make_order(rng, customer.segment, customer.customer_id, category, products_by_category)
        seconds = int(rng.integers(0, 24 * 60 * 60))
        order["event_time"] = datetime.combine(day, time(), tzinfo=timezone.utc) + timedelta(
            seconds=seconds
        )
        rows.append(order)

    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    day_orders = pd.DataFrame(rows).sort_values("event_time", ignore_index=True)
    day_orders.insert(
        0,
        "transaction_id",
        [f"TX-{day:%Y%m%d}-{i:05d}" for i in range(1, len(day_orders) + 1)],
    )
    day_orders["event_time"] = day_orders["event_time"].map(format_event_time)
    return day_orders[COLUMNS]


def generate_history(
    ref: ReferenceData,
    seed: int = 42,
    start: date = DEFAULT_START,
    days: int = DEFAULT_DAYS,
) -> dict[date, pd.DataFrame]:
    """Clean transactions for each day, as {day: table}. Every day is present, even if empty."""
    # A separate random stream from the reference data, still fully set by `seed`.
    rng = np.random.default_rng([seed, 2])
    history = {}
    for offset in range(days):
        day = start + timedelta(days=offset)
        history[day] = _orders_for_day(rng, day, start, ref)
    return history


def daily_file_name(day: date) -> str:
    return f"sales_{day:%Y-%m-%d}.csv"


def write_daily_files(history: dict[date, pd.DataFrame], directory: Path) -> list[Path]:
    """Write one CSV per day into `directory` and return the paths, oldest first."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for day in sorted(history):
        path = directory / daily_file_name(day)
        history[day].to_csv(path, index=False, lineterminator="\n")
        paths.append(path)
    return paths
