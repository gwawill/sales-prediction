"""Reference data: regions, channels, products and customers (requirement FR-1).

Everything is made up. The same seed always gives exactly the same tables (FR-5), so a
transaction generated later can safely point at a customer or product by its ID.

    from sales_poc.datagen.reference import build_reference_data
    ref = build_reference_data(seed=42)
    ref.customers.head()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from faker import Faker

# --- Regions and the countries in each -------------------------------------------------

REGIONS = pd.DataFrame(
    [
        ("NA", "North America"),
        ("EMEA", "Europe, Middle East and Africa"),
        ("APJ", "Asia Pacific and Japan"),
        ("LATAM", "Latin America"),
    ],
    columns=["region_id", "region_name"],
)

COUNTRIES_BY_REGION: dict[str, list[str]] = {
    "NA": ["United States", "Canada"],
    "EMEA": ["United Kingdom", "Germany", "France", "Netherlands", "United Arab Emirates", "South Africa"],
    "APJ": ["Japan", "Australia", "India", "Singapore", "South Korea"],
    "LATAM": ["Brazil", "Mexico", "Chile", "Colombia"],
}

# Share of customers in each region (adds up to 1).
REGION_WEIGHTS = {"NA": 0.40, "EMEA": 0.30, "APJ": 0.20, "LATAM": 0.10}

# --- Sales channels ---------------------------------------------------------------------

CHANNELS = pd.DataFrame(
    [
        ("ONLINE", "Online"),
        ("RETAIL", "Retail Partner"),
        ("DIRECT", "Direct Sales"),
    ],
    columns=["channel_id", "channel_name"],
)

# --- Customer segments ------------------------------------------------------------------

SEGMENTS = ["Consumer", "SMB", "Enterprise"]
SEGMENT_WEIGHTS = [0.50, 0.35, 0.15]

# --- Products ---------------------------------------------------------------------------

# category: (how many products, lowest list price, highest list price, name pattern)
PRODUCT_CATALOG: dict[str, tuple[int, float, float, str]] = {
    "Laptop": (10, 600, 2500, "{series} Laptop {num}"),
    "Desktop": (7, 500, 2000, "{series} Desktop {num}"),
    "Monitor": (7, 150, 900, "{series} Monitor {size}in"),
    "Server": (5, 3000, 20000, "{series} Rack Server R{num}"),
    "Storage": (4, 1500, 15000, "{series} Storage Array S{num}"),
    "Accessory": (7, 20, 200, "{accessory}"),
}
PRODUCT_SERIES = ["Aero", "Vector", "Summit", "Nova", "Atlas", "Orion", "Crest"]
MONITOR_SIZES = [22, 24, 27, 32, 34, 38, 43]
ACCESSORIES = [
    "Wireless Mouse",
    "Keyboard",
    "USB-C Dock",
    "Laptop Backpack",
    "Webcam",
    "Headset",
    "Power Adapter",
]


@dataclass(frozen=True)
class ReferenceData:
    regions: pd.DataFrame
    channels: pd.DataFrame
    products: pd.DataFrame
    customers: pd.DataFrame


def build_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for category, (count, low, high, pattern) in PRODUCT_CATALOG.items():
        prices = np.sort(rng.uniform(low, high, size=count))
        for i in range(count):
            name = pattern.format(
                series=PRODUCT_SERIES[i % len(PRODUCT_SERIES)],
                num=(i + 1) * 100 + int(rng.integers(0, 99)),
                size=MONITOR_SIZES[i % len(MONITOR_SIZES)],
                accessory=ACCESSORIES[i % len(ACCESSORIES)],
            )
            rows.append((category, name, round(float(prices[i]), 2)))

    products = pd.DataFrame(rows, columns=["category", "product_name", "list_price"])
    products.insert(0, "product_id", [f"P{i:04d}" for i in range(1, len(products) + 1)])
    return products


def build_customers(rng: np.random.Generator, fake: Faker, n_customers: int) -> pd.DataFrame:
    segments = rng.choice(SEGMENTS, size=n_customers, p=SEGMENT_WEIGHTS)
    region_ids = rng.choice(
        list(REGION_WEIGHTS), size=n_customers, p=list(REGION_WEIGHTS.values())
    )

    rows = []
    for i in range(n_customers):
        segment, region_id = str(segments[i]), str(region_ids[i])
        # Businesses get company names, consumers get person names.
        name = fake.name() if segment == "Consumer" else fake.company()
        country = str(rng.choice(COUNTRIES_BY_REGION[region_id]))
        rows.append((f"C{i + 1:05d}", name, segment, region_id, country))

    return pd.DataFrame(
        rows, columns=["customer_id", "customer_name", "segment", "region_id", "country"]
    )


def build_reference_data(seed: int = 42, n_customers: int = 500) -> ReferenceData:
    """All four reference tables, fully determined by `seed`."""
    rng = np.random.default_rng(seed)
    fake = Faker("en_US")
    fake.seed_instance(seed)

    return ReferenceData(
        regions=REGIONS.copy(),
        channels=CHANNELS.copy(),
        products=build_products(rng),
        customers=build_customers(rng, fake, n_customers),
    )
