"""Attach customer, region, product and channel details to each transaction (FR-13).

An ID that is present but not in the reference data is noted as "unknown:<column>", so the
quarantine step (T3.6) can reject the row. An ID that is simply missing is not noted again
here, because validation already reports it as "missing:<column>".

Every row is kept, in order, with its index: this step only adds columns and notes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from sales_poc.datagen.reference import ReferenceData
from sales_poc.etl.clean import UNKNOWN_CHANNEL

# The columns this step adds, in order. All are text.
JOINED_COLUMNS = [
    "customer_name",
    "segment",
    "region_id",
    "region_name",
    "country",
    "product_name",
    "category",
    "channel_name",
]

UNKNOWN_CHANNEL_NAME = "Unknown"


@dataclass(frozen=True)
class JoinResult:
    rows: pd.DataFrame     # input columns + JOINED_COLUMNS, same index and order
    unknown: pd.Series     # per row, "; "-joined notes like "unknown:customer_id"


def join_reference(rows: pd.DataFrame, ref: ReferenceData) -> JoinResult:
    customers = ref.customers.set_index("customer_id")
    products = ref.products.set_index("product_id")
    regions = ref.regions.set_index("region_id")["region_name"]
    channel_names = ref.channels.set_index("channel_id")["channel_name"].copy()
    channel_names[UNKNOWN_CHANNEL] = UNKNOWN_CHANNEL_NAME

    out = rows.copy()
    customer_id, product_id, channel_id = rows["customer_id"], rows["product_id"], rows["channel_id"]

    for column in ("customer_name", "segment", "region_id", "country"):
        out[column] = customer_id.map(customers[column])
    out["region_name"] = out["region_id"].map(regions)
    for column in ("product_name", "category"):
        out[column] = product_id.map(products[column])
    out["channel_name"] = channel_id.map(channel_names)

    for column in JOINED_COLUMNS:
        out[column] = out[column].astype("string")
    out = out[list(rows.columns) + JOINED_COLUMNS]

    checks = [
        (customer_id.notna() & ~customer_id.isin(customers.index), "unknown:customer_id"),
        (product_id.notna() & ~product_id.isin(products.index), "unknown:product_id"),
        (channel_id.notna() & ~channel_id.isin(channel_names.index), "unknown:channel_id"),
    ]
    notes = pd.Series("", index=rows.index, dtype="string")
    for mask, note in checks:
        mask = mask.fillna(False).astype(bool)
        notes[mask] = notes[mask].where(notes[mask] == "", notes[mask] + "; ") + note

    return JoinResult(rows=out, unknown=notes.rename("unknown"))
