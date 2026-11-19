"""Cleaning: fix formats (FR-11), then fill or flag missing values (FR-33).

1. `clean_formats`: raw rows come from CSV files or Kafka messages, where every value is
   text or empty. It fixes formats that can be fixed safely and turns the text into real
   types (CLEAN_SCHEMA). For each row it records:
     - fixes:       what it changed, e.g. "fixed:event_time" (counted in the quality report)
     - unparseable: values that were present but could not be understood, e.g.
                    "unparseable:quantity". Those become empty, and the quarantine step
                    (T3.6) uses the note to explain why.

2. `fill_missing`: fills the few missing values that can be filled safely (FILL_RULES) and
   flags each row with what was filled (quality_flags). Values that cannot be guessed are
   left missing, so validation rejects the row.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from sales_poc.etl.schema import CLEAN_COLUMNS, CLEAN_SCHEMA

# Accepted time formats, tried in order. All are read as UTC.
STANDARD_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
OTHER_TIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",   # no T and Z
    "%m/%d/%Y %H:%M:%S",   # US style: month/day/year
]

MONEY_COLUMNS = ["unit_price", "total_amount", "discount_pct"]
UPPERCASE_COLUMNS = ["customer_id", "product_id", "channel_id", "currency"]

# The channel code used when the channel is missing. The join step names it "Unknown".
UNKNOWN_CHANNEL = "UNKNOWN"

# FR-33: which missing values may be filled, with what, and the flag added to the row.
# "list_price" means: the product's list price from the reference data.
# Every other missing value cannot be guessed safely, so validation rejects the row.
FILL_RULES: dict[str, tuple[object, str]] = {
    "discount_pct": (0.0, "discount_filled"),
    "channel_id": (UNKNOWN_CHANNEL, "channel_unknown"),
    "unit_price": ("list_price", "unit_price_filled"),
}


@dataclass(frozen=True)
class FormatResult:
    rows: pd.DataFrame        # typed, exactly CLEAN_SCHEMA, same index as the input
    fixes: pd.Series          # per row, "; "-joined notes like "fixed:event_time"
    unparseable: pd.Series    # per row, "; "-joined notes like "unparseable:quantity"


def _text(raw: pd.Series) -> pd.Series:
    """Trimmed text, with empty strings treated as missing."""
    text = raw.astype("string").str.strip()
    return text.mask(text == "")


def _parse_time(text: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(UTC timestamps, mask of values that were in a non-standard format)."""
    parsed = pd.to_datetime(text, format=STANDARD_TIME_FORMAT, utc=True, errors="coerce")
    standard = parsed.notna()
    for fmt in OTHER_TIME_FORMATS:
        todo = text.notna() & parsed.isna()
        if not todo.any():
            break
        parsed = parsed.where(
            ~todo, pd.to_datetime(text.where(todo), format=fmt, utc=True, errors="coerce")
        )
    fixed = parsed.notna() & ~standard
    return parsed.astype(CLEAN_SCHEMA["event_time"]), fixed


def _parse_number(text: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(numbers, mask of values that needed '$' or ',' removed)."""
    stripped = text.str.replace(r"[$,\s]", "", regex=True)
    numbers = pd.to_numeric(stripped, errors="coerce")
    fixed = numbers.notna() & (stripped != text)
    return numbers, fixed


def _join_notes(index: pd.Index, notes: list[tuple[pd.Series, str]]) -> pd.Series:
    lists: dict = {i: [] for i in index}
    for mask, note in notes:
        for i in mask[mask.fillna(False).astype(bool)].index:
            lists[i].append(note)
    return pd.Series({i: "; ".join(v) for i, v in lists.items()}, dtype="string").reindex(index)


def clean_formats(raw: pd.DataFrame) -> FormatResult:
    """Clean and type the raw rows. Columns other than CLEAN_COLUMNS are ignored."""
    missing = [c for c in CLEAN_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw rows are missing columns: {missing}")

    text = {c: _text(raw[c]) for c in CLEAN_COLUMNS}
    out = pd.DataFrame(index=raw.index)
    fixes: list[tuple[pd.Series, str]] = []

    out["transaction_id"] = text["transaction_id"]

    out["event_time"], time_fixed = _parse_time(text["event_time"])
    fixes.append((time_fixed, "fixed:event_time"))

    for column in UPPERCASE_COLUMNS:
        upper = text[column].str.upper()
        out[column] = upper
        fixes.append((text[column].notna() & (upper != text[column]), f"fixed:{column}"))

    quantity, _ = _parse_number(text["quantity"])
    whole = quantity.notna() & (quantity % 1 == 0)
    out["quantity"] = quantity.where(whole)

    for column in MONEY_COLUMNS:
        out[column], money_fixed = _parse_number(text[column])
        fixes.append((money_fixed, f"fixed:{column}"))

    out = out[CLEAN_COLUMNS].astype(CLEAN_SCHEMA)

    # Present in the raw text, but empty after parsing: we could not understand it.
    unparseable = [
        (text[c].notna() & out[c].isna(), f"unparseable:{c}") for c in CLEAN_COLUMNS
    ]
    return FormatResult(
        rows=out,
        fixes=_join_notes(raw.index, fixes).rename("fixes"),
        unparseable=_join_notes(raw.index, unparseable).rename("unparseable"),
    )


@dataclass(frozen=True)
class FillResult:
    rows: pd.DataFrame        # same shape and types as the input, with values filled
    quality_flags: pd.Series  # per row, "; "-joined flags like "discount_filled"


def fill_missing(rows: pd.DataFrame, products: pd.DataFrame) -> FillResult:
    """Apply FILL_RULES to typed rows (the output of `clean_formats`).

    `products` is the reference table with `product_id` and `list_price`. The input table
    is not changed.
    """
    filled = rows.copy()
    flags: list[tuple[pd.Series, str]] = []
    list_price = filled["product_id"].map(products.set_index("product_id")["list_price"])

    for column, (value, flag) in FILL_RULES.items():
        was_missing = filled[column].isna()
        replacement = list_price if value == "list_price" else value
        filled[column] = filled[column].fillna(replacement).astype(rows[column].dtype)
        # Only flag rows that really got a value (a list price may not exist).
        flags.append((was_missing & filled[column].notna(), flag))

    return FillResult(
        rows=filled,
        quality_flags=_join_notes(rows.index, flags).rename("quality_flags"),
    )
