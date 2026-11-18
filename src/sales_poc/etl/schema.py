"""What a clean transaction looks like, and the check for it (requirement FR-10).

The ETL runs in this order (see `run_etl` in task T3.7):
    fix formats (T3.2) -> fill or flag missing values (T3.3) -> validate (here)
    -> remove duplicates (T3.4) -> link to reference data (T3.5) -> quarantine (T3.6)

So `validate` receives *typed* data: real timestamps and numbers, not raw text. A value
that could not be parsed in T3.2 arrives here as missing, and T3.2 records why.

Checking that a customer or product really exists needs the reference tables, so it
happens in the join step (T3.5), not here.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

# Column -> pandas type after cleaning. The capitalised types ("Int64", "Float64", "string")
# are pandas' nullable types: they can hold a missing value without turning into floats.
CLEAN_SCHEMA: dict[str, str] = {
    "transaction_id": "string",
    "event_time": "datetime64[ns, UTC]",
    "customer_id": "string",
    "product_id": "string",
    "channel_id": "string",
    "quantity": "Int64",
    "unit_price": "Float64",
    "discount_pct": "Float64",
    "total_amount": "Float64",
    "currency": "string",
}
CLEAN_COLUMNS = list(CLEAN_SCHEMA)

# Missing-value filling (T3.3) runs before validation, so by now every field must be set.
REQUIRED_COLUMNS = CLEAN_COLUMNS

ID_PATTERNS = {
    "customer_id": r"^C\d{5}$",
    "product_id": r"^P\d{4}$",
}

ALLOWED_CURRENCY = "USD"

# An order time more than this far in the future is a mistake, not a real order.
MAX_FUTURE = timedelta(days=1)


def empty_clean_frame() -> pd.DataFrame:
    """A table with no rows and exactly the clean columns and types."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in CLEAN_SCHEMA.items()})


def check_schema(rows: pd.DataFrame) -> None:
    """Raise if the table does not have the clean columns and types.

    This catches mistakes in our own code (a step returning the wrong shape), not bad data.
    """
    missing = [c for c in CLEAN_COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    wrong = {
        c: str(rows[c].dtype) for c, t in CLEAN_SCHEMA.items() if str(rows[c].dtype) != t
    }
    if wrong:
        raise TypeError(f"Wrong column types {wrong}; expected {CLEAN_SCHEMA}")


def validate(rows: pd.DataFrame, now: datetime | None = None) -> pd.Series:
    """For each row, every rule it breaks, joined by "; ". An empty string means valid.

    Reasons look like "missing:customer_id" or "invalid:quantity". `now` is used for the
    "not in the future" rule (default: the current time).
    """
    check_schema(rows)
    now = now or datetime.now(timezone.utc)
    problems: list[tuple[pd.Series, str]] = []

    for column in REQUIRED_COLUMNS:
        problems.append((rows[column].isna(), f"missing:{column}"))

    # Rules below only judge values that are present; missing ones are reported above.
    for column, pattern in ID_PATTERNS.items():
        bad = rows[column].notna() & ~rows[column].str.fullmatch(pattern).fillna(False)
        problems.append((bad, f"invalid:{column}"))

    problems += [
        (rows["quantity"] <= 0, "invalid:quantity"),
        (rows["unit_price"] <= 0, "invalid:unit_price"),
        ((rows["discount_pct"] < 0) | (rows["discount_pct"] > 100), "invalid:discount_pct"),
        (rows["total_amount"] < 0, "invalid:total_amount"),
        (rows["currency"] != ALLOWED_CURRENCY, "invalid:currency"),
        (rows["event_time"] > pd.Timestamp(now + MAX_FUTURE), "invalid:event_time_future"),
    ]

    reasons = pd.Series([[] for _ in range(len(rows))], index=rows.index, dtype=object)
    for mask, reason in problems:
        # Comparisons with a missing value give <NA>; treat that as "rule not broken".
        hit = pd.Series(mask, index=rows.index).fillna(False).astype(bool)
        for index in hit[hit].index:
            reasons[index].append(reason)
    return reasons.map("; ".join).astype("string").rename("reject_reasons")
