"""Deliberate data problems, so the ETL has real work to do (FR-4, and FR-33 for missing values).

`add_problems` takes the clean daily history and returns a messy copy plus a *problem log*:
one row per problem added, saying which transaction, what kind, and in which file. The ETL
tests (task T3) use that log to check every problem was caught. The clean history passed in
is never changed.

Each damaged row gets exactly one problem, which keeps the log easy to check.

Problem kinds:
  duplicate          the same row again, in the same file or the next day's file
  missing_fillable   discount_pct, channel_id or unit_price is empty (ETL fills or flags it)
  missing_required   transaction_id, customer_id, product_id, event_time or quantity is empty
                     (cannot be guessed, so the ETL must quarantine the row)
  bad_format         a value in a fixable wrong format (dates, "$1,234.56", lowercase channel)
  unknown_id         customer_id or product_id that does not exist
  negative_quantity  quantity below zero
  late               the row lands in a file 1-3 days after its real date
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from sales_poc.datagen.history import COLUMNS

# Share of rows that get each kind of problem. The rest stay clean (about 95%).
PROBLEM_RATES: dict[str, float] = {
    "duplicate": 0.010,
    "missing_fillable": 0.012,
    "missing_required": 0.005,
    "bad_format": 0.015,
    "unknown_id": 0.003,
    "negative_quantity": 0.003,
    "late": 0.005,
}

FILLABLE_FIELDS = ["discount_pct", "channel_id", "unit_price"]
REQUIRED_FIELDS = ["transaction_id", "customer_id", "product_id", "event_time", "quantity"]
BAD_FORMATS = ["date_us", "date_space", "amount_dollar", "channel_lower"]
UNKNOWN_IDS = {"customer_id": "C99999", "product_id": "P9999"}

LOG_COLUMNS = ["transaction_id", "problem", "detail", "file_day", "event_day"]


@dataclass(frozen=True)
class MessyHistory:
    files: dict[date, pd.DataFrame]  # {file day: rows as text}; empty values are None/NaN
    log: pd.DataFrame                # one row per problem added (LOG_COLUMNS)


# How each number column is written as text, like a real CSV export.
_TEXT_FORMATS = {
    "unit_price": "{:.2f}",
    "total_amount": "{:.2f}",
    "discount_pct": "{:g}",  # 3.0 -> "3", 0.5 -> "0.5"
    "quantity": "{}",
}


def _as_text(rows: pd.DataFrame) -> pd.DataFrame:
    """Every value as a string, like a real CSV export, so formats can be broken freely."""
    text = rows[COLUMNS].astype(object)
    for column, fmt in _TEXT_FORMATS.items():
        text[column] = rows[column].map(fmt.format)
    return text


def order_as_text(order: dict) -> dict[str, str | None]:
    """One clean order as text values, in column order, formatted exactly like `_as_text`."""
    return {
        column: _TEXT_FORMATS.get(column, "{}").format(order[column]) for column in COLUMNS
    }


# Problems that change the row's values in place. Duplicate and late are handled by the
# caller, because what they mean depends on files (history) or timing (live orders).
IN_PLACE_PROBLEMS = {"missing_fillable", "missing_required", "bad_format", "unknown_id",
                     "negative_quantity"}


def damage_row(row: pd.Series, kind: str, rng: np.random.Generator) -> tuple[pd.Series, str]:
    """Apply one in-place problem to a copy of `row`. Returns (damaged row, detail)."""
    row = row.copy()
    if kind == "missing_fillable":
        detail = str(rng.choice(FILLABLE_FIELDS))
        row[detail] = None
    elif kind == "missing_required":
        detail = str(rng.choice(REQUIRED_FIELDS))
        row[detail] = None
    elif kind == "bad_format":
        detail = str(rng.choice(BAD_FORMATS))
        row = _break_format(row, detail)
    elif kind == "unknown_id":
        detail = str(rng.choice(list(UNKNOWN_IDS)))
        row[detail] = UNKNOWN_IDS[detail]
    elif kind == "negative_quantity":
        detail = "quantity"
        row["quantity"] = str(-int(row["quantity"]))
    else:
        raise ValueError(f"'{kind}' is not an in-place problem")
    return row, detail


def _money(value: str) -> str:
    return f"${float(value):,.2f}"


def _break_format(row: pd.Series, kind: str) -> pd.Series:
    row = row.copy()
    if kind in ("date_us", "date_space"):
        t = datetime.strptime(row["event_time"], "%Y-%m-%dT%H:%M:%SZ")
        row["event_time"] = t.strftime(
            "%m/%d/%Y %H:%M:%S" if kind == "date_us" else "%Y-%m-%d %H:%M:%S"
        )
    elif kind == "amount_dollar":
        row["unit_price"] = _money(row["unit_price"])
        row["total_amount"] = _money(row["total_amount"])
    elif kind == "channel_lower":
        row["channel_id"] = row["channel_id"].lower()
    return row


def add_problems(
    history: dict[date, pd.DataFrame],
    seed: int = 42,
    rates: dict[str, float] | None = None,
) -> MessyHistory:
    rates = rates or PROBLEM_RATES
    kinds = list(rates)
    probabilities = [*rates.values(), 1 - sum(rates.values())]
    rng = np.random.default_rng([seed, 3])

    days = sorted(history)
    output: dict[date, list[pd.Series]] = {day: [] for day in days}
    log: list[dict] = []

    def record(row_id, problem, detail, file_day, event_day):
        log.append(
            {"transaction_id": row_id, "problem": problem, "detail": detail,
             "file_day": file_day, "event_day": event_day}
        )

    for index, day in enumerate(days):
        rows = _as_text(history[day]) if len(history[day]) else pd.DataFrame(columns=COLUMNS)
        choices = rng.choice([*kinds, None], size=len(rows), p=probabilities)

        for (_, row), kind in zip(rows.iterrows(), choices):
            tx = row["transaction_id"]

            if kind is None:
                output[day].append(row)

            elif kind == "duplicate":
                output[day].append(row)
                # Re-sent in the same file, or again the next day (if there is one).
                resend_day = day
                if index + 1 < len(days) and rng.random() < 0.5:
                    resend_day = days[index + 1]
                output[resend_day].append(row.copy())
                record(tx, kind, "next_day_file" if resend_day != day else "same_file", resend_day, day)

            elif kind in IN_PLACE_PROBLEMS:
                damaged, detail = damage_row(row, kind, rng)
                output[day].append(damaged)
                record(tx, kind, detail, day, day)

            elif kind == "late":
                delay = int(rng.integers(1, 4))
                if index + delay < len(days):
                    late_day = days[index + delay]
                    output[late_day].append(row)
                    record(tx, kind, f"{delay}_days", late_day, day)
                else:
                    output[day].append(row)  # no later file exists; it stays on time

    files = {
        day: pd.DataFrame(rows, columns=COLUMNS).reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=COLUMNS)
        for day, rows in output.items()
    }
    return MessyHistory(files=files, log=pd.DataFrame(log, columns=LOG_COLUMNS))
