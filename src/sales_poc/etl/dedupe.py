"""Remove duplicate transactions (FR-12).

Rows with the same transaction_id are the same transaction, so only one is kept: the
*last* one in the table. Callers put older data first and newer data last (for example,
the lake's existing rows, then the new batch), so the newest copy wins.

If two copies share an ID but differ in content, that is a *conflict*: the newest copy is
still kept, and the number of conflicts is reported so the quality report can show it.

Rows without a transaction_id cannot be matched to anything and are never removed here;
validation rejects them instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DedupeResult:
    rows: pd.DataFrame       # the rows kept, in their original order and index
    removed: pd.Series       # bool per input row: True if it was dropped as a duplicate
    conflicts: int           # IDs whose copies had different content


def remove_duplicates(rows: pd.DataFrame, compare: list[str] | None = None) -> DedupeResult:
    """Keep the last row for each transaction_id.

    `compare` lists the columns used to decide whether copies are identical (default:
    every column). Only conflicts depend on it; which row is kept does not.
    """
    has_id = rows["transaction_id"].notna()
    removed = has_id & rows["transaction_id"].duplicated(keep="last")

    compare = compare or list(rows.columns)
    with_id = rows.loc[has_id, compare]
    distinct_versions = with_id.astype("string").drop_duplicates().groupby("transaction_id").size()
    conflicts = int((distinct_versions > 1).sum())

    return DedupeResult(rows=rows[~removed], removed=removed.rename("removed"), conflicts=conflicts)
