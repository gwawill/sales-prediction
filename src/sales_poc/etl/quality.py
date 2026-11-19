"""Quarantine and counts (FR-14, FR-20, FR-21).

  - `combine_reasons` merges the rejection notes from the earlier steps into one per row.
  - `split_quarantine` separates rows with a reason. Quarantined rows keep their *raw*
    text plus the reasons, because the raw value is what someone investigating needs.
  - `QualityCounts` holds the numbers for the quality report, and `reconcile` checks the
    FR-21 rule: rows in = clean + duplicates removed + quarantined.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

import pandas as pd

REASONS_COLUMN = "reject_reasons"


class ReconciliationError(RuntimeError):
    """The counts do not add up, so some rows were lost or counted twice."""


def _split(notes: pd.Series) -> pd.Series:
    """'a; b' -> ['a', 'b'];  '' -> []."""
    return notes.fillna("").map(lambda s: [n for n in s.split("; ") if n])


def combine_reasons(*notes: pd.Series) -> pd.Series:
    """Merge "; "-joined notes per row, in order, without repeats.

    "missing:X" is dropped when "unparseable:X" is also present: the value was there but
    could not be read, which explains more than "missing".
    """
    index = notes[0].index
    merged = []
    for parts in zip(*(_split(n.reindex(index)) for n in notes)):
        reasons = list(dict.fromkeys(r for part in parts for r in part))
        unreadable = {r.split(":", 1)[1] for r in reasons if r.startswith("unparseable:")}
        reasons = [r for r in reasons if not (r.startswith("missing:") and r.split(":", 1)[1] in unreadable)]
        merged.append("; ".join(reasons))
    return pd.Series(merged, index=index, dtype="string", name=REASONS_COLUMN)


def split_quarantine(
    rows: pd.DataFrame, raw: pd.DataFrame, reasons: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(good typed rows, quarantined raw rows with a reject_reasons column)."""
    bad = reasons.fillna("") != ""
    quarantine = raw.loc[bad].copy()
    quarantine[REASONS_COLUMN] = reasons[bad]
    return rows.loc[~bad], quarantine


def count_notes(notes: pd.Series) -> dict[str, int]:
    """How often each note appears, e.g. {"missing:quantity": 3, "invalid:quantity": 5}."""
    counter = Counter(n for parts in _split(notes) for n in parts)
    return dict(sorted(counter.items()))


@dataclass
class QualityCounts:
    rows_in: int
    clean: int
    clean_flagged: int
    duplicates_removed: int
    quarantined: int
    conflicts: int = 0
    rows_with_fixes: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    flags: dict[str, int] = field(default_factory=dict)
    fixes: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def reconcile(counts: QualityCounts, clean_ids: pd.Series | None = None) -> None:
    """Raise ReconciliationError unless every row is accounted for exactly once."""
    accounted = counts.clean + counts.duplicates_removed + counts.quarantined
    if accounted != counts.rows_in:
        raise ReconciliationError(
            f"Counts do not add up: {counts.rows_in} rows in, but clean {counts.clean} + "
            f"duplicates {counts.duplicates_removed} + quarantined {counts.quarantined} "
            f"= {accounted}."
        )
    if counts.clean_flagged > counts.clean:
        raise ReconciliationError(
            f"More flagged rows ({counts.clean_flagged}) than clean rows ({counts.clean})."
        )
    if clean_ids is not None and not clean_ids.is_unique:
        repeated = clean_ids[clean_ids.duplicated()].unique()[:5]
        raise ReconciliationError(f"Clean rows still contain duplicate IDs, e.g. {list(repeated)}.")
