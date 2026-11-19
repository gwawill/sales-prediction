"""The one ETL entry point used by both the batch pipeline and the stream consumer (FR-9).

    result = run_etl(raw_rows, reference, source="batch")
    result.clean        # rows ready for the data lake (OUTPUT_COLUMNS)
    result.quarantine   # rejected rows, raw text + reject_reasons
    result.counts       # QualityCounts, already reconciled (FR-21)

Steps, in order:
    1. clean_formats      raw text -> typed values, note fixes and unreadable values
    2. fill_missing       fill discount / channel / price, flag the row (FR-33)
    3. validate           required fields, types and ranges (FR-10)
    4. join_reference     attach customer, region, product, channel details (FR-13)
    5. split_quarantine   any reason from 1, 3 or 4 -> quarantine with raw text (FR-14)
    6. remove_duplicates  one row per transaction_id, newest wins (FR-12)
    7. add event_date, source, ingested_at, quality_flags
    8. reconcile          rows in = clean + duplicates + quarantined, or raise (FR-21)

Because event_date comes from the order's own time, not from the file or message it
arrived in, a late row lands on its real date (FR-15).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import pandas as pd

from sales_poc.datagen.reference import ReferenceData
from sales_poc.etl.clean import clean_formats, fill_missing
from sales_poc.etl.dedupe import remove_duplicates
from sales_poc.etl.join import JOINED_COLUMNS, join_reference
from sales_poc.etl.quality import (
    QualityCounts,
    combine_reasons,
    count_notes,
    reconcile,
    split_quarantine,
)
from sales_poc.etl.schema import CLEAN_COLUMNS, validate

Source = Literal["batch", "stream"]

# Columns of a clean row, in the order they are stored in the lake.
EXTRA_COLUMNS = ["event_date", "source", "ingested_at", "quality_flags"]
OUTPUT_COLUMNS = CLEAN_COLUMNS + JOINED_COLUMNS + EXTRA_COLUMNS


@dataclass(frozen=True)
class EtlResult:
    clean: pd.DataFrame
    quarantine: pd.DataFrame
    counts: QualityCounts


def run_etl(
    raw: pd.DataFrame,
    ref: ReferenceData,
    source: Source,
    ingested_at: datetime | None = None,
    now: datetime | None = None,
) -> EtlResult:
    """Clean one group of raw rows (a batch file, or a micro-batch of Kafka messages).

    `ingested_at` is when our pipeline received the rows (default: now). `now` is used by
    the "not in the future" rule (default: the current time).
    """
    if source not in ("batch", "stream"):
        raise ValueError(f"source must be 'batch' or 'stream', not {source!r}")
    ingested_at = ingested_at or datetime.now(timezone.utc)
    raw = raw.reset_index(drop=True)  # a clean 0..n-1 index keeps every step lined up

    formats = clean_formats(raw)
    filled = fill_missing(formats.rows, ref.products)
    invalid = validate(filled.rows, now=now)
    joined = join_reference(filled.rows, ref)

    reasons = combine_reasons(formats.unparseable, invalid, joined.unknown)
    good, quarantine = split_quarantine(joined.rows, raw, reasons)
    # Only the transaction's own fields decide whether two copies conflict.
    deduped = remove_duplicates(good, compare=CLEAN_COLUMNS)

    stamp = pd.Timestamp(ingested_at)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")

    clean = deduped.rows.copy()
    clean["event_date"] = clean["event_time"].dt.date
    clean["source"] = pd.Series(source, index=clean.index, dtype="string")
    clean["ingested_at"] = pd.Series(stamp, index=clean.index, dtype="datetime64[ns, UTC]")
    clean["quality_flags"] = filled.quality_flags.loc[clean.index]
    clean = clean[OUTPUT_COLUMNS].reset_index(drop=True)

    quarantine = quarantine.reset_index(drop=True)
    quarantine["source"] = source
    quarantine["ingested_at"] = stamp

    flags = clean["quality_flags"]
    counts = QualityCounts(
        rows_in=len(raw),
        clean=len(clean),
        clean_flagged=int((flags != "").sum()),
        duplicates_removed=int(deduped.removed.sum()),
        quarantined=len(quarantine),
        conflicts=deduped.conflicts,
        rows_with_fixes=int((formats.fixes != "").sum()),
        reasons=count_notes(quarantine["reject_reasons"]),
        flags=count_notes(flags),
        fixes=count_notes(formats.fixes),
    )
    reconcile(counts, clean_ids=clean["transaction_id"])
    return EtlResult(clean=clean, quarantine=quarantine, counts=counts)
