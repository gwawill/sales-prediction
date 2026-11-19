import pandas as pd
import pytest

from sales_poc.datagen.history import generate_history
from sales_poc.datagen.problems import add_problems
from sales_poc.datagen.reference import build_reference_data
from sales_poc.etl.clean import clean_formats, fill_missing
from sales_poc.etl.dedupe import remove_duplicates
from sales_poc.etl.join import join_reference
from sales_poc.etl.quality import (
    QualityCounts,
    ReconciliationError,
    combine_reasons,
    count_notes,
    reconcile,
    split_quarantine,
)
from sales_poc.etl.schema import validate


def notes(*values):
    return pd.Series(values, dtype="string")


def test_reasons_are_merged_in_order_without_repeats():
    merged = combine_reasons(
        notes("", "unparseable:event_time", "a"),
        notes("", "invalid:quantity", "a; b"),
        notes("", "", "c"),
    )
    assert list(merged) == ["", "unparseable:event_time; invalid:quantity", "a; b; c"]


def test_missing_is_dropped_when_the_value_was_unreadable():
    merged = combine_reasons(
        notes("unparseable:quantity"),
        notes("missing:quantity; missing:customer_id"),
    )
    assert merged.iloc[0] == "unparseable:quantity; missing:customer_id"


def test_quarantine_keeps_raw_text_and_reasons():
    typed = pd.DataFrame({"transaction_id": ["A", "B", "C"], "quantity": [1, None, 3]})
    raw = pd.DataFrame({"transaction_id": ["A", "B", "C"], "quantity": ["1", "two", "3"]})
    reasons = notes("", "unparseable:quantity", "")

    good, quarantine = split_quarantine(typed, raw, reasons)

    assert list(good["transaction_id"]) == ["A", "C"]
    assert quarantine.to_dict("records") == [
        {"transaction_id": "B", "quantity": "two", "reject_reasons": "unparseable:quantity"}
    ]


def test_count_notes():
    assert count_notes(notes("a; b", "", "a")) == {"a": 2, "b": 1}


def test_reconcile_accepts_counts_that_add_up():
    reconcile(QualityCounts(rows_in=10, clean=7, clean_flagged=2, duplicates_removed=1, quarantined=2))


def test_reconcile_rejects_lost_rows():
    counts = QualityCounts(rows_in=10, clean=7, clean_flagged=0, duplicates_removed=1, quarantined=1)
    with pytest.raises(ReconciliationError, match="10 rows in.*= 9"):
        reconcile(counts)


def test_reconcile_rejects_more_flagged_than_clean():
    counts = QualityCounts(rows_in=3, clean=1, clean_flagged=2, duplicates_removed=0, quarantined=2)
    with pytest.raises(ReconciliationError, match="flagged"):
        reconcile(counts)


def test_reconcile_rejects_duplicate_ids_left_in_clean_data():
    counts = QualityCounts(rows_in=2, clean=2, clean_flagged=0, duplicates_removed=0, quarantined=0)
    with pytest.raises(ReconciliationError, match="duplicate IDs"):
        reconcile(counts, clean_ids=pd.Series(["A", "A"]))


def test_counts_match_the_problem_log_exactly():
    """All ETL steps by hand on generated data; every number must match what was injected.

    Each damaged row has exactly one problem, so the expected counts can be read straight
    from the problem log.
    """
    ref = build_reference_data(seed=42)
    messy = add_problems(generate_history(ref, seed=42, days=120), seed=42)
    raw = pd.concat([messy.files[d] for d in sorted(messy.files)], ignore_index=True)

    formats = clean_formats(raw)
    filled = fill_missing(formats.rows, ref.products)
    joined = join_reference(filled.rows, ref)
    reasons = combine_reasons(formats.unparseable, validate(filled.rows), joined.unknown)
    good, quarantine = split_quarantine(joined.rows, raw, reasons)
    deduped = remove_duplicates(good)
    flags = filled.quality_flags.loc[deduped.rows.index]

    counts = QualityCounts(
        rows_in=len(raw),
        clean=len(deduped.rows),
        clean_flagged=int((flags != "").sum()),
        duplicates_removed=int(deduped.removed.sum()),
        quarantined=len(quarantine),
        conflicts=deduped.conflicts,
        reasons=count_notes(quarantine["reject_reasons"]),
        flags=count_notes(flags),
    )
    reconcile(counts, clean_ids=deduped.rows["transaction_id"])

    log = messy.log
    by_kind = log["problem"].value_counts()
    assert counts.duplicates_removed == by_kind["duplicate"]
    assert counts.quarantined == by_kind["missing_required"] + by_kind["unknown_id"] + by_kind["negative_quantity"]
    assert counts.clean_flagged == by_kind["missing_fillable"]
    assert counts.clean == len(raw) - by_kind["duplicate"] - counts.quarantined

    required = log[log["problem"] == "missing_required"]["detail"].value_counts()
    for field, n in required.items():
        assert counts.reasons[f"missing:{field}"] == n
    unknown = log[log["problem"] == "unknown_id"]["detail"].value_counts()
    for field, n in unknown.items():
        assert counts.reasons[f"unknown:{field}"] == n
    assert counts.reasons["invalid:quantity"] == by_kind["negative_quantity"]

    fillable = log[log["problem"] == "missing_fillable"]["detail"].value_counts()
    flag_name = {"discount_pct": "discount_filled", "channel_id": "channel_unknown", "unit_price": "unit_price_filled"}
    assert counts.flags == {flag_name[f]: int(n) for f, n in sorted(fillable.items(), key=lambda x: flag_name[x[0]])}

    # Late rows and fixed formats are not problems that cost a row: they all end up clean.
    clean_ids = set(deduped.rows["transaction_id"])
    for kind in ("late", "bad_format"):
        assert set(log.loc[log["problem"] == kind, "transaction_id"]) <= clean_ids
