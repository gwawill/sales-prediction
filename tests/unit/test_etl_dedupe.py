import pandas as pd

from sales_poc.datagen.history import generate_history
from sales_poc.datagen.problems import add_problems
from sales_poc.datagen.reference import build_reference_data
from sales_poc.etl.clean import clean_formats, fill_missing
from sales_poc.etl.dedupe import remove_duplicates


def table(*rows):
    return pd.DataFrame(rows, columns=["transaction_id", "quantity"]).astype(
        {"transaction_id": "string", "quantity": "Int64"}
    )


def test_no_duplicates_means_nothing_removed():
    result = remove_duplicates(table(("A", 1), ("B", 2)))

    assert list(result.rows["transaction_id"]) == ["A", "B"]
    assert not result.removed.any()
    assert result.conflicts == 0


def test_exact_copies_keep_one():
    result = remove_duplicates(table(("A", 1), ("B", 2), ("A", 1), ("A", 1)))

    assert list(result.rows["transaction_id"]) == ["B", "A"]
    assert list(result.removed) == [True, False, True, False]
    assert result.conflicts == 0


def test_the_last_copy_wins_and_a_conflict_is_counted():
    result = remove_duplicates(table(("A", 1), ("A", 5)))

    [kept] = result.rows.itertuples()
    assert kept.quantity == 5
    assert result.conflicts == 1


def test_rows_without_an_id_are_never_removed():
    result = remove_duplicates(table((None, 1), (None, 1), ("A", 1)))

    assert len(result.rows) == 3
    assert not result.removed.any()


def test_index_and_order_are_kept():
    rows = table(("A", 1), ("B", 2), ("A", 1))
    rows.index = [10, 20, 30]

    result = remove_duplicates(rows)

    assert list(result.rows.index) == [20, 30]
    assert list(result.removed.index) == [10, 20, 30]


def test_conflicts_can_ignore_columns_that_are_allowed_to_differ():
    rows = table(("A", 1), ("A", 1))
    rows["ingested_at"] = ["09:00", "09:05"]  # a resend arrives later; that's not a conflict

    assert remove_duplicates(rows).conflicts == 1
    assert remove_duplicates(rows, compare=["transaction_id", "quantity"]).conflicts == 0


def test_generated_duplicates_are_all_removed():
    """End to end with the real generator: each logged duplicate disappears, nothing else."""
    ref = build_reference_data(seed=42)
    messy = add_problems(generate_history(ref, seed=42, days=90), seed=42)
    raw = pd.concat([messy.files[d] for d in sorted(messy.files)], ignore_index=True)
    rows = fill_missing(clean_formats(raw).rows, ref.products).rows

    result = remove_duplicates(rows)

    duplicates = (messy.log["problem"] == "duplicate").sum()
    assert duplicates > 50
    assert result.removed.sum() == duplicates
    kept_ids = result.rows["transaction_id"].dropna()
    assert kept_ids.is_unique
    assert result.conflicts == 0  # the generator only makes exact copies
