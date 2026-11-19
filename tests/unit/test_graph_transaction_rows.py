import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from sales_poc.datagen.history import generate_history
from sales_poc.datagen.problems import add_problems
from sales_poc.datagen.reference import build_reference_data
from sales_poc.etl.pipeline import run_etl
from sales_poc.graph.loader import transaction_rows
from sales_poc.lake.clean import read_all_clean, write_clean
from sales_poc.lake.store import LocalStore


@pytest.fixture(scope="module")
def clean():
    ref = build_reference_data(seed=42)
    messy = add_problems(generate_history(ref, seed=42, start=date(2026, 3, 1), days=5), seed=42)
    raw = pd.concat([messy.files[d] for d in sorted(messy.files)], ignore_index=True)
    return run_etl(raw, ref, "batch", ingested_at=datetime(2026, 11, 19, tzinfo=timezone.utc)).clean


def test_one_row_per_transaction_with_plain_values(clean):
    rows = transaction_rows(clean)

    assert len(rows) == len(clean)
    json.dumps(rows)  # plain Python values only
    first = rows[0]
    assert type(first["quantity"]) is int and type(first["total_amount"]) is float
    assert set(first) == {
        "transaction_id", "customer_id", "product_id", "channel_id", "event_time", "event_date",
        "quantity", "unit_price", "discount_pct", "total_amount", "source", "quality_flags",
    }


def test_times_and_dates_are_iso_text_cypher_can_read(clean):
    first = transaction_rows(clean)[0]
    assert first["event_time"].startswith("2026-03-01T") and first["event_time"].endswith("+00:00")
    assert first["event_date"] == "2026-03-01"


def test_flagged_rows_keep_their_flags(clean):
    rows = transaction_rows(clean)
    flagged = [r for r in rows if r["quality_flags"]]
    assert flagged and all(isinstance(r["quality_flags"], str) for r in rows)


def test_read_all_clean_returns_every_day(tmp_path, clean):
    store = LocalStore(tmp_path / "bucket")
    write_clean(store, clean)

    back = read_all_clean(store, workers=4)

    assert len(back) == len(clean)
    assert sorted(back["transaction_id"]) == sorted(clean["transaction_id"])
    assert back["event_date"].is_monotonic_increasing


def test_read_all_clean_on_an_empty_lake(tmp_path):
    assert len(read_all_clean(LocalStore(tmp_path / "empty"))) == 0
