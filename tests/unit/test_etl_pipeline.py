from datetime import date, datetime, timezone

import pandas as pd
import pytest

from sales_poc.datagen.history import COLUMNS, generate_history
from sales_poc.datagen.problems import add_problems
from sales_poc.datagen.reference import build_reference_data
from sales_poc.etl.pipeline import OUTPUT_COLUMNS, run_etl

INGESTED = datetime(2026, 11, 18, 9, 30, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def ref():
    return build_reference_data(seed=42)


@pytest.fixture(scope="module")
def generated(ref):
    messy = add_problems(generate_history(ref, seed=42, days=120), seed=42)
    raw = pd.concat([messy.files[d] for d in sorted(messy.files)], ignore_index=True)
    return raw, messy.log


@pytest.fixture(scope="module")
def result(ref, generated):
    raw, _ = generated
    return run_etl(raw, ref, source="batch", ingested_at=INGESTED)


def test_counts_match_the_problem_log(result, generated):
    raw, log = generated
    by_kind = log["problem"].value_counts()
    c = result.counts

    assert c.rows_in == len(raw)
    assert c.duplicates_removed == by_kind["duplicate"]
    assert c.quarantined == by_kind["missing_required"] + by_kind["unknown_id"] + by_kind["negative_quantity"]
    assert c.clean_flagged == by_kind["missing_fillable"]
    assert c.rows_with_fixes == by_kind["bad_format"]
    assert c.clean + c.duplicates_removed + c.quarantined == c.rows_in
    assert c.conflicts == 0


def test_clean_rows_have_the_lake_columns_and_types(result):
    clean = result.clean
    assert list(clean.columns) == OUTPUT_COLUMNS
    assert str(clean["event_time"].dtype) == "datetime64[ns, UTC]"
    assert str(clean["ingested_at"].dtype) == "datetime64[ns, UTC]"
    assert clean["transaction_id"].is_unique
    assert list(clean.index) == list(range(len(clean)))
    assert not clean.drop(columns="quality_flags").isna().any().any()


def test_source_ingested_at_and_event_date_are_set(result):
    clean = result.clean
    assert (clean["source"] == "batch").all()
    assert (clean["ingested_at"] == pd.Timestamp(INGESTED)).all()
    assert (clean["event_date"] == clean["event_time"].dt.date).all()


def test_late_rows_get_their_real_date_not_the_file_date(result, generated):
    _, log = generated
    late = log[log["problem"] == "late"].set_index("transaction_id")
    clean = result.clean.set_index("transaction_id")

    for tx, entry in late.iterrows():
        assert clean.loc[tx, "event_date"] == entry["event_day"]
        assert clean.loc[tx, "event_date"] < entry["file_day"]


def test_quarantine_has_raw_text_reasons_and_source(result):
    q = result.quarantine
    assert len(q) == result.counts.quarantined
    assert (q["reject_reasons"] != "").all()
    assert (q["source"] == "batch").all()
    assert set(COLUMNS) <= set(q.columns)
    negative = q[q["reject_reasons"] == "invalid:quantity"]
    assert (negative["quantity"].str.startswith("-")).all()  # still the raw text


def test_same_input_gives_the_same_clean_rows(ref, generated, result):
    raw, _ = generated
    again = run_etl(raw, ref, source="batch", ingested_at=INGESTED)
    pd.testing.assert_frame_equal(again.clean, result.clean)


def test_input_index_does_not_matter(ref, generated, result):
    raw, _ = generated
    shuffled_index = raw.set_axis(range(len(raw) * 3, len(raw) * 4))
    again = run_etl(shuffled_index, ref, source="batch", ingested_at=INGESTED)
    pd.testing.assert_frame_equal(again.clean, result.clean)


def test_stream_source_is_recorded(ref, generated):
    raw, _ = generated
    out = run_etl(raw.head(50), ref, source="stream", ingested_at=INGESTED)
    assert (out.clean["source"] == "stream").all()


def test_ingested_at_without_time_zone_is_treated_as_utc(ref, generated):
    raw, _ = generated
    out = run_etl(raw.head(5), ref, source="batch", ingested_at=datetime(2026, 11, 18, 9, 30))
    assert (out.clean["ingested_at"] == pd.Timestamp(INGESTED)).all()


def test_empty_input_gives_empty_results(ref):
    out = run_etl(pd.DataFrame(columns=COLUMNS), ref, source="stream", ingested_at=INGESTED)

    assert list(out.clean.columns) == OUTPUT_COLUMNS
    assert len(out.clean) == len(out.quarantine) == 0
    assert out.counts.rows_in == 0


def test_unknown_source_is_rejected(ref):
    with pytest.raises(ValueError, match="source"):
        run_etl(pd.DataFrame(columns=COLUMNS), ref, source="ftp")
