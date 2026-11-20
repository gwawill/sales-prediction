from datetime import date, datetime, timezone

import pandas as pd
import pytest

from sales_poc.datagen.history import generate_history
from sales_poc.datagen.problems import add_problems
from sales_poc.datagen.reference import build_reference_data
from sales_poc.etl.pipeline import run_etl
from sales_poc.graph.loader import TRANSACTIONS_CYPHER
from sales_poc.lake.clean import clean_key, write_clean
from sales_poc.lake.store import LocalStore
from sales_poc.pipelines.sync import (
    STATE_KEY,
    StoreSync,
    changed_summaries,
    day_stamps,
    describe,
    read_graph_state,
)
from sales_poc.vectors.store import VectorIndex
from sales_poc.vectors.summaries import Summary
from tests.unit.test_vector_store import FakeS3Vectors, embedder

NOW = datetime(2026, 11, 20, 9, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def ref():
    return build_reference_data(seed=42)


@pytest.fixture(scope="module")
def clean(ref):
    """Clean rows for 1 to 6 March 2026."""
    messy = add_problems(generate_history(ref, seed=42, start=date(2026, 3, 1), days=6), seed=42)
    raw = pd.concat([messy.files[d] for d in sorted(messy.files)], ignore_index=True)
    return run_etl(raw, ref, "batch", ingested_at=NOW).clean


def days(rows, *wanted):
    return rows[rows["event_date"].isin(wanted)]


FIRST_FIVE = [date(2026, 3, d) for d in range(1, 6)]
SIXTH = date(2026, 3, 6)


class FakeDriver:
    """Records what would be sent to Neo4j. `fail=True` makes the next transaction load fail."""

    def __init__(self):
        self.transaction_ids: list[str] = []
        self.queries = 0
        self.fail = False

    def execute_query(self, query, **params):
        self.queries += 1
        if query == TRANSACTIONS_CYPHER:
            if self.fail:
                raise ConnectionError("Neo4j is unavailable")
            self.transaction_ids += [r["transaction_id"] for r in params["rows"]]
        return [], None, None


class CountingEmbedder:
    """The fake word embedder, counting how many texts it was asked to embed."""

    def __init__(self):
        self.inner, self.texts = embedder(), 0

    def embed_documents(self, texts):
        self.texts += len(texts)
        return self.inner.embed_documents(texts)


@pytest.fixture
def store(tmp_path, clean):
    store = LocalStore(tmp_path / "bucket")
    write_clean(store, days(clean, *FIRST_FIVE))
    return store


@pytest.fixture
def index():
    return VectorIndex(FakeS3Vectors(), "bucket", "sales-summaries", dim=5)


def make_sync(store, index, ref, driver=None, embed=None):
    return StoreSync(store, driver or FakeDriver(), index, embed or CountingEmbedder(), ref, workers=2)


def test_first_pass_loads_every_day_and_every_summary(store, index, ref, clean):
    sync = make_sync(store, index, ref)

    result = sync.run_once()

    assert result.days_read == 5 and result.graph_days == 5
    assert sorted(sync.driver.transaction_ids) == sorted(days(clean, *FIRST_FIVE)["transaction_id"])
    assert result.summaries_stored == result.summaries_total > 0 and result.summaries_deleted == 0
    assert set(read_graph_state(store)) == set(FIRST_FIVE)


def test_a_pass_with_nothing_new_does_no_work(store, index, ref):
    sync = make_sync(store, index, ref)
    sync.run_once()
    queries, embedded = sync.driver.queries, sync.embedder.texts

    result = sync.run_once()

    assert not result.changed and result.days_read == 0
    assert (sync.driver.queries, sync.embedder.texts) == (queries, embedded)
    assert describe(result).startswith("Nothing new")


def test_a_new_day_is_loaded_into_the_graph_alone(store, index, ref, clean):
    sync = make_sync(store, index, ref)
    sync.run_once()
    sync.driver.transaction_ids.clear()
    embedded_before = sync.embedder.texts

    write_clean(store, days(clean, SIXTH))
    result = sync.run_once()

    assert result.days_read == 1 and result.graph_days == 1
    assert sorted(sync.driver.transaction_ids) == sorted(days(clean, SIXTH)["transaction_id"])
    # A new day moves the end of the period that every customer summary names, so they are
    # all stored again (at most once a day in practice).
    assert result.summaries_stored > 0
    assert sync.embedder.texts - embedded_before == result.summaries_stored
    assert set(read_graph_state(store)) == set(FIRST_FIVE) | {SIXTH}


def test_a_new_order_on_a_known_day_re_embeds_only_the_summaries_it_touches(tmp_path, index, ref, clean):
    store = LocalStore(tmp_path / "bucket")
    rows = days(clean, *FIRST_FIVE)
    late_order = rows[rows["event_date"] == date(2026, 3, 3)].tail(1)
    write_clean(store, rows.drop(index=late_order.index))
    sync = make_sync(store, index, ref)
    first = sync.run_once()
    embedded_before = sync.embedder.texts

    write_clean(store, late_order)       # e.g. a live order that arrived late for 3 March
    result = sync.run_once()

    assert result.graph_days == 1 and result.graph_transactions == len(days(rows, date(2026, 3, 3)))
    # Its category in its region, its region, and its customer: a handful, not all of them.
    assert 0 < result.summaries_stored <= 5 < first.summaries_total
    assert sync.embedder.texts - embedded_before == result.summaries_stored


def test_a_changed_day_is_loaded_again(store, index, ref, clean):
    sync = make_sync(store, index, ref)
    sync.run_once()
    sync.driver.transaction_ids.clear()

    # A corrected copy of one order on 2 March: same ID, different quantity.
    fixed = days(clean, date(2026, 3, 2)).head(1).copy()
    fixed["quantity"] = fixed["quantity"] + 1
    write_clean(store, fixed)
    result = sync.run_once()

    assert result.graph_days == 1
    assert fixed["transaction_id"].iloc[0] in sync.driver.transaction_ids


def test_a_restart_finds_the_graph_and_the_index_already_up_to_date(store, index, ref):
    make_sync(store, index, ref).run_once()

    again = make_sync(store, index, ref)
    result = again.run_once()

    assert result.days_read == 5          # the new process reads the lake into memory once
    assert not result.changed             # but sends nothing to Neo4j and embeds nothing
    assert again.driver.transaction_ids == [] and again.embedder.texts == 0


def test_a_failed_graph_load_is_tried_again_next_pass(store, index, ref, clean):
    driver = FakeDriver()
    driver.fail = True
    sync = make_sync(store, index, ref, driver=driver)

    with pytest.raises(ConnectionError):
        sync.run_once()
    assert not store.exists(STATE_KEY)   # nothing marked as done

    driver.fail = False
    result = sync.run_once()
    assert result.graph_days == 5


def test_summaries_that_no_longer_exist_are_removed(store, index, ref):
    index.ensure()
    index.put([Summary("catreg:Laptop:EMEA:2020-01", "old", {"kind": "catreg"})], [[1.0] * 5])

    result = make_sync(store, index, ref).run_once()

    assert result.summaries_deleted == 1
    assert "catreg:Laptop:EMEA:2020-01" not in index.stored()


def test_an_empty_lake_leaves_the_index_alone(tmp_path, index, ref):
    index.ensure()
    index.put([Summary("region:NA:2026-03", "kept", {"kind": "region"})], [[1.0] * 5])

    result = make_sync(LocalStore(tmp_path / "empty"), index, ref).run_once()

    assert not result.changed
    assert set(index.stored()) == {"region:NA:2026-03"}


def test_day_stamps_change_only_when_a_day_file_changes(store, clean):
    before = day_stamps(store)
    assert sorted(before) == FIRST_FIVE

    write_clean(store, days(clean, date(2026, 3, 1)))      # the same rows again: file untouched
    assert day_stamps(store) == before

    store.put(clean_key(date(2026, 3, 1)), b"different bytes")
    after = day_stamps(store)
    assert after[date(2026, 3, 1)] != before[date(2026, 3, 1)]
    assert {d: s for d, s in after.items() if d != date(2026, 3, 1)} == {
        d: s for d, s in before.items() if d != date(2026, 3, 1)}


def test_changed_summaries_compares_text_and_labels():
    same = Summary("a", "text a", {"kind": "region"})
    new_text = Summary("b", "text b, updated", {"kind": "region"})
    brand_new = Summary("c", "text c", {"kind": "region"})
    stored = {"a": {"kind": "region", "text": "text a"},
              "b": {"kind": "region", "text": "text b"},
              "gone": {"kind": "region", "text": "old"}}

    to_store, stale = changed_summaries([same, new_text, brand_new], stored)

    assert [s.summary_id for s in to_store] == ["b", "c"]
    assert stale == ["gone"]
