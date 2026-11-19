import shutil
from datetime import date, datetime, timezone

import pytest

from sales_poc.datagen.cli import generate
from sales_poc.datagen.reference import load_reference_data
from sales_poc.lake.runs import list_run_ids
from sales_poc.lake.store import LocalStore
from sales_poc.pipelines.batch import find_daily_files, load_new_files
from sales_poc.pipelines.batch_state import STATE_KEY, ChangedFileError, read_state

NOW = datetime(2026, 11, 18, 9, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    generate(out, seed=42, start=date(2026, 3, 1), days=8)
    return out


@pytest.fixture
def incoming(tmp_path, generated):
    """A private copy of the incoming folder, so tests can add or change files."""
    folder = tmp_path / "incoming"
    shutil.copytree(generated / "incoming", folder)
    return folder


@pytest.fixture
def ref(generated):
    return load_reference_data(generated / "reference")


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "bucket")


def everything(store):
    return {k: store.get(k) for k in store.list()}


def test_first_run_loads_all_and_records_them(store, incoming, ref):
    files = find_daily_files(incoming)

    run, skipped = load_new_files(store, files, ref, now=NOW)

    assert len(run.files) == 8 and skipped == []
    state = read_state(store)
    assert sorted(state) == [p.name for p in files]
    assert {entry["run_id"] for entry in state.values()} == {run.run_id}


def test_second_run_loads_nothing_and_changes_nothing(store, incoming, ref):
    files = find_daily_files(incoming)
    load_new_files(store, files, ref, now=NOW)
    before = everything(store)

    run, skipped = load_new_files(store, files, ref, now=NOW)

    assert run is None
    assert len(skipped) == 8
    assert everything(store) == before  # not even a new report


def test_only_a_newly_added_file_is_loaded(store, incoming, ref):
    files = find_daily_files(incoming)
    load_new_files(store, files[:-1], ref, now=NOW)

    run, skipped = load_new_files(store, find_daily_files(incoming), ref, now=NOW)

    assert [p.name for p in run.files] == [files[-1].name]
    assert len(skipped) == 7
    assert len(read_state(store)) == 8
    assert len(list_run_ids(store)) == 2


def test_a_changed_file_stops_the_run(store, incoming, ref):
    files = find_daily_files(incoming)
    load_new_files(store, files, ref, now=NOW)
    files[2].write_text(files[2].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    before = everything(store)

    with pytest.raises(ChangedFileError, match=files[2].name):
        load_new_files(store, files, ref, now=NOW)
    assert everything(store) == before


def test_losing_the_list_is_safe(store, incoming, ref):
    files = find_daily_files(incoming)
    load_new_files(store, files, ref, now=NOW)
    clean_before = {k: store.get(k) for k in store.list("clean/")}
    store.delete(STATE_KEY)

    run, skipped = load_new_files(store, files, ref, now=NOW)

    assert len(run.files) == 8 and skipped == []
    assert {k: store.get(k) for k in store.list("clean/")} == clean_before
    assert run.report["lake"]["rows_added"] == 0
    assert len(read_state(store)) == 8


def test_a_failed_run_does_not_mark_files_as_loaded(store, incoming, ref, monkeypatch):
    import sales_poc.pipelines.batch as batch

    def broken(*args, **kwargs):
        raise RuntimeError("S3 went away")

    monkeypatch.setattr(batch, "write_clean", broken)
    with pytest.raises(RuntimeError):
        load_new_files(store, find_daily_files(incoming), ref, now=NOW)

    assert read_state(store) == {}
