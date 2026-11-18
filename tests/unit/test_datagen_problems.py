from datetime import timedelta

import pandas as pd
import pytest

from sales_poc.datagen.history import COLUMNS, generate_history, write_daily_files
from sales_poc.datagen.problems import (
    BAD_FORMATS,
    FILLABLE_FIELDS,
    PROBLEM_RATES,
    REQUIRED_FIELDS,
    add_problems,
)
from sales_poc.datagen.reference import build_reference_data


@pytest.fixture(scope="module")
def ref():
    return build_reference_data(seed=42)


@pytest.fixture(scope="module")
def clean(ref):
    # 120 days: enough rows (~9,000) for every kind and detail to appear.
    return generate_history(ref, seed=42, days=120)


@pytest.fixture(scope="module")
def messy(clean):
    return add_problems(clean, seed=42)


def all_rows(files):
    return pd.concat(files.values(), ignore_index=True)


def logged(messy, problem):
    return messy.log[messy.log["problem"] == problem]


def test_every_problem_kind_is_present(messy):
    assert set(messy.log["problem"]) == set(PROBLEM_RATES)


def test_every_detail_is_present(messy):
    assert set(logged(messy, "missing_fillable")["detail"]) == set(FILLABLE_FIELDS)
    assert set(logged(messy, "missing_required")["detail"]) == set(REQUIRED_FIELDS)
    assert set(logged(messy, "bad_format")["detail"]) == set(BAD_FORMATS)
    assert set(logged(messy, "unknown_id")["detail"]) == {"customer_id", "product_id"}
    assert set(logged(messy, "duplicate")["detail"]) == {"same_file", "next_day_file"}


def test_about_five_percent_of_rows_get_a_problem(clean, messy):
    share = len(messy.log) / len(all_rows(clean))
    assert 0.035 <= share <= 0.07


def test_same_seed_gives_same_mess(clean):
    a = add_problems(clean, seed=5)
    b = add_problems(clean, seed=5)

    pd.testing.assert_frame_equal(a.log, b.log)
    for day in a.files:
        pd.testing.assert_frame_equal(a.files[day], b.files[day])


def test_clean_history_is_not_changed(ref, clean):
    before = {day: rows.copy() for day, rows in clean.items()}
    add_problems(clean, seed=42)

    for day in clean:
        pd.testing.assert_frame_equal(clean[day], before[day])


def test_only_duplicates_add_rows(clean, messy):
    assert len(all_rows(messy.files)) == len(all_rows(clean)) + len(logged(messy, "duplicate"))


def test_every_day_still_has_a_file(clean, messy):
    assert sorted(messy.files) == sorted(clean)
    for rows in messy.files.values():
        assert list(rows.columns) == COLUMNS


def test_duplicates_appear_twice(messy):
    rows = all_rows(messy.files)
    counts = rows["transaction_id"].value_counts()
    for tx in logged(messy, "duplicate")["transaction_id"]:
        assert counts[tx] == 2


def test_missing_values_are_really_empty(messy):
    rows = all_rows(messy.files)
    for problem in ("missing_fillable", "missing_required"):
        for entry in logged(messy, problem).itertuples():
            if entry.detail == "transaction_id":
                continue  # the row can't be looked up by an ID it no longer has
            [value] = rows.loc[rows["transaction_id"] == entry.transaction_id, entry.detail]
            assert pd.isna(value)  # None or NaN: both are an empty cell in the CSV


def test_missing_transaction_ids_leave_rows_without_an_id(messy):
    rows = all_rows(messy.files)
    missing_ids = logged(messy, "missing_required").query("detail == 'transaction_id'")
    assert rows["transaction_id"].isna().sum() == len(missing_ids)


def test_bad_formats_look_as_described(messy):
    rows = all_rows(messy.files).set_index("transaction_id", drop=False)
    for entry in logged(messy, "bad_format").itertuples():
        row = rows.loc[entry.transaction_id]
        if entry.detail == "date_us":
            assert row["event_time"][2] == "/" and row["event_time"][5] == "/"
        elif entry.detail == "date_space":
            assert " " in row["event_time"] and not row["event_time"].endswith("Z")
        elif entry.detail == "amount_dollar":
            assert row["total_amount"].startswith("$")
        elif entry.detail == "channel_lower":
            assert row["channel_id"].islower()


def test_unknown_ids_and_negative_quantities(messy):
    rows = all_rows(messy.files).set_index("transaction_id", drop=False)
    for entry in logged(messy, "unknown_id").itertuples():
        assert rows.loc[entry.transaction_id, entry.detail] in ("C99999", "P9999")
    for entry in logged(messy, "negative_quantity").itertuples():
        assert int(rows.loc[entry.transaction_id, "quantity"]) < 0


def test_late_rows_sit_in_a_later_file_with_their_real_time(messy):
    late = logged(messy, "late")
    assert len(late) > 0
    for entry in late.itertuples():
        assert timedelta(days=1) <= entry.file_day - entry.event_day <= timedelta(days=3)
        file_rows = messy.files[entry.file_day]
        [event_time] = file_rows.loc[file_rows["transaction_id"] == entry.transaction_id, "event_time"]
        assert event_time.startswith(entry.event_day.isoformat())


def test_messy_files_can_be_written(messy, tmp_path):
    first_days = dict(list(messy.files.items())[:3])
    paths = write_daily_files(first_days, tmp_path)

    assert len(paths) == 3
    assert list(pd.read_csv(paths[0], dtype=str).columns) == COLUMNS
