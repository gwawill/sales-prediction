import json

import pandas as pd
import pytest

from sales_poc.datagen.cli import REFERENCE_TABLES, main


def run(tmp_path, *extra):
    return main(["--out", str(tmp_path), "--seed", "42", "--start", "2026-03-01", "--days", "5", *extra])


def test_writes_every_expected_file(tmp_path, capsys):
    assert run(tmp_path) == 0

    for name in REFERENCE_TABLES:
        assert (tmp_path / "reference" / f"{name}.csv").is_file()
    daily = sorted(p.name for p in (tmp_path / "incoming").iterdir())
    assert daily == [f"sales_2026-03-0{d}.csv" for d in range(1, 6)]
    assert (tmp_path / "problem_log.csv").is_file()
    assert "Wrote data for 2026-03-01 to 2026-03-05" in capsys.readouterr().out


def test_summary_matches_the_files(tmp_path):
    run(tmp_path)

    summary = json.loads((tmp_path / "generation.json").read_text(encoding="utf-8"))
    rows = sum(len(pd.read_csv(p, dtype=str)) for p in (tmp_path / "incoming").iterdir())
    log = pd.read_csv(tmp_path / "problem_log.csv")

    assert summary["daily_files"] == 5
    assert summary["rows_in_files"] == rows
    assert summary["problems"] == len(log)
    assert summary["rows_in_files"] == summary["clean_transactions"] + summary["problems_by_kind"].get(
        "duplicate", 0
    )


def test_refuses_to_mix_with_an_earlier_run(tmp_path, capsys):
    run(tmp_path)
    capsys.readouterr()

    assert run(tmp_path) == 1
    assert "--overwrite" in capsys.readouterr().out


def test_overwrite_replaces_old_files(tmp_path):
    run(tmp_path)
    stray = tmp_path / "incoming" / "sales_1999-01-01.csv"
    stray.write_text("old", encoding="utf-8")

    assert run(tmp_path, "--overwrite") == 0
    assert not stray.exists()
    assert len(list((tmp_path / "incoming").iterdir())) == 5


def test_same_seed_writes_identical_files(tmp_path):
    run(tmp_path / "a")
    run(tmp_path / "b")

    a = sorted((tmp_path / "a").rglob("*.csv"))
    b = sorted((tmp_path / "b").rglob("*.csv"))
    assert [p.relative_to(tmp_path / "a") for p in a] == [p.relative_to(tmp_path / "b") for p in b]
    assert [p.read_bytes() for p in a] == [p.read_bytes() for p in b]


def test_days_must_be_positive(tmp_path):
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path), "--days", "0"])
