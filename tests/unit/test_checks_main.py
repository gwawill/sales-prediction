import pytest

from sales_poc import checks
from sales_poc.checks import CheckError, main


def ok():
    return "fine"


def broken():
    raise CheckError("bucket missing")


def crashes():
    raise ZeroDivisionError("oops")


@pytest.fixture
def fake_checks(monkeypatch):
    monkeypatch.setattr(checks, "CHECKS", {"a": ok, "b": broken, "c": crashes})


def test_all_pass_gives_exit_0_and_summary(fake_checks, capsys):
    assert main(["a"]) == 0

    out = capsys.readouterr().out
    assert "a: OK - fine" in out
    assert "1/1 checks passed." in out


def test_failures_are_reported_and_do_not_stop_other_checks(fake_checks, capsys):
    assert main([]) == 1

    out = capsys.readouterr().out
    assert "a: OK" in out
    assert "b: FAILED - bucket missing" in out
    assert "c: FAILED (unexpected ZeroDivisionError) - oops" in out
    assert "1/3 checks passed. Failed: b, c" in out


def test_unknown_name_gives_exit_2(fake_checks, capsys):
    assert main(["nope"]) == 2

    assert "Unknown check: nope. Choose from: a, b, c" in capsys.readouterr().out
