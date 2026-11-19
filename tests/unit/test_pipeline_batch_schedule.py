import pytest

from sales_poc.pipelines.batch import parse_args, run_every


class FakeClock:
    def __init__(self):
        self.waits = []

    def sleep(self, seconds):
        self.waits.append(seconds)


def test_runs_now_then_waits_between_passes():
    clock, calls = FakeClock(), []

    failures = run_every(lambda: calls.append("pass"), 900, max_passes=3, sleep=clock.sleep)

    assert calls == ["pass"] * 3
    assert clock.waits == [900, 900]  # no wait after the last pass
    assert failures == 0


def test_a_failed_pass_does_not_stop_the_schedule(capsys):
    clock, calls = FakeClock(), []

    def one_pass():
        calls.append(len(calls))
        if len(calls) == 2:
            raise ConnectionError("network down")

    failures = run_every(one_pass, 60, max_passes=4, sleep=clock.sleep)

    assert calls == [0, 1, 2, 3]
    assert failures == 1
    assert "Pass 2 failed: network down" in capsys.readouterr().out


def test_ctrl_c_stops_cleanly(capsys):
    def sleep(_):
        raise KeyboardInterrupt

    assert run_every(lambda: None, 60, sleep=sleep) == 0
    assert "Stopped by Ctrl+C" in capsys.readouterr().out


def test_command_line_modes():
    assert parse_args(["--once"]).once
    assert parse_args(["--every", "0.25", "--max-passes", "2"]).every == 0.25
    for bad in ([], ["--once", "--every", "5"], ["--every", "0"], ["--once", "--max-passes", "2"]):
        with pytest.raises(SystemExit):
            parse_args(bad)
