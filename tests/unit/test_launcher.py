"""run-app: the commands it builds, and that it stops every part when one stops."""

from __future__ import annotations

from sales_poc.launcher import build_parts, run_parts


class FakeProc:
    def __init__(self, exit_after: int | None = None) -> None:
        self.exit_after = exit_after   # polls until it "exits", or None to run forever
        self.polls = 0
        self.code: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        if self.code is None and self.exit_after is not None:
            self.polls += 1
            if self.polls > self.exit_after:
                self.code = 3
        return self.code

    def terminate(self) -> None:
        self.terminated = True
        self.code = -1

    def wait(self, timeout: float | None = None) -> int | None:
        return self.code

    def kill(self) -> None:
        self.code = -9


def test_build_parts_with_tunnel() -> None:
    parts = build_parts("127.0.0.1", 8000, 8501, tunnel=True)
    assert [p.name for p in parts] == ["api", "dashboard", "tunnel"]
    assert parts[0].command[-2:] == ["--port", "8000"]
    assert parts[1].command[-4:] == ["--port", "8501", "--api", "http://127.0.0.1:8000"]
    assert parts[2].command == ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8501"]


def test_build_parts_without_tunnel_and_custom_ports() -> None:
    parts = build_parts("127.0.0.1", 8077, 8600, tunnel=False)
    assert [p.name for p in parts] == ["api", "dashboard"]
    assert parts[1].command[-4:] == ["--port", "8600", "--api", "http://127.0.0.1:8077"]


def test_one_part_stopping_stops_the_others() -> None:
    procs = [FakeProc(), FakeProc(exit_after=2), FakeProc()]
    started = iter(procs)
    code = run_parts(build_parts("127.0.0.1", 8000, 8501, tunnel=True),
                     popen=lambda cmd: next(started), sleep=lambda s: None)
    assert code == 3
    assert procs[0].terminated and procs[2].terminated
    assert not procs[1].terminated


def test_ctrl_c_stops_everything() -> None:
    procs = [FakeProc(), FakeProc(), FakeProc()]
    started = iter(procs)

    def sleep(_: float) -> None:
        raise KeyboardInterrupt

    code = run_parts(build_parts("127.0.0.1", 8000, 8501, tunnel=True),
                     popen=lambda cmd: next(started), sleep=sleep)
    assert code == 0
    assert all(p.terminated for p in procs)


def test_missing_program_stops_what_already_started() -> None:
    first = FakeProc()

    def popen(cmd: list[str]) -> FakeProc:
        if cmd[0] == "cloudflared":
            raise FileNotFoundError(2, "not found", "cloudflared")
        return first

    code = run_parts(build_parts("127.0.0.1", 8000, 8501, tunnel=True), popen=popen, sleep=lambda s: None)
    assert code == 1
    assert first.terminated
