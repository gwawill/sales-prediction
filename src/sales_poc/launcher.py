"""run-app: start the web API, the dashboard and a public Cloudflare tunnel with one command.

    run-app                                   API on 8000, dashboard on 8501, tunnel to the dashboard
    run-app --no-tunnel                       only the API and the dashboard, nothing public
    run-app --api-port 8077 --dashboard-port 8600

Each part runs as its own program (a "child process"). Their output all goes to this terminal.
Cloudflare prints the public https://....trycloudflare.com address among its log lines.

Press Ctrl+C to stop everything. If one part stops by itself, the others are stopped too,
so nothing is left running in the background.

The tunnel points at 127.0.0.1 rather than "localhost". On Windows "localhost" can mean the
IPv6 address ::1, and the dashboard only listens on 127.0.0.1.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

POLL_SECONDS = 0.5
STOP_WAIT_SECONDS = 10


@dataclass(frozen=True)
class Part:
    name: str
    command: list[str]


def _python_main(module: str, args: list[str]) -> list[str]:
    """A command that runs `module.main(args)` with this same Python."""
    code = f"import sys; from {module} import main; sys.exit(main(sys.argv[1:]))"
    return [sys.executable, "-c", code, *args]


def build_parts(api_host: str, api_port: int, dashboard_port: int, tunnel: bool) -> list[Part]:
    api_url = f"http://{api_host}:{api_port}"
    parts = [
        Part("api", _python_main("sales_poc.api.app", ["--port", str(api_port)])),
        Part("dashboard", _python_main("sales_poc.dashboard.app",
                                       ["--port", str(dashboard_port), "--api", api_url])),
    ]
    if tunnel:
        parts.append(Part("tunnel", ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{dashboard_port}"]))
    return parts


def _stop(procs: dict[str, subprocess.Popen]) -> None:
    for proc in procs.values():
        if proc.poll() is None:
            proc.terminate()
    for proc in procs.values():
        try:
            proc.wait(timeout=STOP_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_parts(parts: Sequence[Part], popen: Callable[..., subprocess.Popen] = subprocess.Popen,
              sleep: Callable[[float], None] = time.sleep) -> int:
    """Start every part, wait until one stops or Ctrl+C, then stop the rest. Returns an exit code."""
    procs: dict[str, subprocess.Popen] = {}
    try:
        for part in parts:
            print(f"run-app: starting {part.name}", flush=True)
            procs[part.name] = popen(part.command)
        while True:
            for name, proc in procs.items():
                code = proc.poll()
                if code is not None:
                    print(f"run-app: {name} stopped with exit code {code}; stopping the rest", flush=True)
                    return code or 1
            sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("run-app: Ctrl+C, stopping everything", flush=True)
        return 0
    except FileNotFoundError as exc:
        print(f"run-app: could not start a part, program not found: {exc.filename}", flush=True)
        return 1
    finally:
        _stop(procs)


def main(argv: list[str] | None = None) -> int:
    from sales_poc.config import get_app_settings

    settings = get_app_settings()
    parser = argparse.ArgumentParser(prog="run-app",
                                     description="Run the web API, the dashboard and a Cloudflare tunnel together.")
    parser.add_argument("--api-port", type=int, default=settings.api_port)
    parser.add_argument("--dashboard-port", type=int, default=8501)
    parser.add_argument("--no-tunnel", action="store_true", help="do not start the public Cloudflare tunnel")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    parts = build_parts(settings.api_host, args.api_port, args.dashboard_port, tunnel=not args.no_tunnel)
    return run_parts(parts)


if __name__ == "__main__":
    sys.exit(main())
