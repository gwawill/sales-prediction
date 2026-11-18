import io
import json
import logging
from datetime import date
from pathlib import Path

import pytest

from sales_poc.log import get_logger, setup_logging


@pytest.fixture
def stream():
    """Capture log output in memory, and leave the root logger clean afterwards."""
    buffer = io.StringIO()
    setup_logging("DEBUG", stream=buffer)
    yield buffer
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() == "sales_poc_json":
            root.removeHandler(handler)


def lines(buffer: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buffer.getvalue().splitlines()]


def test_each_line_is_json_with_time_level_and_message(stream):
    get_logger("test").info("hello %s", "world")

    [entry] = lines(stream)
    assert entry["message"] == "hello world"
    assert entry["level"] == "INFO"
    assert entry["logger"] == "test"
    assert entry["time"].endswith("Z")


def test_extra_fields_become_json_fields(stream):
    get_logger("test").info("load_done", extra={"rows": 500, "run_id": "r1"})

    [entry] = lines(stream)
    assert entry["rows"] == 500
    assert entry["run_id"] == "r1"


def test_values_json_cannot_handle_are_turned_into_text(stream):
    get_logger("test").info("x", extra={"day": date(2026, 3, 1), "path": Path("data")})

    [entry] = lines(stream)
    assert entry["day"] == "2026-03-01"
    assert entry["path"] == "data"


def test_exception_is_included(stream):
    try:
        raise ValueError("bad row")
    except ValueError:
        get_logger("test").exception("etl_failed")

    [entry] = lines(stream)
    assert entry["level"] == "ERROR"
    assert "ValueError: bad row" in entry["exception"]


def test_calling_setup_twice_does_not_duplicate_lines(stream):
    second = io.StringIO()
    setup_logging("INFO", stream=second)

    get_logger("test").info("once")

    assert stream.getvalue() == ""
    assert len(lines(second)) == 1


def test_level_filters_lower_messages(stream):
    setup_logging("WARNING", stream=stream)

    get_logger("test").info("hidden")
    get_logger("test").warning("shown")

    assert [e["message"] for e in lines(stream)] == ["shown"]
