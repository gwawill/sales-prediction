"""JSON logging: every log line is one JSON object (requirement FR-22).

This stands in for the ELK Stack used in the real project: tools can search and count
fields such as `rows` or `run_id` without guessing where they are in a sentence.

Usage:

    from sales_poc.log import get_logger, setup_logging

    setup_logging()                      # once, at program start
    log = get_logger(__name__)
    log.info("load_done", extra={"rows": 500, "date": "2026-03-01"})

prints:

    {"time": "2026-11-18T10:00:00.123Z", "level": "INFO", "logger": "...",
     "message": "load_done", "rows": 500, "date": "2026-03-01"}
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import IO

# Attributes every LogRecord has. Anything else on a record came from `extra=` and is
# copied into the JSON output as its own field.
_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}

# Marks the handler we install, so calling setup_logging() twice does not add a second one.
_HANDLER_NAME = "sales_poc_json"


class JsonFormatter(logging.Formatter):
    """Formats a log record as a single line of JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        # default=str turns values JSON can't handle (dates, paths, ...) into text
        # instead of crashing the program.
        return json.dumps(entry, default=str, ensure_ascii=False)


def setup_logging(level: str | int = "INFO", stream: IO[str] | None = None) -> None:
    """Send all logs to `stream` (default: standard error) as JSON lines.

    Safe to call more than once: it replaces its own earlier handler.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() == _HANDLER_NAME:
            root.removeHandler(handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """A normal Python logger. Use `extra={...}` to add fields."""
    return logging.getLogger(name)
