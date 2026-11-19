"""Which daily files the batch pipeline has already loaded (task T5.2).

Kept in the lake at state/batch_loaded.json:

    {"files": {"sales_2026-03-01.csv": {"sha256": "...", "run_id": "...", "loaded_at": "..."}}}

  - A file whose name and content fingerprint (SHA-256) are listed is skipped.
  - A listed name with a *different* fingerprint is an error: the raw area never overwrites a
    file, so a changed file needs a new name.
  - The list is only updated after a run succeeds. If a run fails, or the list is lost, the
    files are simply loaded again, which is safe because every load is idempotent (FR-17).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from sales_poc.lake.store import ObjectStore

STATE_KEY = "state/batch_loaded.json"


class ChangedFileError(RuntimeError):
    """A file was loaded before, but its content is different now."""


def fingerprint(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_state(store: ObjectStore) -> dict[str, dict]:
    """{file name: details} for every loaded file. Empty if there is no list yet."""
    if not store.exists(STATE_KEY):
        return {}
    return json.loads(store.get(STATE_KEY))["files"]


def write_state(store: ObjectStore, files: dict[str, dict]) -> None:
    body = {"files": dict(sorted(files.items()))}
    store.put(STATE_KEY, (json.dumps(body, indent=2) + "\n").encode("utf-8"))


def new_files(store: ObjectStore, files: list[Path]) -> tuple[list[Path], list[Path]]:
    """(files still to load, files already loaded). Raises ChangedFileError if a loaded
    file's content has changed."""
    loaded = read_state(store)
    pending, skipped, changed = [], [], []
    for path in files:
        entry = loaded.get(path.name)
        if entry is None:
            pending.append(path)
        elif entry["sha256"] == fingerprint(path):
            skipped.append(path)
        else:
            changed.append(path.name)
    if changed:
        raise ChangedFileError(
            f"Already loaded, but the content is different now: {', '.join(changed)}. "
            "Raw data is never overwritten, so give a changed file a new name."
        )
    return pending, skipped


def mark_loaded(store: ObjectStore, files: list[Path], run_id: str, when: datetime | None = None) -> None:
    """Add `files` to the list. Call only after the run that loaded them succeeded."""
    when = (when or datetime.now(timezone.utc)).astimezone(timezone.utc)
    state = read_state(store)
    for path in files:
        state[path.name] = {"sha256": fingerprint(path), "run_id": run_id, "loaded_at": when.isoformat()}
    write_state(store, state)
