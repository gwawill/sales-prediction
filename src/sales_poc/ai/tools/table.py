"""The small result table that the query tools hand back to the model."""

from __future__ import annotations

from dataclasses import dataclass


def cell(value) -> str:
    """Text for one value; decimals rounded to 2 places so money reads cleanly."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


@dataclass(frozen=True)
class Table:
    query: str
    columns: list[str]
    rows: list[tuple]
    truncated: bool          # True if there were more rows than the tool returns
    note: str = ""           # optional line shown above the table, e.g. how to read a column

    def to_text(self) -> str:
        """A compact table for the model to read."""
        lines = [self.note] if self.note else []
        lines.append(" | ".join(self.columns))
        lines += [" | ".join(cell(v) for v in row) for row in self.rows]
        if not self.rows:
            lines.append("(no rows)")
        if self.truncated:
            lines.append(f"(only the first {len(self.rows)} rows are shown)")
        return "\n".join(lines)
