"""The `vector_search` tool: find sales summaries by meaning (task T12.3, FR-27).

The same hybrid search the RAG answer uses (vectors/store.py): closest by meaning first, then
summaries whose labels the question names move up. Only reads the index.

Each result is one monthly summary with its ID (e.g. "catreg:Laptop:EMEA:2026-03"), which the
agent can list as a source.
"""

from __future__ import annotations

import re

from sales_poc.ai.tools import ToolError
from sales_poc.ai.tools.table import Table

DEFAULT_TOP_K = 5
MAX_TOP_K = 10
REGIONS = ["NA", "EMEA", "APJ", "LATAM"]
CATEGORIES = ["Laptop", "Desktop", "Server", "Storage", "Monitor", "Accessory"]
MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "vector_search",
        "description": (
            "Search the monthly sales summaries by meaning and get the best matches with their IDs. "
            "There are three kinds: one category in one region for a month, a whole region for a month, "
            "and one customer for a month. Each gives orders, revenue, the change from the month before, "
            "and top channel, product or segment. Good for overviews and trends; use sql_query for exact "
            "figures. Some summaries cover only part of a month and say so."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "what to look for, in plain words"},
                "region": {"type": "string", "enum": REGIONS, "description": "only this region"},
                "category": {"type": "string", "enum": CATEGORIES, "description": "only this category"},
                "month": {"type": "string", "description": "only this month, as YYYY-MM"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": MAX_TOP_K,
                          "description": f"how many summaries (default {DEFAULT_TOP_K})"},
            },
            "required": ["query"],
        },
    },
}


class SearchToolError(ToolError):
    """The search was refused or failed."""


def check_arguments(query: str, region=None, category=None, month=None, top_k=None) -> dict:
    """The search arguments, checked, with friendly messages for the model."""
    if not (query or "").strip():
        raise SearchToolError("The search text is empty.")
    if region and region not in REGIONS:
        raise SearchToolError(f"Unknown region {region!r}. Use one of: {', '.join(REGIONS)}.")
    if category and category not in CATEGORIES:
        raise SearchToolError(f"Unknown category {category!r}. Use one of: {', '.join(CATEGORIES)}.")
    if month and not MONTH.match(str(month)):
        raise SearchToolError(f"Month {month!r} is not in the form YYYY-MM.")
    if top_k is None:
        top_k = DEFAULT_TOP_K
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= MAX_TOP_K:
        raise SearchToolError(f"top_k must be a whole number from 1 to {MAX_TOP_K}.")
    return {"question": query.strip(), "top_k": top_k, "region": region or None,
            "category": category or None, "month": month or None}


class VectorSearchTool:
    def __init__(self, index=None, embedder=None):
        """Give an `index` and `embedder` (tests), or the real ones are opened on first use."""
        self._index, self._embedder = index, embedder

    def run(self, query: str, region=None, category=None, month=None, top_k=None) -> Table:
        from sales_poc.vectors import store

        args = check_arguments(query, region, category, month, top_k)
        if self._index is None:
            self._index = store.default_index()
        if self._embedder is None:
            from sales_poc.vectors.embeddings import get_embedder
            self._embedder = get_embedder()
        try:
            results = store.search(self._index, self._embedder, args.pop("question"), args.pop("top_k"), **args)
        except Exception as exc:  # S3 Vectors or the embedding model; the agent just needs the reason
            raise SearchToolError(f"The search failed: {exc}") from None
        rows = [(r.summary_id, round(r.score, 2), r.text) for r in results]
        return Table(query, ["summary_id", "match", "text"], rows, truncated=False)
