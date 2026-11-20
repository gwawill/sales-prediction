"""Hybrid search: meaning first, then labels named in the question (decided in T9.3).

Why: all summaries share one sentence pattern and differ in a few words ("Laptop" vs
"Accessory", "March" vs "May"). A small embedding model sees them as nearly identical, so
plain search by meaning often puts the wrong category or month first. We already know each
summary's labels exactly, so we use them:

    1. ask the vector store for the CANDIDATES closest summaries by meaning,
    2. find the labels the question names (category, region, month, segment, customer),
    3. rank by meaning score + LABEL_BONUS for every named label a result matches,
    4. return the best top_k.

The meaning score still orders results that match the same number of labels. Each result
keeps its pure meaning `score` (0 to 1) and gets `matched_labels`, so both stay readable.
"""

from __future__ import annotations

import re
from dataclasses import replace

from sales_poc.datagen.reference import COUNTRIES_BY_REGION

CANDIDATES = 30
LABEL_BONUS = 0.1

# word stem (lower case) -> label value
CATEGORY_WORDS = {
    "laptop": "Laptop", "notebook": "Laptop",
    "desktop": "Desktop",
    "monitor": "Monitor", "display": "Monitor",
    "server": "Server",
    "storage": "Storage",
    "accessor": "Accessory",  # accessory, accessories
}

REGION_WORDS = {
    "emea": "EMEA", "europe": "EMEA", "european": "EMEA", "middle east": "EMEA", "africa": "EMEA",
    "apj": "APJ", "asia": "APJ", "pacific": "APJ",
    "north america": "NA", "usa": "NA",
    "latam": "LATAM", "latin america": "LATAM", "south america": "LATAM",
    # Every country a customer can be in, from the reference data ("germany" -> EMEA).
    **{country.lower(): region for region, countries in COUNTRIES_BY_REGION.items() for country in countries},
}

SEGMENT_WORDS = {
    "enterprise": "Enterprise",
    "smb": "SMB", "small business": "SMB", "small and medium": "SMB",
    "consumer": "Consumer",
}

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]


def _has(text: str, phrase: str) -> bool:
    """`phrase` appears as whole words (a trailing 's' or 'ies' plural is allowed)."""
    return re.search(rf"\b{re.escape(phrase)}\w*\b", text) is not None


def labels_in(question: str) -> dict[str, list[str]]:
    """The labels a question names: {"category": [...], "region_id": [...], ...}."""
    q = question.lower()
    found: dict[str, list[str]] = {}

    def add(key, value):
        found.setdefault(key, [])
        if value not in found[key]:
            found[key].append(value)

    for word, value in CATEGORY_WORDS.items():
        if _has(q, word):
            add("category", value)
    for word, value in REGION_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", q):
            add("region_id", value)
    # "NA" only as a capitalised word of its own, never "na" inside other words.
    if re.search(r"\bNA\b", question):
        add("region_id", "NA")
    for word, value in SEGMENT_WORDS.items():
        if _has(q, word):
            add("segment", value)

    for number, name in enumerate(MONTHS, start=1):
        # With a year ("march 2026", any case): that exact month.
        for match in re.finditer(rf"\b{name}\s+(\d{{4}})\b", q):
            add("month", f"{match.group(1)}-{number:02d}")
        # Without a year, only when capitalised ("March"), so "how may I..." is not May.
        # It then matches that month in any year ("*-03").
        if re.search(rf"\b{name.capitalize()}\b(?!\s+\d{{4}})", question):
            add("month", f"*-{number:02d}")
    for match in re.finditer(r"\b(\d{4})-(\d{2})\b", q):
        add("month", f"{match.group(1)}-{match.group(2)}")

    for match in re.finditer(r"\bc\d{5}\b", q):
        add("customer_id", match.group(0).upper())
    return found


def _matches(value: str | None, wanted: list[str]) -> bool:
    if value is None:
        return False
    return any(value.endswith(w[1:]) if w.startswith("*") else value == w for w in wanted)


def label_matches(metadata: dict, wanted: dict[str, list[str]]) -> int:
    """How many of the named labels this result matches."""
    return sum(_matches(metadata.get(key), values) for key, values in wanted.items())


def rerank(question: str, results: list, top_k: int) -> list:
    """Re-order search results by meaning score + LABEL_BONUS per matched label.

    Each returned result has `matched_labels` set; its `score` stays the meaning score.
    """
    wanted = labels_in(question)
    if not wanted:
        return results[:top_k]
    marked = [replace(r, matched_labels=label_matches(r.metadata, wanted)) for r in results]
    return sorted(marked, key=lambda r: r.score + LABEL_BONUS * r.matched_labels, reverse=True)[:top_k]
