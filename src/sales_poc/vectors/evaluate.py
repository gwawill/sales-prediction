"""How good is the summary search? A small fixed set of questions with the right answers.

    python -m sales_poc.vectors.evaluate          search the real index and print a report

Each question lists the summary it should find, as an ID or the start of one: a question
without a month accepts that category and region in any month ("catreg:Laptop:EMEA:").
A question counts as found when one of the first TOP_K results matches. The report also
shows how often the match came first, and the questions that missed, so a change to the
search can be judged by numbers instead of one example.

The questions name places the way people do: regions ("EMEA"), but also countries
("Germany") and other words for a product ("notebook").
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

TOP_K = 3

QUESTIONS: list[tuple[str, str]] = [
    ("laptop sales in EMEA", "catreg:Laptop:EMEA:"),
    ("How did laptop sales in Germany do?", "catreg:Laptop:EMEA:"),
    ("server sales in Japan in March 2026", "catreg:Server:APJ:2026-03"),
    ("APJ desktop sales in March 2026", "catreg:Desktop:APJ:2026-03"),
    ("storage revenue in Brazil", "catreg:Storage:LATAM:"),
    ("monitor sales in the United States in December 2025", "catreg:Monitor:NA:2025-12"),
    ("accessories in France", "catreg:Accessory:EMEA:"),
    ("notebook sales in India in June 2026", "catreg:Laptop:APJ:2026-06"),
    ("How did all sales in Latin America do in October 2026?", "region:LATAM:2026-10"),
    ("overall sales in EMEA in May 2026", "region:EMEA:2026-05"),
    ("total sales in North America in January 2026", "region:NA:2026-01"),
    ("servers in Canada", "catreg:Server:NA:"),
    ("desktop sales in Mexico in August 2026", "catreg:Desktop:LATAM:2026-08"),
    ("What does customer C00012 buy?", "customer:C00012"),
    ("storage sales in Australia in September 2026", "catreg:Storage:APJ:2026-09"),
    ("laptop sales in the United Kingdom in November 2025", "catreg:Laptop:EMEA:2025-11"),
    ("monitors in South Korea", "catreg:Monitor:APJ:"),
    ("server sales in Singapore in December 2025", "catreg:Server:APJ:2025-12"),
    ("all sales in APJ in March 2026", "region:APJ:2026-03"),
    ("accessory sales in Colombia in April 2026", "catreg:Accessory:LATAM:2026-04"),
]


@dataclass(frozen=True)
class Outcome:
    question: str
    expected: str
    found_ids: list[str]

    @property
    def rank(self) -> int | None:
        """1 if the first result matches, 2 if the second, ...; None if none of them do."""
        for position, summary_id in enumerate(self.found_ids, start=1):
            if summary_id.startswith(self.expected):
                return position
        return None


def evaluate(search_fn, questions: list[tuple[str, str]] = QUESTIONS, top_k: int = TOP_K) -> list[Outcome]:
    """Run every question through `search_fn(question, top_k) -> results with .summary_id`."""
    return [Outcome(q, expected, [r.summary_id for r in search_fn(q, top_k)]) for q, expected in questions]


def report(outcomes: list[Outcome]) -> str:
    found = sum(o.rank is not None for o in outcomes)
    first = sum(o.rank == 1 for o in outcomes)
    lines = [f"Found in the top {TOP_K}: {found} of {len(outcomes)}. Ranked first: {first} of {len(outcomes)}."]
    for o in outcomes:
        if o.rank != 1:
            where = f"ranked {o.rank}" if o.rank else "MISSED"
            lines.append(f"  {where:9} {o.question!r} wanted {o.expected}, got {o.found_ids}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from sales_poc.vectors.embeddings import get_embedder
    from sales_poc.vectors.store import default_index, search

    index, embedder = default_index(), get_embedder()
    outcomes = evaluate(lambda q, k: search(index, embedder, q, top_k=k))
    print(report(outcomes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
