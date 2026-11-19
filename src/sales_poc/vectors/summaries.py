"""Short text summaries of the sales data, for the vector store (FR-19).

The vector store holds readable summaries, not raw rows: summaries are what a search by
meaning can find and what an AI can quote. Three kinds:

  catreg    one category in one region in one month
            "Laptop sales in EMEA, March 2026: 210 orders, $1.20M ..."
  region    everything in one region in one month
  customer  one customer over the whole period

Each summary has:
  summary_id  stable, e.g. "catreg:Laptop:EMEA:2026-03", so re-loading overwrites, not copies
  text        the summary itself
  metadata    labels a search can be narrowed by: kind, region_id, category, month,
              segment, customer_id (only the ones that apply)

A month that does not have data for every day says so ("partial month: data for 4 of 30
days"), and is compared with the month before by average revenue per day with data, so a
short month is not mistaken for a sales collapse.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

KINDS = ("catreg", "region", "customer")


@dataclass(frozen=True)
class Summary:
    summary_id: str
    text: str
    metadata: dict[str, str]


def money(value: float) -> str:
    """$1.23M, $45.6K, $812."""
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def _compare(now: float, before: float | None, month: pd.Period, coverage: dict) -> str:
    """How revenue compares with the month before.

    For a partial month, compare average revenue per day with data, not the totals.
    """
    label = _month_label(month - 1)
    if before is None or before == 0:
        return f"No data for {label} to compare with."
    if coverage[month].partial:
        days_before = coverage[month - 1].days_with_data if (month - 1) in coverage else 1
        per_day_now = now / coverage[month].days_with_data
        per_day_before = before / max(days_before, 1)
        pct = (per_day_now - per_day_before) / per_day_before * 100
        direction = "higher" if pct >= 0 else "lower"
        return (f"Average revenue per day with data was {abs(pct):.0f}% {direction} than in "
                f"{label} ({money(per_day_now)} vs {money(per_day_before)} a day).")
    pct = (now - before) / before * 100
    direction = "more" if pct >= 0 else "less"
    return f"Revenue was {abs(pct):.0f}% {direction} than {label} ({money(before)})."


def _month_label(period: pd.Period) -> str:
    return period.strftime("%B %Y")


@dataclass(frozen=True)
class Coverage:
    days_with_data: int
    days_in_month: int

    @property
    def partial(self) -> bool:
        return self.days_with_data < self.days_in_month

    @property
    def note(self) -> str:
        if not self.partial:
            return ""
        return f"partial month: data for {self.days_with_data} of {self.days_in_month} days"


def _coverage(rows: pd.DataFrame) -> dict[pd.Period, Coverage]:
    """For each month, how many of its days have any data at all."""
    days = pd.to_datetime(pd.Series(sorted(set(rows["event_date"]))))
    per_month = days.dt.to_period("M").value_counts()
    return {
        period: Coverage(int(count), period.days_in_month) for period, count in per_month.items()
    }


def _top(series: pd.Series) -> tuple[str, float]:
    """(largest label, its share of the total)."""
    totals = series.sort_values(ascending=False)
    return str(totals.index[0]), float(totals.iloc[0] / totals.sum()) if totals.sum() else 0.0


def _prepared(clean: pd.DataFrame) -> pd.DataFrame:
    rows = clean.copy()
    rows["month"] = pd.to_datetime(rows["event_date"]).dt.to_period("M")
    rows["total_amount"] = rows["total_amount"].astype(float)
    rows["quantity"] = rows["quantity"].astype(int)
    return rows


def category_region_summaries(rows: pd.DataFrame, coverage: dict) -> list[Summary]:
    out = []
    revenue = rows.groupby(["category", "region_id", "month"])["total_amount"].sum()
    for (category, region, month), group in rows.groupby(["category", "region_id", "month"]):
        region_name = group["region_name"].iloc[0]
        total = float(group["total_amount"].sum())
        previous = revenue.get((category, region, month - 1))
        channel, channel_share = _top(group.groupby("channel_name")["total_amount"].sum())
        product, product_share = _top(group.groupby("product_name")["total_amount"].sum())
        segment, segment_share = _top(group.groupby("segment")["total_amount"].sum())
        note = coverage[month].note
        text = (
            f"{category} sales in {region_name} ({region}), {_month_label(month)}"
            f"{f' ({note})' if note else ''}: {len(group)} orders, {money(total)} revenue, "
            f"{int(group['quantity'].sum())} units. "
            f"{_compare(total, previous, month, coverage)} "
            f"Top channel: {channel} ({channel_share:.0%} of revenue). "
            f"Top product: {product} ({product_share:.0%}). "
            f"Largest customer segment: {segment} ({segment_share:.0%})."
        )
        out.append(Summary(
            summary_id=f"catreg:{category}:{region}:{month}",
            text=text,
            metadata={"kind": "catreg", "category": category, "region_id": region, "month": str(month)},
        ))
    return out


def region_summaries(rows: pd.DataFrame, coverage: dict) -> list[Summary]:
    out = []
    revenue = rows.groupby(["region_id", "month"])["total_amount"].sum()
    for (region, month), group in rows.groupby(["region_id", "month"]):
        region_name = group["region_name"].iloc[0]
        total = float(group["total_amount"].sum())
        previous = revenue.get((region, month - 1))
        by_category = group.groupby("category")["total_amount"].sum().sort_values(ascending=False)
        top3 = ", ".join(f"{c} {money(v)}" for c, v in by_category.head(3).items())
        note = coverage[month].note
        text = (
            f"All sales in {region_name} ({region}), {_month_label(month)}"
            f"{f' ({note})' if note else ''}: {len(group)} orders, {money(total)} revenue "
            f"from {group['customer_id'].nunique()} customers. "
            f"{_compare(total, previous, month, coverage)} "
            f"Biggest categories: {top3}."
        )
        out.append(Summary(
            summary_id=f"region:{region}:{month}",
            text=text,
            metadata={"kind": "region", "region_id": region, "month": str(month)},
        ))
    return out


def customer_summaries(rows: pd.DataFrame) -> list[Summary]:
    out = []
    first, last = min(rows["event_date"]), max(rows["event_date"])
    for customer_id, group in rows.groupby("customer_id"):
        info = group.iloc[0]
        by_category = group.groupby("category")["total_amount"].sum().sort_values(ascending=False)
        top3 = ", ".join(f"{c} {money(v)}" for c, v in by_category.head(3).items())
        channel, channel_share = _top(group.groupby("channel_name")["total_amount"].sum())
        text = (
            f"Customer {customer_id}, {info['customer_name']}: {info['segment']} customer in "
            f"{info['country']} ({info['region_name']}). From {first:%d %b %Y} to {last:%d %b %Y}: "
            f"{len(group)} orders, {money(float(group['total_amount'].sum()))} revenue. "
            f"Main categories: {top3}. Preferred channel: {channel} ({channel_share:.0%}). "
            f"Last order: {max(group['event_date']):%d %b %Y}."
        )
        out.append(Summary(
            summary_id=f"customer:{customer_id}",
            text=text,
            metadata={"kind": "customer", "customer_id": str(customer_id), "segment": str(info["segment"]),
                      "region_id": str(info["region_id"])},
        ))
    return out


def build_summaries(clean: pd.DataFrame) -> list[Summary]:
    """Every summary for the clean rows (the lake), in a stable order."""
    if clean.empty:
        return []
    rows = _prepared(clean)
    coverage = _coverage(rows)
    return (
        category_region_summaries(rows, coverage)
        + region_summaries(rows, coverage)
        + customer_summaries(rows)
    )
