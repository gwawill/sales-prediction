"""Live orders, made one at a time for the Kafka producer (FR-3), with the same kinds of
problems as the history (FR-4).

    maker = LiveOrderMaker(ref, seed=42)
    order = maker.next_order()        # LiveOrder(payload={...}, problem=None, detail=None)
    json.dumps(order.payload)         # ready to send

How problems work for live orders:
  - duplicate: the order is sent normally, and the *next* call returns the same order again
    (like a client retrying). That second copy is the one marked "duplicate".
  - late: the order's time is 1 to 3 days in the past, so it belongs to an older day (FR-15).
  - every other kind changes the order's values, exactly as in the history.

Transaction IDs look like LV-20261118142530-000001: the time the maker started plus a
counter, so restarting the producer does not reuse old IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from sales_poc.datagen.history import (
    expected_orders,
    format_event_time,
    make_order,
    pick_category,
)
from sales_poc.datagen.problems import (
    IN_PLACE_PROBLEMS,
    PROBLEM_RATES,
    damage_row,
    order_as_text,
)
from sales_poc.datagen.reference import ReferenceData


@dataclass(frozen=True)
class LiveOrder:
    payload: dict[str, str | None]  # the message to send; same fields as a history row
    problem: str | None             # which problem was added, or None if clean
    detail: str | None              # which field or format, as in the history problem log


class LiveOrderMaker:
    def __init__(
        self,
        ref: ReferenceData,
        seed: int | None = None,
        rates: dict[str, float] | None = None,
        started_at: datetime | None = None,
    ):
        """`seed=None` gives different orders every run; a number makes them repeatable."""
        self._rng = np.random.default_rng(None if seed is None else [seed, 4])
        self._rates = rates if rates is not None else PROBLEM_RATES
        self._kinds = [*self._rates, None]
        self._probabilities = [*self._rates.values(), 1 - sum(self._rates.values())]

        customers = ref.customers
        self._consumers = customers[customers["segment"] == "Consumer"]
        self._businesses = customers[customers["segment"] != "Consumer"]
        weights = np.where(self._businesses["segment"] == "Enterprise", 3.0, 1.0)
        self._business_weights = weights / weights.sum()
        self._products_by_category = {c: g for c, g in ref.products.groupby("category")}

        started = started_at or datetime.now(timezone.utc)
        self._id_prefix = f"LV-{started:%Y%m%d%H%M%S}"
        self._counter = 0
        self._pending_duplicate: dict | None = None

    def _new_clean_order(self, now: datetime) -> dict:
        # Same consumer/business balance as the history would expect on this day.
        expected = expected_orders(now.date(), start=now.date())
        business_share = expected["Business"] / (expected["Business"] + expected["Consumer"])
        if self._rng.random() < business_share:
            customer = self._businesses.iloc[
                int(self._rng.choice(len(self._businesses), p=self._business_weights))
            ]
        else:
            customer = self._consumers.iloc[int(self._rng.integers(0, len(self._consumers)))]

        segment = customer["segment"]
        category = pick_category(self._rng, segment)
        order = make_order(
            self._rng, segment, customer["customer_id"], category, self._products_by_category
        )
        self._counter += 1
        order["transaction_id"] = f"{self._id_prefix}-{self._counter:06d}"
        order["event_time"] = format_event_time(now)
        return order

    def next_order(self, now: datetime | None = None) -> LiveOrder:
        """The next order to send. `now` sets its time (default: the current UTC time)."""
        if self._pending_duplicate is not None:
            payload, self._pending_duplicate = self._pending_duplicate, None
            return LiveOrder(payload=payload, problem="duplicate", detail="resent")

        now = now or datetime.now(timezone.utc)
        order = self._new_clean_order(now)
        kind = self._rng.choice(self._kinds, p=self._probabilities)

        if kind is None:
            return LiveOrder(payload=order_as_text(order), problem=None, detail=None)

        if kind == "duplicate":
            payload = order_as_text(order)
            self._pending_duplicate = dict(payload)
            return LiveOrder(payload=payload, problem=None, detail=None)

        if kind == "late":
            delay = timedelta(seconds=int(self._rng.integers(24 * 3600, 3 * 24 * 3600)))
            order["event_time"] = format_event_time(now - delay)
            return LiveOrder(
                payload=order_as_text(order), problem="late", detail=f"{delay.days}_days"
            )

        assert kind in IN_PLACE_PROBLEMS
        damaged, detail = damage_row(pd.Series(order_as_text(order)), str(kind), self._rng)
        payload = {k: (None if pd.isna(v) else v) for k, v in damaged.to_dict().items()}
        return LiveOrder(payload=payload, problem=str(kind), detail=detail)
