import json
from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest

from sales_poc.datagen.history import COLUMNS
from sales_poc.datagen.live import LiveOrderMaker
from sales_poc.datagen.problems import PROBLEM_RATES
from sales_poc.datagen.reference import build_reference_data

START = datetime(2026, 11, 18, 14, 0, 0, tzinfo=timezone.utc)
N = 20_000


@pytest.fixture(scope="module")
def ref():
    return build_reference_data(seed=42)


def make_orders(ref, n, seed=42, rates=None):
    maker = LiveOrderMaker(ref, seed=seed, rates=rates, started_at=START)
    return [maker.next_order(START + timedelta(seconds=i)) for i in range(n)]


@pytest.fixture(scope="module")
def many(ref):
    return make_orders(ref, N)


def test_payload_has_the_history_fields_and_is_json_ready(ref):
    [order] = make_orders(ref, 1, rates={})

    assert list(order.payload) == COLUMNS
    assert json.loads(json.dumps(order.payload)) == order.payload
    assert all(isinstance(v, str) for v in order.payload.values())


def test_clean_order_is_valid(ref):
    [order] = make_orders(ref, 1, rates={})
    p = order.payload

    assert order.problem is None
    assert p["transaction_id"] == "LV-20261118140000-000001"
    assert p["event_time"] == "2026-11-18T14:00:00Z"
    assert p["customer_id"] in set(ref.customers["customer_id"])
    assert p["product_id"] in set(ref.products["product_id"])
    assert int(p["quantity"]) > 0
    assert p["currency"] == "USD"


def test_ids_are_unique_apart_from_duplicates(many):
    ids = [o.payload["transaction_id"] for o in many if o.payload["transaction_id"]]
    repeated = sum(c - 1 for c in Counter(ids).values() if c > 1)
    assert repeated == sum(o.problem == "duplicate" for o in many)


def test_same_seed_gives_same_orders(ref):
    assert make_orders(ref, 300, seed=3) == make_orders(ref, 300, seed=3)


def test_no_seed_gives_different_orders(ref):
    a = LiveOrderMaker(ref, started_at=START)
    b = LiveOrderMaker(ref, started_at=START)
    first_a = [a.next_order(START).payload["customer_id"] for _ in range(20)]
    first_b = [b.next_order(START).payload["customer_id"] for _ in range(20)]
    assert first_a != first_b


@pytest.mark.parametrize("kind", list(PROBLEM_RATES))
def test_each_problem_appears_at_about_its_set_rate(many, kind):
    count = sum(o.problem == kind for o in many)
    expected = PROBLEM_RATES[kind] * N
    # Allow for chance: within 40% of the expected count.
    assert 0.6 * expected <= count <= 1.4 * expected


def test_duplicate_is_the_same_payload_sent_again(many):
    for previous, order in zip(many, many[1:]):
        if order.problem == "duplicate":
            assert order.payload == previous.payload


def test_late_orders_are_one_to_three_days_old(many):
    late = [(i, o) for i, o in enumerate(many) if o.problem == "late"]
    assert late
    for i, order in late:
        sent_at = START + timedelta(seconds=i)
        event = datetime.strptime(order.payload["event_time"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        assert timedelta(days=1) <= sent_at - event <= timedelta(days=3)


def test_missing_values_are_none_in_the_payload(many):
    for order in many:
        if order.problem in ("missing_fillable", "missing_required"):
            assert order.payload[order.detail] is None
            assert json.dumps(order.payload)  # None becomes JSON null


def test_rate_can_be_turned_off(ref):
    assert all(o.problem is None for o in make_orders(ref, 500, rates={}))
