import json
from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest

from sales_poc.datagen.history import COLUMNS
from sales_poc.datagen.live import LiveOrderMaker
from sales_poc.datagen.problems import PROBLEM_RATES
from sales_poc.datagen.reference import build_reference_data
from sales_poc.pipelines.producer import PROBLEM_HEADER, encode, parse_args, produce_orders

START = datetime(2026, 11, 18, 14, 0, tzinfo=timezone.utc)


class FakeClock:
    """Time only moves when someone sleeps."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture(scope="module")
def ref():
    return build_reference_data(seed=42)


def maker(ref, rates=None):
    """rates=None: the standard problem mix. rates={}: clean."""
    return LiveOrderMaker(ref, seed=1, rates=rates, started_at=START)


def collect():
    sent = []
    return sent, lambda key, value, headers: sent.append((key, value, dict(headers)))


def run(ref, count, rates=None):
    sent, send = collect()
    clock = FakeClock()
    stats = produce_orders(send, maker(ref, rates), rate=100, count=count, clock=clock, sleep=clock.sleep)
    return sent, stats


def test_message_key_value_and_header(ref):
    order = maker(ref, rates={}).next_order()
    key, value, headers = encode(order)

    assert key == order.payload["transaction_id"].encode()
    decoded = json.loads(value)
    assert decoded == order.payload and list(decoded) == COLUMNS
    assert headers == [(PROBLEM_HEADER, b"clean")]


def test_missing_transaction_id_means_no_key(ref):
    order = maker(ref, rates={}).next_order()
    order.payload["transaction_id"] = None
    key, value, _ = encode(order)

    assert key is None
    assert json.loads(value)["transaction_id"] is None


def test_count_limit(ref):
    sent, stats = run(ref, 25, rates={})
    assert stats.sent == len(sent) == 25
    assert stats.by_problem == {"clean": 25}


def test_rate_is_kept_without_drift(ref):
    sent, send = collect()
    clock = FakeClock()

    produce_orders(send, maker(ref, {}), rate=4, count=9, clock=clock, sleep=clock.sleep)

    assert clock.now == pytest.approx(2.0)  # 9 orders at 4/s: the last is due at 8/4 = 2s


def test_seconds_limit(ref):
    sent, send = collect()
    clock = FakeClock()

    produce_orders(send, maker(ref, {}), rate=5, seconds=3, clock=clock, sleep=clock.sleep)

    assert len(sent) == 15  # due at 0, 0.2, ... 2.8


def test_clean_mode_sends_no_problems(ref):
    sent, stats = run(ref, 500, rates={})

    assert stats.by_problem == {"clean": 500}
    assert len({k for k, _, _ in sent}) == 500


def test_problems_are_on_by_default_and_labelled(ref):
    sent, stats = run(ref, 5000)

    labels = Counter(h[PROBLEM_HEADER].decode() for _, _, h in sent)
    assert labels == Counter(stats.by_problem)
    assert set(labels) == {"clean", *PROBLEM_RATES}
    damaged = 1 - labels["clean"] / len(sent)
    assert 0.03 <= damaged <= 0.07


def test_duplicates_are_sent_twice_in_a_row_with_the_same_key(ref):
    sent, _ = run(ref, 5000)

    for previous, (key, value, headers) in zip(sent, sent[1:]):
        if headers[PROBLEM_HEADER] == b"duplicate":
            assert (key, value) == previous[:2]


def test_late_orders_are_one_to_three_days_old(ref):
    sent, _ = run(ref, 5000)
    late = [json.loads(v) for _, v, h in sent if h[PROBLEM_HEADER] == b"late"]

    assert late
    for order in late:
        event = datetime.strptime(order["event_time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - event
        assert timedelta(days=1) <= age <= timedelta(days=3, minutes=5)


def test_ctrl_c_stops_cleanly(ref, capsys):
    sent, send = collect()

    def sleep(_):
        raise KeyboardInterrupt

    stats = produce_orders(send, maker(ref, {}), rate=1, clock=FakeClock(), sleep=sleep)

    assert stats.sent == 1  # the first order goes immediately, then Ctrl+C during the wait
    assert "Stopped by Ctrl+C" in capsys.readouterr().out


def test_command_line():
    args = parse_args(["--rate", "2", "--count", "10"])
    assert (args.rate, args.count, args.seconds, args.clean) == (2.0, 10, None, False)
    assert parse_args(["--clean"]).clean
    with pytest.raises(SystemExit):
        parse_args(["--rate", "0"])
