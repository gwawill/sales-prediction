import json
from datetime import datetime, timezone

import pytest

from sales_poc.datagen.history import COLUMNS
from sales_poc.datagen.live import LiveOrderMaker
from sales_poc.datagen.reference import build_reference_data
from sales_poc.pipelines.producer import encode, parse_args, produce_orders

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
    return LiveOrderMaker(ref, seed=1, rates={} if rates is None else rates, started_at=START)


def collect():
    sent = []
    return sent, lambda key, value: sent.append((key, value))


def test_message_key_is_the_transaction_id_and_value_is_json(ref):
    order = maker(ref).next_order()
    key, value = encode(order)

    assert key == order.payload["transaction_id"].encode()
    decoded = json.loads(value)
    assert decoded == order.payload
    assert list(decoded) == COLUMNS


def test_missing_transaction_id_means_no_key(ref):
    order = maker(ref).next_order()
    order.payload["transaction_id"] = None
    key, value = encode(order)

    assert key is None
    assert json.loads(value)["transaction_id"] is None


def test_count_limit(ref):
    sent, send = collect()
    clock = FakeClock()

    stats = produce_orders(send, maker(ref), rate=10, count=25, clock=clock, sleep=clock.sleep)

    assert stats.sent == len(sent) == 25
    assert stats.by_problem == {"clean": 25}


def test_rate_is_kept_without_drift(ref):
    sent, send = collect()
    clock = FakeClock()

    produce_orders(send, maker(ref), rate=4, count=9, clock=clock, sleep=clock.sleep)

    assert clock.now == pytest.approx(2.0)  # 9 orders at 4/s: the last is due at 8/4 = 2s


def test_seconds_limit(ref):
    sent, send = collect()
    clock = FakeClock()

    produce_orders(send, maker(ref), rate=5, seconds=3, clock=clock, sleep=clock.sleep)

    assert len(sent) == 15  # due at 0, 0.2, ... 2.8


def test_clean_by_default_in_this_version(ref):
    sent, send = collect()
    clock = FakeClock()

    produce_orders(send, maker(ref), rate=100, count=500, clock=clock, sleep=clock.sleep)

    keys = [k for k, _ in sent]
    assert len(set(keys)) == 500  # no duplicates
    assert all(json.loads(v)["quantity"] for _, v in sent)


def test_ctrl_c_stops_cleanly(ref, capsys):
    sent, send = collect()

    def sleep(_):
        raise KeyboardInterrupt

    stats = produce_orders(send, maker(ref), rate=1, clock=FakeClock(), sleep=sleep)

    assert stats.sent == 1  # the first order goes immediately, then Ctrl+C during the wait
    assert "Stopped by Ctrl+C" in capsys.readouterr().out


def test_command_line():
    args = parse_args(["--rate", "2", "--count", "10"])
    assert (args.rate, args.count, args.seconds) == (2.0, 10, None)
    with pytest.raises(SystemExit):
        parse_args(["--rate", "0"])
