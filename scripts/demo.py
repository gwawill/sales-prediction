"""Run the PoC's acceptance checks in order and say PASS or FAIL for each (task T15.1).

    python scripts/demo.py                   all checks (about 10 minutes, a few cents of LLM use)
    python scripts/demo.py --checks 1,2,3    only some checks
    python scripts/demo.py --skip-ai         everything except check 8 (the agent questions)
    python scripts/demo.py --keep            keep the demo's S3 folder afterwards, to look at it

The checks are the ones in specs/spec.md, section 6.

Checks 1 to 5 build everything fresh, so they mean the same on every run:
    - data is generated into a temporary folder on this PC;
    - the batch and stream pipelines load it into their own folder in the real S3 bucket,
      _demo/<run id>/, which is deleted at the end (unless --keep);
    - the stream check uses a brand-new Kafka consumer group that starts at the current end
      of the topic, so it reads only the orders this demo sends.
    Note: the demo's live orders stay in the real Kafka topic. Like any stream-produce run,
    they are loaded into the real lake the next time stream-consume runs.

Checks 6 to 8 look at the live platform: the real lake, the Neo4j graph, the saved forecast
and the agent. They only read. Build those first (see README: batch-load, graph-load,
vector-load, forecast-run).

Check 9 runs the unit tests with every cloud address pointed at nothing, so a test that
secretly needed the cloud would fail.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
STORED_KINDS = {"clean", "bad_format", "missing_fillable", "late", "duplicate"}   # live orders the lake keeps
REPLAY_LABEL = "batch_replay"
REPLAYED_ROWS = 5
# Live orders in check 4: the usual problem kinds, but far more often (about 5% each instead
# of 0.3-1.5%) and from a fixed seed, so every kind really appears in a short run.
DEMO_PROBLEM_RATE = 0.05
MUST_APPEAR = {"duplicate", "late", "bad_format", "missing_fillable", "missing_required"}

QUESTIONS = [
    "What were laptop sales in EMEA last month, and how does that compare to the month before?",
    "Which enterprise customers who bought servers also bought storage?",
    "What do we expect monitor sales in North America to be over the next four weeks?",
    "Why did APJ desktop sales drop in March?",
]


class CheckFailed(Exception):
    """A check ran and the result was wrong."""


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def say(line: str = "") -> None:
    print(f"    {line}", flush=True)


# --- shared state, built once and reused by later checks -----------------------------------

@dataclass
class Demo:
    run_id: str
    workdir: Path
    orders: int
    keep: bool
    _store: object = None
    _ref: object = None
    _loaded: bool = False
    _stream: dict | None = None
    _lake_rows: object = None
    _sql: object = None
    facts: dict = field(default_factory=dict)       # results later checks compare against

    @property
    def data(self) -> Path:
        return self.workdir / "data"

    @property
    def store(self):
        """The demo's own folder in the real S3 bucket."""
        if self._store is None:
            from sales_poc.lake.store import s3_store
            self._store = s3_store(prefix=f"_demo/{self.run_id}/")
        return self._store

    def generate(self) -> dict:
        from sales_poc.datagen.cli import generate
        if not (self.data / "generation.json").exists():
            return generate(self.data, SEED)
        return json.loads((self.data / "generation.json").read_text())

    @property
    def ref(self):
        if self._ref is None:
            from sales_poc.datagen.reference import load_reference_data
            self.generate()
            self._ref = load_reference_data(self.data / "reference")
        return self._ref

    def files(self) -> list[Path]:
        from sales_poc.pipelines.batch import find_daily_files
        self.generate()
        return find_daily_files(self.data / "incoming")

    def batch_loaded(self):
        """Load the history once (check 2 does it; later checks need it)."""
        from sales_poc.pipelines.batch import load_new_files
        if not self._loaded:
            run, skipped = load_new_files(self.store, self.files(), self.ref)
            self.facts["first_batch"] = run
            self._loaded = True
        return self.facts["first_batch"]

    def real_lake(self):
        """The real lake, read once, as a read-only SQL tool (checks 6 and 8)."""
        if self._sql is None:
            from sales_poc.ai.tools.sql import SqlTool
            from sales_poc.lake.clean import read_all_clean
            from sales_poc.lake.store import s3_store
            self._lake_rows = read_all_clean(s3_store())
            self._sql = SqlTool(rows=self._lake_rows)
        return self._sql

    def clean_up(self) -> None:
        if self._store is None or self.keep:
            if self.keep and self._store is not None:
                print(f"\nKept the demo's S3 folder: _demo/{self.run_id}/")
            return
        keys = self.store.list()
        prefix = self.store.prefix
        for start in range(0, len(keys), 1000):
            chunk = [{"Key": prefix + k} for k in keys[start:start + 1000]]
            self.store.client.delete_objects(Bucket=self.store.bucket, Delete={"Objects": chunk, "Quiet": True})
        print(f"\nDeleted the demo's S3 folder _demo/{self.run_id}/ ({len(keys)} files).")


def read_days(store, days) -> "pd.DataFrame":
    """Several clean days from the lake, read in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd

    from sales_poc.lake.clean import read_clean_day

    with ThreadPoolExecutor(16) as pool:
        return pd.concat(list(pool.map(lambda d: read_clean_day(store, d), days)), ignore_index=True)


def folder_fingerprint(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


# --- the checks ----------------------------------------------------------------------------

def check_1(demo: Demo) -> None:
    """One command generates data, and the same seed gives the same files."""
    from sales_poc.datagen.cli import main as generate_data

    second = demo.workdir / "again"
    started = time.monotonic()
    expect(generate_data(["--seed", str(SEED), "--out", str(demo.data)]) == 0, "generate-data failed")
    say(f"generate-data took {time.monotonic() - started:.0f}s")
    expect(generate_data(["--seed", str(SEED), "--out", str(second)]) == 0, "second generate-data failed")
    a, b = folder_fingerprint(demo.data), folder_fingerprint(second)
    summary = demo.generate()
    say(f"{summary['daily_files']} daily files, {summary['rows_in_files']} rows, "
        f"{summary['problems']} deliberate problems; {len(a)} files in all")
    different = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    expect(not different, f"the same seed gave different files: {different[:5]}")
    say(f"generated twice with seed {SEED}: all {len(a)} files are byte-for-byte the same")


def check_2(demo: Demo) -> None:
    """The batch pipeline loads 12 months of history into S3, and the quality report adds up."""
    from sales_poc.lake.clean import list_clean_days
    from sales_poc.lake.runs import latest_report

    started = time.monotonic()
    run = demo.batch_loaded()
    expect(run is not None, "the batch loaded nothing")
    c = run.report["counts"]
    say(f"loaded {len(run.files)} files into s3://{demo.store.bucket}/_demo/{demo.run_id}/ "
        f"in {time.monotonic() - started:.0f}s")
    say(f"rows in {c['rows_in']} = clean {c['clean']} + duplicates {c['duplicates_removed']} "
        f"+ quarantined {c['quarantined']}  (flagged but kept: {c['clean_flagged']})")
    expect(c["rows_in"] == c["clean"] + c["duplicates_removed"] + c["quarantined"], "the report does not add up")
    saved = latest_report(demo.store, "batch")
    expect(saved is not None and saved["counts"] == c, "the saved report differs from the run")
    days = list_clean_days(demo.store)
    months = sorted({d.strftime("%Y-%m") for d in days})
    say(f"lake: {len(days)} days in {len(months)} months ({months[0]} to {months[-1]})")
    expect(len(months) == 12, f"expected 12 months, found {len(months)}")


def check_3(demo: Demo) -> None:
    """A second run changes nothing; scheduled mode picks up a new file and skips loaded ones."""
    from sales_poc.datagen.cli import generate
    from sales_poc.pipelines.batch import load_new_files, run_batch, run_every

    demo.batch_loaded()
    files = demo.files()
    again = run_batch(demo.store, files, demo.ref)           # deliberately ignore "already loaded"
    lake = again.report["lake"]
    say(f"re-ran all {len(files)} files: {lake['rows_added']} added, {lake['rows_replaced']} replaced, "
        f"{lake['rows_unchanged']} unchanged, {len(lake['days_written'])} days rewritten")
    expect(lake["rows_added"] == 0 and lake["rows_replaced"] == 0 and not lake["days_written"],
           "a second run changed the clean data")

    # Scheduled mode: two passes; between them a new day's file arrives (the "sleep" drops it in).
    summary = demo.generate()
    new_day = date.fromisoformat(summary["end"]) + timedelta(days=1)
    extra = demo.workdir / "new_day"
    generate(extra, SEED, start=new_day, days=1)
    passes = []

    def one_pass():
        passes.append(load_new_files(demo.store, demo.files(), demo.ref))

    def new_file_arrives(_seconds):
        (demo.data / "incoming" / f"sales_{new_day}.csv").write_bytes(
            (extra / "incoming" / f"sales_{new_day}.csv").read_bytes())

    failures = run_every(one_pass, every_seconds=0, max_passes=2, sleep=new_file_arrives)
    expect(failures == 0, "a scheduled pass failed")
    (run1, skipped1), (run2, skipped2) = passes
    say(f"scheduled pass 1: {0 if run1 is None else len(run1.files)} loaded, {len(skipped1)} skipped")
    say(f"new file sales_{new_day}.csv dropped in")
    say(f"scheduled pass 2: {0 if run2 is None else len(run2.files)} loaded "
        f"({'' if run2 is None else run2.files[0].name}), {len(skipped2)} skipped")
    expect(run1 is None and len(skipped1) == len(files), "pass 1 should load nothing")
    expect(run2 is not None and [p.name for p in run2.files] == [f"sales_{new_day}.csv"],
           "pass 2 should load only the new file")
    expect(len(skipped2) == len(files), "pass 2 should skip every file loaded before")


def check_3a(demo: Demo) -> None:
    """A row with only a missing discount or channel is kept and flagged, not quarantined."""
    import pandas as pd

    from sales_poc.lake.clean import read_clean_day
    from sales_poc.lake.runs import read_quarantine

    run = demo.batch_loaded()
    log = pd.read_csv(demo.data / "problem_log.csv", dtype=str, keep_default_na=False)
    once = log["transaction_id"].map(log["transaction_id"].value_counts()) == 1     # only one problem
    picked = log[once & (log["problem"] == "missing_fillable") & log["detail"].isin(["discount_pct", "channel_id"])]
    expect(len(picked) > 0, "the generated data has no such rows to check")
    days = sorted({date.fromisoformat(d) for d in picked["event_day"]})
    clean = read_days(demo.store, days)
    found = clean[clean["transaction_id"].isin(picked["transaction_id"])]
    quarantined = read_quarantine(demo.store, run.run_id) if run.report.get("quarantine_key") else pd.DataFrame()
    in_quarantine = (set(quarantined.get("transaction_id", [])) & set(picked["transaction_id"]))
    flagged = found[found["quality_flags"].fillna("") != ""]
    by_field = picked["detail"].value_counts().to_dict()
    say(f"{len(picked)} rows had only a missing discount or channel {by_field}")
    say(f"in the clean lake: {len(found)}, flagged: {len(flagged)}, quarantined: {len(in_quarantine)}")
    say(f"example flags: {sorted(set(flagged['quality_flags']))[:3]}")
    expect(len(found) == len(picked) == len(flagged) and not in_quarantine,
           "some of these rows were dropped, not flagged, or quarantined")


def start_group_at_end(group: str) -> int:
    """Commit the topic's current end for a new consumer group, so it reads only newer messages."""
    from confluent_kafka import Consumer, TopicPartition

    from sales_poc.config import get_kafka_settings
    from sales_poc.kafka import connection_config

    s = get_kafka_settings()
    c = Consumer({**connection_config(s), "group.id": group, "enable.auto.commit": False, "log_level": 0})
    try:
        partitions = c.list_topics(s.kafka_topic, timeout=20).topics[s.kafka_topic].partitions
        ends = [TopicPartition(s.kafka_topic, p, c.get_watermark_offsets(TopicPartition(s.kafka_topic, p),
                                                                          timeout=20)[1]) for p in partitions]
        c.commit(offsets=ends, asynchronous=False)
        return len(ends)
    finally:
        c.close()


def run_stream(demo: Demo) -> dict:
    """Send live orders (and a few replayed batch rows) while the consumer loads them.
    The consumer is stopped after its first batch and started again, as after a crash."""
    import pandas as pd

    from sales_poc.config import get_kafka_settings
    from sales_poc.datagen.live import LiveOrderMaker
    from sales_poc.datagen.problems import PROBLEM_RATES
    from sales_poc.lake.clean import read_clean_day
    from sales_poc.pipelines.batch import read_daily_file
    from sales_poc.pipelines.consumer import consume_loop, make_consumer, process_batch
    from sales_poc.pipelines.producer import PROBLEM_HEADER, KafkaSender, produce_orders

    if demo._stream is not None:
        return demo._stream
    demo.batch_loaded()
    topic = get_kafka_settings().kafka_topic
    group = f"sales-poc-demo-{demo.run_id}"
    partitions = start_group_at_end(group)

    # Rows the batch already stored, to send again through Kafka (check 5).
    last_file = demo.files()[-1]
    log = pd.read_csv(demo.data / "problem_log.csv", dtype=str, keep_default_na=False)
    raw = read_daily_file(last_file)
    raw = raw[~raw["transaction_id"].isin(log["transaction_id"])].head(REPLAYED_ROWS)
    replay_day = date.fromisoformat(last_file.stem.removeprefix("sales_"))

    sent: list[tuple[str | None, str]] = []
    sender = KafkaSender()

    def send(key, value, headers):
        sent.append((json.loads(value).get("transaction_id"), dict(headers)[PROBLEM_HEADER].decode()))
        sender(key, value, headers)

    def producer():
        rates = {kind: DEMO_PROBLEM_RATE for kind in PROBLEM_RATES}
        produce_orders(send, LiveOrderMaker(demo.ref, seed=SEED, rates=rates), rate=10, count=demo.orders)
        for row in raw.to_dict("records"):
            payload = {k: ("" if pd.isna(v) else v) for k, v in row.items()}
            send(row["transaction_id"].encode(), json.dumps(payload).encode(),
                 [(PROBLEM_HEADER, REPLAY_LABEL.encode())])
        sender.close()

    thread = threading.Thread(target=producer)
    thread.start()
    process = lambda batch: process_batch(demo.store, batch, demo.ref)  # noqa: E731
    first = make_consumer(group_id=group)
    stats1 = consume_loop(first, process, topic, batch_size=40, batch_seconds=4, max_batches=1)
    first.close()   # stop, as if the consumer crashed after its first batch
    second = make_consumer(group_id=group)
    stats2 = consume_loop(second, process, topic, batch_size=40, batch_seconds=4, idle_batches=3)
    second.close()
    thread.join()

    today = datetime.now(timezone.utc).date()
    days = [today - timedelta(days=n) for n in range(0, 5)]
    lake = read_days(demo.store, days)
    demo._stream = {
        "group": group, "partitions": partitions, "sent": sent, "delivered": sender.delivered,
        "failed": sender.failed, "consumed": (stats1, stats2), "lake": lake,
        "replayed": list(raw["transaction_id"]), "replay_day": replay_day,
    }
    return demo._stream


def check_4(demo: Demo) -> None:
    """Live orders with duplicates and late events are each stored exactly once."""
    stream = run_stream(demo)
    live = [(tx, label) for tx, label in stream["sent"] if label != REPLAY_LABEL]
    labels = Counter(label for _, label in live)
    s1, s2 = stream["consumed"]
    say(f"consumer group {stream['group']} started at the end of {stream['partitions']} partitions")
    say(f"sent {len(stream['sent'])} messages, Kafka confirmed {stream['delivered']}; by kind: {dict(labels)}")
    say(f"consumer: {s1.messages} messages in its first batch, then stopped and restarted; "
        f"{s2.messages} more in {s2.batches} batches")
    absent = sorted(MUST_APPEAR - set(labels))
    expect(not absent, f"this run sent no {absent} orders, so it would not test them; use more --orders")
    expect(stream["failed"] == 0 and stream["delivered"] == len(stream["sent"]), "Kafka did not confirm every message")
    expect(s1.messages + s2.messages == len(stream["sent"]), "the consumer did not read every message exactly once")
    prefix = next(tx for tx, _ in live if tx).rsplit("-", 1)[0]
    expected = {tx for tx, label in live if label in STORED_KINDS and tx}
    lake = stream["lake"]
    ours = lake[lake["transaction_id"].str.startswith(prefix)]
    counts = Counter(ours["transaction_id"])
    missing, extra = expected - set(counts), set(counts) - expected
    say(f"expected in the lake: {len(expected)}; found {len(ours)} rows, "
        f"{len(counts)} distinct; missing {len(missing)}, unexpected {len(extra)}")
    late = [tx for tx, label in live if label == "late"]
    if late:
        dates = sorted({str(d) for d in ours[ours["transaction_id"].isin(late)]["event_date"]})
        say(f"late orders were stored under their own (earlier) dates: {dates}")
    expect(not missing and not extra, f"missing {sorted(missing)[:3]}, unexpected {sorted(extra)[:3]}")
    expect(max(counts.values(), default=1) == 1, "a transaction was stored more than once")


def check_5(demo: Demo) -> None:
    """The same transaction loaded by batch and by stream appears exactly once."""
    from sales_poc.lake.clean import read_clean_day

    stream = run_stream(demo)
    rows = read_clean_day(demo.store, stream["replay_day"])
    hits = rows[rows["transaction_id"].isin(stream["replayed"])]
    say(f"sent {len(stream['replayed'])} rows from {stream['replay_day']}'s batch file through Kafka again")
    say(f"in the lake: {len(hits)} rows for them, sources {sorted(set(hits['source']))}; "
        f"day {stream['replay_day']} has {len(rows)} rows, all transaction IDs unique: "
        f"{rows['transaction_id'].is_unique}")
    expect(len(hits) == len(stream["replayed"]), "a replayed transaction is missing or doubled")
    expect(set(hits["source"]) == {"batch"}, "the identical stream copy should not replace the batch row")
    expect(rows["transaction_id"].is_unique, "the day has a transaction more than once")


CHECK_6_SQL = """
SELECT string_agg(customer_id, ',' ORDER BY customer_id) FROM (
  SELECT DISTINCT customer_id FROM sales WHERE segment = 'Enterprise' AND category = 'Server'
  INTERSECT
  SELECT DISTINCT customer_id FROM sales WHERE segment = 'Enterprise' AND category = 'Storage')
"""
CHECK_6_CYPHER = """
MATCH (c:Customer {segment: 'Enterprise'})-[:MADE]->(:Transaction)-[:FOR]->(:Product {category: 'Server'})
WHERE EXISTS { (c)-[:MADE]->(:Transaction)-[:FOR]->(:Product {category: 'Storage'}) }
RETURN DISTINCT c.customer_id AS customer_id ORDER BY customer_id
"""


def check_6(demo: Demo) -> None:
    """Neo4j's answer to "enterprise customers who bought servers and storage" matches SQL on the lake."""
    from sales_poc.graph.connection import driver

    started = time.monotonic()
    sql = demo.real_lake()
    [(lake_rows,)] = sql.run("SELECT COUNT(*) FROM sales").rows
    say(f"read the real lake: {lake_rows} rows in {time.monotonic() - started:.0f}s")
    [(lake_list,)] = sql.run(CHECK_6_SQL).rows
    lake_ids = lake_list.split(",") if lake_list else []
    with driver() as d:
        records, _, _ = d.execute_query(CHECK_6_CYPHER)
        graph_ids = [r["customer_id"] for r in records]
        graph_rows = d.execute_query("MATCH (t:Transaction) RETURN count(t) AS n").records[0]["n"]
    say(f"graph holds {graph_rows} transactions{'' if graph_rows == lake_rows else ' (run graph-load to catch up)'}")
    say(f"SQL on the lake: {len(lake_ids)} customers; Cypher on Neo4j: {len(graph_ids)} customers")
    expect(graph_ids, "the graph found no such customers")
    expect(lake_ids == graph_ids, "the two answers differ (run graph-load after the latest lake load)")
    say("the two lists are identical")
    demo.facts["check_6_count"] = len(graph_ids)


def check_7(demo: Demo) -> None:
    """The forecast covers every category and region for 4 weeks and shows its error."""
    import math

    from sales_poc.forecast.publish import latest_forecast
    from sales_poc.lake.store import s3_store

    found = latest_forecast(s3_store())
    expect(found is not None, "no forecast saved (run forecast-run)")
    table, summary = found
    series = table.groupby(["category", "region_id"])["week_start"].nunique()
    say(f"forecast run {summary['run_id']}, weeks {', '.join(summary['forecast_weeks'])}")
    say(f"{len(series)} series (6 categories x 4 regions), each with {sorted(set(series))} weeks")
    expect(len(series) == 24 and set(series) == {4}, "not every series has 4 weeks")
    errors_ok = all(not math.isnan(v) for v in table["backtest_wape"])
    expect(errors_ok, "some rows have no backtest error")
    overall = ", ".join(f"{m['model']} {m['overall_wape']:.0%}" for m in summary["models"])
    say(f"every row carries its model's backtest error (WAPE); overall: {overall}")
    say(f"the chosen model beats the baseline in {summary['best_beats_baseline']} of {summary['series']} series")


def check_8(demo: Demo) -> None:
    """Each example question gets a sensible answer with sources listed."""
    from sales_poc.ai.agent import Agent, ToolBox, default_makers
    from sales_poc.ai.tools.sql import TOOL_SPEC

    sql = demo.real_lake()
    makers = default_makers()
    makers["sql_query"] = (TOOL_SPEC, lambda: sql)       # reuse the lake already in memory
    agent = Agent(tools=ToolBox(makers))
    problems = []
    try:
        for n, question in enumerate(QUESTIONS, start=1):
            started = time.monotonic()
            answer = agent.ask(question)
            kinds = sorted({s.tool for s in answer.sources})
            say(f"Q{n}. {question}")
            for line in answer.text.splitlines():
                if line.strip():
                    say(f"      {line.strip()}")
            say(f"      sources: {len(answer.sources)} ({', '.join(kinds) or 'none'}); "
                f"{len(answer.steps)} tool calls, {answer.usage.total_tokens} tokens, {time.monotonic() - started:.0f}s")
            if not answer.text or not answer.sources:
                problems.append(f"Q{n} has no answer or no sources")
            if answer.hit_step_limit:
                problems.append(f"Q{n} hit the step limit")
            if n == 1 and "sql_query" not in kinds:
                problems.append("Q1 should use the order data (sql_query)")
            if n == 2 and "check_6_count" in demo.facts and str(demo.facts["check_6_count"]) not in answer.text:
                problems.append(f"Q2 should give the count from check 6 ({demo.facts['check_6_count']})")
            if n == 3 and "get_forecast" not in kinds:
                problems.append("Q3 should use the saved forecast (get_forecast)")
    finally:
        agent.close()
    say("Read the answers above too: the script checks sources and key facts, a person judges 'sensible'.")
    expect(not problems, "; ".join(problems))


def check_9(demo: Demo) -> None:
    """Unit tests for the transform logic pass without any cloud connection."""
    nowhere = {
        "AWS_PROFILE": "demo-no-cloud", "AWS_CONFIG_FILE": os.devnull, "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
        "NEO4J_URI": "neo4j://127.0.0.1:9", "KAFKA_BOOTSTRAP_SERVERS": "127.0.0.1:9",
        "OPENROUTER_BASE_URL": "http://127.0.0.1:9/v1", "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9", "NO_PROXY": "",
    }
    started = time.monotonic()
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "not integration", "-p", "no:cacheprovider"],
                            cwd=ROOT, env={**os.environ, **nowhere}, capture_output=True, text=True)
    last = [line for line in result.stdout.splitlines() if line.strip()][-1:] or ["(no output)"]
    say("cloud addresses pointed at nothing: AWS profile, Neo4j, Kafka, OpenRouter, and any web proxy")
    say(f"pytest (unit tests only): {last[0]}  [{time.monotonic() - started:.0f}s]")
    if result.returncode != 0:
        for line in result.stdout.splitlines()[-15:]:
            say(f"  {line}")
    expect(result.returncode == 0, "some unit tests failed")


CHECKS: list[tuple[str, Callable[[Demo], None]]] = [
    ("1", check_1), ("2", check_2), ("3", check_3), ("3a", check_3a), ("4", check_4),
    ("5", check_5), ("6", check_6), ("7", check_7), ("8", check_8), ("9", check_9),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="demo", description="Run the PoC's acceptance checks.")
    parser.add_argument("--checks", help="comma-separated, e.g. 1,2,3a (default: all)")
    parser.add_argument("--skip-ai", action="store_true", help="skip check 8 (the agent questions)")
    parser.add_argument("--keep", action="store_true", help="keep the demo's S3 folder afterwards")
    parser.add_argument("--orders", type=int, default=80, help="live orders to send in check 4 (default 80)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True, errors="replace")

    names = [n for n, _ in CHECKS]
    chosen = args.checks.split(",") if args.checks else names
    unknown = [c for c in chosen if c not in names]
    if unknown:
        parser.error(f"unknown checks {unknown}; choose from {names}")
    if args.skip_ai and "8" in chosen:
        chosen.remove("8")

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    results: list[tuple[str, str, float]] = []
    print(f"Sales PoC demo {run_id}: checks {', '.join(chosen)}")
    with tempfile.TemporaryDirectory(prefix="sales-poc-demo-") as tmp:
        demo = Demo(run_id, Path(tmp), args.orders, args.keep)
        try:
            for name, check in CHECKS:
                if name not in chosen:
                    continue
                print(f"\n[{name}] {check.__doc__.strip()}")
                started = time.monotonic()
                try:
                    check(demo)
                    outcome = "PASS"
                except CheckFailed as exc:
                    outcome = "FAIL"
                    say(f"FAILED: {exc}")
                except Exception as exc:  # noqa: BLE001 - report it and go on to the next check
                    outcome = "ERROR"
                    say(f"ERROR: {type(exc).__name__}: {exc}")
                seconds = time.monotonic() - started
                results.append((name, outcome, seconds))
                print(f"  -> {outcome} ({seconds:.0f}s)")
        finally:
            demo.clean_up()

    print("\nSummary")
    for name, outcome, seconds in results:
        print(f"  check {name:<3} {outcome:<5} {seconds:6.0f}s")
    passed = sum(outcome == "PASS" for _, outcome, _ in results)
    print(f"{passed} of {len(results)} checks passed.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
