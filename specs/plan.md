# Plan — Dell-style Sales Prediction AI (Proof of Concept)

This file says **how** we build what `spec.md` describes: which tools, how the parts fit, and
where the code lives. Requirement IDs such as FR-12 point back to `spec.md`.

---

## 1. The whole flow

```
  Data generator (Python script)
     |                         |
  Live orders             Batch files (12 months)
     |                         |
  Aiven Kafka              upload to S3 raw/
     |                         |
  Stream consumer          Batch runner
  (micro-batches)              |
     |                         |
     +------> shared ETL <-----+      validate -> clean -> dedupe -> join
                  |                   bad rows -> S3 quarantine/
                  |                   report   -> S3 reports/
     +------------+-------------+
     |            |             |
  S3 clean/    S3 Vectors     Neo4j AuraDB
  (Parquet)    (summaries)    (relationships)
     |            |             |
     +---- Forecast (reads clean/, writes S3 forecasts/)
                  |
          AI layer on OpenRouter
          RAG + agent with read-only tools
                  |
          FastAPI service  (+ optional dashboard)
```

## 2. Tool choices

| Need | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Matches dell.md and runs on Windows |
| Data handling | pandas + pyarrow | Small data, no Java needed |
| SQL on the lake | DuckDB | Runs SQL directly on Parquet files, no server |
| S3 access | boto3 | Official AWS library, also covers S3 Vectors |
| Kafka client | confluent-kafka | Has ready-made Windows builds, works with Aiven SSL |
| Graph | neo4j Python driver | Official driver for AuraDB |
| LLM | openai Python SDK pointed at OpenRouter | OpenRouter speaks the same API, including tool calling |
| Embeddings | Local `BAAI/bge-small-en-v1.5` (384 numbers per vector) run by fastembed | Decided in T1.5: free, no API key, text stays on the PC, no server or PyTorch needed |
| Forecast | statsmodels (Holt-Winters) with a simple baseline | Easy to explain, enough for weekly data |
| Web API | FastAPI + uvicorn | Matches dell.md |
| Settings | pydantic-settings + `.env` | One place for all keys and names |
| Fake data | Faker + numpy | Realistic names, repeatable with a seed |
| Tests | pytest | Standard |
| Dashboard (optional) | Streamlit | Fast to build in Python |

## 3. Folder layout

```
SalesPrediction-dell/
  dell.md
  CLAUDE.md
  specs/                 spec.md, plan.md, tasks.md
  pyproject.toml
  .env.example           names of every setting, no real values
  .gitignore             ignores .env, secrets/, data/, .venv/
  secrets/               Aiven certificate files (not committed)
  data/                  locally generated files (not committed)
  src/sales_poc/
    config.py            reads settings from .env
    log.py               JSON logging setup (not "logging.py", which would clash with Python's own module)
    datagen/             reference data, history, live orders
    etl/                 schema, validate, clean, dedupe, join (pure code, no cloud)
    lake/                S3 raw / clean / quarantine / reports read and write
    pipelines/           batch runner, stream producer, stream consumer
    graph/               Neo4j constraints and loader
    vectors/             summary builder, embeddings, S3 Vectors loader and search
    forecast/            weekly aggregation, model, evaluation
    ai/                  LLM client, RAG, agent, tools
    api/                 FastAPI app
    dashboard/           optional Streamlit app
  scripts/               small command-line entry points (check_cloud, demo)
  tests/
    unit/                no cloud needed
    integration/         needs real credentials, marked and skipped by default
```

## 4. Data model

Reference tables:

- `customers`: customer_id, name, segment (Consumer, SMB, Enterprise), region_id, country
- `products`: product_id, name, category (Laptop, Desktop, Monitor, Server, Storage, Accessory),
  list_price
- `regions`: region_id, name (NA, EMEA, APJ, LATAM)
- `channels`: channel_id, name (Online, Retail Partner, Direct Sales)

Transaction (one row per order line, after cleaning):

| Field | Meaning |
|---|---|
| transaction_id | Unique ID, used to remove duplicates |
| event_time | When the order happened (UTC) |
| event_date | Date part of event_time, used to split files |
| customer_id, product_id, channel_id | Links to reference data |
| quantity | Whole number, greater than 0 |
| unit_price, discount_pct, total_amount | Money values in USD |
| source | `batch` or `stream` |
| ingested_at | When our pipeline received it |
| quality_flags | List of values the ETL filled in, empty if none (FR-33) |
| + joined fields | segment, region, category, product name, channel name |

Target size: about 500 customers, 40 products and 25,000 to 30,000 transactions. This keeps
Neo4j inside the AuraDB Free limits.

## 5. Key design decisions

**Shared transform (FR-9).** All cleaning code lives in `etl/` as plain functions that take a
pandas table and return a clean table, a quarantine table and counts. Neither pipeline has its
own cleaning code. Because `etl/` touches no cloud service, it can be unit tested fully offline.

**Missing values (FR-33).** The fill rules live in one table inside `etl/clean.py`, so they are
easy to read and change. Each filled value adds a short tag to `quality_flags`, for example
`discount_filled` or `channel_unknown`. A row is only quarantined when a required field is
missing or a value is impossible, such as a negative quantity.

**Daily batch files and schedule (FR-2, FR-34).** The generator writes one file per day, named
`sales_YYYY-MM-DD.csv`, into a local `data/incoming/` folder. The batch command has two modes.
`--once` loads all new files and stops. `--every N` repeats that every N minutes until stopped.
To know which files are already loaded, it keeps a small list in S3 at
`s3://<bucket>/state/batch_loaded.json`. Even if that list is lost, re-loading is safe because
the load is idempotent (FR-17).

**S3 layout (FR-8, FR-14, FR-16, FR-20).**

```
s3://<bucket>/raw/batch/<file name>
s3://<bucket>/raw/stream/date=YYYY-MM-DD/<batch id>.jsonl
s3://<bucket>/clean/transactions/event_date=YYYY-MM-DD/data.parquet
s3://<bucket>/quarantine/run=<run id>/rows.parquet
s3://<bucket>/reports/run=<run id>.json
s3://<bucket>/forecasts/run=<run id>/forecast.parquet
```

**Idempotent load (FR-12, FR-15, FR-17).** For each date touched by a run: read the existing
clean file for that date, add the new rows, remove duplicates by `transaction_id`, and write the
file back whole. Running twice gives the same file. A late event simply touches an older date.
This read-merge-write approach is fine at PoC size and easy to explain. It would be too slow at
real Dell scale, where a table format such as Apache Iceberg or Delta Lake with MERGE would be
used instead. The README should say this honestly.

**Stream delivery (FR-7).** The consumer collects events until it has N of them or T seconds
pass, then runs ETL and writes to S3. It tells Kafka "done" (commits the offset) only after the
write succeeds. If it crashes, some events are read again, and the dedupe step removes them. In
short: at-least-once delivery plus dedupe gives each transaction exactly once in the lake.

**Aiven connection.** Aiven Kafka uses SSL with three files (CA certificate, client certificate,
client key). They live in `secrets/` and their paths are settings in `.env`.

**Neo4j model (FR-18).**

```
(:Customer)-[:LOCATED_IN]->(:Region)
(:Customer)-[:MADE]->(:Transaction)-[:FOR]->(:Product)
(:Transaction)-[:VIA]->(:Channel)
```

Every node has a uniqueness constraint on its ID, and the loader uses `MERGE`, so re-loading does
not create copies.

**Hybrid search (decided in T9.3).** Plain search by meaning put the wrong category first (for
"laptop sales in EMEA" it returned an Accessory summary first), because every summary shares one
sentence pattern. So `search` takes the 30 closest summaries by meaning, then ranks results
higher for each label the question names (category, region, month, segment, customer). On five
test questions the right summary came first 5/5 times, against 2/5 for plain search. Code:
`vectors/hybrid.py`. `vector-search --plain` shows plain search for comparison.

**Vector documents (FR-19).** The vector store holds short text summaries built from the clean
lake, not raw rows. For example: one per category, region and month, and one per customer. Each
summary is stored with metadata (region, category, month, segment) so search can be filtered.
The summary's ID is built from its keys, so re-loading overwrites instead of duplicating.

**Forecast (FR-23 to FR-25).** Group clean sales into weekly totals per category and region
(complete Monday-to-Sunday weeks only). Fit a "last 4 weeks average" baseline and three
exponential smoothing models: simple, Holt (damped trend) and Holt-Winters (damped trend +
13-week season). Measure error with a rolling backtest: the last 3 windows of 4 weeks, each
model trained only on the weeks before each window (decided in T10.3, so one lucky window
cannot pick the winner). The best model per series is chosen by WAPE (total error / total
sales), because MAPE cannot handle weeks with zero sales; MAPE is still reported where it can
be calculated.

**AI layer (FR-26 to FR-30).**

- One small `llm.py` wraps OpenRouter. The model name, temperature and limits are settings.
- `rag.py`: embed the question, search S3 Vectors for the top matches, and answer only from them.
- `agent.py`: a tool-calling loop with at most N steps. Tools:
  - `sql_query` runs DuckDB on the clean Parquet files and only allows a single `SELECT`.
  - `graph_query` runs Cypher inside a Neo4j read-only transaction.
  - `vector_search` searches the summaries.
  - `get_forecast` reads the latest forecast.
- Every answer returns the text plus a list of sources used.

**API (FR-31).** Async FastAPI with `POST /ask`, `GET /forecast`, `GET /quality/latest` and
`GET /health`.

## 6. Testing approach

- Unit tests cover all of `etl/`, the data generator's seed behaviour, the SQL safety check and
  the forecast aggregation. They need no network.
- Integration tests talk to real S3, Kafka, Neo4j and OpenRouter. They are marked
  `integration` and only run when asked (`pytest -m integration`).
- A `scripts/demo` command runs the acceptance checks from `spec.md` section 6 in order.

## 7. Risks

- **Free tier limits** on Aiven and AuraDB. Keep data small. Check limits in task T1.
- **S3 Vectors region.** Not in every region. Check in task T1.
- **LLM cost.** Use a cheap model by default and the agent step limit.
- **Embedding model is fixed once loaded.** The S3 Vectors index is created with 384
  dimensions for bge-small-en-v1.5. Changing the model means rebuilding the index. The model
  files live in `models/` (ignored by Git), downloaded on first use.
- **Interview honesty.** The real project most likely used a cloud embedding service. The PoC
  uses a local model to stay free; the idea (text in, meaning vector out) is the same.
- **Windows quirks.** Use `pathlib` for paths and avoid shell-specific scripts.
