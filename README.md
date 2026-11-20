# Sales prediction PoC

A small, working proof of concept (PoC) of a sales data and AI platform. Made-up sales from a
company that sells laptops, desktops, monitors, servers, storage and accessories flow in two
ways, as daily files (batch) and as live orders (streaming). They are cleaned, stored in three
places, forecast, and made available to plain-language questions.

```
 made-up data ──► daily CSV files ──► batch pipeline ─┐
              └─► live orders ──► Kafka ──► stream ───┤
                                                      ▼
                         ETL: fix formats, fill gaps, validate, join, de-duplicate
                                                      │
          ┌───────────────────────────────┬───────────┴────────────────┐
          ▼                               ▼                            ▼
   S3 data lake (Parquet)          Neo4j graph                S3 Vectors (summaries)
          │                               │                            │
          ├──► weekly forecast            │                            │
          ▼                               ▼                            ▼
   AI agent with four read-only tools (SQL, graph, search, forecast) and RAG answers
                                          │
                          FastAPI web API  +  Streamlit dashboard
```

Some words used below:

- **ETL** (extract, transform, load): reading raw data, cleaning it, and storing the result.
- **Data lake**: plain files in cloud storage (here, Amazon S3), organised by date.
- **Parquet**: a compact file format that stores a table column by column, with types.
- **Kafka**: a service that carries a stream of messages (here, live orders) between programs.
- **Graph database** (Neo4j): stores things and the links between them, such as
  customer → bought → product.
- **Vector store** (S3 Vectors): finds text by meaning, using numbers ("embeddings") that a
  language model computes for each piece of text.
- **RAG** (retrieval-augmented generation): find the most relevant text first, then let the
  language model answer only from it.
- **Agent**: a language model that can call tools (here, read-only queries) to answer bigger
  questions.

The full plan is in [`specs/`](specs): what to build ([spec.md](specs/spec.md)), how
([plan.md](specs/plan.md)) and in which steps ([tasks.md](specs/tasks.md)).

## What you need

- Python 3.11 or newer (developed on 3.14, on Windows). No Docker.
- Four cloud services. Every one has a free or very cheap option:

  | Service | Used for | Settings in `.env` |
  |---|---|---|
  | Amazon S3 and S3 Vectors | the data lake and the vector store | `AWS_*`, `S3_*` |
  | Aiven for Apache Kafka | the live order stream | `KAFKA_*`, files in `secrets/` |
  | Neo4j AuraDB Free | the graph | `NEO4J_*` |
  | OpenRouter | the language model | `OPENROUTER_API_KEY`, `LLM_MODEL` |

- Embeddings run on your own PC with the free `BAAI/bge-small-en-v1.5` model. It downloads by
  itself into `models/` the first time.

## Set up

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
copy .env.example .env          # then fill in the values; each one is explained in the file
```

Put Aiven's CA certificate in `secrets/ca.pem`. `.env`, `secrets/`, `data/` and `models/` are
never committed.

Then check that every service answers:

```powershell
.venv\Scripts\python scripts\check_cloud.py          # or name some: s3 kafka neo4j ...
```

Each check says OK or explains what to fix.

## Run it, step by step

The commands below are installed with the package (in `.venv\Scripts\`). Run them in this
order the first time.

| Step | Command | What it does |
|---|---|---|
| 1 | `generate-data` | Writes a year of made-up sales into `data/`: reference tables, one messy CSV per day (with deliberate problems such as bad dates and duplicates), and a log of every problem. The same seed always gives the same files. |
| 2 | `batch-load --once` | Loads the daily files into the S3 lake: raw copy, ETL, clean Parquet per day, rejected rows in quarantine, and a quality report that must add up. Files already loaded are skipped. `--every 60` runs it on a schedule. |
| 3 | `stream-produce --rate 2` and `stream-consume` | In two terminals: send live orders to Kafka, and load them into the lake in small batches. The consumer only tells Kafka a batch is done after it is safely stored. |
| 4 | `graph-load` | Copies the lake into Neo4j as customers, products, regions, channels and transactions. Running it again changes nothing. |
| 5 | `vector-load` | Writes monthly text summaries of the lake and stores them in S3 Vectors. |
| 6 | `forecast-run` | Forecasts the next 4 weeks of revenue for every category and region, choosing the best of four models by how well each did on past weeks. |
| 7 | `api-serve` | Starts the web API on http://127.0.0.1:8000 (see below). |
| 8 | `dashboard` | Starts the dashboard on http://127.0.0.1:8501 (needs the API for questions). |

Smaller commands for trying things out:

```powershell
vector-search "laptop sales in EMEA"                  # search the summaries
rag-ask "How did laptop sales in EMEA change in March 2026?"
sql-query "SELECT region_id, SUM(total_amount) FROM sales GROUP BY 1"
graph-query "MATCH (c:Customer)-[:LOCATED_IN]->(r:Region) RETURN r.region_id, count(c)"
agent-ask "Which enterprise customers who bought servers also bought storage?" --steps
```

## Simulate incoming data

Once steps 1 to 6 have run, you can show both ways data arrives, live, in separate terminals.

**Real-time orders.** The producer makes up new orders on the spot and sends them to Kafka.
About 5% of them have a deliberate problem (a duplicate, a late time, a missing value, ...).
The consumer cleans each small batch and stores it in the lake.

```powershell
stream-produce --rate 2          # 2 new orders per second, until Ctrl+C
stream-consume                   # in a second terminal
```

Useful producer options: `--count 100` or `--seconds 60` stop it automatically, `--seed 42`
makes the orders repeatable, and `--clean` sends no problems. Every command shows all its
options with `--help`.

**A new daily file.** Start the batch pipeline on a schedule. It checks `data/incoming/` every
minute and loads only files it has not seen before.

```powershell
batch-load --every 1             # in a third terminal
```

Then play the source system. Make the next day's file in a separate folder, and copy it into
the watched folder in one move, so the pipeline never sees a half-written file:

```powershell
generate-data --out data\new_day --start 2026-11-01 --days 1
copy data\new_day\incoming\sales_2026-11-01.csv data\incoming\
```

On its next check, `batch-load` loads that one file and writes a quality report for it. For
the day after, use `--start 2026-11-02`, and add `--overwrite` so `generate-data` may replace
the earlier file in `data\new_day\`.

**See the result.** Click **Reload data** in the dashboard to see the new sales in the charts.
The graph and the summaries do not update by themselves: run `graph-load` and `vector-load` so
the agent knows about the new data too.

## The web API and the dashboard

`api-serve` starts a FastAPI service. Its page http://127.0.0.1:8000/docs lets you try every
endpoint in the browser.

| Endpoint | Answer |
|---|---|
| `GET /health` | Is the service up? Add `?deep=true` to also test S3, S3 Vectors and Neo4j. |
| `GET /quality/latest` | The newest ETL quality report (`?source=batch` or `stream`). |
| `GET /forecast` | The latest 4-week forecast with each model's past error (`?category=`, `?region=`). |
| `POST /ask` | `{"question": "..."}`: the agent's answer and its sources. Add `"mode": "rag"` for a faster answer from the summaries only. |

If port 8000 is taken on your PC, set `API_PORT` in `.env`, or run `api-serve --port 8077`
and `dashboard --api http://127.0.0.1:8077`.

`dashboard` shows last month's numbers, monthly revenue by region, actual and forecast weekly
revenue for any category and region, and a question box. The first load reads the whole lake
into memory, which takes about 30 seconds.

## Safety: the AI layer only reads

The agent's tools cannot change data. The SQL tool accepts exactly one `SELECT` and runs it on
an in-memory copy, with file and network access switched off. The graph tool lets Neo4j check
each query without running it, accepts only read queries, and runs them in a read-only
transaction. The agent stops after `AGENT_MAX_STEPS` tool calls (default 6), and every answer
lists its sources: the SQL or Cypher it ran, the summaries it used, or the forecast run.

## Tests and the demo

```powershell
.venv\Scripts\python -m pytest                    # unit tests, no cloud needed (about 2 minutes)
.venv\Scripts\python -m pytest -m integration     # tests against the real services
.venv\Scripts\python scripts\demo.py              # every acceptance check, in order
```

`scripts/demo.py` runs the acceptance checks from [spec.md](specs/spec.md) section 6 and prints
PASS or FAIL for each one. It takes about 10 minutes and a few cents of language-model use
(`--skip-ai` leaves out the four agent questions; `--checks 1,2,3a` runs only some).

- Checks 1 to 5 (data generation, batch, re-runs, scheduled mode, flagged rows, streaming
  with duplicates and late orders, batch and stream overlap) are built fresh each time. The
  demo loads into its own folder in your bucket, `_demo/<run id>/`, and deletes it afterwards.
  Its Kafka consumer group starts at the end of the topic, so it reads only the demo's orders.
  Those orders do stay in the topic, so the next `stream-consume` loads them into the real
  lake, like any other live orders.
- Checks 6 to 8 read the live platform (the lake, Neo4j, the saved forecast and the agent),
  so run steps 1 to 6 above first. If you stream new orders, run `graph-load` again before the
  demo, or check 6 will find the graph a step behind the lake.
- Check 9 runs the unit tests with every cloud address pointed at nothing.

## How this differs from a real platform

This is a PoC. It shows how the parts fit together, not how they behave at full size.

- **Data size.** About 28,000 orders a year, made up by a script. A real company has millions
  of orders, messier data and many more source systems.
- **One computer.** Everything runs on one PC. A real platform would run the pipelines on
  managed services (for example AWS Glue, EMR or Kinesis) with an orchestrator such as Airflow,
  and the API in containers behind a load balancer.
- **SQL engine.** DuckDB reads the whole lake into memory, which only works because the data is
  small. At scale this would be Amazon Athena or Redshift querying the Parquet files in place.
- **No Spark.** pandas is enough here. Real volumes would need Spark or a similar engine.
- **Streaming.** One Kafka topic on a free plan, with micro-batches of a few seconds and
  "at least once" delivery made safe by removing duplicates. A real system would have more
  partitions, a schema registry, and monitoring of consumer lag.
- **Forecasting.** Simple statistical models (exponential smoothing) on 12 months of weekly
  totals. On past weeks they were off by 42% for a typical category and region (from 23% to
  89% across the 24 series), so treat the numbers as rough. Real forecasting would use years of history,
  promotions, prices and seasons, and would be retrained and checked regularly.
- **AI layer.** A small, cheap language model through OpenRouter and a local embedding model.
  There is no user login, no per-user data access rules and no cost limits per user, all of
  which a real deployment would need. Answers are only as good as the made-up data, so the
  agent can say what happened but cannot know real-world causes.
- **Monitoring.** JSON log lines and a quality report per ETL run stand in for tools such as
  Prometheus, Grafana or the ELK Stack.
- **Free tiers.** Neo4j AuraDB Free and the Aiven free plan have size and speed limits that a
  real system would outgrow quickly.
