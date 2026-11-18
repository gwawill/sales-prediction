# Tasks — Dell-style Sales Prediction AI (Proof of Concept)

Each task is one branch and one commit. Do them in order. A task is finished only when its
"Done when" check has actually been run and passed.

Requirement IDs such as FR-12 point to `spec.md`. Design details are in `plan.md`.

---

## T0 — Project setup

Create `pyproject.toml`, the folder layout from `plan.md`, `.gitignore`, `.env.example`,
`config.py` (settings from `.env`) and JSON logging.

Done when: `pip install -e .[dev]` works and `pytest` runs (with zero or one trivial test).

Note: this folder is not a Git repository yet. Run `git init` before this task.

## T1 — Cloud connection check

Write `scripts/check_cloud.py`. It tests each service with one small action: write and read a
file in S3, create and query a tiny S3 Vectors index, send and receive one Kafka message on Aiven,
run `RETURN 1` on Neo4j, and get one chat reply and one embedding.

Decide the embedding model here (OpenRouter or Bedrock Titan) and record the choice in `plan.md`.

Done when: the script prints OK for every service.

## T2 — Data generator (FR-1 to FR-5)

Generate reference data, 12 months of batch history with injected problems, and a function that
makes one live order at a time.

Done when: unit tests show the same seed gives the same output, and that each kind of problem
from FR-4 is present.

## T3 — Shared ETL transform (FR-9 to FR-15)

Pure functions for validate, clean, dedupe, join and quarantine, returning clean rows, rejected
rows with reasons, and counts.

Done when: unit tests cover every problem type from FR-4 and the reconciliation rule from FR-21.
No network is used.

## T4 — Data lake writer (FR-8, FR-16, FR-17, FR-20, FR-21)

Write raw, clean, quarantine and report files to S3 using the layout in `plan.md`, with the
read-merge-write idempotent load.

Done when: an integration test writes the same rows twice and the clean files are unchanged.

## T5 — Batch pipeline (FR-6)

A command that uploads a batch file to `raw/`, runs ETL and loads the lake.

Done when: 12 months of history load, the report adds up, and a second run changes nothing
(acceptance checks 2 and 3).

## T6 — Stream producer (FR-3, FR-4)

Send live orders to Aiven Kafka at a chosen rate, including some duplicates and late events.

Done when: messages can be seen arriving in the Aiven console or with a test consumer.

## T7 — Stream consumer (FR-7, FR-15)

Micro-batch consumer using the shared ETL, writing to the lake, committing offsets only after a
successful write.

Done when: acceptance checks 4 and 5 pass, including after stopping and restarting the consumer
in the middle of a run.

## T8 — Graph loader (FR-18)

Create Neo4j constraints and load nodes and relationships from the clean lake with `MERGE`.

Done when: a re-load does not change node counts, and acceptance check 6 passes.

## T9 — Vector loader (FR-19)

Build text summaries from the clean lake, create embeddings, and store them in S3 Vectors with
metadata. Add a search function.

Done when: searching "laptop sales in EMEA" returns EMEA laptop summaries at the top, and a
re-load does not add copies.

## T10 — Forecasting (FR-23 to FR-25)

Weekly aggregation, Holt-Winters and the baseline, a 4-week holdout error report, and saving the
forecast to S3.

Done when: acceptance check 7 passes and the error report prints for model and baseline.

## T11 — LLM client and RAG (FR-26, FR-28)

OpenRouter wrapper and a RAG function that answers only from retrieved summaries and lists them.

Done when: a sample question gets an answer that names the summaries it used.

## T12 — Agent and tools (FR-27 to FR-30)

Tool-calling loop with `sql_query`, `graph_query`, `vector_search` and `get_forecast`, with
read-only guards and a step limit.

Done when: unit tests show write queries are rejected, and the four example questions in
`spec.md` section 3 get sensible answers with sources (acceptance check 8).

## T13 — Web API (FR-31)

FastAPI app with `/ask`, `/forecast`, `/quality/latest` and `/health`.

Done when: each endpoint answers correctly when called locally.

## T14 — Dashboard (FR-32, optional)

Simple Streamlit page: sales over time, forecast, and a question box that calls the API.

Done when: the page loads and shows real data from the lake.

## T15 — Demo script and README

A `scripts/demo.py` that runs the acceptance checks in order, and a README explaining how to set
up and run everything. Also add a short note linking each part of the PoC to the matching section
of `dell.md`, for interview practice.

Done when: a fresh run of the demo script passes every acceptance check.
