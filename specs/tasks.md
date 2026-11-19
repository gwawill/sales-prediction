# Tasks — Dell-style Sales Prediction AI (Proof of Concept)

The work is split into tasks (T0, T1, ...), and each task is split into subtasks (T0.1, T0.2, ...).

**Each subtask is one commit.** Do subtasks in order. A subtask is finished only when its
"Done when" check has actually been run and passed. After each subtask, stop, report, and give
the commit message.

Requirement IDs such as FR-12 point to `spec.md`. Design details are in `plan.md`.

---

## T0 — Project setup

- **T0.1 — Package skeleton.** `pyproject.toml`, `src/sales_poc/` with empty sub-packages from
  `plan.md`, `tests/` folders, `.gitignore`.
  Done when: `pip install -e .[dev]` works.
- **T0.2 — Settings.** `config.py` reading `.env`, plus `.env.example` listing every setting.
  Done when: a unit test loads settings from a sample env and fails clearly when one is missing.
- **T0.3 — JSON logging.** `log.py` that prints one JSON object per line.
  Done when: a unit test checks a log line is valid JSON with time, level and message.
- **T0.4 — Test setup.** pytest config with the `integration` marker skipped by default.
  Done when: `pytest` runs unit tests and skips integration tests.
- **T0.5 — Settings split into groups.** Added after T0.4: one settings group per service (AWS,
  vector store, Kafka, Neo4j, LLM, embeddings, app), each loaded only when needed, so one
  service can be tested before the others are set up.
  Done when: unit tests show each group loads with only its own values.

## T1 — Cloud connection check

- **T1.1 — S3 check.** Write, read and delete a small file.
- **T1.2 — S3 Vectors check.** Create a tiny index, add one vector, query it, delete it.
- **T1.3 — Aiven Kafka check.** Send and receive one message using the SSL files in `secrets/`.
- **T1.4 — Neo4j check.** Run `RETURN 1` on AuraDB.
- **T1.5 — Local embedding check.** Decision: local `bge-small-en-v1.5` via fastembed (recorded
  in `plan.md`). The check embeds three sentences and confirms the vector size is 384 and that
  similar sentences score closer than unrelated ones.
- **T1.6 — OpenRouter chat check.** Get one short chat reply from `LLM_MODEL`.
- **T1.7 — One command.** `scripts/check_cloud.py` runs all checks and prints OK or the error.

Done when (each subtask): its check prints OK against the real service.

## T2 — Data generator (FR-1 to FR-5)

- **T2.1 — Reference data.** Customers, products, regions, channels with stable IDs.
  Done when: unit test shows same seed gives same tables.
- **T2.2 — Clean daily history.** 12 months of transactions, one CSV file per day, with
  realistic seasonality.
  Done when: unit test shows one file per day and same seed gives same files.
- **T2.3 — Injected problems.** Duplicates, missing values, bad formats, unknown IDs, negative
  quantities, late events (FR-4).
  Done when: unit test finds each problem type in the output.
- **T2.4 — Live order maker.** A function that returns one new order at a time, with the same
  kinds of problems.
  Done when: unit test checks the order shape and that problems appear at the set rate.
- **T2.5 — Command.** `generate-data` command that writes everything to `data/`.
  Done when: running it creates the expected files.

## T3 — Shared ETL transform (FR-9 to FR-15, FR-33)

All pure functions, no network.

- **T3.1 — Schema and validation.** Required fields, types and ranges (FR-10).
- **T3.2 — Format cleaning.** Dates, currency strings, text case (FR-11).
- **T3.3 — Fill or flag missing values.** Rules from FR-33, with `quality_flags`.
- **T3.4 — Duplicates.** Remove by `transaction_id` (FR-12).
- **T3.5 — Joins.** Attach customer, product, region and channel details (FR-13).
- **T3.6 — Quarantine and counts.** Rejected rows with reasons, and the reconciliation rule
  (FR-14, FR-21).
- **T3.7 — One entry point.** A single `run_etl(rows)` that chains all steps.

Done when (each subtask): its unit tests pass.

## T4 — Data lake writer (FR-8, FR-16, FR-17, FR-20, FR-21)

- **T4.1 — Raw writer.** Save inputs unchanged to `raw/`.
- **T4.2 — Clean writer.** Read-merge-write per date partition, deduped by `transaction_id`.
- **T4.3 — Quarantine and report writers.**

Done when: an integration test writes the same rows twice and the clean files are unchanged.

## T5 — Batch pipeline (FR-6, FR-34)

- **T5.1 — Run once.** Upload new daily files, run ETL, load the lake.
  Done when: 12 months load and the report adds up (acceptance check 2).
- **T5.2 — Loaded-file list.** Remember loaded files in `state/batch_loaded.json` and skip them.
  Done when: a second run loads nothing and changes nothing (acceptance check 3).
- **T5.3 — Scheduled mode.** `--every N` repeats the run.
  Done when: a new daily file dropped in is picked up on the next run (acceptance checks 3, 3a).

## T6 — Stream producer (FR-3, FR-4)

- **T6.1 — Producer.** Send live orders to Aiven Kafka at a chosen rate.
- **T6.2 — Problem events.** Add duplicates and late events on purpose.

Done when: messages can be seen arriving with a test consumer.

## T7 — Stream consumer (FR-7, FR-15)

- **T7.1 — Micro-batch reading.** Collect N events or wait T seconds.
- **T7.2 — ETL and load.** Run the shared ETL and write to the lake.
- **T7.3 — Safe offset commit.** Tell Kafka "done" only after the write succeeds.

Done when: acceptance checks 4 and 5 pass, including after a restart in the middle of a run.

## T8 — Graph loader (FR-18)

- **T8.1 — Constraints.** Uniqueness constraints for every node type.
- **T8.2 — Reference nodes.** Customers, products, regions, channels.
- **T8.3 — Transactions and links.** With `MERGE`, in batches.

Done when: a re-load does not change node counts, and acceptance check 6 passes.

## T9 — Vector loader (FR-19)

- **T9.1 — Summary builder.** Text summaries from the clean lake, with metadata.
- **T9.2 — Embeddings.** Wrapper for the chosen embedding model.
- **T9.3 — Load and search.** Put vectors in S3 Vectors with stable IDs, and a search function.
  Search is hybrid: meaning first, then labels named in the question (see plan.md).

Done when: searching "laptop sales in EMEA" returns EMEA laptop summaries first, and a re-load
adds no copies.

## T10 — Forecasting (FR-23 to FR-25)

- **T10.1 — Weekly aggregation.** Sales per category, region and week.
- **T10.2 — Models.** The 4-week-average baseline plus three exponential smoothing models:
  simple, Holt (damped trend) and Holt-Winters (damped trend + 13-week season). T10.3 picks the
  best per series by measured error.
- **T10.3 — Evaluation.** Error on the last 4 held-back weeks, for both.
- **T10.4 — Save forecast.** Write to `forecasts/` in S3.

Done when: acceptance check 7 passes.

## T11 — LLM client and RAG (FR-26, FR-28)

- **T11.1 — LLM client.** OpenRouter wrapper with model name and limits from settings.
- **T11.2 — RAG answer.** Retrieve summaries, answer only from them, list the sources.

Done when: a sample question gets an answer that names the summaries it used.

## T12 — Agent and tools (FR-27 to FR-30)

- **T12.1 — `sql_query` tool.** DuckDB on clean Parquet, single `SELECT` only.
- **T12.2 — `graph_query` tool.** Cypher in a read-only transaction.
- **T12.3 — `vector_search` and `get_forecast` tools.**
- **T12.4 — Agent loop.** Tool calling with a step limit and a sources list.

Done when: unit tests show write queries are rejected, and the four example questions in
`spec.md` section 3 get sensible answers with sources (acceptance check 8).

## T13 — Web API (FR-31)

- **T13.1 — `/health` and `/quality/latest`.**
- **T13.2 — `/forecast`.**
- **T13.3 — `/ask`.**

Done when (each subtask): its endpoint answers correctly when called locally.

## T14 — Dashboard (FR-32, optional)

- **T14.1 — Sales and forecast charts.**
- **T14.2 — Question box** that calls `/ask`.

Done when: the page loads and shows real data from the lake.

## T15 — Demo script and README

- **T15.1 — Demo script.** `scripts/demo.py` runs the acceptance checks in order.
- **T15.2 — README.** Setup and run steps, plus honest notes on what differs from real scale.
- **T15.3 — Interview map.** A short note linking each PoC part to the matching section of
  `dell.md`.

Done when: a fresh run of the demo script passes every acceptance check.
