"""The web API (task T13, FR-31). A small FastAPI service in front of the platform.

    api-serve                       start it on http://127.0.0.1:8000 (API_HOST / API_PORT)
    http://127.0.0.1:8000/docs      FastAPI's built-in page to try every endpoint

Endpoints:
    GET  /health                    is the service up? (instant, touches nothing; HEAD works too)
    GET  /health?deep=true          also checks S3, S3 Vectors and Neo4j for real (a few seconds)
    GET  /quality/latest            the newest ETL quality report (FR-20); ?source=batch|stream
    GET  /forecast                  the latest saved 4-week forecast; ?category=...&region=...
    POST /ask                       {"question": "..."} answered by the agent (default) or by
                                    RAG ({"mode": "rag"}), with the sources it used (FR-28)

How it fits together: the endpoints are `async`, and each hands its slow, blocking work
(S3, Neo4j, the language model) to a worker thread, so one long question does not stop the
service from answering /health. The agent's tools are created once and shared by every
request, so the lake is read into DuckDB only on the first SQL question, not on each one.
After that, the SQL tool re-reads only the days that changed, at most every
LAKE_CHECK_SECONDS, so answers include new orders without restarting the API.

Everything here only reads. There is no login (out of scope in spec.md), so the service
listens on 127.0.0.1 by default: only this PC can reach it.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from sales_poc.ai.llm import LlmError
from sales_poc.log import get_logger

log = get_logger(__name__)

Region = Literal["NA", "EMEA", "APJ", "LATAM"]
Category = Literal["Laptop", "Desktop", "Server", "Storage", "Monitor", "Accessory"]
FORECAST_CACHE_SECONDS = 300.0


# --- what the endpoints use, swappable in tests --------------------------------------------

class Services:
    """The platform behind the API. Each part is created on first use and then reused."""

    def __init__(
        self,
        store=None,
        llm=None,
        toolbox=None,
        checks: dict[str, Callable[[], str]] | None = None,
        rag_answer: Callable | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._store, self._llm, self._toolbox = store, llm, toolbox
        self._checks, self._rag_answer, self._clock = checks, rag_answer, clock
        self._forecast: tuple[float, Any] | None = None
        self._lock = threading.Lock()

    @property
    def store(self):
        if self._store is None:
            from sales_poc.lake.store import s3_store
            self._store = s3_store()
        return self._store

    @property
    def llm(self):
        if self._llm is None:
            from sales_poc.ai.llm import get_llm
            self._llm = get_llm()
        return self._llm

    @property
    def toolbox(self):
        with self._lock:
            if self._toolbox is None:
                from sales_poc.ai.agent import ToolBox
                self._toolbox = ToolBox()
            return self._toolbox

    @property
    def checks(self) -> dict[str, Callable[[], str]]:
        if self._checks is None:
            from sales_poc.checks import CHECKS
            # Free and quick. Kafka (slow) and the LLM (costs tokens) are left to check_cloud.py.
            self._checks = {name: CHECKS[name] for name in ("s3", "s3vectors", "neo4j")}
        return self._checks

    def latest_report(self, source: str | None):
        from sales_poc.lake.runs import latest_report
        return latest_report(self.store, source)

    def latest_forecast(self):
        """(table, summary) of the newest forecast, re-read at most every FORECAST_CACHE_SECONDS."""
        from sales_poc.forecast.publish import latest_forecast

        now = self._clock()
        if self._forecast is None or now - self._forecast[0] > FORECAST_CACHE_SECONDS:
            self._forecast = (now, latest_forecast(self.store))
        return self._forecast[1]

    def ask_agent(self, question: str, max_steps: int | None):
        from sales_poc.ai.agent import Agent
        return Agent(self.llm, self.toolbox, max_steps=max_steps).ask(question)

    def ask_rag(self, question: str):
        if self._rag_answer is not None:
            return self._rag_answer(question, llm=self.llm)
        from sales_poc.ai.rag import answer
        return answer(question, llm=self.llm)

    def close(self) -> None:
        if self._toolbox is not None:
            self._toolbox.close()


# --- request and response shapes (they also fill the /docs page) ---------------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, examples=["What were laptop sales in EMEA last month?"])
    mode: Literal["agent", "rag"] = "agent"
    max_steps: int | None = Field(default=None, ge=1, description="agent only; at most AGENT_MAX_STEPS")


class AskSource(BaseModel):
    kind: str          # sql_query, graph_query, vector_search, get_forecast, or summary (RAG)
    detail: str        # the query, summary ID or forecast run


class AskStep(BaseModel):
    tool: str
    arguments: dict
    ok: bool


class AskResponse(BaseModel):
    question: str
    mode: str
    answer: str
    sources: list[AskSource]
    steps: list[AskStep]
    hit_step_limit: bool
    tokens: int
    seconds: float


# --- the app -------------------------------------------------------------------------------

def create_app(services: Services | None = None) -> FastAPI:
    services = services or Services()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        services.close()   # close the Neo4j connection and the rest on shutdown

    app = FastAPI(title="Sales PoC API", version="0.1.0", lifespan=lifespan,
                  description="Read-only API over the sales data lake, forecast and AI layer.")
    app.state.services = services

    # HEAD too: free uptime monitors (used to keep a sleeping host awake) send only HEAD.
    @app.head("/health", include_in_schema=False)
    @app.get("/health")
    async def health(deep: bool = False):
        body: dict[str, Any] = {"status": "ok", "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if not deep:
            return body

        def run_checks() -> dict:
            results = {}
            for name, check in services.checks.items():
                started = time.monotonic()
                try:
                    results[name] = {"ok": True, "detail": check()}
                except Exception as exc:  # each check explains its own failure
                    results[name] = {"ok": False, "detail": str(exc)}
                results[name]["seconds"] = round(time.monotonic() - started, 2)
            return results

        body["checks"] = await run_in_threadpool(run_checks)
        if not all(c["ok"] for c in body["checks"].values()):
            body["status"] = "failing"
            log.warning("health_failing", extra={"checks": [n for n, c in body["checks"].items() if not c["ok"]]})
            raise HTTPException(status_code=503, detail=body)
        return body

    @app.get("/quality/latest")
    async def quality_latest(source: Literal["batch", "stream"] | None = None):
        report = await run_in_threadpool(services.latest_report, source)
        if report is None:
            raise HTTPException(status_code=404, detail="No quality report has been saved yet. Run batch-load.")
        return report

    @app.get("/forecast")
    async def forecast(category: Category | None = None, region: Region | None = None):
        found = await run_in_threadpool(services.latest_forecast)
        if found is None:
            raise HTTPException(status_code=404, detail="No forecast has been saved yet. Run forecast-run.")
        table, summary = found
        if category:
            table = table[table["category"] == category]
        if region:
            table = table[table["region_id"] == region]
        rows = [
            {"category": r.category, "region_id": r.region_id, "week_start": str(r.week_start),
             "forecast_revenue": round(float(r.forecast_revenue), 2), "model": r.model,
             "backtest_wape": float(r.backtest_wape), "baseline_wape": float(r.baseline_wape)}
            for r in table.sort_values(["category", "region_id", "week_start"]).itertuples()
        ]
        keys = ("run_id", "forecast_weeks", "weeks_used", "test_weeks", "backtest", "models",
                "series", "best_beats_baseline")
        return {**{k: summary.get(k) for k in keys}, "rows": rows}

    @app.post("/ask", response_model=AskResponse)
    async def ask(request: AskRequest):
        started = time.monotonic()
        limit = services.llm.settings.agent_max_steps
        if request.max_steps and request.max_steps > limit:
            raise HTTPException(status_code=422, detail=f"max_steps can be at most {limit} (AGENT_MAX_STEPS).")
        try:
            if request.mode == "rag":
                result = await run_in_threadpool(services.ask_rag, request.question)
                sources = [AskSource(kind="summary", detail=s.summary_id) for s in result.sources]
                steps, hit_limit = [], False
            else:
                result = await run_in_threadpool(services.ask_agent, request.question, request.max_steps)
                sources = [AskSource(kind=s.tool, detail=" ".join(s.detail.split())) for s in result.sources]
                steps = [AskStep(tool=s.tool, arguments=s.arguments, ok=s.ok) for s in result.steps]
                hit_limit = result.hit_step_limit
        except LlmError as exc:
            log.error("ask_failed", extra={"mode": request.mode, "error": str(exc)})
            raise HTTPException(status_code=502, detail=str(exc)) from None

        seconds = round(time.monotonic() - started, 2)
        log.info("ask_done", extra={"mode": request.mode, "tool_calls": len(steps), "sources": len(sources),
                                    "tokens": result.usage.total_tokens, "seconds": seconds})
        return AskResponse(question=request.question, mode=request.mode, answer=result.text, sources=sources,
                           steps=steps, hit_step_limit=hit_limit, tokens=result.usage.total_tokens, seconds=seconds)

    return app


def main(argv: list[str] | None = None) -> int:
    """api-serve: run the API with uvicorn (a small web server for Python apps)."""
    import uvicorn

    from sales_poc.config import get_app_settings
    from sales_poc.log import setup_logging

    settings = get_app_settings()
    parser = argparse.ArgumentParser(prog="api-serve", description="Run the sales PoC web API.")
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    setup_logging()
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0
