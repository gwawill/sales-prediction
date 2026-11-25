import logging
from datetime import date
from types import SimpleNamespace as NS

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sales_poc.ai.agent import AgentAnswer, Source, Step
from sales_poc.ai.llm import LlmError, Usage
from sales_poc.api import app as api
from sales_poc.api.app import Services, create_app
from sales_poc.lake.store import LocalStore
from sales_poc.lake.runs import write_report


def forecast_saved():
    rows = [{"category": c, "region_id": r, "week_start": date(2026, 11, d), "forecast_revenue": 100.123,
             "model": "holt", "backtest_wape": 0.34, "baseline_wape": 0.5}
            for c in ("Laptop", "Server") for r in ("EMEA", "NA") for d in (9, 2)]
    summary = {"run_id": "run-1", "forecast_weeks": ["2026-11-02", "2026-11-09"], "series": 4,
               "models": [{"model": "holt", "overall_wape": 0.4}], "not_shown": "x"}
    return pd.DataFrame(rows), summary


class FakeServices(Services):
    def __init__(self, tmp_path, **kwargs):
        super().__init__(store=LocalStore(tmp_path), llm=NS(settings=NS(agent_max_steps=6)), **kwargs)
        self.forecast_saved = forecast_saved()
        self.agent_calls, self.closed = [], False
        self.agent_error = None

    def latest_forecast(self):
        return self.forecast_saved

    def ask_agent(self, question, max_steps):
        self.agent_calls.append((question, max_steps))
        if self.agent_error:
            raise self.agent_error
        return AgentAnswer(
            question, "There were 68.",
            steps=[Step("graph_query", {"query": "MATCH ..."}, True, "68")],
            sources=[Source("graph_query", "MATCH (c)\n   RETURN count(c)")],
            usage=Usage(100, 20, 2))

    def ask_rag(self, question):
        return NS(text="Up 11% [S1].", sources=[NS(summary_id="catreg:Laptop:EMEA:2026-03")],
                  usage=Usage(50, 10, 1))

    def close(self):
        self.closed = True


@pytest.fixture
def services(tmp_path):
    return FakeServices(tmp_path)


@pytest.fixture
def client(services):
    with TestClient(create_app(services)) as c:
        yield c


# --- /health -------------------------------------------------------------------------------

def test_health_is_instant_and_touches_nothing(tmp_path):
    services = FakeServices(tmp_path, checks={"s3": lambda: pytest.fail("should not run")})
    with TestClient(create_app(services)) as c:
        response = c.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok" and "checks" not in response.json()


def test_health_answers_head_for_uptime_monitors(tmp_path):
    services = FakeServices(tmp_path, checks={"s3": lambda: pytest.fail("should not run")})
    with TestClient(create_app(services)) as c:
        response = c.head("/health")
    assert response.status_code == 200
    assert response.content == b""


def test_deep_health_runs_the_checks(tmp_path):
    services = FakeServices(tmp_path, checks={"s3": lambda: "bucket ok", "neo4j": lambda: "graph ok"})
    with TestClient(create_app(services)) as c:
        body = c.get("/health?deep=true").json()
    assert body["status"] == "ok"
    assert body["checks"]["s3"]["ok"] and body["checks"]["neo4j"]["detail"] == "graph ok"


def test_deep_health_fails_with_503_and_says_why(tmp_path):
    def broken():
        raise RuntimeError("S3 refused the key")

    services = FakeServices(tmp_path, checks={"s3": broken, "neo4j": lambda: "graph ok"})
    with TestClient(create_app(services)) as c:
        response = c.get("/health?deep=true")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["status"] == "failing"
    assert detail["checks"]["s3"] == {"ok": False, "detail": "S3 refused the key", "seconds": pytest.approx(0, abs=1)}
    assert detail["checks"]["neo4j"]["ok"]


def test_the_default_deep_checks_are_free_and_quick():
    assert list(Services().checks) == ["s3", "s3vectors", "neo4j"]


# --- /quality/latest -----------------------------------------------------------------------

def report(run_id, source):
    return {"run_id": run_id, "source": source, "counts": {"rows_in": 10}}


def test_quality_latest_returns_the_newest_report(client, services):
    write_report(services.store, report("20261101T000000Z-batch-aaaaaa", "batch"))
    write_report(services.store, report("20261102T000000Z-stream-bbbbbb", "stream"))

    assert client.get("/quality/latest").json()["run_id"] == "20261102T000000Z-stream-bbbbbb"
    assert client.get("/quality/latest?source=batch").json()["run_id"] == "20261101T000000Z-batch-aaaaaa"


def test_quality_latest_without_reports_is_404(client):
    response = client.get("/quality/latest")
    assert response.status_code == 404 and "batch-load" in response.json()["detail"]


def test_quality_latest_rejects_an_unknown_source(client):
    assert client.get("/quality/latest?source=kafka").status_code == 422


# --- /forecast -----------------------------------------------------------------------------

def test_forecast_returns_the_summary_and_sorted_rows(client):
    body = client.get("/forecast").json()
    assert body["run_id"] == "run-1" and body["forecast_weeks"] == ["2026-11-02", "2026-11-09"]
    assert "not_shown" not in body
    assert len(body["rows"]) == 8
    assert body["rows"][0] == {"category": "Laptop", "region_id": "EMEA", "week_start": "2026-11-02",
                               "forecast_revenue": 100.12, "model": "holt", "backtest_wape": 0.34,
                               "baseline_wape": 0.5}


def test_forecast_filters(client):
    rows = client.get("/forecast?category=Server&region=NA").json()["rows"]
    assert [(r["category"], r["region_id"], r["week_start"]) for r in rows] == [
        ("Server", "NA", "2026-11-02"), ("Server", "NA", "2026-11-09")]


@pytest.mark.parametrize("query", ["category=Phone", "region=Europe"])
def test_forecast_rejects_unknown_filters(client, query):
    assert client.get(f"/forecast?{query}").status_code == 422


def test_no_forecast_is_404(client, services):
    services.forecast_saved = None
    response = client.get("/forecast")
    assert response.status_code == 404 and "forecast-run" in response.json()["detail"]


def test_the_forecast_is_cached_for_a_while(tmp_path, monkeypatch):
    import sales_poc.forecast.publish as publish

    reads, now = [], [1000.0]
    monkeypatch.setattr(publish, "latest_forecast", lambda store: reads.append(1) or forecast_saved())
    services = Services(store=LocalStore(tmp_path), clock=lambda: now[0])

    services.latest_forecast()
    now[0] += api.FORECAST_CACHE_SECONDS - 1
    services.latest_forecast()
    assert len(reads) == 1
    now[0] += 2
    services.latest_forecast()
    assert len(reads) == 2


# --- /ask ----------------------------------------------------------------------------------

def test_ask_uses_the_agent_by_default(client, services):
    body = client.post("/ask", json={"question": "How many?"}).json()

    assert services.agent_calls == [("How many?", None)]
    assert body["mode"] == "agent" and body["answer"] == "There were 68."
    assert body["sources"] == [{"kind": "graph_query", "detail": "MATCH (c) RETURN count(c)"}]
    assert body["steps"] == [{"tool": "graph_query", "arguments": {"query": "MATCH ..."}, "ok": True}]
    assert body["tokens"] == 120 and body["hit_step_limit"] is False


def test_ask_with_rag(client):
    body = client.post("/ask", json={"question": "Laptops?", "mode": "rag"}).json()
    assert body["answer"] == "Up 11% [S1]."
    assert body["sources"] == [{"kind": "summary", "detail": "catreg:Laptop:EMEA:2026-03"}]
    assert body["steps"] == [] and body["tokens"] == 60


def test_ask_passes_a_lower_step_limit(client, services):
    client.post("/ask", json={"question": "q", "max_steps": 2})
    assert services.agent_calls == [("q", 2)]


@pytest.mark.parametrize(
    "body",
    [{}, {"question": ""}, {"question": "x" * 1001}, {"question": "q", "mode": "magic"},
     {"question": "q", "max_steps": 0}, {"question": "q", "max_steps": 7}],
)
def test_bad_questions_are_rejected(client, services, body):
    assert client.post("/ask", json=body).status_code == 422
    assert services.agent_calls == []


def test_a_model_problem_is_a_502_with_the_reason(client, services):
    services.agent_error = LlmError("OpenRouter refused the key.")
    response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 502 and response.json()["detail"] == "OpenRouter refused the key."


def test_each_question_is_logged_as_json(client, caplog):
    with caplog.at_level(logging.INFO, logger="sales_poc.api.app"):
        client.post("/ask", json={"question": "q"})
    [record] = [r for r in caplog.records if r.message == "ask_done"]
    assert (record.mode, record.tool_calls, record.tokens) == ("agent", 1, 120)


# --- life cycle ----------------------------------------------------------------------------

def test_services_are_closed_on_shutdown(tmp_path):
    services = FakeServices(tmp_path)
    with TestClient(create_app(services)):
        pass
    assert services.closed


def test_the_docs_page_lists_every_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/health", "/quality/latest", "/forecast", "/ask"}
    assert list(paths["/ask"]) == ["post"]
