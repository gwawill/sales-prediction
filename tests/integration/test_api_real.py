"""The web API with the real services behind it (one RAG question costs a fraction of a cent)."""

import pytest
from fastapi.testclient import TestClient

from sales_poc.api.app import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


def test_deep_health_passes(client):
    response = client.get("/health?deep=true")
    assert response.status_code == 200, response.text
    assert set(response.json()["checks"]) == {"s3", "s3vectors", "neo4j"}


def test_the_latest_quality_report_adds_up(client):
    report = client.get("/quality/latest?source=batch").json()
    c = report["counts"]
    assert c["rows_in"] == c["clean"] + c["duplicates_removed"] + c["quarantined"]


def test_the_forecast_covers_every_series_for_four_weeks(client):
    body = client.get("/forecast").json()
    assert body["series"] == 24 and len(body["forecast_weeks"]) == 4
    assert len(body["rows"]) == 24 * 4


def test_a_rag_question_names_its_sources(client):
    body = client.post("/ask", json={"question": "How did laptop sales in EMEA change in March 2026?",
                                     "mode": "rag"}).json()
    assert {"kind": "summary", "detail": "catreg:Laptop:EMEA:2026-03"} in body["sources"]
