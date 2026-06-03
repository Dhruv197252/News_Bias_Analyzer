"""
Integration tests for FastAPI endpoints.
Run with: pytest tests/test_api.py -v
Requires the backend to be running on localhost:8000
OR use TestClient for in-process testing.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from backend.main import app
    return TestClient(app)


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_text(client):
    response = client.post("/api/analyze/text", json={
        "text": "The radical regime imposed draconian laws that devastated citizens.",
        "headline": "Test Headline"
    })
    assert response.status_code == 200
    data = response.json()
    assert "composite_score" in data
    assert 0.0 <= data["composite_score"] <= 1.0
    assert data["ml_label"] in [0, 1]


def test_get_domains(client):
    response = client.get("/api/domains")
    assert response.status_code == 200
    data = response.json()
    assert "domains" in data
    assert len(data["domains"]) > 0
