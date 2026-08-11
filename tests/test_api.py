"""FastAPI TestClient integration tests for ridm_ultra.api routes."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ridm_ultra.api import create_app


@pytest.fixture
def api_client():
    app = create_app()
    return TestClient(app)


def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "6.0.0"


def test_session_crud_endpoints(api_client):
    # 1. Create Session
    create_resp = api_client.post("/api/v1/sessions/new", json={"title": "Pytest Session", "system_prompt": "Test bot"})
    assert create_resp.status_code == 200
    session_data = create_resp.json()
    session_id = session_data["session_id"]
    assert session_data["title"] == "Pytest Session"

    # 2. Get Session
    get_resp = api_client.get(f"/api/v1/sessions/{session_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["session_id"] == session_id

    # 3. List Sessions
    list_resp = api_client.get("/api/v1/sessions")
    assert list_resp.status_code == 200
    assert list_resp.json()["total"] >= 1

    # 4. Delete Session
    del_resp = api_client.delete(f"/api/v1/sessions/{session_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted_session_id"] == session_id


def test_chat_stream_sse_endpoint(api_client):
    payload = {
        "message": "Hello from pytest SSE test!",
        "temperature": 0.7,
        "max_tokens": 100
    }
    response = api_client.post("/api/v1/chat/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    lines = response.text.strip().split("\n\n")
    assert len(lines) > 0
    first_chunk = json.loads(lines[0].replace("data: ", ""))
    assert "delta" in first_chunk
