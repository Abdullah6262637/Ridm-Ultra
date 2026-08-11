"""Automated Verification Test for FastAPI SSE API & Streamlit Web UI Integration."""
import json
import socket
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn

from ridm_ultra.api import create_app


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def run_server(host: str, port: int):
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="error")


def main():
    print("=== RIDM ULTRA FASTAPI & SSE STREAMING VERIFICATION ===")

    port = find_free_port()
    host = "127.0.0.1"

    # 1. Start FastAPI server in background thread
    server_thread = threading.Thread(target=run_server, args=(host, port), daemon=True)
    server_thread.start()
    time.sleep(1.5)  # Wait for server startup

    base_url = f"http://{host}:{port}"

    # 2. Test GET /health
    print(f"\n--- Test 1: GET {base_url}/health ---")
    req = urllib.request.Request(f"{base_url}/health")
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print(f"Health Response: {data}")
        assert data["status"] == "ok"

    # 3. Test POST /api/v1/sessions/new
    print(f"\n--- Test 2: POST {base_url}/api/v1/sessions/new ---")
    session_req = urllib.request.Request(
        f"{base_url}/api/v1/sessions/new",
        data=json.dumps({"title": "Smoke Test Session", "system_prompt": "You are a test bot."}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(session_req, timeout=5) as resp:
        session_data = json.loads(resp.read().decode("utf-8"))
        session_id = session_data["session_id"]
        print(f"Created Session ID: {session_id}")
        assert session_id is not None

    # 4. Test POST /api/v1/chat/stream SSE Streaming
    print(f"\n--- Test 3: POST {base_url}/api/v1/chat/stream (SSE Stream) ---")
    stream_payload = {
        "message": "Hello from smoke test!",
        "session_id": session_id
    }
    stream_req = urllib.request.Request(
        f"{base_url}/api/v1/chat/stream",
        data=json.dumps(stream_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    received_chunks = []
    with urllib.request.urlopen(stream_req, timeout=15) as resp:
        content_type = resp.headers.get("Content-Type", "")
        assert "text/event-stream" in content_type, f"Expected text/event-stream in Content-Type, got '{content_type}'"

        for line in resp:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data: "):
                chunk_data = json.loads(line_str[6:])
                received_chunks.append(chunk_data.get("delta", ""))
                print(chunk_data.get("delta", ""), end="", flush=True)
    print()
    assert len(received_chunks) > 0, "No SSE stream chunks received!"

    # 5. Test GET /api/v1/sessions/{session_id} History Reload
    print(f"\n--- Test 4: GET {base_url}/api/v1/sessions/{session_id} ---")
    get_sess_req = urllib.request.Request(f"{base_url}/api/v1/sessions/{session_id}")
    with urllib.request.urlopen(get_sess_req, timeout=5) as resp:
        sess_history = json.loads(resp.read().decode("utf-8"))
        print(f"Reloaded Messages Count: {len(sess_history['messages'])}")
        assert len(sess_history["messages"]) == 2  # User + Assistant

    # 6. Verify Streamlit UI App File Existence
    print("\n--- Test 5: Verify Streamlit App Module ---")
    ui_path = Path("ridm_ultra/ui/app.py")
    assert ui_path.exists(), "Streamlit app.py does not exist!"
    print(f"Streamlit UI Module Verified: {ui_path.resolve()}")

    print("\n[SUCCESS] FastAPI SSE API & Web UI Verification Completed Successfully!")


if __name__ == "__main__":
    main()
