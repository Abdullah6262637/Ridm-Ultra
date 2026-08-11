"""Production-Grade Streamlit Interactive Web UI for RIDM Ultra Chat System."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import streamlit as st

# Configure page layout and visual theme
st.set_page_config(
    page_title="RIDM Ultra — Conversational AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = "http://127.0.0.1:8000"


def fetch_json(url: str, method: str = "GET", data: dict = None) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8") if data else None,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stream_sse_events(url: str, payload: dict):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start_time = time.time()
    first_token_time = None

    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data: "):
                raw_json = line_str[6:]
                try:
                    data = json.loads(raw_json)
                    if first_token_time is None and data.get("delta"):
                        first_token_time = time.time()
                    yield data, (first_token_time - start_time if first_token_time else 0.0)
                except json.JSONDecodeError:
                    continue


def main():
    st.title("⚡ RIDM Ultra Conversational Engine")
    st.caption("Hardware-Aware Closed-Form & Transformer LLM Intelligence Layer")

    # Wait for Backend to Load Massive DB
    try:
        health_data = fetch_json(f"{API_BASE_URL}/health")
        if not health_data.get("is_loaded", True):
            with st.status("🚀 Devasa SVD Veritabanı Yükleniyor (1.3 GB)...", expanded=True) as status:
                st.write("Sistem arka planda 1.2 Milyar kelimelik bağlantılarını kuruyor.")
                st.write("Lütfen birkaç saniye bekleyin, arayüz otomatik açılacaktır...")
                while not health_data.get("is_loaded", True):
                    time.sleep(2)
                    try:
                        health_data = fetch_json(f"{API_BASE_URL}/health")
                    except Exception:
                        pass
                status.update(label="Model başarıyla yüklendi!", state="complete", expanded=False)
    except Exception:
        # If API is down, just skip health check and continue rendering UI
        pass

    # Initialize Session State Variables
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = {"ttft_ms": 0, "total_tokens": 0, "tier": "UNKNOWN", "model": "N/A"}

    # Sidebar: Session Manager & System Dashboard
    with st.sidebar:
        st.header("💬 Conversation Manager")

        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            try:
                new_session = fetch_json(f"{API_BASE_URL}/api/v1/sessions/new", method="POST", data={"title": "New Conversation"})
                st.session_state.current_session_id = new_session["session_id"]
                st.session_state.messages = []
                st.session_state.last_metrics = {"ttft_ms": 0, "total_tokens": 0, "tier": "UNKNOWN", "model": "N/A"}
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create new session: {e}")

        st.divider()

        # Load Historical Sessions List
        st.subheader("History")
        try:
            sessions_data = fetch_json(f"{API_BASE_URL}/api/v1/sessions?limit=20")
            sessions_list = sessions_data.get("sessions", [])

            for sess in sessions_list:
                s_id = sess["session_id"]
                title = sess.get("title") or f"Session {s_id[:6]}"
                num_msgs = len(sess.get("messages", []))
                label = f"{title} ({num_msgs} msgs)"

                btn_type = "secondary" if s_id != st.session_state.current_session_id else "primary"
                if st.button(label, key=f"sess_{s_id}", use_container_width=True, type=btn_type):
                    st.session_state.current_session_id = s_id
                    st.session_state.messages = [
                        {"role": m["role"], "content": m["content"]}
                        for m in sess.get("messages", [])
                    ]
                    st.rerun()
        except Exception:
            st.warning("⚠️ FastAPI Backend offline or unreachable at http://localhost:8000")

        st.divider()

        # Real-time System Metrics Dashboard
        st.subheader("📊 Engine Metrics")
        metrics = st.session_state.last_metrics
        m1, m2 = st.columns(2)
        with m1:
            st.metric(label="TTFT Latency", value=f"{metrics['ttft_ms']:.0f} ms")
        with m2:
            st.metric(label="Tokens", value=f"{metrics['total_tokens']}")

        tier_color = "🟢" if metrics["tier"] == "fast" else ("🔵" if metrics["tier"] == "balanced" else "🟣")
        st.info(f"**Router Tier**: {tier_color} `{metrics['tier'].upper()}`\n\n**Model**: `{metrics['model']}`")

    # Render Active Chat Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input Box
    if prompt := st.chat_input("Ask RIDM Ultra anything..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant Streaming Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = []

            payload = {
                "message": prompt,
                "session_id": st.session_state.current_session_id,
            }

            try:
                ttft_seconds = 0.0
                last_model = "unknown"
                total_tokens = 0
                last_tier = "fast"

                for chunk, latency in stream_sse_events(f"{API_BASE_URL}/api/v1/chat/stream", payload):
                    delta = chunk.get("delta", "")
                    full_response.append(delta)
                    message_placeholder.markdown("".join(full_response) + "▌")
                    ttft_seconds = latency
                    if chunk.get("model"):
                        last_model = chunk["model"]
                    if chunk.get("usage"):
                        total_tokens = chunk["usage"].get("total_tokens", total_tokens)

                final_text = "".join(full_response)
                message_placeholder.markdown(final_text)

                # Infer tier from response or content length
                if "Synthesizing detailed reasoning" in final_text:
                    last_tier = "reasoning"
                else:
                    last_tier = "fast"

                st.session_state.messages.append({"role": "assistant", "content": final_text})
                st.session_state.last_metrics = {
                    "ttft_ms": ttft_seconds * 1000,
                    "total_tokens": total_tokens if total_tokens else len(final_text.split()),
                    "tier": last_tier,
                    "model": last_model,
                }
            except Exception as e:
                st.error(f"Error communicating with Chat API: {e}")


if __name__ == "__main__":
    main()
