"""FastAPI Application Factory for RIDM Ultra API Services."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="RIDM Ultra Chat API",
        description="Production-Grade Asynchronous SSE & REST Chat API Service",
        version="6.0.0",
    )

    # Enable CORS for local Streamlit UI and frontend clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
