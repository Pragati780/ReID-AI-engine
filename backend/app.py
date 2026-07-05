"""
backend/app.py

FastAPI application entrypoint. This file wires together CORS, the API
router, and startup housekeeping (ensuring local storage folders exist).
It contains no face-recognition logic -- see services/pipeline_service.py
for the boundary between this web layer and the AI pipeline.

Run with:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

(from inside the backend/ directory)
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("face_reid_api")

app = FastAPI(
    title="Face Re-Identification API",
    description=(
        "Thin FastAPI wrapper around an existing, unmodified face-recognition "
        "pipeline (detection -> alignment -> ArcFace embedding -> cosine "
        "similarity matching -> timestamp aggregation). This API only "
        "orchestrates uploads and serves results; all AI logic lives in "
        "core/*.py and pipeline/run_pipeline.py, invoked as a black-box "
        "subprocess by services/pipeline_service.py."
    ),
    version="1.0.0",
)

# --------------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------------
# Allows the Vite dev server (default http://localhost:5173) and a locally
# built frontend to call this API from the browser. Tighten this list to
# your actual deployed frontend origin(s) in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.on_event("startup")
def ensure_local_storage_dirs() -> None:
    """Creates backend-local storage folders if they don't already exist.
    (outputs/ itself is the AI pipeline's own directory and is created by
    the pipeline, not by this backend.)"""
    backend_root = Path(__file__).resolve().parent
    for folder in ("uploads", "responses", "static"):
        (backend_root / folder).mkdir(parents=True, exist_ok=True)
    logger.info("Backend storage folders ready.")


@app.get("/")
def root():
    return {
        "service": "Face Re-Identification API",
        "docs": "/docs",
        "health": "/api/health",
    }
