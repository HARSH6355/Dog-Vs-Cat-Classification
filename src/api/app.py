"""
app.py
------
FastAPI inference service for the Cat vs Dog classifier.

Endpoints:
    GET  /          → API info & available endpoints
    GET  /health    → Health check (model loaded status)
    POST /predict   → Upload image → returns class label + probabilities
    GET  /metrics   → Request count, average latency (in-memory counters)

Run locally (development):
    uvicorn src.api.app:app --reload --port 8000

Run via Docker:
    docker run -p 8000:8000 cat-dog-classifier:v1
"""

import io
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cat_dog_api")


# ---------------------------------------------------------------------------
# Global state  (in-memory monitoring — satisfies M5 basic monitoring)
# ---------------------------------------------------------------------------
class _AppState:
    predictor = None
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    # Keep last 1000 latencies for rolling average
    latencies_ms: deque = deque(maxlen=1000)


state = _AppState()

# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; release on shutdown."""
    logger.info("Starting up Cat vs Dog API — loading model...")
    from src.api.predict import Predictor  # lazy import to avoid circular
    state.predictor = Predictor()
    if state.predictor.model_loaded:
        logger.info("Model loaded successfully. API is ready.")
    else:
        logger.warning("Model failed to load. /predict will return 503.")
    yield
    logger.info("Shutting down Cat vs Dog API.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Cat vs Dog Classifier API",
    description=(
        "MLOps Assignment — Module 2\n\n"
        "A production-ready REST API that classifies uploaded images "
        "as either **cat** or **dog** using a trained Baseline CNN model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins for local testing / assignment demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Allowed image types
# ---------------------------------------------------------------------------
ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp", "image/gif", "image/webp",
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=JSONResponse, tags=["Info"])
async def root() -> Dict[str, Any]:
    """Welcome endpoint — returns API info and available routes."""
    return {
        "service": "Cat vs Dog Classifier API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": state.predictor.model_loaded if state.predictor else False,
        "endpoints": {
            "GET  /health":  "Health check",
            "POST /predict": "Classify an uploaded image (multipart/form-data, field='file')",
            "GET  /metrics": "Basic request metrics",
            "GET  /docs":    "Interactive Swagger UI",
        },
    }


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns 200 if the service is running and the model is loaded.
    Returns 503 if the model failed to load.
    """
    model_loaded = state.predictor.model_loaded if state.predictor else False

    response = {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "total_requests": state.total_requests,
        "uptime": "running",
    }

    if not model_loaded:
        return JSONResponse(status_code=503, content=response)
    return response


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(..., description="Image file (JPEG, PNG, BMP, WEBP)")) -> Dict[str, Any]:
    """
    Classify an uploaded image as **cat** or **dog**.

    - **file**: Upload an image file via multipart/form-data.

    **Response example:**
    ```json
    {
      "label": "dog",
      "confidence": 0.9832,
      "probabilities": { "cat": 0.0168, "dog": 0.9832 },
      "latency_ms": 12.4,
      "filename": "my_pet.jpg"
    }
    ```
    """
    state.total_requests += 1
    t_request_start = time.perf_counter()

    logger.info(f"POST /predict — file='{file.filename}' content_type='{file.content_type}'")

    # ── Guard: model must be loaded ───────────────────────────────────────
    if not state.predictor or not state.predictor.model_loaded:
        state.failed_requests += 1
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs.",
        )

    # ── Guard: file type ──────────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        state.failed_requests += 1
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}"
            ),
        )

    # ── Read and decode image ─────────────────────────────────────────────
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        state.failed_requests += 1
        raise HTTPException(
            status_code=422,
            detail="Could not decode image. File may be corrupt or not a valid image.",
        )
    except Exception as e:
        state.failed_requests += 1
        raise HTTPException(status_code=500, detail=f"Failed to read image: {str(e)}")

    # ── Run inference ─────────────────────────────────────────────────────
    try:
        result = state.predictor.predict(image)
    except Exception as e:
        state.failed_requests += 1
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # ── Track metrics ─────────────────────────────────────────────────────
    state.successful_requests += 1
    total_latency_ms = (time.perf_counter() - t_request_start) * 1000
    state.latencies_ms.append(total_latency_ms)

    logger.info(
        f"Prediction: label={result['label']} confidence={result['confidence']} "
        f"latency={result['latency_ms']}ms"
    )

    return {
        **result,
        "filename": file.filename,
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics() -> Dict[str, Any]:
    """
    Basic in-memory request metrics.

    Tracks total requests, successes, failures, and average latency.
    Satisfies the M5 monitoring requirement.
    """
    avg_latency = (
        round(sum(state.latencies_ms) / len(state.latencies_ms), 2)
        if state.latencies_ms
        else 0.0
    )
    return {
        "total_requests": state.total_requests,
        "successful_requests": state.successful_requests,
        "failed_requests": state.failed_requests,
        "success_rate_pct": (
            round(state.successful_requests / state.total_requests * 100, 1)
            if state.total_requests > 0
            else 0.0
        ),
        "avg_latency_ms": avg_latency,
        "recent_latencies_count": len(state.latencies_ms),
    }
