"""
test_api.py
-----------
Unit tests for the Cat vs Dog FastAPI inference service.

Uses FastAPI's TestClient (backed by httpx) for synchronous testing
without needing a running server.

Run:
    pytest tests/test_api.py -v
"""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.app import app, state
from src.api.predict import Predictor


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_image_bytes(
    width: int = 224,
    height: int = 224,
    color: tuple = (200, 100, 50),
    fmt: str = "JPEG",
) -> bytes:
    """Create a solid-colour in-memory image and return its bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


def _make_upload_file(image_bytes: bytes, filename: str = "test.jpg", content_type: str = "image/jpeg"):
    """Return a dict suitable for httpx files= parameter."""
    return {"file": (filename, io.BytesIO(image_bytes), content_type)}


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create a single TestClient for the entire module (loads model once)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_image_bytes():
    """224x224 orange JPEG — generic enough to get a valid prediction."""
    return _make_image_bytes()


# ─────────────────────────────────────────────────────────────────────────────
# Tests — /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    """Tests for GET /health"""

    def test_health_returns_200_when_model_loaded(self, client: TestClient):
        response = client.get("/health")
        # Accept 200 (ok) or 503 (model not found in CI); just verify structure
        assert response.status_code in (200, 503)

    def test_health_response_has_required_keys(self, client: TestClient):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_health_status_is_string(self, client: TestClient):
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["status"], str)
        assert data["status"] in ("ok", "degraded")

    def test_health_model_loaded_is_bool(self, client: TestClient):
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["model_loaded"], bool)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — /predict
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    """Tests for POST /predict"""

    def test_predict_valid_jpeg_returns_200(self, client: TestClient, sample_image_bytes: bytes):
        if not (state.predictor and state.predictor.model_loaded):
            pytest.skip("Model not loaded — skipping inference tests.")
        response = client.post(
            "/predict",
            files=_make_upload_file(sample_image_bytes),
        )
        assert response.status_code == 200

    def test_predict_response_has_required_keys(self, client: TestClient, sample_image_bytes: bytes):
        if not (state.predictor and state.predictor.model_loaded):
            pytest.skip("Model not loaded — skipping inference tests.")
        response = client.post("/predict", files=_make_upload_file(sample_image_bytes))
        data = response.json()
        for key in ("label", "confidence", "probabilities", "latency_ms"):
            assert key in data, f"Missing key: {key}"

    def test_predict_label_is_cat_or_dog(self, client: TestClient, sample_image_bytes: bytes):
        if not (state.predictor and state.predictor.model_loaded):
            pytest.skip("Model not loaded — skipping inference tests.")
        response = client.post("/predict", files=_make_upload_file(sample_image_bytes))
        data = response.json()
        assert data["label"] in ("cat", "dog")

    def test_predict_confidence_in_range(self, client: TestClient, sample_image_bytes: bytes):
        if not (state.predictor and state.predictor.model_loaded):
            pytest.skip("Model not loaded — skipping inference tests.")
        response = client.post("/predict", files=_make_upload_file(sample_image_bytes))
        confidence = response.json()["confidence"]
        assert 0.0 <= confidence <= 1.0

    def test_predict_probabilities_sum_to_one(self, client: TestClient, sample_image_bytes: bytes):
        if not (state.predictor and state.predictor.model_loaded):
            pytest.skip("Model not loaded — skipping inference tests.")
        response = client.post("/predict", files=_make_upload_file(sample_image_bytes))
        probs = response.json()["probabilities"]
        assert "cat" in probs and "dog" in probs
        total = probs["cat"] + probs["dog"]
        assert abs(total - 1.0) < 0.01, f"Probs don't sum to 1: {total}"

    def test_predict_latency_is_positive(self, client: TestClient, sample_image_bytes: bytes):
        if not (state.predictor and state.predictor.model_loaded):
            pytest.skip("Model not loaded — skipping inference tests.")
        response = client.post("/predict", files=_make_upload_file(sample_image_bytes))
        latency = response.json()["latency_ms"]
        assert latency > 0

    def test_predict_png_image(self, client: TestClient):
        """PNG images should also work."""
        if not (state.predictor and state.predictor.model_loaded):
            pytest.skip("Model not loaded — skipping inference tests.")
        png_bytes = _make_image_bytes(fmt="PNG")
        response = client.post(
            "/predict",
            files=_make_upload_file(png_bytes, filename="test.png", content_type="image/png"),
        )
        assert response.status_code == 200

    def test_predict_invalid_content_type_returns_415(self, client: TestClient):
        """Uploading a text file should return 415 Unsupported Media Type."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert response.status_code == 415

    def test_predict_corrupt_image_returns_422(self, client: TestClient):
        """Sending random bytes as JPEG should return 422 Unprocessable."""
        response = client.post(
            "/predict",
            files={"file": ("bad.jpg", io.BytesIO(b"these are not image bytes"), "image/jpeg")},
        )
        assert response.status_code == 422

    def test_predict_includes_filename(self, client: TestClient, sample_image_bytes: bytes):
        """Response should echo back the uploaded filename."""
        if not (state.predictor and state.predictor.model_loaded):
            pytest.skip("Model not loaded — skipping inference tests.")
        response = client.post(
            "/predict",
            files=_make_upload_file(sample_image_bytes, filename="my_pet.jpg"),
        )
        data = response.json()
        assert data.get("filename") == "my_pet.jpg"


# ─────────────────────────────────────────────────────────────────────────────
# Tests — /metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    """Tests for GET /metrics"""

    def test_metrics_returns_200(self, client: TestClient):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_has_required_keys(self, client: TestClient):
        response = client.get("/metrics")
        data = response.json()
        for key in ("total_requests", "successful_requests", "failed_requests", "avg_latency_ms"):
            assert key in data, f"Missing key: {key}"

    def test_metrics_counts_are_non_negative(self, client: TestClient):
        response = client.get("/metrics")
        data = response.json()
        assert data["total_requests"] >= 0
        assert data["successful_requests"] >= 0
        assert data["failed_requests"] >= 0

    def test_metrics_avg_latency_is_non_negative(self, client: TestClient):
        response = client.get("/metrics")
        data = response.json()
        assert data["avg_latency_ms"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Root /
# ─────────────────────────────────────────────────────────────────────────────

class TestRootEndpoint:
    def test_root_returns_200(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_service_key(self, client: TestClient):
        data = client.get("/").json()
        assert "service" in data
        assert "endpoints" in data


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Predictor Unit Tests (no HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictorUnit:
    """Direct unit tests for the Predictor class."""

    def test_predictor_instantiates(self):
        predictor = Predictor()
        assert predictor is not None

    def test_predictor_model_loaded_is_bool(self):
        predictor = Predictor()
        assert isinstance(predictor.model_loaded, bool)

    def test_predictor_predict_returns_dict_when_loaded(self):
        predictor = Predictor()
        if not predictor.model_loaded:
            pytest.skip("Model not found — skipping predictor unit test.")
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        result = predictor.predict(img)
        assert isinstance(result, dict)
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_predictor_converts_non_rgb_image(self):
        """Predictor should handle RGBA images gracefully by converting to RGB."""
        predictor = Predictor()
        if not predictor.model_loaded:
            pytest.skip("Model not found — skipping predictor unit test.")
        rgba_img = Image.new("RGBA", (224, 224), color=(100, 150, 200, 128))
        result = predictor.predict(rgba_img)
        assert result["label"] in ("cat", "dog")
