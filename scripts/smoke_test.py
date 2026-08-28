#!/usr/bin/env python3
"""
smoke_test.py
-------------
Post-deployment smoke test for the Cat vs Dog Classifier API.

Runs two checks:
  1. GET  /health  — service is up and model is loaded
  2. POST /predict — returns a valid cat/dog prediction

Exits with code 0 if ALL checks pass.
Exits with code 1 if ANY check fails (causes CI pipeline to fail).

Usage:
    python scripts/smoke_test.py                       # default http://localhost:8000
    python scripts/smoke_test.py --url http://HOST:PORT
"""

import argparse
import io
import sys
import time

try:
    import requests
    from PIL import Image
except ImportError:
    print("ERROR: Missing dependencies. Run: pip install requests Pillow")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def _pass(msg: str) -> None:
    print(f"  ✅  {msg}")


def _fail(msg: str) -> None:
    print(f"  ❌  {msg}")


def _make_test_image_bytes() -> bytes:
    """Create a simple 224x224 RGB image as JPEG bytes for the prediction test."""
    img = Image.new("RGB", (224, 224), color=(120, 80, 40))  # brownish
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def wait_for_service(base_url: str, timeout: int = 60) -> bool:
    """Poll /health until service responds or timeout is reached."""
    print(f"\n  Waiting for service at {base_url} (up to {timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code in (200, 503):
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Smoke Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_health(base_url: str) -> bool:
    """Check 1: GET /health must return 200 with model_loaded=true."""
    url = f"{base_url}/health"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            _fail(f"/health returned HTTP {r.status_code} (expected 200)")
            return False

        data = r.json()
        if not data.get("model_loaded"):
            _fail(f"/health: model_loaded is False — model failed to load!")
            return False
        if data.get("status") != "ok":
            _fail(f"/health: status='{data.get('status')}' (expected 'ok')")
            return False

        _pass(f"/health → status=ok, model_loaded=True  [HTTP 200]")
        return True

    except Exception as e:
        _fail(f"/health request failed: {e}")
        return False


def test_predict(base_url: str) -> bool:
    """Check 2: POST /predict must return a valid cat/dog prediction."""
    url = f"{base_url}/predict"
    try:
        img_bytes = _make_test_image_bytes()
        files = {"file": ("smoke_test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        r = requests.post(url, files=files, timeout=30)

        if r.status_code != 200:
            _fail(f"/predict returned HTTP {r.status_code} (expected 200)")
            _fail(f"Response body: {r.text[:300]}")
            return False

        data = r.json()

        # Validate response structure
        required_keys = {"label", "confidence", "probabilities", "latency_ms"}
        missing = required_keys - set(data.keys())
        if missing:
            _fail(f"/predict response missing keys: {missing}")
            return False

        # Validate label value
        if data["label"] not in ("cat", "dog"):
            _fail(f"/predict: label='{data['label']}' (expected 'cat' or 'dog')")
            return False

        # Validate confidence is a probability
        conf = data["confidence"]
        if not (0.0 <= conf <= 1.0):
            _fail(f"/predict: confidence={conf} is out of [0, 1] range")
            return False

        # Validate probabilities sum to ~1
        probs = data["probabilities"]
        total = probs.get("cat", 0) + probs.get("dog", 0)
        if abs(total - 1.0) > 0.01:
            _fail(f"/predict: probabilities sum to {total:.4f} (expected ~1.0)")
            return False

        _pass(
            f"/predict → label={data['label']}, "
            f"confidence={conf:.2%}, "
            f"latency={data['latency_ms']}ms  [HTTP 200]"
        )
        return True

    except Exception as e:
        _fail(f"/predict request failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke tests for Cat vs Dog API")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the running API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=60,
        help="Seconds to wait for service to become available (default: 60)",
    )
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    _banner("Cat vs Dog API — Smoke Tests")
    print(f"  Target: {base_url}")

    # Wait for service to be ready
    if not wait_for_service(base_url, timeout=args.wait):
        _fail(f"Service did not become available within {args.wait}s")
        return 1

    results = []

    _banner("Check 1: Health Endpoint")
    results.append(test_health(base_url))

    _banner("Check 2: Prediction Endpoint")
    results.append(test_predict(base_url))

    _banner("Results")
    passed = sum(results)
    total = len(results)

    if all(results):
        print(f"  🎉  ALL {total}/{total} smoke tests PASSED")
        print(f"  🚀  Deployment is healthy and serving predictions!")
        return 0
    else:
        print(f"  💥  {total - passed}/{total} smoke tests FAILED")
        print(f"  🛑  Deployment is NOT healthy — rolling back recommended!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
