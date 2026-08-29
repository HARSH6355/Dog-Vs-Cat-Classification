#!/usr/bin/env python3
"""
simulate_requests.py
--------------------
M5 — Post-Deployment Model Performance Tracking

Sends a batch of simulated (and optionally real) requests to the running
Cat vs Dog API and collects predictions, latencies, and a performance report.

Usage:
    # Basic simulation (synthetic images):
    python scripts/simulate_requests.py

    # With real image directory:
    python scripts/simulate_requests.py --images-dir data/raw/

    # Against a different API:
    python scripts/simulate_requests.py --url http://localhost:8000 --n 30

Output:
    - Console: per-request log + summary table
    - File:    monitoring/performance_report.json  (saved automatically)
"""

import argparse
import io
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
    from PIL import Image
except ImportError:
    print("ERROR: pip install requests Pillow")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic image generators  (simulate different scene types)
# ─────────────────────────────────────────────────────────────────────────────

# Colours loosely associated with cat/dog photos in training data
_CAT_COLOURS = [
    (200, 170, 130),  # tabby beige
    (80,  60,  40),   # dark brown
    (230, 210, 180),  # light cream
    (120, 100,  80),  # grey-brown
]
_DOG_COLOURS = [
    (190, 150,  90),  # golden retriever
    (40,  30,  20),   # black lab
    (220, 200, 170),  # yellow lab
    (100, 80,  60),   # chocolate
]

# Simulated ground-truth labels (the "true" label each colour profile represents)
_SIMULATED_ITEMS: List[Tuple[str, tuple]] = (
    [("cat", c) for c in _CAT_COLOURS] * 5 +
    [("dog", c) for c in _DOG_COLOURS] * 5
)


def _make_synthetic_image(colour: tuple, size: int = 224) -> bytes:
    """Return JPEG bytes for a solid-colour test image."""
    # Add noise to prevent identical images
    r = max(0, min(255, colour[0] + random.randint(-20, 20)))
    g = max(0, min(255, colour[1] + random.randint(-20, 20)))
    b = max(0, min(255, colour[2] + random.randint(-20, 20)))
    img = Image.new("RGB", (size, size), color=(r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf.read()


def _collect_real_images(directory: Path, max_n: int) -> List[Tuple[str, Path]]:
    """Collect up to max_n real image files with inferred labels."""
    items = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in sorted(directory.rglob("*")):
        if p.suffix.lower() in exts:
            name_lower = p.name.lower()
            if "cat" in name_lower:
                label = "cat"
            elif "dog" in name_lower:
                label = "dog"
            else:
                label = "unknown"
            items.append((label, p))
            if len(items) >= max_n:
                break
    return items


# ─────────────────────────────────────────────────────────────────────────────
# API client
# ─────────────────────────────────────────────────────────────────────────────

def _predict(base_url: str, image_bytes: bytes, filename: str = "test.jpg") -> Optional[dict]:
    """POST image to /predict, return response dict or None on failure."""
    try:
        r = requests.post(
            f"{base_url}/predict",
            files={"file": (filename, io.BytesIO(image_bytes), "image/jpeg")},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _save_report(records: List[dict], report_path: Path) -> None:
    """Save performance report as JSON."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [r["predicted"] for r in records if r["predicted"]]
    true_labels = [r["true_label"] for r in records]
    latencies = [r["latency_ms"] for r in records if r["latency_ms"] is not None]
    confidences = [r["confidence"] for r in records if r["confidence"] is not None]

    # Accuracy only when true label is known
    known = [(t, p) for t, p in zip(true_labels, labels) if t != "unknown" and p]
    accuracy = sum(t == p for t, p in known) / len(known) if known else None

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_requests": len(records),
        "successful_predictions": sum(1 for r in records if r["predicted"]),
        "failed_predictions": sum(1 for r in records if not r["predicted"]),
        "accuracy_on_labeled": round(accuracy, 4) if accuracy is not None else "N/A",
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else 0,
        "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0,
        "class_distribution": {
            "cat": labels.count("cat"),
            "dog": labels.count("dog"),
        },
        "true_label_distribution": {
            "cat": true_labels.count("cat"),
            "dog": true_labels.count("dog"),
            "unknown": true_labels.count("unknown"),
        },
        "records": records,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def _print_summary(report: dict) -> None:
    """Print a formatted summary to stdout."""
    print("\n" + "=" * 60)
    print("  Cat vs Dog API — Post-Deployment Performance Report")
    print("=" * 60)
    print(f"  Generated at      : {report['generated_at']}")
    print(f"  Total requests    : {report['total_requests']}")
    print(f"  Successful        : {report['successful_predictions']}")
    print(f"  Failed            : {report['failed_predictions']}")
    print(f"  Accuracy (labeled): {report['accuracy_on_labeled']}")
    print(f"  Avg latency       : {report['avg_latency_ms']} ms")
    print(f"  P95 latency       : {report['p95_latency_ms']} ms")
    print(f"  Avg confidence    : {report['avg_confidence']:.1%}")
    print()
    print("  Predicted class distribution:")
    print(f"    🐱 cat : {report['class_distribution']['cat']}")
    print(f"    🐶 dog : {report['class_distribution']['dog']}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="M5 — Post-deployment model performance tracking"
    )
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the API (default: http://localhost:8000)")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of simulated requests (default: 20)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Optional: path to real image directory")
    parser.add_argument("--output", type=str,
                        default="monitoring/performance_report.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    report_path = Path(args.output)

    print(f"\n{'=' * 60}")
    print(f"  Cat vs Dog — Post-Deployment Simulation")
    print(f"{'=' * 60}")
    print(f"  API:     {base_url}")
    print(f"  N:       {args.n} requests")
    print(f"  Output:  {report_path}")

    # ── Check API health first ────────────────────────────────────────────────
    try:
        health = requests.get(f"{base_url}/health", timeout=10).json()
        if not health.get("model_loaded"):
            print("\n❌ API /health reports model is NOT loaded. Aborting.")
            return 1
        print(f"\n  ✅ API healthy — model_loaded=True")
    except Exception as e:
        print(f"\n❌ Cannot reach API at {base_url}: {e}")
        return 1

    # ── Prepare items ─────────────────────────────────────────────────────────
    if args.images_dir:
        img_dir = Path(args.images_dir)
        items = _collect_real_images(img_dir, args.n)
        print(f"  [DIR] Found {len(items)} real images in {img_dir}")
        if not items:
            print("  [WARN] No images found, falling back to synthetic.")
            items = None

    if not args.images_dir or not items:
        items = None  # use synthetic

    # ── Run requests ──────────────────────────────────────────────────────────
    records = []
    print(f"\n  {'#':<5} {'True':^8} {'Pred':^8} {'Conf':^8} {'Lat(ms)':^10} Status")
    print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*10} ------")

    for i in range(args.n):
        if items:
            true_label, img_path = items[i % len(items)]
            img_bytes = img_path.read_bytes()
            filename = img_path.name
        else:
            true_label, colour = _SIMULATED_ITEMS[i % len(_SIMULATED_ITEMS)]
            img_bytes = _make_synthetic_image(colour)
            filename = f"synthetic_{i:03d}.jpg"

        t0 = time.perf_counter()
        result = _predict(base_url, img_bytes, filename)
        total_ms = round((time.perf_counter() - t0) * 1000, 1)

        if result:
            pred = result["label"]
            conf = result["confidence"]
            lat  = result.get("latency_ms", total_ms)
            status = "PASS"
        else:
            pred = None
            conf = None
            lat  = None
            status = "FAIL"

        records.append({
            "request_id":  i + 1,
            "filename":    filename,
            "true_label":  true_label,
            "predicted":   pred,
            "confidence":  conf,
            "latency_ms":  lat,
            "timestamp":   datetime.utcnow().isoformat() + "Z",
        })

        conf_str = f"{conf:.2%}" if conf is not None else "   —  "
        lat_str  = f"{lat:.1f}" if lat is not None else "  —  "
        print(f"  {i+1:<5} {true_label:^8} {(pred or '—'):^8} {conf_str:^8} {lat_str:^10} {status}")

    # ── Save & print report ───────────────────────────────────────────────────
    report = _save_report(records, report_path)
    _print_summary(report)

    print(f"\n  📄 Full report saved to: {report_path}\n")

    failed = report["failed_predictions"]
    if failed > 0:
        print(f"  [WARN] {failed} requests failed. Check API logs.\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
