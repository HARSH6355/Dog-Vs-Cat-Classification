#!/usr/bin/env python3
"""
simulate_requests.py
--------------------
M5 -- Post-Deployment Model Performance Tracking

Sends a balanced batch of REAL images from data/raw/PetImages/ to the
running Cat vs Dog API and collects predictions, latencies, and a
performance report.

Image source priority:
  1. data/raw/PetImages/Cat/ and data/raw/PetImages/Dog/  (real labelled images)
  2. Any directory passed via --images-dir
  3. Synthetic fallback (only if no real images found anywhere)

Usage:
    # Default -- uses data/raw/PetImages automatically:
    python scripts/simulate_requests.py

    # Override number of requests (balanced: N/2 cats, N/2 dogs):
    python scripts/simulate_requests.py --n 40

    # Custom image directory:
    python scripts/simulate_requests.py --images-dir /some/other/path

    # Against a different API:
    python scripts/simulate_requests.py --url http://host:8000 --n 30

Output:
    - Console : per-request table + summary
    - File    : monitoring/performance_report.json
"""

import argparse
import io
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import requests
    from PIL import Image, UnidentifiedImageError
except ImportError:
    print("ERROR: pip install requests Pillow")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_RAW_CAT = Path("data/raw/PetImages/Cat")
_DEFAULT_RAW_DOG = Path("data/raw/PetImages/Dog")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Image collection
# ─────────────────────────────────────────────────────────────────────────────

def _sample_from_folder(folder: Path, label: str, n: int) -> List[Tuple[str, Path]]:
    """Randomly sample up to n valid JPEG/PNG files from a folder."""
    all_files = [p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS]
    random.shuffle(all_files)
    valid = []
    for p in all_files:
        if len(valid) >= n:
            break
        try:
            # Quick validity check -- skip corrupted files (Kaggle dataset has a few)
            with Image.open(p) as img:
                img.verify()
            valid.append((label, p))
        except (UnidentifiedImageError, Exception):
            continue
    return valid


def _collect_balanced_real_images(n: int) -> List[Tuple[str, Path]]:
    """
    Collect n/2 cat images + n/2 dog images from the default raw data folders.
    Returns list of (true_label, path) tuples, shuffled.
    """
    half = n // 2
    remainder = n - half

    cat_items = _sample_from_folder(_DEFAULT_RAW_CAT, "cat", half)
    dog_items = _sample_from_folder(_DEFAULT_RAW_DOG, "dog", remainder)

    items = cat_items + dog_items
    random.shuffle(items)
    return items


def _collect_from_custom_dir(directory: Path, n: int) -> List[Tuple[str, Path]]:
    """Collect images from a custom directory, inferring labels from subfolder names."""
    items = []
    # Check for Cat/Dog subfolders
    cat_dir = directory / "Cat"
    dog_dir = directory / "Dog"
    if cat_dir.exists() and dog_dir.exists():
        half = n // 2
        items.extend(_sample_from_folder(cat_dir, "cat", half))
        items.extend(_sample_from_folder(dog_dir, "dog", n - half))
        random.shuffle(items)
        return items

    # Flat directory -- infer label from filename
    for p in sorted(directory.rglob("*")):
        if p.suffix.lower() not in _IMAGE_EXTS:
            continue
        name_lower = p.name.lower()
        if "cat" in name_lower:
            label = "cat"
        elif "dog" in name_lower:
            label = "dog"
        else:
            label = "unknown"
        items.append((label, p))
        if len(items) >= n:
            break
    random.shuffle(items)
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK: List[Tuple[str, tuple]] = (
    [("cat", (200, 170, 130))] * 5 +
    [("cat", (80,  60,  40))]  * 5 +
    [("dog", (190, 150,  90))] * 5 +
    [("dog", (40,  30,  20))]  * 5
)


def _make_synthetic_bytes(colour: tuple) -> bytes:
    r = max(0, min(255, colour[0] + random.randint(-20, 20)))
    g = max(0, min(255, colour[1] + random.randint(-20, 20)))
    b = max(0, min(255, colour[2] + random.randint(-20, 20)))
    img = Image.new("RGB", (224, 224), color=(r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# API client
# ─────────────────────────────────────────────────────────────────────────────

def _predict(base_url: str, image_bytes: bytes, filename: str) -> Optional[dict]:
    """POST image to /predict, return response dict or None on failure."""
    try:
        r = requests.post(
            f"{base_url}/predict",
            files={"file": (filename, io.BytesIO(image_bytes), "image/jpeg")},
            timeout=30,
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _build_and_save_report(records: list, report_path: Path) -> dict:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    preds      = [r["predicted"]  for r in records if r["predicted"]]
    truths     = [r["true_label"] for r in records]
    latencies  = [r["latency_ms"] for r in records if r["latency_ms"] is not None]
    confs      = [r["confidence"] for r in records if r["confidence"] is not None]

    known = [(t, p) for t, p in zip(truths, preds) if t != "unknown" and p]
    correct = sum(t == p for t, p in known)
    accuracy = round(correct / len(known), 4) if known else None

    latencies_sorted = sorted(latencies)
    p95_idx = int(len(latencies_sorted) * 0.95) if latencies_sorted else 0

    report = {
        "generated_at":          datetime.utcnow().isoformat() + "Z",
        "image_source":          "real (data/raw/PetImages/)",
        "total_requests":        len(records),
        "successful_predictions": len(preds),
        "failed_predictions":    len(records) - len(preds),
        "labeled_samples":       len(known),
        "correct_predictions":   correct,
        "accuracy":              accuracy if accuracy is not None else "N/A",
        "avg_latency_ms":        round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "p95_latency_ms":        round(latencies_sorted[p95_idx], 2) if latencies_sorted else 0,
        "avg_confidence":        round(sum(confs) / len(confs), 4) if confs else 0,
        "class_distribution":    {"cat": preds.count("cat"), "dog": preds.count("dog")},
        "true_label_distribution": {
            "cat":     truths.count("cat"),
            "dog":     truths.count("dog"),
            "unknown": truths.count("unknown"),
        },
        "records": records,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def _print_summary(report: dict) -> None:
    print("\n" + "=" * 60)
    print("  Cat vs Dog API -- Post-Deployment Performance Report")
    print("=" * 60)
    print(f"  Image source      : {report['image_source']}")
    print(f"  Generated at      : {report['generated_at']}")
    print(f"  Total requests    : {report['total_requests']}")
    print(f"  Successful        : {report['successful_predictions']}")
    print(f"  Failed            : {report['failed_predictions']}")
    print(f"  Accuracy          : {report['accuracy']} ({report['correct_predictions']}/{report['labeled_samples']})")
    print(f"  Avg latency       : {report['avg_latency_ms']} ms")
    print(f"  P95 latency       : {report['p95_latency_ms']} ms")
    print(f"  Avg confidence    : {report['avg_confidence']:.1%}")
    print()
    print("  Predicted class distribution:")
    print(f"    cat : {report['class_distribution']['cat']}")
    print(f"    dog : {report['class_distribution']['dog']}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="M5 -- Post-deployment model performance tracking with real images"
    )
    parser.add_argument("--url",        default="http://localhost:8000",
                        help="API base URL (default: http://localhost:8000)")
    parser.add_argument("--n",          type=int, default=20,
                        help="Total requests to send -- balanced cat/dog (default: 20)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Optional: override image directory")
    parser.add_argument("--output",     type=str,
                        default="monitoring/performance_report.json",
                        help="Output JSON report path")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for reproducible sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    base_url    = args.url.rstrip("/")
    report_path = Path(args.output)

    print(f"\n{'=' * 60}")
    print(f"  Cat vs Dog -- Post-Deployment Simulation")
    print(f"{'=' * 60}")
    print(f"  API     : {base_url}")
    print(f"  N       : {args.n} requests")
    print(f"  Output  : {report_path}")

    # -- Health check ─────────────────────────────────────────────────────────
    try:
        health = requests.get(f"{base_url}/health", timeout=10).json()
        if not health.get("model_loaded"):
            print("\n[ERROR] API /health reports model NOT loaded. Aborting.")
            return 1
        print("\n  [OK] API healthy -- model_loaded=True")
    except Exception as e:
        print(f"\n[ERROR] Cannot reach API at {base_url}: {e}")
        return 1

    # -- Collect images ───────────────────────────────────────────────────────
    use_synthetic = False
    items: List[Tuple[str, Path]] = []

    if args.images_dir:
        custom_dir = Path(args.images_dir)
        print(f"\n  [SRC] Custom directory: {custom_dir}")
        items = _collect_from_custom_dir(custom_dir, args.n)

    elif _DEFAULT_RAW_CAT.exists() and _DEFAULT_RAW_DOG.exists():
        print(f"\n  [SRC] Real images: data/raw/PetImages/ (Cat + Dog)")
        items = _collect_balanced_real_images(args.n)

    if not items:
        print(f"  [WARN] No real images found -- falling back to synthetic images.")
        use_synthetic = True

    if use_synthetic:
        source_label = "synthetic (colour patches)"
    else:
        print(f"  [INFO] Sampled {len(items)} images "
              f"({sum(1 for t,_ in items if t=='cat')} cats, "
              f"{sum(1 for t,_ in items if t=='dog')} dogs)")
        source_label = "real (data/raw/PetImages/)"

    # -- Run requests ─────────────────────────────────────────────────────────
    records = []
    header = f"  {'#':<5} {'True':^8} {'Pred':^8} {'Conf':^8} {'Lat(ms)':^10} {'Status':^6} Filename"
    print(f"\n{header}")
    print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*6} --------")

    for i in range(args.n):
        if use_synthetic:
            true_label, colour = _FALLBACK[i % len(_FALLBACK)]
            img_bytes = _make_synthetic_bytes(colour)
            filename  = f"synthetic_{i:03d}.jpg"
        else:
            true_label, img_path = items[i]
            img_bytes = img_path.read_bytes()
            filename  = img_path.name

        t0     = time.perf_counter()
        result = _predict(base_url, img_bytes, filename)
        total_ms = round((time.perf_counter() - t0) * 1000, 1)

        if result:
            pred   = result["label"]
            conf   = result["confidence"]
            lat    = result.get("latency_ms", total_ms)
            status = "PASS"
        else:
            pred = conf = lat = None
            status = "FAIL"

        records.append({
            "request_id":  i + 1,
            "filename":    filename,
            "true_label":  true_label,
            "predicted":   pred,
            "confidence":  conf,
            "latency_ms":  lat,
            "correct":     (pred == true_label) if pred and true_label != "unknown" else None,
            "timestamp":   datetime.utcnow().isoformat() + "Z",
        })

        conf_str = f"{conf:.1%}" if conf is not None else "  --  "
        lat_str  = f"{lat:.1f}"  if lat  is not None else "  --  "
        print(f"  {i+1:<5} {true_label:^8} {(pred or '--'):^8} "
              f"{conf_str:^8} {lat_str:^10} {status:^6} {filename[:30]}")

    # -- Report ───────────────────────────────────────────────────────────────
    report = _build_and_save_report(records, report_path)
    report["image_source"] = source_label
    # Re-save with updated source label
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    _print_summary(report)
    print(f"\n  [SAVED] Full report: {report_path}\n")

    if report["failed_predictions"] > 0:
        print(f"  [WARN] {report['failed_predictions']} requests failed.\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
