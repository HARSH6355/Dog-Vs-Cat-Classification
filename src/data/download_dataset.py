"""
download_dataset.py
-------------------
Downloads the Kaggle dataset directly into the project's data/raw/ directory
so the dataset is co-located with the project and not re-downloaded each session.

Usage (standalone):
    python src/data/download_dataset.py

Usage (from another script):
    from src.data.download_dataset import download_dataset
    path = download_dataset()
"""

import os
import shutil
import logging
from pathlib import Path

import kagglehub
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_dataset(
    kaggle_id: str = "bhavikjikadara/dog-and-cat-classification-dataset",
    target_dir: str = "data/raw",
    force_redownload: bool = False,
) -> Path:
    """
    Downloads a Kaggle dataset to the project's data/raw directory.

    If the data already exists in target_dir and force_redownload=False,
    the download is skipped — no re-downloading each session.

    Args:
        kaggle_id:        Kaggle dataset ID (e.g., 'user/dataset-name').
        target_dir:       Destination directory relative to project root.
        force_redownload: If True, re-downloads even if data already exists.

    Returns:
        Path: Absolute path to the downloaded data directory.
    """
    target_path = Path(target_dir).resolve()

    # ---- Check if data already exists ----------------------------------------
    if target_path.exists() and any(target_path.iterdir()) and not force_redownload:
        logger.info(
            f"Dataset already present at '{target_path}'. Skipping download.\n"
            f"  → Pass force_redownload=True to force a fresh download."
        )
        return target_path

    # ---- Download via kagglehub -----------------------------------------------
    logger.info(f"Downloading dataset '{kaggle_id}' from Kaggle …")
    kaggle_cache_path = kagglehub.dataset_download(kaggle_id)
    logger.info(f"Kaggle download cached at: {kaggle_cache_path}")

    # ---- Move/copy from Kaggle cache to project data/raw ----------------------
    target_path.mkdir(parents=True, exist_ok=True)
    src = Path(kaggle_cache_path)

    # Copy all files/folders from the cache path into target_path
    logger.info(f"Copying dataset to project directory: {target_path} …")
    for item in src.iterdir():
        dest = target_path / item.name
        if dest.exists():
            logger.info(f"  [SKIP] {item.name} already exists in target.")
            continue
        if item.is_dir():
            shutil.copytree(str(item), str(dest))
        else:
            shutil.copy2(str(item), str(dest))
        logger.info(f"  [COPIED] {item.name}")

    logger.info(f"✅ Dataset ready at: {target_path}")
    return target_path


def verify_dataset_structure(data_dir: str = "data/raw") -> None:
    """
    Verifies that the expected class folders exist in the dataset directory
    and prints a summary of image counts per class.

    Args:
        data_dir: Path to the raw data directory.
    """
    data_path = Path(data_dir).resolve()
    config = load_config()
    classes = config["dataset"]["classes"]

    logger.info("=" * 50)
    logger.info("Dataset Verification Summary")
    logger.info("=" * 50)

    total = 0
    for cls in classes:
        # Dataset might have class folders directly or inside a sub-folder
        # Try common layouts:
        candidates = [
            data_path / cls,
            data_path / "training_set" / cls,
            data_path / "train" / cls,
            data_path / "PetImages" / cls.capitalize(),
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found:
            n = len(list(found.glob("*.jpg")) + list(found.glob("*.jpeg")) + list(found.glob("*.png")))
            total += n
            logger.info(f"  {cls:>8}: {n:>6} images  ({found})")
        else:
            logger.warning(f"  {cls:>8}: NOT FOUND — checked {[str(c) for c in candidates]}")

    logger.info(f"  {'TOTAL':>8}: {total:>6} images")
    logger.info("=" * 50)


if __name__ == "__main__":
    # Ensure we run from project root
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)

    path = download_dataset()
    verify_dataset_structure()
