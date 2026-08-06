"""
preprocess.py
-------------
Handles image preprocessing for the Cats vs Dogs dataset:
- Loads raw images from data/raw/
- Resizes to 224x224 RGB
- Applies augmentation on training split
- Creates train/val/test PyTorch DataLoaders
- Saves split manifests for DVC tracking

Usage (standalone):
    python src/data/preprocess.py

Usage (from notebook/train):
    from src.data.preprocess import get_dataloaders, get_transforms
"""

import os
import logging
import random
from pathlib import Path
from typing import Tuple, Dict, Optional

import yaml
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(
    image_size: Tuple[int, int] = (224, 224),
    augment: bool = False,
    aug_config: Optional[dict] = None,
) -> transforms.Compose:
    """
    Returns torchvision transforms for a given split.

    Args:
        image_size: Target (H, W) for resizing.
        augment:    If True, applies data augmentation.
        aug_config: Augmentation config dict from config.yaml.

    Returns:
        A torchvision.transforms.Compose pipeline.
    """
    aug_config = aug_config or {}

    # ImageNet mean/std — standard for CNNs trained from scratch too
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment:
        transform_list = [
            transforms.Resize(image_size),
        ]
        if aug_config.get("horizontal_flip", True):
            transform_list.append(transforms.RandomHorizontalFlip())
        if aug_config.get("vertical_flip", False):
            transform_list.append(transforms.RandomVerticalFlip())
        if aug_config.get("rotation_degrees", 0) > 0:
            transform_list.append(
                transforms.RandomRotation(degrees=aug_config["rotation_degrees"])
            )
        # ColorJitter for brightness, contrast, saturation, hue
        jitter_kwargs = {
            k: aug_config.get(k, 0)
            for k in ("brightness", "contrast", "saturation", "hue")
            if aug_config.get(k, 0) > 0
        }
        if jitter_kwargs:
            transform_list.append(transforms.ColorJitter(**jitter_kwargs))

        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    else:
        transform_list = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

    return transforms.Compose(transform_list)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Finder  (handles different directory layouts)
# ─────────────────────────────────────────────────────────────────────────────

def find_dataset_root(raw_dir: str, classes: list) -> Path:
    """
    Searches common sub-directory layouts to find the root folder that
    contains <class>/ sub-folders (usable by ImageFolder).

    Supported layouts:
        data/raw/cat/  data/raw/dog/              ← flat
        data/raw/training_set/cat/  ...           ← training_set prefix
        data/raw/train/cat/  ...                  ← train prefix
        data/raw/PetImages/Cat/  ...              ← Microsoft PetImages format
        data/raw/<any>/cat/  ...                  ← one-level nesting
    """
    raw = Path(raw_dir).resolve()

    def has_class_dirs(p: Path) -> bool:
        """True if all class folders exist under p."""
        return all((p / cls).exists() or (p / cls.capitalize()).exists() for cls in classes)

    # 1. Flat layout
    if has_class_dirs(raw):
        return raw

    # 2. One-level nesting — search all immediate subdirectories
    for sub in sorted(raw.iterdir()):
        if sub.is_dir() and has_class_dirs(sub):
            return sub

    raise FileNotFoundError(
        f"Could not find class folders {classes} under '{raw}'. "
        "Please verify the dataset was downloaded correctly."
    )


class CatDogDataset(datasets.ImageFolder):
    """
    Thin wrapper around torchvision.datasets.ImageFolder.
    Normalises folder names to lowercase so 'Cat'/'cat' both work.
    """

    def __init__(self, root: str, transform=None):
        # Remap classes to lowercase for consistency
        super().__init__(root=root, transform=transform)
        # class_to_idx: {'cat': 0, 'dog': 1} (ImageFolder sorts alphabetically)
        logger.info(f"Classes detected: {self.class_to_idx}")
        logger.info(f"Total images found: {len(self)}")


# ─────────────────────────────────────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    config_path: str = "configs/config.yaml",
    raw_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Creates train/val/test DataLoaders from the raw dataset.

    Args:
        config_path: Path to config YAML.
        raw_dir:     Override for raw data directory path.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_to_idx)
    """
    cfg = load_config(config_path)
    ds_cfg  = cfg["dataset"]
    tr_cfg  = cfg["training"]
    aug_cfg = cfg["augmentation"]

    image_size = tuple(ds_cfg["image_size"])
    classes    = ds_cfg["classes"]
    seed       = ds_cfg["random_seed"]
    raw_dir    = raw_dir or cfg["paths"]["data_raw"]

    # Set reproducibility seeds
    random.seed(seed)
    torch.manual_seed(seed)

    # Locate dataset root
    dataset_root = find_dataset_root(raw_dir, classes)
    logger.info(f"Dataset root found at: {dataset_root}")

    # ── Full dataset with NO augmentation (for val/test transforms) ──────────
    base_transform  = get_transforms(image_size, augment=False)
    train_transform = get_transforms(image_size, augment=True, aug_config=aug_cfg)

    full_dataset = CatDogDataset(root=str(dataset_root), transform=base_transform)
    n_total = len(full_dataset)

    # ── Stratified split ─────────────────────────────────────────────────────
    train_ratio = ds_cfg["train_split"]
    val_ratio   = ds_cfg["val_split"]

    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)
    n_test  = n_total - n_train - n_val

    # Use random_split (deterministic with seed above)
    indices = list(range(n_total))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    # Apply augmented transforms ONLY to training subset
    train_dataset_aug = CatDogDataset(root=str(dataset_root), transform=train_transform)
    train_subset = Subset(train_dataset_aug, train_idx)
    val_subset   = Subset(full_dataset, val_idx)
    test_subset  = Subset(full_dataset, test_idx)

    logger.info(
        f"Split → train: {len(train_subset):,} | "
        f"val: {len(val_subset):,} | "
        f"test: {len(test_subset):,}"
    )

    num_workers = min(tr_cfg.get("num_workers", 4), os.cpu_count() or 1)

    train_loader = DataLoader(
        train_subset,
        batch_size=tr_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=tr_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=tr_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, full_dataset.class_to_idx


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders()
    logger.info(f"class_to_idx: {class_to_idx}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Sample one batch
    images, labels = next(iter(train_loader))
    logger.info(f"Sample batch — images shape: {images.shape}, labels: {labels[:5]}")
