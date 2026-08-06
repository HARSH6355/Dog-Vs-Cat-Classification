"""
helpers.py
----------
Shared utility functions used across the MLOps pipeline:
- Device detection (CPU/GPU)
- Metric calculation
- Plot generation (loss curves, confusion matrix)
- MLflow artifact logging helpers
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for notebooks & scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Returns CUDA if available, else MPS (Apple Silicon), else CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Returns the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Loss & Accuracy Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str = "models/training_curves.png",
    title: str = "Training & Validation Curves",
) -> str:
    """
    Plots and saves training & validation loss/accuracy curves.

    Args:
        history:   Dict with keys 'train_loss', 'val_loss',
                   'train_acc', 'val_acc' — each a list of epoch values.
        save_path: Output PNG path.
        title:     Plot title.

    Returns:
        Absolute path to the saved figure.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── Loss ────────────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Accuracy ─────────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs, history["val_acc"],   "r-o", label="Val Acc",   markersize=4)
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    abs_path = str(Path(save_path).resolve())
    plt.savefig(abs_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved → {abs_path}")
    return abs_path


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str = "models/confusion_matrix.png",
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> str:
    """
    Computes and saves a confusion matrix heatmap.

    Args:
        y_true:      Ground-truth labels.
        y_pred:      Predicted labels.
        class_names: List of class name strings.
        save_path:   Output PNG path.
        normalize:   If True, shows row-normalized percentages.
        title:       Plot title.

    Returns:
        Absolute path to the saved figure.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_plot = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
        fmt = ".1f"
        val_label = "(%)"
    else:
        cm_plot = cm
        fmt = "d"
        val_label = "(count)"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"{title} {val_label}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    abs_path = str(Path(save_path).resolve())
    plt.tight_layout()
    plt.savefig(abs_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {abs_path}")
    return abs_path


# ─────────────────────────────────────────────────────────────────────────────
# Classification Report
# ─────────────────────────────────────────────────────────────────────────────

def get_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
) -> str:
    """Returns sklearn classification report as a string."""
    return classification_report(y_true, y_pred, target_names=class_names)


# ─────────────────────────────────────────────────────────────────────────────
# Model checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: dict,
    save_path: str,
    is_best: bool = False,
) -> None:
    """Saves model + optimizer state to disk."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }
    torch.save(state, save_path)
    if is_best:
        best_path = str(Path(save_path).parent / "best_model.pt")
        torch.save(state, best_path)
        logger.info(f"🏆 New best model saved → {best_path}")
    logger.info(f"Checkpoint saved → {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int, dict]:
    """Loads model + optimizer from a checkpoint file."""
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    epoch   = state.get("epoch", 0)
    history = state.get("history", {})
    logger.info(f"Loaded checkpoint from epoch {epoch}: {checkpoint_path}")
    return model, optimizer, epoch, history
