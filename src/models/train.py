"""
train.py
--------
End-to-end training script for the Baseline CNN with full MLflow tracking.

Logs to MLflow:
  - Parameters: epochs, batch_size, lr, optimizer, scheduler, model config
  - Metrics:    train_loss, val_loss, train_acc, val_acc (per epoch)
                test_loss, test_acc (final)
  - Artifacts:  best_model.pt, training_curves.png, confusion_matrix.png,
                classification_report.txt

Usage (from project root):
    python src/models/train.py
    python src/models/train.py --config configs/config.yaml --epochs 10
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

# Make src importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.preprocess import get_dataloaders
from src.models.baseline_cnn import build_model
from src.utils.helpers import (
    get_device,
    count_parameters,
    plot_training_curves,
    plot_confusion_matrix,
    get_classification_report,
    save_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    training: bool = True,
) -> Tuple[float, float]:
    """
    Runs one epoch (train or eval).

    Args:
        model:     The CNN model.
        loader:    DataLoader for this split.
        criterion: Loss function.
        optimizer: Optimizer (only used when training=True).
        device:    Compute device.
        training:  If True, computes gradients and updates weights.

    Returns:
        Tuple of (avg_loss, accuracy_percent).
    """
    model.train() if training else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct  += predicted.eq(labels).sum().item()
            total    += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Tuple[List[int], List[int]]:
    """Runs inference and returns (y_true, y_pred) lists."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    return y_true, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(config_path: str = "configs/config.yaml") -> None:
    """
    Full training pipeline with MLflow experiment tracking.

    Steps:
      1. Load config & data
      2. Build model
      3. For each epoch: train → validate → log metrics → checkpoint
      4. Evaluate on test set
      5. Log artifacts (curves, confusion matrix, model) to MLflow
    """
    cfg = load_config(config_path)
    tr_cfg  = cfg["training"]
    ml_cfg  = cfg["mlflow"]
    ds_cfg  = cfg["dataset"]

    device = get_device()
    class_names = ds_cfg["classes"]

    # ── Data ─────────────────────────────────────────────────────────────────
    logger.info("Loading data …")
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(config_path)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    logger.info(model.summary())

    # ── Loss / Optimizer / Scheduler ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=tr_cfg["learning_rate"],
        weight_decay=tr_cfg["weight_decay"],
    )
    scheduler = StepLR(
        optimizer,
        step_size=tr_cfg["scheduler_step_size"],
        gamma=tr_cfg["scheduler_gamma"],
    )

    # ── MLflow Setup ─────────────────────────────────────────────────────────
    tracking_uri = str(Path(ml_cfg["tracking_uri"]).resolve())
    mlflow.set_tracking_uri(f"file:///{tracking_uri}")
    mlflow.set_experiment(ml_cfg["experiment_name"])

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{ml_cfg['run_name_prefix']}_{timestamp}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started → run_id: {run_id}")

        # ── Log hyperparameters ───────────────────────────────────────────────
        mlflow.log_params({
            "epochs":            tr_cfg["epochs"],
            "batch_size":        tr_cfg["batch_size"],
            "learning_rate":     tr_cfg["learning_rate"],
            "weight_decay":      tr_cfg["weight_decay"],
            "optimizer":         tr_cfg["optimizer"],
            "scheduler":         tr_cfg["scheduler"],
            "scheduler_step":    tr_cfg["scheduler_step_size"],
            "scheduler_gamma":   tr_cfg["scheduler_gamma"],
            "model_filters":     str(cfg["model"]["filters"]),
            "model_fc_units":    cfg["model"]["fc_units"],
            "model_dropout":     cfg["model"]["dropout_rate"],
            "num_parameters":    count_parameters(model),
            "image_size":        str(ds_cfg["image_size"]),
            "train_split":       ds_cfg["train_split"],
            "augmentation":      str(cfg["augmentation"]),
        })

        # ── Training Loop ────────────────────────────────────────────────────
        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }

        best_val_acc  = 0.0
        patience_ctr  = 0
        patience      = tr_cfg.get("early_stopping_patience", 5)
        best_ckpt_path = str(Path(cfg["paths"]["models"]) / "best_model.pt")

        for epoch in range(1, tr_cfg["epochs"] + 1):
            t_start = time.time()

            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, device, training=True
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, optimizer, device, training=False
            )
            scheduler.step()

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            elapsed = time.time() - t_start
            logger.info(
                f"Epoch [{epoch:>3}/{tr_cfg['epochs']}] "
                f"| Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% "
                f"| Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% "
                f"| {elapsed:.1f}s"
            )

            # Log per-epoch metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
                "lr":         scheduler.get_last_lr()[0],
            }, step=epoch)

            # Checkpoint if best validation accuracy
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                patience_ctr = 0
                save_checkpoint(
                    model, optimizer, epoch, history,
                    save_path=best_ckpt_path,
                    is_best=True,
                )
            else:
                patience_ctr += 1

            # Save latest checkpoint every 5 epochs
            if epoch % 5 == 0:
                ckpt = str(Path(cfg["paths"]["models"]) / f"checkpoint_epoch_{epoch:03d}.pt")
                save_checkpoint(model, optimizer, epoch, history, ckpt)

            # Early stopping
            if patience_ctr >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        # ── Test Evaluation ──────────────────────────────────────────────────
        logger.info("Evaluating on test set …")
        y_true, y_pred = evaluate_on_loader(model, test_loader, device, class_names)
        test_loss, test_acc = run_epoch(
            model, test_loader, criterion, optimizer, device, training=False
        )

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_acc":  test_acc,
            "best_val_acc": best_val_acc,
        })

        logger.info(f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.2f}%")

        # ── Artifacts ────────────────────────────────────────────────────────
        # 1. Training curves
        curves_path = plot_training_curves(
            history, save_path="models/training_curves.png"
        )
        mlflow.log_artifact(curves_path, artifact_path="plots")

        # 2. Confusion matrix
        cm_path = plot_confusion_matrix(
            y_true, y_pred, class_names=class_names,
            save_path="models/confusion_matrix.png"
        )
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # 3. Classification report
        report = get_classification_report(y_true, y_pred, class_names)
        report_path = "models/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        logger.info(f"\nClassification Report:\n{report}")

        # 4. Log model (best checkpoint)
        if ml_cfg.get("log_model", True):
            # Load best weights before logging
            best_state = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(best_state["model_state_dict"])
            mlflow.pytorch.log_model(model, artifact_path="model")

        logger.info(f"✅ Training complete! Best Val Acc: {best_val_acc:.2f}%")
        logger.info(f"   MLflow run_id: {run_id}")
        logger.info(f"   To view results: mlflow ui --port 5000")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline CNN — Cats vs Dogs")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to YAML config file (default: configs/config.yaml)"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--lr",     type=float, default=None, help="Override learning rate")
    return parser.parse_args()


if __name__ == "__main__":
    # Run from project root
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)

    args = parse_args()
    cfg = load_config(args.config)

    # Allow CLI overrides
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr

    train(args.config)
