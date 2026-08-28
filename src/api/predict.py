"""
predict.py
----------
Core inference engine for the Cat vs Dog classifier.

Loads best_model.pt once at startup and exposes a `predict_image()` function
that accepts a PIL Image and returns a structured prediction dict.

Usage:
    from src.api.predict import Predictor
    predictor = Predictor()
    result = predictor.predict(pil_image)
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from torchvision import transforms
from PIL import Image

# Ensure project root imports work when running from any directory
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.baseline_cnn import build_model

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pt"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

CLASS_NAMES = ["cat", "dog"]


# ─────────────────────────────────────────────────────────────────────────────
# Inference Transform  (must match training normalisation exactly)
# ─────────────────────────────────────────────────────────────────────────────

def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    """
    Returns the deterministic transform pipeline used at inference time.
    No augmentation — only resize, center-crop, tensor conversion and
    ImageNet normalisation (same statistics used during training).
    """
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),   # slight over-resize
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet means
            std=[0.229, 0.224, 0.225],    # ImageNet stds
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Predictor Class
# ─────────────────────────────────────────────────────────────────────────────

class Predictor:
    """
    Stateful predictor that loads the model once and serves predictions.

    Args:
        model_path: Path to best_model.pt checkpoint.
        config_path: Path to configs/config.yaml.
        device:      Torch device (auto-detected if None).
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        config_path: Path = DEFAULT_CONFIG_PATH,
        device: torch.device = None,
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self.model_loaded = False
        self._load()

    # ── Private ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load model weights and build the preprocessing transform."""
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            return

        try:
            # Load config
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f)

            image_size = cfg["dataset"]["image_size"][0]  # e.g. 224

            # Build model architecture then load weights
            self.model = build_model(cfg)
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            # Handle both raw state_dict and checkpoint dicts
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()

            self.transform = get_inference_transform(image_size)
            self.model_loaded = True
            logger.info(
                f"Model loaded from {self.model_path} on device={self.device}"
            )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference on a single PIL Image.

        Args:
            image: A PIL Image (RGB).

        Returns:
            dict with keys:
                label        (str)  — "cat" or "dog"
                confidence   (float)— probability of the predicted class  [0, 1]
                probabilities (dict) — {"cat": float, "dog": float}
                latency_ms   (float)— inference time in milliseconds
        """
        if not self.model_loaded:
            raise RuntimeError("Model is not loaded. Check server logs for details.")

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        t0 = time.perf_counter()

        # Preprocess
        tensor = self.transform(image).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # Inference
        with torch.no_grad():
            probs = self.model.predict_proba(tensor)          # (1, 2)
            probs_np = probs.squeeze(0).cpu().numpy().tolist()  # [p_cat, p_dog]

        latency_ms = (time.perf_counter() - t0) * 1000
        pred_idx = int(probs_np.index(max(probs_np)))

        return {
            "label": CLASS_NAMES[pred_idx],
            "confidence": round(float(max(probs_np)), 4),
            "probabilities": {
                "cat": round(float(probs_np[0]), 4),
                "dog": round(float(probs_np[1]), 4),
            },
            "latency_ms": round(latency_ms, 2),
        }
