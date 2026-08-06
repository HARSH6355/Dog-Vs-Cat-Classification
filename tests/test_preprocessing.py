"""
test_preprocessing.py
---------------------
Unit tests for src/data/preprocess.py

Run: pytest tests/ -v
"""

import sys
import os
from pathlib import Path
import pytest
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import get_transforms


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_image():
    """Creates a simple 256x256 RGB PIL Image for testing."""
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def tiny_image():
    """Creates a 64x64 RGB PIL Image for edge-case testing."""
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: get_transforms
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTransforms:
    """Tests for the get_transforms() function."""

    def test_base_transform_returns_compose(self):
        """get_transforms should return a torchvision Compose object."""
        t = get_transforms((224, 224), augment=False)
        assert isinstance(t, transforms.Compose)

    def test_aug_transform_returns_compose(self):
        """get_transforms with augment=True should also return Compose."""
        t = get_transforms((224, 224), augment=True)
        assert isinstance(t, transforms.Compose)

    def test_output_tensor_shape_224(self, sample_image):
        """Transform should produce (3, 224, 224) tensor from a 256x256 PIL image."""
        t = get_transforms((224, 224), augment=False)
        result = t(sample_image)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224), f"Expected (3,224,224), got {result.shape}"

    def test_output_tensor_shape_tiny_image(self, tiny_image):
        """Tiny input (64x64) should still resize to 224x224 correctly."""
        t = get_transforms((224, 224), augment=False)
        result = t(tiny_image)
        assert result.shape == (3, 224, 224)

    def test_output_tensor_dtype(self, sample_image):
        """Output tensor should be float32."""
        t = get_transforms((224, 224), augment=False)
        result = t(sample_image)
        assert result.dtype == torch.float32

    def test_output_values_normalized(self, sample_image):
        """After ImageNet normalization, values should be centered around 0."""
        t = get_transforms((224, 224), augment=False)
        result = t(sample_image)
        # With ImageNet normalization, most pixel values should be in [-3, 3]
        assert result.min() > -5.0
        assert result.max() < 5.0

    def test_augmented_transform_different_from_base(self, sample_image):
        """Augmented and base transforms may differ (due to randomness)."""
        t_base = get_transforms((224, 224), augment=False)
        t_aug  = get_transforms((224, 224), augment=True,
                                aug_config={"horizontal_flip": True, "rotation_degrees": 30})
        base_out = t_base(sample_image)
        # Run augmentation multiple times — at least one should differ from base
        results = [t_aug(sample_image) for _ in range(10)]
        any_different = any(not torch.allclose(base_out, r) for r in results)
        assert any_different, "Augmented transform never produced a different result"

    def test_custom_image_size(self, sample_image):
        """Custom image size (128x128) should work."""
        t = get_transforms((128, 128), augment=False)
        result = t(sample_image)
        assert result.shape == (3, 128, 128)

    def test_augmentation_config_no_flip(self, sample_image):
        """With horizontal_flip=False, transform should still run without error."""
        t = get_transforms(
            (224, 224), augment=True,
            aug_config={"horizontal_flip": False, "rotation_degrees": 0}
        )
        result = t(sample_image)
        assert result.shape == (3, 224, 224)
