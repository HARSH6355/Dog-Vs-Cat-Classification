"""
test_model.py
-------------
Unit tests for src/models/baseline_cnn.py

Run: pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest
import torch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baseline_cnn import BaselineCNN, ConvBlock, build_model


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def model():
    """Default BaselineCNN instance."""
    return BaselineCNN(num_classes=2, filters=[32, 64, 128], fc_units=512, dropout_rate=0.5)


@pytest.fixture
def dummy_batch():
    """A batch of 4 RGB images at 224x224."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def single_image():
    """A single RGB image at 224x224."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def minimal_config():
    """Minimal config dict for build_model()."""
    return {
        "model": {
            "num_classes": 2,
            "filters": [16, 32],
            "fc_units": 64,
            "dropout_rate": 0.3,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests: ConvBlock
# ─────────────────────────────────────────────────────────────────────────────

class TestConvBlock:
    """Tests for the ConvBlock sub-module."""

    def test_output_channels(self):
        """ConvBlock should produce the correct number of output channels."""
        block = ConvBlock(in_channels=3, out_channels=32)
        x = torch.randn(2, 3, 64, 64)
        out = block(x)
        assert out.shape[1] == 32

    def test_spatial_downsampling(self):
        """MaxPool(2) should halve the spatial dimensions."""
        block = ConvBlock(in_channels=3, out_channels=32)
        x = torch.randn(1, 3, 64, 64)
        out = block(x)
        assert out.shape[-2:] == (32, 32), f"Expected (32,32), got {out.shape[-2:]}"

    def test_no_nan_in_output(self):
        """ConvBlock should not produce NaN values."""
        block = ConvBlock(in_channels=3, out_channels=32)
        x = torch.randn(1, 3, 32, 32)
        out = block(x)
        assert not torch.isnan(out).any()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: BaselineCNN
# ─────────────────────────────────────────────────────────────────────────────

class TestBaselineCNN:
    """Tests for the full BaselineCNN model."""

    def test_output_shape_batch(self, model, dummy_batch):
        """Forward pass on a batch of 4 should produce (4, 2) logits."""
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"

    def test_output_shape_single(self, model, single_image):
        """Forward pass on a single image should produce (1, 2) logits."""
        model.eval()
        with torch.no_grad():
            out = model(single_image)
        assert out.shape == (1, 2), f"Expected (1, 2), got {out.shape}"

    def test_output_no_nan(self, model, dummy_batch):
        """Model output must not contain NaN values."""
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert not torch.isnan(out).any(), "Model output contains NaN"

    def test_output_no_inf(self, model, dummy_batch):
        """Model output must not contain Inf values."""
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert not torch.isinf(out).any(), "Model output contains Inf"

    def test_num_classes_2(self, dummy_batch):
        """Model with num_classes=2 should output 2 logits per sample."""
        model = BaselineCNN(num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape[1] == 2

    def test_custom_filters(self, dummy_batch):
        """Custom filter configuration should work end-to-end."""
        model = BaselineCNN(filters=[16, 32, 64], fc_units=128)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, 2)

    def test_trainable_parameters_positive(self, model):
        """Model must have at least one trainable parameter."""
        from src.utils.helpers import count_parameters
        n = count_parameters(model)
        assert n > 0, "Model has no trainable parameters"

    def test_predict_proba_sums_to_one(self, model, single_image):
        """predict_proba() should return probabilities that sum to 1."""
        model.eval()
        with torch.no_grad():
            probs = model.predict_proba(single_image)
        assert probs.shape == (1, 2)
        assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5), \
            f"Probabilities don't sum to 1: {probs.sum(dim=1).item()}"

    def test_predict_proba_in_range(self, model, dummy_batch):
        """All probabilities must be in [0, 1]."""
        model.eval()
        with torch.no_grad():
            probs = model.predict_proba(dummy_batch)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_gradient_flows(self, model, dummy_batch):
        """Gradients must flow back through the model during training."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        labels = torch.randint(0, 2, (4,))

        optimizer.zero_grad()
        out = model(dummy_batch)
        loss = criterion(out, labels)
        loss.backward()

        # Check that at least one parameter has a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradients flowed through the model"

    def test_build_model_from_config(self, minimal_config, dummy_batch):
        """build_model() should create a working model from a config dict."""
        model = build_model(minimal_config)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, 2)
