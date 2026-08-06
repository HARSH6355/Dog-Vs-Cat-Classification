"""
baseline_cnn.py
---------------
Defines the Baseline CNN architecture for binary image classification.

Architecture:
  • 3 Convolutional blocks: Conv2d → BatchNorm → ReLU → MaxPool
  • Global Average Pooling
  • 2 Fully-connected layers with Dropout
  • Single output neuron (sigmoid) for binary classification

Usage:
    from src.models.baseline_cnn import BaselineCNN
    model = BaselineCNN(num_classes=2)
"""

import torch
import torch.nn as nn
from typing import List


class ConvBlock(nn.Module):
    """
    A single convolutional block: Conv → BN → ReLU → MaxPool.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output (filter) channels.
        kernel_size:  Convolution kernel size (default 3).
        pool_size:    Max-pooling window size (default 2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    Baseline CNN for binary image classification (Cats vs Dogs).

    Network layout:
        Input: (B, 3, 224, 224)
        ConvBlock 1: 3  → filters[0]  | output: (B, 32,  112, 112)
        ConvBlock 2: f0 → filters[1]  | output: (B, 64,   56,  56)
        ConvBlock 3: f1 → filters[2]  | output: (B, 128,  28,  28)
        GlobalAvgPool                 | output: (B, 128)
        FC1: 128 → fc_units + Dropout | output: (B, 512)
        FC2: fc_units → num_classes   | output: (B, 2)

    Args:
        num_classes:   Number of output classes (default 2 for binary).
        filters:       List of output channels per conv block.
        fc_units:      Hidden units in the fully-connected layer.
        dropout_rate:  Dropout probability (default 0.5).
        in_channels:   Input image channels (default 3 for RGB).
    """

    def __init__(
        self,
        num_classes: int = 2,
        filters: List[int] = [32, 64, 128],
        fc_units: int = 512,
        dropout_rate: float = 0.5,
        in_channels: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.filters = filters
        self.fc_units = fc_units

        # ── Convolutional feature extractor ───────────────────────────────
        channels = [in_channels] + filters
        conv_blocks = []
        for i in range(len(filters)):
            conv_blocks.append(ConvBlock(channels[i], channels[i + 1]))
        self.features = nn.Sequential(*conv_blocks)

        # ── Global Average Pooling ─────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # ── Classifier ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[-1], fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_units, num_classes),
        )

        # Weight initialisation
        self._initialise_weights()

    def _initialise_weights(self) -> None:
        """Applies He / Kaiming Normal initialisation to Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.features(x)  # (B, filters[-1], H', W')
        x = self.gap(x)       # (B, filters[-1], 1, 1)
        x = self.classifier(x)  # (B, num_classes)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities (for inference)."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def summary(self) -> str:
        """Returns a string summary of the model architecture."""
        from src.utils.helpers import count_parameters
        lines = [
            "=" * 50,
            f"  BaselineCNN — Binary Image Classifier",
            "=" * 50,
            f"  Filters:      {self.filters}",
            f"  FC Units:     {self.fc_units}",
            f"  Num Classes:  {self.num_classes}",
            f"  Parameters:   {count_parameters(self):,}",
            "=" * 50,
        ]
        return "\n".join(lines)


def build_model(config: dict) -> BaselineCNN:
    """
    Builds a BaselineCNN from a loaded config dict.

    Args:
        config: Full config dict from configs/config.yaml.

    Returns:
        An initialised BaselineCNN model.
    """
    m = config["model"]
    return BaselineCNN(
        num_classes=m["num_classes"],
        filters=m["filters"],
        fc_units=m["fc_units"],
        dropout_rate=m["dropout_rate"],
    )


if __name__ == "__main__":
    import torch
    model = BaselineCNN()
    print(model.summary())

    # Dry-run forward pass
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # Expected: (4, 2)
