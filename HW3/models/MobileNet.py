"""
MobileNetV2 for CIFAR-10 (32×32 inputs).

Adapted from the original MobileNetV2 paper (Sandler et al., 2018) with
modifications for small input resolution:
  - First conv uses stride 1 instead of stride 2
  - No aggressive downsampling at the stem
This matches the same philosophy as Option 2 in transfer learning.

Reference:
    Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018).
    MobileNetV2: Inverted residuals and linear bottlenecks.
    CVPR 2018, pp. 4510-4520.
"""

import torch
import torch.nn as nn
from typing import List


class InvertedResidual(nn.Module):
    """Inverted residual block (bottleneck) used in MobileNetV2.

    Expands channels by expansion factor, applies depthwise conv, then
    projects back down. A residual connection is added when input and
    output shapes match (stride == 1 and in_channels == out_channels).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the depthwise convolution (1 or 2).
        expansion: Channel expansion factor applied before depthwise conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion: int,
    ) -> None:
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden = in_channels * expansion

        layers: List[nn.Module] = []

        # Pointwise expansion (skipped when expansion == 1)
        if expansion != 1:
            layers += [
                nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]

        # Depthwise convolution
        layers += [
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride,
                      padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
        ]

        # Pointwise projection (linear, no activation)
        layers += [
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection."""
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class MobileNetV2(nn.Module):
    """MobileNetV2 adapted for CIFAR-10 classification.

    Args:
        num_classes: Number of output classes. Default: 10.
        width_mult: Width multiplier to scale channel counts. Default: 1.0.

    Attributes:
        features: Sequential stem + inverted residual blocks + final conv.
        classifier: Global average pool + dropout + linear head.

    Shape:
        Input:  (N, 3, 32, 32)
        Output: (N, num_classes)

    Example:
        >>> model = MobileNetV2(num_classes=10)
        >>> x = torch.randn(4, 3, 32, 32)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([4, 10])
    """

    # Each row: [expansion, out_channels, num_blocks, stride]
    # Stride 2 blocks perform spatial downsampling
    _INVERTED_RESIDUAL_SETTINGS = [
        [1,  16, 1, 1],
        [6,  24, 2, 1],  # stride 1 instead of 2 for 32×32 inputs
        [6,  32, 3, 2],
        [6,  64, 4, 2],
        [6,  96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()

        def _scaled(c: int) -> int:
            """Scale channel count by width multiplier."""
            return int(c * width_mult)

        # Stem: stride 1 to preserve spatial resolution on 32×32 images
        input_channels = _scaled(32)
        self.features: List[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(3, input_channels, kernel_size=3, stride=1,
                          padding=1, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU6(inplace=True),
            )
        ]

        # Inverted residual blocks
        for expansion, out_channels, num_blocks, stride in self._INVERTED_RESIDUAL_SETTINGS:
            out_channels = _scaled(out_channels)
            for i in range(num_blocks):
                self.features.append(
                    InvertedResidual(
                        in_channels  = input_channels,
                        out_channels = out_channels,
                        stride       = stride if i == 0 else 1,
                        expansion    = expansion,
                    )
                )
                input_channels = out_channels

        # Final conv to expand to 1280 channels
        last_channels = _scaled(1280)
        self.features.append(
            nn.Sequential(
                nn.Conv2d(input_channels, last_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(last_channels),
                nn.ReLU6(inplace=True),
            )
        )

        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(last_channels, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize conv, batchnorm, and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor and classifier."""
        x = self.features(x)
        x = self.classifier(x)
        return x
