"""A small 3D CNN for action recognition on short clips.

The network applies 3D convolutions over the time and space axes so that the
temporal pattern of motion drives the prediction. It accepts clips in the
(B, T, C, H, W) layout used throughout this repo and returns class logits of
shape (B, num_classes).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Conv3DBlock(nn.Module):
    """Conv3d -> BatchNorm3d -> ReLU, optionally followed by pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)
        # pool over space only, keep the time axis until the final pooling so
        # short clips do not collapse too early
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.conv(x)))
        if self.pool is not None:
            x = self.pool(x)
        return x


class Action3DCNN(nn.Module):
    """Compact 3D CNN action classifier.

    Args:
        num_classes: number of action classes to predict.
        in_channels: number of input channels per frame (3 for RGB).
        widths: channel width of each 3D conv stage.
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 3,
        widths: tuple[int, ...] = (16, 32, 64),
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        self.num_classes = num_classes
        self.in_channels = in_channels

        blocks = []
        prev = in_channels
        for w in widths:
            blocks.append(Conv3DBlock(prev, w, pool=True))
            prev = w
        self.features = nn.Sequential(*blocks)

        # collapse the remaining time, height, width into a single vector
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a clip batch to logits.

        Args:
            x: tensor of shape (B, T, C, H, W).

        Returns:
            logits of shape (B, num_classes).
        """
        if x.dim() != 5:
            raise ValueError(
                f"expected a 5D clip batch (B, T, C, H, W), got shape {tuple(x.shape)}"
            )
        b, t, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, got {c}"
            )

        # Conv3d wants (B, C, T, H, W); our clips are (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.features(x)
        x = self.global_pool(x)            # (B, C', 1, 1, 1)
        x = torch.flatten(x, 1)            # (B, C')
        logits = self.classifier(x)        # (B, num_classes)
        return logits
