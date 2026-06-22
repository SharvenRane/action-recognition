"""Synthetic motion clips for action recognition.

Each clip is a short grayscale-as-RGB video of a bright blob moving across the
frame. The direction and pattern of the blob's motion encodes the action class,
so a model that can read temporal structure can separate the classes while a
model that only looks at single frames cannot.

Shapes follow the convention used everywhere in this repo:

    clip tensor: (T, C, H, W)
    batch:       (B, T, C, H, W)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# The four motion classes. The index in this list is the integer label.
MOTION_CLASSES: List[str] = [
    "move_right",   # blob travels left to right
    "move_left",    # blob travels right to left
    "move_down",    # blob travels top to bottom
    "circle",       # blob travels around a circle
]


def _draw_blob(frame: np.ndarray, cx: float, cy: float, radius: float) -> None:
    """Render a soft Gaussian blob into a single (H, W) frame in place."""
    h, w = frame.shape
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
    frame += np.exp(-dist_sq / (2.0 * radius * radius))


def make_motion_clip(
    label: int,
    num_frames: int = 8,
    height: int = 16,
    width: int = 16,
    radius: float = 2.5,
    noise: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build one synthetic clip whose motion encodes ``label``.

    Returns a float32 array of shape (T, C, H, W) with C == 3 and values in
    roughly [0, 1]. The three channels are identical (the signal is the motion,
    not the color), which keeps a standard 3 channel model usable.
    """
    if rng is None:
        rng = np.random.default_rng()
    if label < 0 or label >= len(MOTION_CLASSES):
        raise ValueError(f"label must be in [0, {len(MOTION_CLASSES)}), got {label}")

    t = np.linspace(0.0, 1.0, num_frames)

    # margin keeps the blob fully inside the frame
    margin = radius * 2.0
    lo_x, hi_x = margin, width - 1 - margin
    lo_y, hi_y = margin, height - 1 - margin
    mid_x = (lo_x + hi_x) / 2.0
    mid_y = (lo_y + hi_y) / 2.0

    # small per-clip jitter on the starting position so the model cannot cheat
    # by memorizing one exact pixel path
    jx = rng.uniform(-1.0, 1.0)
    jy = rng.uniform(-1.0, 1.0)

    name = MOTION_CLASSES[label]
    if name == "move_right":
        cx = lo_x + t * (hi_x - lo_x)
        cy = np.full(num_frames, mid_y + jy)
    elif name == "move_left":
        cx = hi_x - t * (hi_x - lo_x)
        cy = np.full(num_frames, mid_y + jy)
    elif name == "move_down":
        cx = np.full(num_frames, mid_x + jx)
        cy = lo_y + t * (hi_y - lo_y)
    elif name == "circle":
        ang = t * 2.0 * math.pi
        rad_x = (hi_x - lo_x) / 2.0
        rad_y = (hi_y - lo_y) / 2.0
        cx = mid_x + rad_x * np.cos(ang)
        cy = mid_y + rad_y * np.sin(ang)
    else:  # pragma: no cover - guarded by the range check above
        raise AssertionError(name)

    clip = np.zeros((num_frames, height, width), dtype=np.float32)
    for i in range(num_frames):
        _draw_blob(clip[i], float(cx[i]), float(cy[i]), radius)

    if noise > 0:
        clip += rng.normal(0.0, noise, size=clip.shape).astype(np.float32)

    clip = np.clip(clip, 0.0, 1.0)

    # (T, H, W) -> (T, C, H, W) with three identical channels
    clip = np.repeat(clip[:, None, :, :], 3, axis=1)
    return clip.astype(np.float32)


class SyntheticMotionDataset(Dataset):
    """A torch Dataset of synthetic motion clips with balanced classes."""

    def __init__(
        self,
        num_samples: int = 256,
        num_frames: int = 8,
        height: int = 16,
        width: int = 16,
        seed: int = 0,
    ) -> None:
        self.num_frames = num_frames
        self.height = height
        self.width = width
        rng = np.random.default_rng(seed)

        clips = []
        labels = []
        num_classes = len(MOTION_CLASSES)
        for n in range(num_samples):
            label = n % num_classes  # balanced round robin
            clip = make_motion_clip(
                label,
                num_frames=num_frames,
                height=height,
                width=width,
                rng=rng,
            )
            clips.append(clip)
            labels.append(label)

        self.clips = torch.from_numpy(np.stack(clips, axis=0))  # (N, T, C, H, W)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.clips.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.clips[idx], self.labels[idx]
