import numpy as np
import torch

from src.data import make_motion_clip, SyntheticMotionDataset, MOTION_CLASSES


def test_clip_shape_and_range():
    clip = make_motion_clip(0, num_frames=8, height=16, width=16)
    assert clip.shape == (8, 3, 16, 16)
    assert clip.dtype == np.float32
    assert clip.min() >= 0.0
    assert clip.max() <= 1.0
    # the blob must actually be bright somewhere
    assert clip.max() > 0.5


def test_channels_are_identical():
    clip = make_motion_clip(2, num_frames=6, height=12, width=12, noise=0.0)
    # all three channels carry the same grayscale signal
    assert np.allclose(clip[:, 0], clip[:, 1])
    assert np.allclose(clip[:, 0], clip[:, 2])


def test_motion_direction_matches_label():
    # move_right: blob centroid x should increase over time
    clip = make_motion_clip(0, num_frames=8, height=16, width=16, noise=0.0)
    xs = []
    for f in clip[:, 0]:  # one channel
        total = f.sum()
        col = (f.sum(axis=0) * np.arange(f.shape[1])).sum() / total
        xs.append(col)
    assert xs[-1] > xs[0] + 2.0

    # move_left: blob centroid x should decrease over time
    clip_l = make_motion_clip(1, num_frames=8, height=16, width=16, noise=0.0)
    xs_l = []
    for f in clip_l[:, 0]:
        total = f.sum()
        col = (f.sum(axis=0) * np.arange(f.shape[1])).sum() / total
        xs_l.append(col)
    assert xs_l[-1] < xs_l[0] - 2.0


def test_invalid_label_raises():
    try:
        make_motion_clip(99)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_dataset_balanced_and_typed():
    ds = SyntheticMotionDataset(num_samples=40, num_frames=8, height=16, width=16, seed=1)
    assert len(ds) == 40
    clip, label = ds[0]
    assert isinstance(clip, torch.Tensor)
    assert clip.shape == (8, 3, 16, 16)
    assert label.dtype == torch.long

    counts = torch.bincount(ds.labels, minlength=len(MOTION_CLASSES))
    # 40 samples over 4 classes round robin -> 10 each
    assert torch.all(counts == 10)


def test_dataset_is_deterministic_with_seed():
    a = SyntheticMotionDataset(num_samples=8, seed=7)
    b = SyntheticMotionDataset(num_samples=8, seed=7)
    assert torch.allclose(a.clips, b.clips)
