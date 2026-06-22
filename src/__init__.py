"""Action recognition from short video clips with a small 3D CNN."""

from .model import Action3DCNN
from .data import make_motion_clip, SyntheticMotionDataset, MOTION_CLASSES
from .train import train_model, evaluate

__all__ = [
    "Action3DCNN",
    "make_motion_clip",
    "SyntheticMotionDataset",
    "MOTION_CLASSES",
    "train_model",
    "evaluate",
]
