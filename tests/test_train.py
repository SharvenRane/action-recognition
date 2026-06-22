import torch
from torch.utils.data import DataLoader

from src.data import SyntheticMotionDataset, MOTION_CLASSES
from src.model import Action3DCNN
from src.train import train_model, evaluate


def _make_loaders():
    train_ds = SyntheticMotionDataset(num_samples=256, num_frames=8, height=16, width=16, seed=0)
    test_ds = SyntheticMotionDataset(num_samples=64, num_frames=8, height=16, width=16, seed=123)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    return train_loader, test_loader


def test_training_reduces_loss():
    torch.manual_seed(0)
    train_loader, _ = _make_loaders()
    model = Action3DCNN(num_classes=len(MOTION_CLASSES))
    history = train_model(model, train_loader, epochs=6, lr=1e-3)
    assert len(history["loss"]) == 6
    # the last epoch loss should be clearly below the first
    assert history["loss"][-1] < history["loss"][0] * 0.7


def test_accuracy_beats_chance():
    torch.manual_seed(0)
    train_loader, test_loader = _make_loaders()
    model = Action3DCNN(num_classes=len(MOTION_CLASSES))
    train_model(model, train_loader, epochs=6, lr=1e-3)
    acc = evaluate(model, test_loader)
    chance = 1.0 / len(MOTION_CLASSES)  # 0.25 for four classes
    # well above chance on held out synthetic clips
    assert acc > chance + 0.3


def test_evaluate_on_untrained_is_near_chance_or_better():
    # sanity: evaluate returns a probability in [0, 1]
    _, test_loader = _make_loaders()
    model = Action3DCNN(num_classes=len(MOTION_CLASSES))
    acc = evaluate(model, test_loader)
    assert 0.0 <= acc <= 1.0
