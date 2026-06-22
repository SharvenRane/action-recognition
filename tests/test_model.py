import torch

from src.model import Action3DCNN


def test_forward_shape():
    model = Action3DCNN(num_classes=4)
    clip = torch.rand(2, 8, 3, 16, 16)  # (B, T, C, H, W)
    logits = model(clip)
    assert logits.shape == (2, 4)


def test_forward_varied_temporal_length():
    model = Action3DCNN(num_classes=4)
    for t in (4, 6, 8, 12):
        clip = torch.rand(3, t, 3, 16, 16)
        logits = model(clip)
        assert logits.shape == (3, 4)


def test_forward_varied_spatial_size():
    model = Action3DCNN(num_classes=5)
    clip = torch.rand(2, 8, 3, 32, 32)
    logits = model(clip)
    assert logits.shape == (2, 5)


def test_logits_finite_and_differentiable():
    model = Action3DCNN(num_classes=4)
    clip = torch.rand(2, 8, 3, 16, 16, requires_grad=True)
    logits = model(clip)
    assert torch.isfinite(logits).all()
    loss = logits.sum()
    loss.backward()
    # gradients should flow back to the input
    assert clip.grad is not None
    assert torch.isfinite(clip.grad).all()


def test_rejects_wrong_rank():
    model = Action3DCNN(num_classes=4)
    bad = torch.rand(2, 3, 16, 16)  # missing time axis
    try:
        model(bad)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_rejects_wrong_channel_count():
    model = Action3DCNN(num_classes=4, in_channels=3)
    bad = torch.rand(2, 8, 1, 16, 16)
    try:
        model(bad)
        raised = False
    except ValueError:
        raised = True
    assert raised
