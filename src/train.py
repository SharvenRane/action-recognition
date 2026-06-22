"""Training and evaluation helpers for the action recognition model."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import Action3DCNN


def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    """Return classification accuracy of ``model`` over ``loader``."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device)
            labels = labels.to(device)
            logits = model(clips)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
    return correct / max(total, 1)


def train_model(
    model: Action3DCNN,
    train_loader: DataLoader,
    epochs: int = 6,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, List[float]]:
    """Train ``model`` and return per epoch loss and train accuracy history.

    The returned dict has keys ``"loss"`` and ``"acc"``, each a list with one
    entry per epoch. Tests use the history to confirm that loss goes down.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {"loss": [], "acc": []}

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        correct = 0
        for clips, labels in train_loader:
            clips = clips.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch = labels.numel()
            running_loss += float(loss.item()) * batch
            seen += batch
            correct += int((logits.argmax(dim=1) == labels).sum().item())

        history["loss"].append(running_loss / max(seen, 1))
        history["acc"].append(correct / max(seen, 1))

    return history
