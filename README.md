# action-recognition

Recognize actions from short video clips with a small 3D CNN. The idea is the
plain one: stack a few frames into a clip, run 3D convolutions over time and
space together, pool the result down to a vector, and read off a class. Because
the convolutions see several frames at once, the network can pick up on how
things move rather than how any single frame looks.

To keep everything runnable offline with no datasets to download, the clips are
synthetic. Each clip shows a bright blob drifting across a small frame, and the
direction or pattern of that drift is what defines the class. A model that reads
temporal structure can tell the classes apart. A model that only ever sees one
frame at a time cannot, since any single frame is just a blob somewhere in the
image.

## Motion classes

| label | name       | what the blob does          |
|-------|------------|-----------------------------|
| 0     | move_right | travels left to right       |
| 1     | move_left  | travels right to left       |
| 2     | move_down  | travels top to bottom       |
| 3     | circle     | travels around a circle     |

Every clip gets a little position jitter and pixel noise so the model has to
learn the motion pattern instead of memorizing one exact pixel path.

## Shapes

Clips use the layout `(T, C, H, W)` and batches use `(B, T, C, H, W)`, with
`C == 3`. The three channels carry the same grayscale signal, so a standard
three channel model works without changes. Inside the model the batch is
permuted to `(B, C, T, H, W)` for `Conv3d` and permuted back conceptually when
the logits come out.

## Model

`Action3DCNN` is a compact stack of `Conv3d -> BatchNorm3d -> ReLU` blocks. Each
block pools over height and width but leaves the time axis intact, so short
clips do not collapse before the network has looked at the motion. After the
convolutional stages a single adaptive pool flattens whatever time, height, and
width remain into one vector, and a linear layer turns that into class logits of
shape `(B, num_classes)`. The forward pass checks the input rank and channel
count and raises a clear error when they are wrong.

## Usage

```python
import torch
from torch.utils.data import DataLoader
from src.data import SyntheticMotionDataset, MOTION_CLASSES
from src.model import Action3DCNN
from src.train import train_model, evaluate

train_ds = SyntheticMotionDataset(num_samples=256, seed=0)
test_ds = SyntheticMotionDataset(num_samples=64, seed=123)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

model = Action3DCNN(num_classes=len(MOTION_CLASSES))
history = train_model(model, train_loader, epochs=6, lr=1e-3)
print("final train accuracy:", history["acc"][-1])
print("test accuracy:", evaluate(model, test_loader))
```

## Tests

```
C:/Users/sharv/.venvs/cv/Scripts/python.exe -m pytest tests/ -q
```

The suite covers three things. The data tests confirm clip shape and value
range, that the three channels stay identical, and that the blob centroid
actually moves in the direction the label promises. The model tests confirm the
forward pass returns `(B, num_classes)` logits across several clip lengths and
spatial sizes, that gradients flow back to the input, and that bad input shapes
are rejected. The training tests confirm that loss falls over a short run and
that held out accuracy lands well above chance on the synthetic motion classes.

All fifteen tests pass on CPU in a couple of seconds with no downloads.

## Layout

```
src/
  data.py    synthetic motion clips and a torch Dataset
  model.py   the 3D CNN action classifier
  train.py   train loop and accuracy evaluation
tests/       pytest behavior checks
```
