# Kestrel

Kestrel is a small PyTorch detector for rock-paper-scissors with a YOLO-style API.
It is designed to stay lightweight: one shared CNN backbone, one classification head,
one bounding-box regressor, and one simple checkpoint format.

## Features

- Train on a standard YOLO dataset YAML.
- Load from either a config `.yaml` file or a saved `.pt` checkpoint.
- Run prediction on local images, image arrays, or image URLs.
- Save checkpoints or export TorchScript for deployment.
- Use a familiar callable interface:

```python
import kestrel
model = kestrel("kestrel26n.yaml")
```

## Install

Install from github:
```bash
!pip install git+https://github.com/k329272/kestrel -q
```

## Quick Start

```python
import kestrel

model = kestrel("kestrel26n.yaml")

metrics = model.train(
    data="examples/data.yaml",
    epochs=10,
    patience=3,
    save_best=True,
)

print(metrics)
model.save("kestrel26n.pt")

loaded = kestrel("kestrel26n.pt")
results = loaded("path/to/image.jpg")
annotated = results[0].plot()
results[0].save("prediction.jpg")
```

## Dataset Format

Training uses a standard YOLO dataset YAML:

```yaml
path: ../datasets/rps
train: images/train
val: images/val
names: [Rock, Paper, Scissors]
```

Labels use the usual YOLO text format:

```text
class_id x_center y_center width height
```

If `names` is omitted, Kestrel infers class names from the label files.
If `imgsz` is omitted, Kestrel samples the dataset and picks a representative size.

## Training

`train()` supports a few useful quality-of-life options:

- `val_split` controls the fraction of training data held out for validation.
- `patience` and `min_delta` enable early stopping on validation loss.
- `scheduler_factor` and `scheduler_patience` control the learning-rate scheduler.
- `num_workers` and `pin_memory` tune data loading performance.
- `amp=True` enables mixed precision on CUDA.
- `save_best=True` writes the best checkpoint to `kestrel_best.pt` by default.

Example:

```python
model.train(
    data="examples/data.yaml",
    epochs=25,
    batch_size=32,
    lr=1e-3,
    bbox_loss_weight=1.0,
    patience=5,
    save_best=True,
)
```

## Prediction

`predict()` accepts:

- a file path
- a list of file paths
- a NumPy image array
- an `http://` or `https://` image URL

Each prediction returns a `Results` object with:

- `plot()` to draw boxes and labels
- `save()` to write an annotated image
- `to_dict()` for serialization

Example:

```python
results = model.predict("path/to/image.jpg", conf=0.25)
print(results[0].to_dict())
results[0].save("annotated.jpg")
```

## Saving And Exporting

```python
model.save("kestrel26n.pt")
model.export(format="torchscript", path="kestrel_model.torchscript.pt")
```

- `kestrel("something.yaml")` builds a new model from config.
- `kestrel("something.pt")` loads a saved checkpoint.
- `export(format="checkpoint")` writes a checkpoint using `save()`.

## API

```python
model = kestrel("kestrel26n.yaml")
model.train(data="examples/data.yaml", epochs=10)
metrics = model.val()
results = model("image.jpg")
model.save("kestrel26n.pt")
```

## Project Notes

- Default classes are `Rock`, `Paper`, and `Scissors`, for the `hands.ipynb` notebook.
- The package exposes a callable module, so `import kestrel; kestrel(...)` works directly.
- The repository includes `examples/data.yaml` as a ready-to-edit dataset example.
