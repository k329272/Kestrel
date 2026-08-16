"""
kestrel
=======

A tiny YOLO-style API wrapped around a two-head (bbox regressor + classifier)
TensorFlow/Keras rock-paper-scissors detector.

Usage
-----
    import kestrel

    # Build a new model from an architecture config
    model = kestrel("kestrel26n.yaml")

    # ...or load a previously-trained/saved model
    model = kestrel("kestrel26n.pt")

    # Train
    results = model.train(data="rps.yaml", epochs=3)

    # Validate
    results = model.val()

    # Predict
    results = model("https://ultralytics.com/images/bus.jpg")

    # Export
    success = model.export(format="onnx")

The `kestrel` name is both the module and a callable constructor (mirroring
`from ultralytics import YOLO; YOLO(...)`, but collapsed onto the module
object itself so `import kestrel; kestrel(...)` works directly).
"""

import sys as _sys

from .model import Kestrel
from .results import Results

__version__ = "0.1.0"
__all__ = ["Kestrel", "Results", "__version__"]


class _CallableModule(_sys.modules[__name__].__class__):
    """Lets the `kestrel` module itself be called, e.g. `kestrel("x.yaml")`."""

    def __call__(self, *args, **kwargs):
        return Kestrel(*args, **kwargs)


_sys.modules[__name__].__class__ = _CallableModule
