"""Kestrel: a tiny PyTorch detector with a callable-module API."""

import sys as _sys

from .model import Kestrel
from .results import Results

__version__ = "0.2.0"
__all__ = ["Kestrel", "Results", "__version__"]


class _CallableModule(_sys.modules[__name__].__class__):
    """Let ``import kestrel; kestrel(...)`` work directly."""

    def __call__(self, *args, **kwargs):
        return Kestrel(*args, **kwargs)


_sys.modules[__name__].__class__ = _CallableModule
