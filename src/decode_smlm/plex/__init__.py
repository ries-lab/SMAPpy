"""
DECODE
This software package implements a DeepLearning based framework for high-density fitting in SMLM.

"""

__version__ = "1.0.0dev0"  # do not modify by hand set and sync with bumpversion
__author__ = "Lucas-Raphael Mueller, Artur Speiser"
__repo__ = "https://github.com/TuragaLab/DECODE/master/gateway.yaml"  # main repo
__gateway__ = (
    "https://raw.githubusercontent.com/TuragaLab/DECODE/master/gateway.yaml"  # gateway
)

# The subpackages are imported when first used, not here, as in `decode_smlm`'s own __init__:
# DECODE-Plex imported them all, and with them torch, Lightning and the plotting, so reading one
# model's settings loaded everything.  `decode_smlm.plex.simulation`, `.EmitterSet` and the rest
# still work as attributes.  The warning for a GPU of compute capability below 3.7 is gone: current
# torch builds refuse such a GPU themselves.
import importlib

_SUBPACKAGES = ("emitter", "evaluation", "generic", "hardware", "io", "neuralfitter", "plot",
                "renderer", "simulation", "utils", "validation")
_NAMES = {"EmitterSet": ("emitter.emitter", "EmitterSet"),
          "emitter_factory": ("emitter.emitter", "factory")}


def __getattr__(name):
    if name in _SUBPACKAGES:
        return importlib.import_module(f".{name}", __name__)
    if name in _NAMES:
        module, attr = _NAMES[name]
        return getattr(importlib.import_module(f".{module}", __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_SUBPACKAGES, *_NAMES])
