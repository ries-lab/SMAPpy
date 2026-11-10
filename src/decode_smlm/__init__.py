"""
DECODE
This software package implements a DeepLearning based framework for high-density fitting in SMLM.

"""

__version__ = '0.10.2'  # do not modify by hand set and sync with bumpversion
__author__ = 'Lucas-Raphael Mueller, Artur Speiser'
__repo__ = 'https://github.com/TuragaLab/DECODE/master/gateway.yaml'  # main repo
__gateway__ = 'https://raw.githubusercontent.com/TuragaLab/DECODE/master/gateway.yaml'  # gateway

# The subpackages are imported when first used, not here: DECODE imported them all, and torch with
# them, so the port's torch-free parts (`decode_smlm.prefit`) could not be imported without it
# (docs/decode/poc-plan.md (e)).  `decode_smlm.simulation`, `decode_smlm.EmitterSet` and the rest
# still work as attributes.  DECODE's warning for a GPU of compute capability below 3.7 is gone:
# current torch builds refuse such a GPU themselves.
import importlib

_SUBPACKAGES = ("evaluation", "generic", "neuralfitter", "simulation", "utils")
_NAMES = {"CoordinateOnlyEmitter": "generic.emitter", "EmitterSet": "generic.emitter",
          "RandomEmitterSet": "generic.emitter"}


def __getattr__(name):
    if name in _SUBPACKAGES:
        return importlib.import_module(f".{name}", __name__)
    if name in _NAMES:
        return getattr(importlib.import_module(f".{_NAMES[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_SUBPACKAGES, *_NAMES])
