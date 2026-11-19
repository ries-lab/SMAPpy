"""Subpackages imported when first used (decode_smlm.plex, docs/decode/plex-plan.md §3).

DECODE-Plex's ``__init__`` files imported every module of their package, so reading one model
imported Lightning, Hydra, seaborn, torchmetrics and the compiled spline with it.  Each of those
files now names what it held and `attach` imports it on first use (PEP 562), so a module costs
only what it imports itself.
"""
import importlib


def attach(package: str, submodules=(), names=None):
    """``__getattr__`` and ``__dir__`` for a package: ``submodules`` by name, ``names`` as
    ``{attribute: (module relative to the package, its name there)}``."""
    submodules = set(submodules)
    names = dict(names or {})

    def __getattr__(name):
        if name in submodules:
            return importlib.import_module(f"{package}.{name}")
        if name in names:
            module, attr = names[name]
            return getattr(importlib.import_module(module, package), attr)
        raise AttributeError(f"module {package!r} has no attribute {name!r}")

    def __dir__():
        return sorted({*submodules, *names})

    return __getattr__, __dir__
