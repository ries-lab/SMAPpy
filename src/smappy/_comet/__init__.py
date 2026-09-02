"""COMET, vendored: the drift estimator smappy's `drift` module runs on.

COMET -- Cost-function Optimized Maximal Overlap Drift EsTimation, by Lenny
Reinkensmeier and Mark Bates (https://github.com/gpufit/Comet) -- estimates
sample drift by maximising the overlap of localizations between time windows,
with no fiducials and no reference structure.

**It is not smappy's work.**  It is MIT licensed (see ``LICENSE.txt`` beside
this file) and it is a published method: results that used it should cite it,
which is why `smappy.drift` records its name and version in the provenance of
every corrected file.

Only the modules the drift correction actually calls are here -- the CLI, the
batch runner, the IDL and CUDA/C++ bindings, the docs and COMET's own tests are
not.  Upstream's layout is kept so this copy can be diffed against a newer
release; the only edits are relative imports, two lazy ones, and the two
cost-function fast paths in `core.cpu_wrapper` that smappy opts into.

It is vendored rather than depended on because COMET publishes no distribution
on PyPI, and drift correction that only works from a git checkout is drift
correction most people never run.
"""

from ._version import __version__
from .core.backends import (available_backends, best_backend, cuda_available,
                            describe_backends, torch_available)
from .core.drift_optimizer import comet_run_kd
from .core.segmenter import segmentation_wrapper

#: what upstream calls itself, for provenance
UPSTREAM = "COMET (github.com/gpufit/Comet)"

__all__ = [
    "__version__", "UPSTREAM",
    "comet_run_kd", "segmentation_wrapper",
    "available_backends", "best_backend", "cuda_available", "torch_available",
    "describe_backends",
]
