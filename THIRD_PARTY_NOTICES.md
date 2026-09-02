# Third-Party Notices

smappy's own source code and documentation are licensed under the BSD 3-Clause
license in `LICENSE`.

## Vendored: COMET

`src/smappy/_comet/` is a copy of the parts of
[COMET](https://github.com/gpufit/Comet) 1.1.0 ("Cost-function Optimized Maximal
Overlap Drift EsTimation") that smappy's drift correction calls, Copyright (c)
2025 Lenny Reinkensmeier & Mark Bates, MIT licensed.
`src/smappy/_comet/LICENSE.txt` carries their notice and **ships inside every
wheel and source distribution**, which is what that licence asks for.

It is vendored rather than depended on because COMET publishes no distribution
on the Python package index -- the name `comet` there belongs to an unrelated
project -- and drift correction that only works from a git checkout is drift
correction almost nobody runs.  It is in the published package, so
`pip install smappy-smlm[drift]` is all a user needs; the extra brings in numba,
which COMET's cost-function kernels are compiled with.

Only the modules the drift correction reaches are included.  COMET's CLI, batch
runner, utilities, its own test suite, its documentation, and its IDL and
CUDA/C++ bindings are not here; use upstream for those.  The changes to what is
here are: absolute imports made relative, so the copy is self-contained and
cannot collide with an installed COMET; two imports made lazy, so pandas and
tkinter are needed only if COMET's own saving paths are taken; and two
cost-function fast paths added for smappy (see the commit that vendored COMET
for the measurements).  Each file says all of this in its header.

**COMET is a published method, and smappy is not its author.**  Drift-corrected
files record it in the `/drift` group's attributes, so a result can be traced
back to the method that produced it.  If you publish work that used it, cite it
as `externaltools/Comet/CITATION.cff` asks.

## Dependencies

Dependencies are separate works and remain subject to their own licenses. They
are installed from the Python package index; their source is not vendored here.
Exact versions are selected at install time, so distributors should preserve the
license files shipped in the installed distributions and review the notices for
the versions they distribute.

| Dependency | Use | Declared license |
| --- | --- | --- |
| NumPy | runtime | BSD 3-Clause |
| SciPy | runtime | BSD 3-Clause; binary distributions may include components under other compatible licenses |
| tifffile | runtime | BSD 3-Clause |
| h5py | runtime | BSD 3-Clause |
| PyYAML | runtime | MIT |
| Matplotlib | optional `viewer` | Matplotlib license (BSD-compatible, PSF-derived) |
| Pillow | optional `image` | MIT-CMU (HPND) |
| numba | optional `drift` | BSD 2-Clause (pulls in llvmlite, BSD 2-Clause) |
| pybind11 | build only, and its headers are compiled into the extension modules | BSD 3-Clause |
| setuptools | build only | MIT |
| pytest | test only | MIT |
