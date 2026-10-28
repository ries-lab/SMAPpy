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
correction almost nobody runs.  It is in the published package, so `pip install smappy-smlm` is all a user
needs: COMET's cost function is compiled with smappy's own C++ extensions
(`csrc/drift.hpp`) rather than with numba, so drift correction has no optional
dependency at all.

Only the modules the drift correction reaches are included.  COMET's CLI, batch
runner, utilities, its own test suite, its documentation, its IDL and CUDA/C++
bindings, and its CUDA and torch backends are not here; use upstream for those.
The changes to what is here are: absolute imports made relative, so the copy is
self-contained and cannot collide with an installed COMET; two imports made
lazy, so pandas and tkinter are needed only if COMET's own saving paths are
taken; and the CPU cost function moved from numba to smappy's own C++, keeping
upstream's function names and signatures so nothing else changed.  Each file says
all of this in its header.

**COMET is a published method, and smappy is not its author.**  Drift-corrected
files record it in the `/drift` group's attributes, so a result can be traced
back to the method that produced it.  If you publish work that used it, cite it
as `src/smappy/_comet/CITATION.cff` asks (Reinkensmeier and Bates,
bioRxiv 2026, [doi:10.64898/2026.03.27.714864](https://doi.org/10.64898/2026.03.27.714864)).

## Included: uipsf

`src/uipsf/` is uiPSF v2, a re-implementation in PyTorch of uiPSF (Liu et al.,
*Nat. Methods* 2024, [doi:10.1038/s41592-024-02282-x](https://doi.org/10.1038/s41592-024-02282-x)),
BSD 3-Clause licensed as smappy is, Copyright (c) 2026 uiPSF contributors.
`src/uipsf/LICENSE` carries its notice and ships with it.  It needs PyTorch,
the extra `smappy-smlm[uipsf]`.

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
| pybind11 | build only, and its headers are compiled into the extension modules | BSD 3-Clause |
| setuptools | build only | MIT |
| pytest | test only | MIT |
