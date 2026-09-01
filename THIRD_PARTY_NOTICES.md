# Third-Party Notices

smappy's own source code and documentation are licensed under the BSD 3-Clause
license in `LICENSE`.

## Vendored: COMET

`externaltools/Comet/` is a copy of
[COMET](https://github.com/gpufit/Comet) ("Cost-function Optimized Maximal
Overlap Drift EsTimation"), Copyright (c) 2025 Lenny Reinkensmeier & Mark Bates,
licensed under the MIT license (`externaltools/Comet/Python_interface/LICENSE.txt`,
which must be preserved in any redistribution). It carries local changes for the
smappy drift correction; see the commit that vendored it.

COMET is an **optional** dependency, installed separately
(`pip install -e externaltools/Comet/Python_interface`), and is not part of the
published `smappy-smlm` distribution: neither the source distribution nor the
wheels contain it. Only drift correction uses it, and only when it is installed.

If you publish drift-corrected results, cite COMET as its `CITATION.cff` asks.

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
