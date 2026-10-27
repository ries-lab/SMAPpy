"""4Pi bead stacks cut into their quadrants, as MATLAB files.

The Ries-lab 4Pi's bead scans (zenodo 10823308, ``40nm_600``; uiPSF's example
data, ``4pi_40nm_bead``) are MATLAB v7.3 files with one variable per phase
channel, ``qd1`` .. ``qd4``, each a stack over z and the deformable mirrors'
phase steps, and a ``Z Positions.txt`` beside them.  MATLAB writes arrays
column-major, so h5py hands back the axes reversed: (z, phase, x, y).  They
are put the right way round here, (y, x) last, which uiPSF v1 did not do --
so a v1 model learned from these files is transposed against them
(`smappy.fourpi.load_uipsf_v1`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_quadrant_mat(path) -> np.ndarray:
    """The quadrants of one file as (channel, z, phase, y, x), in the order of
    their names (``qd1``, ``qd2``, ...)."""
    import h5py
    with h5py.File(path, "r") as f:
        names = sorted((k for k in f if k.lower().startswith("qd")),
                       key=lambda k: int(k[2:]) if k[2:].isdigit() else k)
        if not names:
            raise ValueError(f"{Path(path).name} has no qd1..qd4 variables")
        stacks = [np.asarray(f[k][...]) for k in names]
    stacks = [s[None, None] if s.ndim == 2 else s for s in stacks]
    stacks = [s[:, None] if s.ndim == 3 else s for s in stacks]
    return np.stack([np.swapaxes(s, -1, -2) for s in stacks])


def read_z_positions(folder) -> np.ndarray:
    """The piezo positions of a bead scan in um, from its ``Z Positions.txt``,
    or None when there is none."""
    path = Path(folder) / "Z Positions.txt"
    if not path.exists():
        return None
    values = [float(w) for w in path.read_text().replace(",", " ").split()
              if _is_number(w)]
    return np.asarray(values, float)


def _is_number(word: str) -> bool:
    try:
        float(word)
        return True
    except ValueError:
        return False
