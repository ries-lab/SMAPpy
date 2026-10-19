"""Registering two MINFLUX measurements by their MBM beads.

A sequential experiment -- Exchange-PAINT, a second label after a wash --
measures the same field twice, and the two tables need not line up: the
stage may have been moved, the sample settled.  The MBM beads are fixed to
the coverslip and recorded in both, so the map between the measurements is
the map between their beads (`docs/survey/minflux.md`, "MBM", item 2).

**Which positions.**  Imspector's localizations (`x_nm`) are corrected by the
beads' displacement since the measurement began (NOTES.md, "MINFLUX
beam-line monitoring beads"), so each measurement's frame is its beads at
its start: a bead's position is the **median of its first samples**
(`first`, 10 by default, minflux-viewer's choice; pyMINFLUX takes the mean
of 3), in each measurement.  After *MBM drift correction*, which replaces
Imspector's correction and keeps its reference at each measurement's first
bead point, the same holds.

**Which bead is which.**  Not by name: Imspector's names and slots (``gri``)
are reused -- R0 was two beads 20 um apart in two measurements of one file
(`io.msr.bead_table`).  So pairs are found by their positions: every pair of
a reference bead and a target bead votes for the shift between them, and the
shift most pairs agree on, within *tolerance*, is taken (`match`); each bead
then pairs with at most one other.  Beads tens of micrometres apart and a
shift of a micrometre or two -- the usual case -- leave no doubt.

**The map.**  With one or two pairs a translation (the mean shift; two
beads cannot pin a rotation against their own noise), with three or more a
rigid one, rotation and translation, by the Kabsch (1976) least squares fit
in x and y; z, where both are 3D, is shifted by the pairs' mean.  Each
bead's residual says how well it agrees.  A scale is not fitted: the beads
are on one coverslip under one objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

FIRST = 10                  # the samples a bead's position is the median of
TOLERANCE_NM = 200.0        # how far two pairs' shifts may differ and agree
MAX_SHIFT_NM = 5000.0       # the largest shift looked for


@dataclass
class BeadPositions:
    """Each bead's start position in one measurement."""
    names: List[str]
    xyz: np.ndarray                     # (n, 2 or 3) nm


@dataclass
class Registration:
    """The map target -> reference, and how the beads agree with it."""
    matrix: np.ndarray                  # (4, 4) affine on (x, y, z, 1)
    kind: str                           # "translation" or "rigid"
    pairs: List[Tuple[int, int]]        # (reference bead, target bead)
    residuals_nm: np.ndarray            # per pair, after the map
    reference: BeadPositions
    target: BeadPositions
    angle_deg: float = 0.0
    shift_nm: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def rms_nm(self) -> float:
        r = self.residuals_nm
        return float(np.sqrt(np.mean(r ** 2))) if len(r) else float("nan")

    def __str__(self) -> str:
        s = self.shift_nm
        text = (f"{self.kind} from {len(self.pairs)} bead{'s' if len(self.pairs) != 1 else ''}"
                f": shift {s[0]:.1f}, {s[1]:.1f}" + (f", {s[2]:.1f}" if s[2] else "")
                + " nm")
        if self.kind == "rigid":
            text += f", rotation {self.angle_deg:.4f} deg"
        return text + f", residual {self.rms_nm:.2f} nm rms"


def start_positions(t, xyz, bead, names: Optional[Sequence[str]] = None,
                    first: int = FIRST) -> BeadPositions:
    """Per bead (``bead``, an id per sample), the median of its ``first``
    samples in time.  ``names`` per bead id in ascending order, else the ids."""
    t = np.asarray(t, float)
    xyz = np.asarray(xyz, float)
    if xyz.ndim == 1:
        xyz = xyz[:, None]
    bead = np.asarray(bead)
    ids = np.unique(bead)
    out = np.empty((len(ids), xyz.shape[1]))
    for k, i in enumerate(ids):
        rows = np.flatnonzero(bead == i)
        rows = rows[np.argsort(t[rows], kind="stable")][:max(int(first), 1)]
        out[k] = np.nanmedian(xyz[rows], axis=0)
    labels = list(names) if names is not None else [str(i) for i in ids]
    return BeadPositions(labels, out)


def match(reference: np.ndarray, target: np.ndarray, tolerance_nm: float = TOLERANCE_NM,
          max_shift_nm: float = MAX_SHIFT_NM) -> List[Tuple[int, int]]:
    """Pairs ``(i, j)`` of reference and target beads that share one shift:
    the shift most candidate pairs agree on within ``tolerance_nm``, then
    each bead with its closest partner under that shift, once."""
    ref = np.asarray(reference, float)[:, :2]
    tgt = np.asarray(target, float)[:, :2]
    if not len(ref) or not len(tgt):
        return []
    shifts = ref[:, None, :] - tgt[None, :, :]               # (n_ref, n_tgt, 2)
    flat = shifts.reshape(-1, 2)
    close = np.hypot(flat[:, 0], flat[:, 1]) <= max_shift_nm
    if not close.any():
        return []
    candidates = flat[close]
    d = np.hypot(*(candidates[:, None, :] - candidates[None, :, :]).transpose(2, 0, 1))
    votes = (d <= tolerance_nm).sum(axis=1)
    agreeing = candidates[d[np.argmax(votes)] <= tolerance_nm]
    shift = np.median(agreeing, axis=0)
    off = np.hypot(*(shifts - shift).transpose(2, 0, 1))      # (n_ref, n_tgt)
    pairs, used_t = [], set()
    for i, j in sorted(((i, j) for i in range(len(ref)) for j in range(len(tgt))
                        if off[i, j] <= tolerance_nm), key=lambda p: off[p]):
        if i in {p[0] for p in pairs} or j in used_t:
            continue
        pairs.append((i, j))
        used_t.add(j)
    return sorted(pairs)


def kabsch_2d(source: np.ndarray, destination: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The rotation ``R`` (2x2) and shift ``t`` minimising |R s + t - d|^2."""
    s = np.asarray(source, float)
    d = np.asarray(destination, float)
    cs, cd = s.mean(axis=0), d.mean(axis=0)
    h = (s - cs).T @ (d - cd)
    u, _, vt = np.linalg.svd(h)
    sign = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rotation = vt.T @ np.diag([1.0, sign]) @ u.T
    return rotation, cd - rotation @ cs


def register(reference: BeadPositions, target: BeadPositions,
             tolerance_nm: float = TOLERANCE_NM, max_shift_nm: float = MAX_SHIFT_NM,
             rigid: Optional[bool] = None) -> Registration:
    """The map of ``target``'s frame onto ``reference``'s (module docstring).
    ``rigid``: None, rigid from three pairs; True or False to choose."""
    pairs = match(reference.xyz, target.xyz, tolerance_nm, max_shift_nm)
    if not pairs:
        raise ValueError(
            f"no bead of the target lies within {max_shift_nm:g} nm of the same "
            f"shift as another: no pair of beads agrees within {tolerance_nm:g} nm")
    i = np.array([p[0] for p in pairs])
    j = np.array([p[1] for p in pairs])
    ref, tgt = reference.xyz[i], target.xyz[j]
    use_rigid = len(pairs) >= 3 if rigid is None else bool(rigid)
    if use_rigid and len(pairs) < 2:
        raise ValueError("a rotation needs two beads at least")
    matrix = np.eye(4)
    if use_rigid:
        rotation, shift = kabsch_2d(tgt[:, :2], ref[:, :2])
        matrix[:2, :2] = rotation
        matrix[:2, 3] = shift
        angle = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    else:
        matrix[:2, 3] = np.mean(ref[:, :2] - tgt[:, :2], axis=0)
        angle = 0.0
    three_d = ref.shape[1] > 2 and tgt.shape[1] > 2
    if three_d:
        matrix[2, 3] = float(np.mean(ref[:, 2] - tgt[:, 2]))
    mapped = (matrix[:2, :2] @ tgt[:, :2].T).T + matrix[:2, 3]
    residual = np.hypot(*(mapped - ref[:, :2]).T)
    if three_d:
        residual = np.sqrt(residual ** 2 + (tgt[:, 2] + matrix[2, 3] - ref[:, 2]) ** 2)
    # the beads' mean displacement, which is what a reader wants to see: the
    # matrix's own translation is about the origin, tens of micrometres away
    moved = np.zeros(3)
    moved[:2] = np.mean(ref[:, :2] - tgt[:, :2], axis=0)
    if three_d:
        moved[2] = matrix[2, 3]
    return Registration(matrix, "rigid" if use_rigid else "translation", pairs, residual,
                        reference, target, angle, moved)
