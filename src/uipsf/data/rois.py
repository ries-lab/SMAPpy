"""Beads in a stack, and ROIs around them: the batch a bead fit takes (plan §5.7).

**Bead finding follows SMAPpy's convention** (`smappy.calibrate.core.
detect_beads`), with the same parameters, so that a standalone uipsf and
SMAPpy's calibration window pick the same beads from the same stacks:
- the stack's maximum over z, smoothed by a Gaussian of
  ``detection_sigma_px``;
- a threshold of ``detection_threshold_sigma`` robust standard deviations
  (1.4826·MAD) above the median of the smoothed projection.  Median and
  MAD make it independent of offset and conversion, so photons and ADU
  find the same beads;
- local maxima over 3 × 3, their centres of mass rounded to pixels;
- kept only if the ROI plus ``padding`` lies inside the image, and only if
  no other detection lies closer than ``min_distance_px``: both members of
  a close pair are dropped, whatever their brightness.
`tests/uipsf_tests/test_rois.py` holds this to SMAPpy's own function while the two
agree; they may diverge later, deliberately and documented.  Brightness is
not filtered here: uipsf's bead rejection judges each bead by its fit
(`workflows/rejection.py`).

**Extraction** cuts ROI × ROI pixels around integer centres, so the bead
sits within half a pixel of the ROI's centre pixel and x, y start at 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .dataset import Batch


@dataclass
class Beads:
    """Bead centres in a list of stacks: integer pixel (x, y) in each stack, and the stack's index."""
    x: np.ndarray
    y: np.ndarray
    stack: np.ndarray
    peak: np.ndarray            # the smoothed projection above its median, photons

    def __len__(self):
        return len(self.x)


def find_beads(stacks, roi_size: int, detection_sigma_px: float = 2.0, detection_threshold_sigma: float = 6.0,
               min_distance_px: float = 25.0, padding: int = 5) -> Beads:
    """Isolated beads in every stack (`BeadStack`s), by SMAPpy's convention (module docstring)."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    xs, ys, ss, hs = [], [], [], []
    radius = roi_size // 2 + padding
    for i, st in enumerate(stacks):
        projection = np.asarray(st.photons, float).max(0)
        filtered = ndimage.gaussian_filter(projection, detection_sigma_px)
        background = np.median(filtered)
        noise = 1.4826 * np.median(np.abs(filtered - background))
        threshold = background + detection_threshold_sigma * max(noise, np.finfo(float).eps)
        peaks = (filtered == ndimage.maximum_filter(filtered, size=3)) & (filtered > threshold)
        labels, n = ndimage.label(peaks)
        centres = ndimage.center_of_mass(filtered, labels, np.arange(1, n + 1)) if n else []
        yx = np.rint(centres).astype(int).reshape(-1, 2)
        good = ((yx >= radius) & (yx < np.array(projection.shape) - radius)).all(axis=1)
        for a, b in cKDTree(yx).query_pairs(min_distance_px):   # a neighbour outside the border counts too
            good[a] = good[b] = False
        yx = yx[good]
        xs.append(yx[:, 1])
        ys.append(yx[:, 0])
        ss.append(np.full(len(yx), i))
        hs.append(filtered[yx[:, 0], yx[:, 1]] - background)
    def cat(parts, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)
    return Beads(cat(xs, int), cat(ys, int), cat(ss, int), cat(hs, float))


def extract_rois(stacks, beads: Beads, roi_size: int, slices=slice(1, None), dtype=torch.float32) -> Batch:
    """The `Batch` [E, 1, K, ROI, ROI] of the beads' ROIs, with the stacks' z, noise and saturation.

    **The first slice is dropped by default** (``slices=slice(1, None)``):
    the first frame of a stack is not like the others.  On the Zenodo
    `1ch_40nm_bead` stacks it is 22 % brighter, shared by all beads
    (docs/uipsf/architecture.md, M3), and v1's settings drop it too.
    ``slices=slice(None)`` keeps every slice.  The nominal focus of each
    stack is the middle of the slices kept; the beads' fitted z are
    relative to it.
    """
    half = roi_size // 2
    data, sat, z = [], [], []
    for x, y, s in zip(beads.x, beads.y, beads.stack, strict=True):
        st = stacks[s]
        sl = (slices, slice(y - half, y - half + roi_size), slice(x - half, x - half + roi_size))
        data.append(st.photons[sl])
        sat.append(st.saturated[sl] if st.saturated is not None else np.zeros(data[-1].shape, bool))
        z.append(st.z_objective_nm[slices])
    if not data:
        raise ValueError("no beads to extract")
    if len({len(v) for v in z}) > 1:
        raise ValueError("stacks of different lengths cannot share a batch")
    z = np.stack(z)
    z = z - z.mean(-1, keepdims=True)          # each stack about its own middle: the nominal focus
    z_obj = torch.tensor(z[0] if np.allclose(z, z[0], atol=1e-3) else z, dtype=dtype)
    s = {float(stacks[i].read_noise_var) for i in beads.stack}
    if len(s) > 1:
        raise ValueError("stacks with different read noise: per-pixel maps come later")
    chip = np.array([[stacks[i].chip_origin[0] + x, stacks[i].chip_origin[1] + y]
                     for x, y, i in zip(beads.x, beads.y, beads.stack, strict=True)], float)
    return Batch(data=torch.tensor(np.stack(data), dtype=dtype)[:, None],
                 z_nm=Batch.z_from_objective(z_obj),
                 s=s.pop(), mask=~torch.tensor(np.stack(sat))[:, None],
                 field_xy=torch.tensor(chip, dtype=dtype), stack_id=torch.tensor(beads.stack))
