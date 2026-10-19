"""Two channels on one chip: the split, the bead pairs and a starting transform (plan §5.7, M4).

**Pairing follows SMAPpy's convention** (`smappy.calibrate.dual.collect_dual_beads`), as `find_beads`
follows its `detect_beads`, so that a standalone uipsf and SMAPpy's calibration window pair the same
beads; `tests/uipsf_tests/test_pairing.py` holds the pairs equal to SMAPpy's while SMAPpy is installed.  The two
may diverge later, deliberately and documented.  The steps:

1. **The layout** (`detect_layout`, SMAPpy's `detect_bead_layout` and `transform.detect_layout`): beads
   are found over each whole image; every within-stack pair of them votes, in a histogram of 2-px bins,
   for the vector between partners -- a difference along both axes (no mirror), or a *sum* along one
   (a mirror on that axis: partners satisfy x₁ + x₂ = const).  Each histogram is scored by a difference
   of Gaussians (σ 2 and 8 bins: a sharp peak stands, the smooth spread of same-half pairs subtracts
   away), short vectors near the origin excluded; the sharpest peak names the mirror, and the axis the
   partners lie far apart along names the split.  The main channel is the left or upper half.
2. **The split** is the middle of the first stack's image, in chip pixels; each stack is cut there.
3. **Beads in each half** by `find_beads` (SMAPpy's `detect_beads`), at sub-pixel positions from the
   mean of the 11 central planes over 13 × 13 pixels.  SMAPpy fits an elliptical Gaussian there; uipsf
   takes the background-free centroid: for pairing within 12 px the two agree.
4. **The offset** between the halves by a vote: the secondary positions mirrored (or shifted by the
   cut), then every main–secondary difference within a stack in a histogram of 2-px bins, smoothed by
   one bin; its peak.  Only beads within SMAPpy's brightness range (a factor 6 about the median) vote.
5. **Pairs**: mutual nearest neighbours within ``coarse_tolerance_px`` (12, SMAPpy's bead default)
   after that offset, stack by stack.
6. **A starting transform**, secondary → main in chip pixels (SMAPpy's `DualColorCalibration.
   transformation`): a projective least-squares fit to the pairs, refitted without pairs more than
   3 robust σ (and 0.5 px) off.  SMAPpy's own fit is RANSAC and a soft-L1 refinement; uipsf refines
   the transform jointly with the PSFs anyway (`model.channels`), so its start need only be close.

`extract_pair_rois` cuts each pair's ROIs, one per channel, into one `Batch` [E, 2, K, Y, X] with each
ROI's chip position (`Batch.roi_xy`), what the channel transform maps between.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .dataset import Batch
from .io import BeadStack
from .rois import find_beads

LAYOUTS = ("right-left", "right-left mirrored", "up-down", "up-down mirrored")
VOTE_BIN_PX, VOTE_SMOOTH_PX, ORIGIN_GUARD_BINS = 2.0, 4.0, 3       # SMAPpy's RegisterSettings, transform.py
COARSE_TOLERANCE_PX = 12.0                                         # SMAPpy's bead_registration()
BRIGHTNESS_RANGE = 6.0                                             # SMAPpy's CalibrationSettings


@dataclass
class Pairs:
    """Bead pairs on a split chip.  Positions are (x, y) in each stack's full image (zero-based pixel
    centres); add the stack's ``chip_origin`` for chip pixels."""
    main_xy: np.ndarray            # [E, 2] sub-pixel positions in the main half
    secondary_xy: np.ndarray       # [E, 2]
    main_px: np.ndarray            # [E, 2] the detections' whole pixels, where the ROIs are cut
    secondary_px: np.ndarray       # [E, 2]
    stack: np.ndarray              # [E]
    layout: str
    split: int                     # chip pixel of the second half's first column or row
    transform: np.ndarray          # [3, 3] secondary → main chip pixels, H₂₂ = 1
    chip_origin: np.ndarray        # [S, 2] each stack's
    image_shape: tuple[int, int] = (0, 0)      # (height, width) of the whole image
    sources: list = None                       # per stack: path, camera ROI, shape, z

    def __len__(self):
        return len(self.stack)

    @property
    def mirror_axis(self) -> int | None:
        """0 for x, 1 for y, as SMAPpy's ``mirror_axis_xy``; None unmirrored."""
        return None if "mirrored" not in self.layout else (0 if "right-left" in self.layout else 1)

    def geometry(self) -> dict:
        """The split as SMAPpy records it with a dual calibration (`DualColorCalibration.geometry`), which its
        global fitter reads (`dualfit`): written into the model file so that SMAPpy's calibration made from
        it is one of its own dual calibrations."""
        main = "upper" if "up-down" in self.layout else "left"
        return {"layout": self.layout, "main_channel": main, "split_position": int(self.split),
                "image_shape": [int(v) for v in self.image_shape], "coordinate_system": "camera-chip",
                "sources": list(self.sources or []), "mirror_axis_xy": self.mirror_axis,
                "convention": "zero-based pixel centers; split is first index of second half"}

    def chip(self, which: str = "main") -> np.ndarray:
        xy = self.main_xy if which == "main" else self.secondary_xy
        return xy + self.chip_origin[self.stack]


def _vote_histogram(x, y, frame, mode):
    """SMAPpy's `transform._vote`: (score, peak (x, y)) of the feature vectors of every within-frame pair."""
    from scipy import ndimage
    vectors = []
    for f in np.unique(frame):
        fx, fy = x[frame == f], y[frame == f]
        k = len(fx)
        if k < 2:
            continue
        dx = (fx[None, :] + fx[:, None]) if mode == "x" else (fx[None, :] - fx[:, None])
        dy = (fy[None, :] + fy[:, None]) if mode == "y" else (fy[None, :] - fy[:, None])
        off = ~np.eye(k, dtype=bool)
        vectors.append(np.c_[dx[off], dy[off]])
    if not vectors or sum(len(v) for v in vectors) < 4:
        return 0.0, np.zeros(2)
    vectors = np.concatenate(vectors)
    ranges = []
    for values, summed in ((x, mode == "x"), (y, mode == "y")):
        if summed:
            ranges.append((2 * values.min() - VOTE_BIN_PX, 2 * values.max() + VOTE_BIN_PX))
        else:
            span = values.max() - values.min() + VOTE_BIN_PX
            ranges.append((-span, span))
    bins = [max(int(round((hi - lo) / VOTE_BIN_PX)), 4) for lo, hi in ranges]
    hist, ex, ey = np.histogram2d(vectors[:, 0], vectors[:, 1], bins=bins, range=ranges)
    narrow = VOTE_SMOOTH_PX / VOTE_BIN_PX
    score = ndimage.gaussian_filter(hist, narrow) - ndimage.gaussian_filter(hist, narrow * 4)
    if mode == "none":
        centre = [np.searchsorted(e, 0.0) - 1 for e in (ex, ey)]
        lo = [max(c - ORIGIN_GUARD_BINS, 0) for c in centre]
        score[lo[0]:centre[0] + ORIGIN_GUARD_BINS + 1, lo[1]:centre[1] + ORIGIN_GUARD_BINS + 1] = -np.inf
    i, j = np.unravel_index(int(np.argmax(score)), score.shape)
    finite = score[np.isfinite(score)]
    spread = float(np.std(finite)) or 1.0
    r = max(int(round(narrow)), 1)
    win = hist[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1]
    cx = ((ex[:-1] + ex[1:]) / 2)[max(i - r, 0):i + r + 1]
    cy = ((ey[:-1] + ey[1:]) / 2)[max(j - r, 0):j + r + 1]
    w = win.sum()
    peak = (np.array([win.sum(1) @ cx / w, win.sum(0) @ cy / w]) if w > 0
            else np.array([cx[min(r, len(cx) - 1)], cy[min(r, len(cy) - 1)]]))
    return float(score[i, j] / spread), peak


def detect_layout(stacks, roi_size: int = 27, **find) -> str:
    """How the chip is split, from where the beads are (step 1 of the module docstring)."""
    beads = find_beads(stacks, roi_size, **find)
    if len(beads) < 4:
        raise ValueError(f"{len(beads)} beads found: too few to detect the layout from; give it")
    x, y = beads.x.astype(float), beads.y.astype(float)
    votes = {mode: _vote_histogram(x, y, beads.stack, mode) for mode in ("none", "x", "y")}
    mode = max(votes, key=lambda m: votes[m][0])
    if not votes[mode][0] > 0:
        raise ValueError("no offset between the halves stands out among the beads")
    if mode != "none":
        axis = "xy".index(mode)
    else:
        extent = np.array([x.max() - x.min(), y.max() - y.min()])
        axis = int(np.argmax(np.abs(votes[mode][1]) / np.maximum(extent, 1.0)))
    return ("right-left" if axis == 0 else "up-down") + (" mirrored" if mode != "none" else "")


def _centroids(stack: BeadStack, beads, cut_axis, cut, half):
    """Sub-pixel (x, y) in the full image: the background-free centroid of the 11 central planes'
    mean over 13 × 13 pixels around each detection (SMAPpy fits a Gaussian to the same)."""
    out = []
    K = stack.photons.shape[0]
    planes = slice(max(0, K // 2 - 5), K // 2 + 6)
    for x, y in zip(beads.x, beads.y, strict=True):
        full = [x, y]
        full[cut_axis] += cut * half
        X, Y = full
        roi = np.asarray(stack.photons[planes, Y - 6:Y + 7, X - 6:X + 7], float).mean(0)
        roi = np.clip(roi - np.median(np.r_[roi[0], roi[-1], roi[1:-1, 0], roi[1:-1, -1]]), 0, None)
        g = np.arange(13) - 6
        w = roi.sum()
        out.append([X + (roi.sum(0) @ g) / w, Y + (roi.sum(1) @ g) / w] if w > 0 else [X, Y])
    return np.asarray(out, float).reshape(-1, 2)


def _brightness_ok(stack, beads, cut_axis, cut, half, radius):
    """SMAPpy's brightness of each bead (the largest background-free plane sum of its padded box)."""
    out = []
    for x, y in zip(beads.x, beads.y, strict=True):
        full = [x, y]
        full[cut_axis] += cut * half
        X, Y = full
        v = np.asarray(stack.photons[:, Y - radius:Y + radius + 1, X - radius:X + radius + 1], float)
        border = np.concatenate((v[:, 0], v[:, -1], v[:, 1:-1, 0], v[:, 1:-1, -1]), axis=1)
        out.append(float((v - np.median(border)).sum((1, 2)).max()))
    return np.asarray(out)


def _split(st: BeadStack, axis, lo, hi):
    region = [slice(None)] * 3
    region[axis + 1] = slice(lo, hi)
    return BeadStack(st.photons[tuple(region)], st.z_objective_nm, st.read_noise_var, st.chip_origin,
                     None if st.saturated is None else st.saturated[tuple(region)], st.source)


def fit_projective(source, target) -> np.ndarray:
    """The projective map source → target (normalised DLT), H₂₂ = 1."""
    def normalise(p):
        c = p.mean(0)
        s = np.sqrt(2) / max(np.sqrt(((p - c) ** 2).sum(1).mean()), 1e-15)
        return np.c_[(p - c) * s, np.ones(len(p))], np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    if len(source) < 4:
        raise ValueError("a projective fit needs at least four pairs")
    a, na = normalise(np.asarray(source, float))
    b, nb = normalise(np.asarray(target, float))
    z = np.zeros((len(a), 3))
    design = np.concatenate([np.c_[-a, z, b[:, :1] * a], np.c_[z, -a, b[:, 1:2] * a]])
    h = np.linalg.svd(design)[2][-1].reshape(3, 3)
    H = np.linalg.inv(nb) @ h @ na
    return H / H[2, 2]


def _apply(H, xy):
    q = np.c_[xy, np.ones(len(xy))] @ H.T
    return q[:, :2] / q[:, 2:]


def pair_beads(stacks, roi_size: int = 27, layout: str = "auto", split: int | None = None,
               coarse_tolerance_px: float = COARSE_TOLERANCE_PX, min_pairs: int = 8, padding: int = 5,
               **find) -> Pairs:
    """Bead pairs in split-chip stacks (`BeadStack`s of the whole image), by SMAPpy's convention (module
    docstring).  ``layout`` one of `LAYOUTS` or "auto"; ``split`` the chip pixel of the second half's
    first column or row (default the middle of the first image)."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    if layout == "auto":
        layout = detect_layout(stacks, roi_size, padding=padding, **find)
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be 'auto' or one of {LAYOUTS}")
    shape = stacks[0].photons.shape[-2:]
    if any(st.photons.shape[-2:] != shape for st in stacks):
        raise ValueError("the stacks must share one image size")
    axis = 1 if "right-left" in layout else 0              # array axis (y, x) the chip is split along
    coord = 1 - axis                                       # the same axis as (x, y)
    origins = np.array([st.chip_origin for st in stacks], float)
    split = int(origins[0][coord]) + shape[axis] // 2 if split is None else int(split)
    cuts = [split - int(o[coord]) for o in origins]
    if not all(0 < c < shape[axis] for c in cuts):
        raise ValueError(f"the split at {split} px (chip) lies outside a bead image")
    radius = roi_size // 2 + padding
    points, pixels, which, bright = [], [], [], []
    for half in (0, 1):                                    # main (left or upper) first
        pts, pix, stk, ok = [], [], [], []
        for s, (st, cut) in enumerate(zip(stacks, cuts, strict=True)):
            part = _split(st, axis, 0, cut) if half == 0 else _split(st, axis, cut, shape[axis])
            b = find_beads([part], roi_size, padding=padding, **find)
            pts.append(_centroids(st, b, coord, cut, half) + origins[s])
            px = np.c_[b.x, b.y]
            px[:, coord] += cut * half
            pix.append(px)
            stk.append(np.full(len(b), s))
            ok.append(_brightness_ok(st, b, coord, cut, half, radius))
        points.append(np.concatenate(pts))
        pixels.append(np.concatenate(pix).astype(int))
        which.append(np.concatenate(stk))
        brightness = np.concatenate(ok)
        positive = brightness > 0
        centre = np.median(brightness[positive]) if positive.any() else 1.0
        half_range = np.sqrt(BRIGHTNESS_RANGE)
        bright.append(positive & (brightness >= centre / half_range) & (brightness <= centre * half_range))
    approx = points[1].copy()
    if "mirrored" in layout:
        approx[:, coord] = 2 * split - 1 - approx[:, coord]
    else:
        approx[:, coord] -= np.array([cuts[s] for s in which[1]], float)
    diffs = []
    for s in range(len(stacks)):
        a = np.flatnonzero((which[0] == s) & bright[0])
        b = np.flatnonzero((which[1] == s) & bright[1])
        if len(a) and len(b):
            diffs.append((points[0][a, None] - approx[None, b]).reshape(-1, 2))
    if not diffs:
        raise ValueError("no beads in both halves of any stack to vote for the offset")
    diffs = np.concatenate(diffs)
    low = np.floor(diffs.min(0) / 2).astype(int) - 4
    bins = np.rint(diffs / 2).astype(int) - low
    hist = np.zeros(tuple(bins.max(0) + 5))
    np.add.at(hist, tuple(bins.T), 1)
    peak = np.array(np.unravel_index(np.argmax(ndimage.gaussian_filter(hist, 1)), hist.shape))
    approx += (peak + low) * 2
    pairs = []
    for s in range(len(stacks)):
        a = np.flatnonzero(which[0] == s)
        b = np.flatnonzero(which[1] == s)
        if not len(a) or not len(b):
            continue
        distance, index = cKDTree(approx[b]).query(points[0][a])
        _, back = cKDTree(points[0][a]).query(approx[b])
        for k, (d, j) in enumerate(zip(distance, index, strict=True)):
            if d <= coarse_tolerance_px and back[j] == k:
                pairs.append((a[k], b[j]))
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    if len(pairs) < min_pairs:
        raise ValueError(f"only {len(pairs)} bead pairs; need {min_pairs}")
    main, second = points[0][pairs[:, 0]], points[1][pairs[:, 1]]
    H = fit_projective(second, main)
    for _ in range(3):                         # drop pairs far off the fit, then refit
        r = np.linalg.norm(_apply(H, second) - main, axis=1)
        good = r <= max(0.5, 3 * 1.4826 * np.median(np.abs(r - np.median(r))) + np.median(r))
        if good.all() or good.sum() < 4:
            break
        H = fit_projective(second[good], main[good])
    return Pairs(main - origins[which[0][pairs[:, 0]]], second - origins[which[1][pairs[:, 1]]],
                 pixels[0][pairs[:, 0]], pixels[1][pairs[:, 1]], which[0][pairs[:, 0]], layout, split, H,
                 origins, tuple(int(v) for v in shape),
                 [{"path": st.source, "axes": None, "shape": list(st.photons.shape),
                   "roi": [int(o[0]), int(o[1]), int(shape[1]), int(shape[0])],
                   "z_nm": np.asarray(st.z_objective_nm, float).tolist()}
                  for st, o in zip(stacks, origins, strict=True)])


def extract_pair_rois(stacks, pairs: Pairs, roi_size: int, slices=slice(1, None), dtype=torch.float32) -> Batch:
    """The `Batch` [E, 2, K, ROI, ROI] of the pairs' ROIs, the main channel first, each cut around its own
    detection; ``roi_xy`` their centres in chip pixels.  Slices, z and noise as `rois.extract_rois`."""
    from .rois import Beads, extract_rois
    parts = []
    for px in (pairs.main_px, pairs.secondary_px):
        beads = Beads(px[:, 0], px[:, 1], pairs.stack, np.zeros(len(pairs)))
        parts.append(extract_rois(stacks, beads, roi_size, slices=slices, dtype=dtype))
    a, b = parts
    origin = pairs.chip_origin[pairs.stack]
    roi = np.stack([pairs.main_px + origin, pairs.secondary_px + origin], 1)
    return Batch(data=torch.cat([a.data, b.data], 1), z_nm=a.z_nm, s=a.s, mask=torch.cat([a.mask, b.mask], 1),
                 field_xy=a.field_xy, stack_id=a.stack_id, roi_xy=torch.tensor(roi, dtype=dtype))
