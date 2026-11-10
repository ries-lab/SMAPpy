"""How far a split camera's channel map is off in the sample, from the raw frames' crosstalk.

A two-colour acquisition on a split camera, one dye per half (*two channels*, *Overlay
channels*), is put together by a map measured on beads.  The beads were imaged at another time
and in another colour than the dyes, so the map may be off in the sample by some nanometres.  The
localizations cannot say by how much: each molecule is found in its own half only.  But a few
percent of its light reaches the other half -- crosstalk, far too dim to be detected (about 15
photons a frame on the Pan1 movie this was made on, 2026-11-10).  Summed over a hundred thousand
molecules it is a bright, sharp spot, and where that spot lies against where the map puts the
partner is the map's error at the dye's own colour, in the sample.

So, for every well-fitted molecule of one half:

* the raw frame of the **other** half is cut, a ``(2r+1)^2`` patch around the whole pixel
  nearest to where the map puts the partner, and the same patch of the frame ``lag`` frames later
  is the control: what does not blink -- autofluorescence, out-of-focus haze seen in both
  halves -- is in both and cancels; a molecule is on for a few frames, not ``lag``.
  On the Pan1 movie the slow part was 45 % of the summed signal and sat 20 nm elsewhere.
* every pixel of the patch is added, its value unchanged, to a grid ``FINE`` times finer than a
  pixel, at its exact offset from the predicted position (each molecule's own sub-pixel position)
  -- no image is resampled.  Each bin also adds up those exact offsets, and the fit puts the
  bin's value at their mean, not at the bin's centre: with the predicted positions all at one
  sub-pixel place (beads, or a sample on a grid) bin centres moved a 10 nm residual to 19 nm,
  bilinear interpolation to 12 nm, and mean positions left it at 9.9 nm, as a fit to every pixel
  does (`PatchSum`; the simulation is in the plugin's page).
* a Gaussian plus a constant fitted to the summed signal less the summed control gives the
  residual (its centre), the width (a check that it is a spot) and, against the same sum at the
  molecule's own position, the fraction of its light that leaked -- the camera's offset and gain
  cancel in the ratio, so no camera is needed.
* the frames are taken in blocks, each block summed apart, and the error of every number is the
  spread over the blocks.

Molecules whose patch holds a localization of the other half -- within the patch's radius plus
two pixels of the predicted partner, in the frame or in the control frame -- are left out: a
molecule of the other dye is on there, and its light is not crosstalk.  Within 3 px only, the
other dye's bright molecules a few pixels away put their tails into the sum: on a simulation
with a 240-photon leak and 3000-photon neighbours the offset came out 4 nm off with an error of
1.4 nm, and right once the whole patch was kept clear.  Excluded in the signal frame only, the
control kept the other dye's light that the signal had lost, and the measurement came out
negative.

Measured over a grid of regions of the field, the residuals give a shift or an affine correction
of the map (`fit_correction`): on the Pan1 movie the y residual went from +20 nm at the top of
the field to +9 nm at the bottom, which no single shift describes.

Everything here is on arrays: the plugin (`plugins/raw_offset.py`) finds the frames, the map and
the table, and hands them in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: fine bins per camera pixel of the summed patch
FINE = 4
#: a localization of the other half this much farther (px) from the predicted partner than the
#: patch's radius leaves the molecule out: a molecule of the other dye is on in the patch
EXCLUDE_MARGIN_PX = 2.0
#: a direction is reported as measured when its leaked light is this many times its error
MIN_SNR = 5.0
#: the frames are summed in this many blocks for the errors
BLOCKS = 8


# ------------------------------------------------------------------ the sum
@dataclass
class PatchSum:
    """Patches summed on a fine grid at each pixel's exact offset from a predicted position.

    ``sig`` and ``ctl`` the summed values of the frame and of the control frame, ``n`` how many
    pixels landed in each bin, ``su`` / ``sv`` the sum of their exact offsets (px), so that the
    fit can place each bin at the mean offset of what is in it.
    """
    radius: int
    fine: int = FINE
    sig: np.ndarray = field(default=None)
    ctl: np.ndarray = field(default=None)
    n: np.ndarray = field(default=None)
    su: np.ndarray = field(default=None)
    sv: np.ndarray = field(default=None)
    molecules: int = 0

    def __post_init__(self):
        nb = (2 * self.radius + 1) * self.fine
        for name in ("sig", "ctl", "n", "su", "sv"):
            if getattr(self, name) is None:
                setattr(self, name, np.zeros((nb, nb)))

    def add(self, signal: np.ndarray, control: np.ndarray, dx: np.ndarray, dy: np.ndarray):
        """Add patches ``(m, 2r+1, 2r+1)`` cut around the whole pixel nearest each predicted
        position; ``dx``, ``dy`` the predicted position within that pixel (-0.5 to 0.5)."""
        if not len(signal):
            return
        r, f = self.radius, self.fine
        nb = (2 * r + 1) * f
        k = np.arange(-r, r + 1, dtype=float)
        u = np.broadcast_to(k[None, None, :] - np.asarray(dx, float)[:, None, None], signal.shape)
        v = np.broadcast_to(k[None, :, None] - np.asarray(dy, float)[:, None, None], signal.shape)
        gx = np.floor((u + r + 0.5) * f).astype(np.int64)
        gy = np.floor((v + r + 0.5) * f).astype(np.int64)
        ok = (gx >= 0) & (gx < nb) & (gy >= 0) & (gy < nb)
        index = (gy[ok] * nb + gx[ok])
        size = nb * nb
        for name, values in (("sig", signal), ("ctl", control), ("su", u), ("sv", v)):
            getattr(self, name).ravel()[:] += np.bincount(index, weights=np.asarray(
                values, float)[ok], minlength=size)
        self.n.ravel()[:] += np.bincount(index, minlength=size)
        self.molecules += len(signal)

    def __iadd__(self, other: "PatchSum"):
        for name in ("sig", "ctl", "n", "su", "sv"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.molecules += other.molecules
        return self

    def image(self) -> np.ndarray:
        """The mean of signal less control per bin (0 where nothing landed)."""
        return np.where(self.n > 0, (self.sig - self.ctl) / np.maximum(self.n, 1), 0.0)

    def positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Each bin's mean offset (px) from the predicted position; its centre where empty."""
        nb = self.n.shape[0]
        centres = (np.arange(nb) + 0.5) / self.fine - self.radius - 0.5
        cu, cv = np.meshgrid(centres, centres)
        safe = np.maximum(self.n, 1)
        return (np.where(self.n > 0, self.su / safe, cu),
                np.where(self.n > 0, self.sv / safe, cv))


@dataclass
class Spot:
    """A Gaussian fitted to a `PatchSum`: centre (px, from the predicted position), width (px),
    amplitude (the summed spot's integral per molecule, raw units) and constant."""
    x: float
    y: float
    sigma: float
    amplitude: float
    background: float
    ok: bool = True


def fit_spot(summed: PatchSum, sigma_px: float = 1.2) -> Spot:
    """A circular Gaussian plus a constant, least squares weighted by how many pixels each bin
    holds, at each bin's mean offset."""
    from scipy.optimize import least_squares
    n = summed.n
    ok = n > 0
    if ok.sum() < 10 or summed.molecules == 0:
        return Spot(np.nan, np.nan, np.nan, np.nan, np.nan, False)
    u, v = summed.positions()
    u, v, img, w = u[ok], v[ok], summed.image()[ok], np.sqrt(n[ok])

    def model(p):
        x0, y0, s, a, b = p
        return a / (2 * np.pi * s * s) * np.exp(-((u - x0) ** 2 + (v - y0) ** 2) / (2 * s * s)) + b

    edge = np.median(img[np.hypot(u, v) > summed.radius - 1]) if np.any(
        np.hypot(u, v) > summed.radius - 1) else 0.0
    p0 = [0.0, 0.0, sigma_px, max(float((img - edge).sum()) / summed.fine ** 2, 1e-6), edge]
    r = summed.radius
    try:
        fit = least_squares(lambda p: (model(p) - img) * w, p0,
                            bounds=([-r / 2, -r / 2, 0.3, -np.inf, -np.inf],
                                    [r / 2, r / 2, r / 1.5, np.inf, np.inf]))
    except ValueError:
        return Spot(np.nan, np.nan, np.nan, np.nan, np.nan, False)
    x0, y0, s, a, b = fit.x
    return Spot(float(x0), float(y0), float(s), float(a), float(b), bool(fit.success))


# ------------------------------------------------------------- the molecules
@dataclass
class Direction:
    """The molecules of one half whose light is looked for in the other.

    Positions in chip pixels: ``source`` where each molecule is, ``target`` where the map puts
    its partner in the other half, ``place`` where it is in the main half's coordinates (the
    region it is counted in); ``frame`` its frame; ``region`` the index of its region."""
    name: str
    source: np.ndarray
    target: np.ndarray
    place: np.ndarray
    frame: np.ndarray
    region: np.ndarray


def exclude_seen(target: np.ndarray, frame: np.ndarray, others: np.ndarray,
                 other_frames: np.ndarray, lag: int, distance: float) -> np.ndarray:
    """True where no localization of the other half lies within ``distance`` px of the
    predicted partner, in the molecule's frame or in its control frame (``frame + lag``)."""
    from scipy.spatial import cKDTree
    keep = np.ones(len(target), bool)
    if not len(others) or not len(target):
        return keep
    big = 4.0 * distance + 1e4        # frames far apart in the search
    tree = cKDTree(np.c_[others, np.asarray(other_frames, float) * big])
    for shift in (0, lag) if lag else (0,):
        d, _ = tree.query(np.c_[target, (np.asarray(frame, float) + shift) * big],
                          distance_upper_bound=distance)
        keep &= ~np.isfinite(d)
    return keep


def regions_of(place: np.ndarray, box: Sequence[float], grid: int) -> np.ndarray:
    """The index (row * grid + column) of a ``grid`` x ``grid`` division of ``box`` = (x0, y0,
    x1, y1) that each position lies in; -1 outside."""
    x0, y0, x1, y1 = (float(v) for v in box)
    col = np.floor((place[:, 0] - x0) / max(x1 - x0, 1e-9) * grid).astype(int)
    row = np.floor((place[:, 1] - y0) / max(y1 - y0, 1e-9) * grid).astype(int)
    inside = (col >= 0) & (col < grid) & (row >= 0) & (row < grid)
    return np.where(inside, row * grid + col, -1)


# ------------------------------------------------------------------ summing
@dataclass
class Sums:
    """The patch sums of one direction: ``leak[(region, block)]`` at the predicted partner and
    ``own[(region, block)]`` at the molecule itself (for the fraction that leaked)."""
    leak: Dict[Tuple[int, int], PatchSum]
    own: Dict[Tuple[int, int], PatchSum]


def sum_patches(blocks: Iterable[Tuple[int, np.ndarray]], directions: Sequence[Direction],
                roi_corner: Tuple[int, int], radius: int, lag: int, first: int, stop: int,
                n_blocks: int = BLOCKS, report: Optional[Callable[[str], None]] = None,
                stop_requested: Optional[Callable[[], bool]] = None) -> List[Sums]:
    """Sum every direction's patches over frames ``[first, stop)`` of ``blocks`` -- the movie's
    ``(first frame, (n, h, w) frames)`` from ``first`` to ``stop + lag``.  A molecule's control
    is its frame ``+ lag``, so the frames are kept ``lag`` past each block."""
    x0, y0 = int(roi_corner[0]), int(roi_corner[1])
    span = max(stop - first, 1)
    out = [Sums({}, {}) for _ in directions]
    pending = None                                    # frames read but not yet used
    pending_first = first
    k = np.arange(-radius, radius + 1)

    def block_of(frames):
        return np.minimum((frames - first) * n_blocks // span, n_blocks - 1)

    def process(stack, stack_first, lo, hi):
        """Molecules of frames [lo, hi): their frames and controls are in ``stack``."""
        for d, sums in zip(directions, out):
            sel = (d.frame >= lo) & (d.frame < hi) & (d.region >= 0)
            if not sel.any():
                continue
            fr = d.frame[sel] - stack_first
            blocks = block_of(d.frame[sel])
            for where, store in ((d.target[sel], sums.leak), (d.source[sel], sums.own)):
                cx = np.rint(where[:, 0]).astype(int)
                cy = np.rint(where[:, 1]).astype(int)
                ix = (cx - x0)[:, None, None] + k[None, None, :]
                iy = (cy - y0)[:, None, None] + k[None, :, None]
                h, w = stack.shape[1:]
                inside = ((ix.min(axis=(1, 2)) >= 0) & (ix.max(axis=(1, 2)) < w)
                          & (iy.min(axis=(1, 2)) >= 0) & (iy.max(axis=(1, 2)) < h))
                sig = stack[fr[:, None, None], iy.clip(0, h - 1), ix.clip(0, w - 1)]
                ctl = (stack[(fr + lag)[:, None, None], iy.clip(0, h - 1), ix.clip(0, w - 1)]
                       if lag else np.zeros_like(sig))
                dx, dy = where[:, 0] - cx, where[:, 1] - cy
                regions = d.region[sel]
                for key in set(zip(regions[inside].tolist(), blocks[inside].tolist())):
                    m = inside & (regions == key[0]) & (blocks == key[1])
                    store.setdefault(key, PatchSum(radius)).add(sig[m], ctl[m], dx[m], dy[m])

    for start, frames in blocks:
        if stop_requested is not None and stop_requested():
            raise InterruptedError("stopped")
        frames = np.asarray(frames, np.float32)
        if pending is None:
            pending, pending_first = frames, start
        else:
            pending = np.concatenate([pending, frames])
        usable_until = pending_first + len(pending) - lag     # frames whose control is read
        hi = min(usable_until, stop)
        if hi > pending_first:
            process(pending, pending_first, pending_first, hi)
            if report is not None:
                report(f"summed frames {first} to {hi} of {stop}")
            pending = pending[hi - pending_first:]
            pending_first = hi
        if pending_first >= stop:
            break
    return out


# ---------------------------------------------------------------- the answer
@dataclass
class Measured:
    """One direction in one region (or all regions, ``region`` -1): the spot's centre in the
    target half (px from the predicted partner) with its error from the blocks, the width (px),
    the fraction of the molecules' light that leaked and its error, and how many molecules."""
    region: int
    x: float
    y: float
    x_err: float
    y_err: float
    sigma: float
    fraction: float
    fraction_err: float
    molecules: int
    spot: Optional[Spot] = None
    summed: Optional[PatchSum] = None

    @property
    def measured(self) -> bool:
        return (np.isfinite(self.x) and np.isfinite(self.x_err) and self.fraction > 0
                and self.fraction > MIN_SNR * self.fraction_err)


def _merge(sums: Dict[Tuple[int, int], PatchSum], region: Optional[int],
           block: Optional[int], radius: int) -> PatchSum:
    out = PatchSum(radius)
    for (r, b), s in sums.items():
        if (region is None or r == region) and (block is None or b == block):
            out += s
    return out


def measure(sums: Sums, region: Optional[int], radius: int, sigma_px: float = 1.2,
            n_blocks: int = BLOCKS) -> Measured:
    """Fit the sums of ``region`` (None: all), and the same per block for the errors."""
    def one(block):
        leak = _merge(sums.leak, region, block, radius)
        own = _merge(sums.own, region, block, radius)
        spot = fit_spot(leak, sigma_px)
        mine = fit_spot(own, sigma_px)
        fraction = (spot.amplitude / mine.amplitude
                    if mine.ok and spot.ok and mine.amplitude > 0 else np.nan)
        return spot, fraction, leak
    spot, fraction, leak = one(None)
    per = [one(b) for b in range(n_blocks)]
    per = [(s, f) for s, f, lk in per if lk.molecules >= 20 and s.ok and np.isfinite(f)]
    if len(per) >= 3:
        xs = np.array([[s.x, s.y, f] for s, f in per])
        err = xs.std(axis=0, ddof=1) / np.sqrt(len(per))
    else:
        err = np.full(3, np.nan)
    return Measured(-1 if region is None else int(region), spot.x, spot.y, float(err[0]),
                    float(err[1]), spot.sigma, float(fraction), float(err[2]),
                    leak.molecules, spot, leak)


# ------------------------------------------------------------- the correction
def to_main(residual_target: np.ndarray, place: np.ndarray, to_reference: Callable,
            to_secondary: Callable, direction: str) -> np.ndarray:
    """A residual measured in the target half (px) as "true main position less mapped
    secondary" at ``place`` (main-half px): what the map should add there.

    ``"main -> secondary"``: the partner is at ``to_secondary(p) + r`` in the secondary half, so
    the map sends it to ``to_reference(that)``, and is off by ``p - to_reference(...)``.
    ``"secondary -> main"``: the partner lies at the mapped position plus ``r`` in the main half,
    which is the residual itself."""
    place = np.atleast_2d(np.asarray(place, float))
    r = np.atleast_2d(np.asarray(residual_target, float))
    if direction == "secondary -> main":
        return r
    q = to_secondary(place)
    return place - to_reference(q + r)


def fit_correction(place: np.ndarray, residual: np.ndarray, error: np.ndarray,
                   model: str = "shift") -> Tuple[np.ndarray, np.ndarray]:
    """``(matrix, residual_after)``: the 3x3 (chip px, acting on main-half positions) that a
    mapped secondary position is moved by -- ``x + r(x)`` with ``r`` a constant (*shift*) or an
    affine function (*affine*) -- weighted by the inverse variance of each region's residual."""
    place = np.atleast_2d(np.asarray(place, float))
    residual = np.atleast_2d(np.asarray(residual, float))
    error = np.atleast_2d(np.asarray(error, float))
    w = 1.0 / np.maximum(error, 1e-6) ** 2                     # (n, 2), per axis
    out = np.eye(3)
    if model == "shift" or len(place) < 3:
        b = (w * residual).sum(axis=0) / w.sum(axis=0)
        out[:2, 2] = b
        return out, residual - b
    if model != "affine":
        raise ValueError(f"no correction {model!r}: shift or affine")
    centre = place.mean(axis=0)
    design = np.c_[place - centre, np.ones(len(place))]
    coef = np.zeros((3, 2))
    for axis in range(2):
        sw = np.sqrt(w[:, axis])
        coef[:, axis], *_ = np.linalg.lstsq(design * sw[:, None], residual[:, axis] * sw,
                                            rcond=None)
    a = coef[:2].T                                                # r = a (x - c) + t
    out[:2, :2] += a
    out[:2, 2] = coef[2] - a @ centre
    after = residual - (design @ coef)
    return out, after


def correction_nm(matrix_px: np.ndarray, pixelsize_nm: Tuple[float, float]) -> np.ndarray:
    """The 3x3 in chip pixels as the 4x4 affine of a `transformation.Transformation` in nm."""
    s = np.diag([float(pixelsize_nm[0]), float(pixelsize_nm[1]), 1.0])
    c = s @ np.asarray(matrix_px, float) @ np.linalg.inv(s)
    m = np.eye(4)
    m[:2, :2] = c[:2, :2]
    m[:2, 3] = c[:2, 2]
    return m
