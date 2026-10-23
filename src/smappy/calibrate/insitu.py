"""A blinking movie to what uipsf learns an in situ PSF from: molecules found, measured and chosen.

In situ learning (`docs/uipsf/plan.md` §9, M5) needs single molecules at every height, the out-of-focus
ones most, since they carry most of what a PSF has to learn.  Detection made for localisation loses
exactly those: uiPSF v1 thresholds each frame relative to its brightest in-focus peak, and a defocused
spot with as many photons has a low peak.  So everything here is chosen to keep them, and to keep out
what would mislead the fit -- overlapping molecules first.  The steps, each a function:

1. **Background** (`temporal_background`): a sliding median over time, per pixel.  A molecule is on for
   a few frames, the median of ``background_frames`` (21) frames does not see it.
2. **Detection** (`detect`): a difference of Gaussians at several widths (``scales_px``, 1.2, 2 and 3
   pixels, SMAPpy's `detect.DoGFilter`), so that a defocused spot is found at its own size.  A
   candidate is kept on its **absolute significance**: the photons above the background in a disc of
   radius 2σ against the background's Poisson noise there, never relative to the frame.  Of candidates
   of several widths at one place the most significant stays.
3. **A first fit** (`measure`): SMAPpy's elliptical Gaussian (C++, axis-aligned) in a ``roi_px`` region:
   photons, widths, position.  Model-free, so nothing favours the start's PSF.  Detections whose fits
   meet (within ``merge_px``) are one molecule: a defocused spot has several local maxima.  A Gaussian
   wider than a third of the region spread over the background and is no detection.
4. **Blinks** (`link`): SMAPpy's grouping (`group.connect`) with its precision test and no dark frame,
   so that what is judged about a molecule is judged on all its frames.  What is fitted is each **frame**
   (``unit="frame"``, the default), its own region about its own position, as v1 does: the selection's
   brightness window keeps the bright central frames and drops the partial first and last.  Summing a
   blink's frames into one region (``unit="blink"``: Poisson with mean (ΣN_k)·I + K·b) is the option, to
   be compared before it is the default (decided October 2026, plan §9 M5, step 4).
5. **Overlaps, by the spot's size** (`overlaps`, `neighbour_mask`): with r = ``overlap_k``·σ of each spot
   (3σ), a neighbour in any frame of the blink closer than r_i + r_j rejects all its frames; a farther
   one's disc is masked in the frames it is on in, and a region with more than ``max_masked`` of it masked
   is rejected.  Only neighbours with at least ``neighbour_fraction`` (a fifth) of the molecule's photons
   count: on dense real data false detections on the background and faint diffuse patches otherwise did
   most of the rejecting.  A blink whose inner frames' photons step by more than ``step_ratio`` (another molecule
   switching on or off under it) is rejected too; its first and last frames are partial and not judged.
   A unit whose own Gaussian, fitted with the mask, is wider than a third of the region is out (*wide*),
   as a detection is in step 3.
6. **Selection** (`select`): a **z proxy** from the summed spot's widths -- σx² − σy² for an astigmatic
   PSF, linear in z, or σx² + σy² for a 2D one, which says |z| -- split into ``bins`` bins.  Per bin:
   spots whose widths are far from the bin's typical ones (``shape_k`` robust sds) are out -- an
   unresolved pair is wider or elongated for its z; then a random draw of the bin's share of ``molecules`` (`quotas`) from a
   **brightness window** ``window_factor`` (3) wide below the bin's ``window_top`` (90th) percentile of
   photons, not the brightest: those are where unresolved pairs collect, and with a Gaussian
   illumination they are the field's centre.  Of the molecules, ``heldout`` (10 %) are held out, all
   their frames.

Every unit is a record with its state (*used*, *held out*, *rejected*, *not drawn*) and the reason,
as the beads are (`uipsf.workflows.rejection`).  `InSituROIs.batch` is the `uipsf.Batch` of the used
ones.  torch is imported only there: this module is imported without torch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..plugins import param


UNITS = (("frame", "frame"), ("blink", "blink, frames summed"))
PROXIES = (("astigmatism", "astigmatism: σx² − σy²"), ("size", "size: σx² + σy²"))


@dataclass
class InSituSettings:
    """How molecules are found and chosen.  One number to set, *molecules*; under *more* the few that depend
    on the data; the rest hidden, the defaults decided in M5 (a script or a chain still sets them)."""
    molecules: int = param(2000, label="molecules", min=21,
                           help="how many spots to learn from, each frame of a blink one: the movies are "
                                "read until this many pass the selection")
    significance: float = param(6.0, label="significance", min=1.0, advanced=True,
                                help="photons above the background in a disc, in standard deviations of its "
                                     "noise, for a detection")
    roi_px: int = param(15, label="region", unit="px", min=7, advanced=True,
                        help="the square region each molecule is fitted in (odd)")
    proxy: str = param("astigmatism", label="z proxy", choices=PROXIES, advanced=True,
                       help="how the first round sorts by height: an astigmatic PSF's elongation, or a 2D "
                            "PSF's size")
    block_frames: int = param(200, label="block", unit="frames", min=21, hidden=True,
                              help="consecutive frames read at once: the background's median runs within one")
    background_frames: int = param(21, label="background window", unit="frames", min=3, hidden=True,
                                   help="the sliding median over time that is the background")
    scales_px: tuple = param((1.2, 2.0, 3.0), label="DoG widths", hidden=True,
                             help="the difference of Gaussians' widths a spot is looked for at, px")
    merge_px: float = param(2.0, label="merge", unit="px", min=0.0, hidden=True,
                            help="detections whose Gaussians meet within this are one molecule")
    overlap_k: float = param(3.0, label="overlap", unit="σ", min=0.0, hidden=True,
                             help="a neighbour within this many of each spot's σ, added, rejects a molecule")
    max_masked: float = param(0.25, label="masked at most", min=0.0, max=1.0, hidden=True,
                              help="the fraction of a region a neighbour's mask may cover")
    neighbour_fraction: float = param(0.2, label="neighbour light", min=0.0, max=1.0, hidden=True,
                                      help="a neighbour dimmer than this fraction of the molecule neither "
                                           "rejects it nor is masked")
    link_k: float = param(2.5, label="linking", unit="sd", min=0.5, hidden=True,
                          help="SMAPpy's precision test for a blink's frames")
    link_dx_px: float = param(1.5, label="linking at most", unit="px", min=0.1, hidden=True,
                              help="the largest step a blink may make between frames")
    link_dx_min_px: float = param(0.5, label="linking at least", unit="px", min=0.0, hidden=True,
                                  help="always linked within this, whatever the precisions say")
    unit: str = param("frame", label="fitted", choices=UNITS, hidden=True,
                      help="each frame its own region, or a blink's frames summed into one")
    step_ratio: float = param(2.0, label="photon step", min=1.0, hidden=True,
                              help="a blink whose inner frames' photons differ by more is out")
    bins: int = param(21, label="z bins", min=1, hidden=True, help="the bins molecules are drawn from")
    window_factor: float = param(3.0, label="brightness window", min=1.0, hidden=True,
                                 help="how wide, as a factor, below the bin's top")
    window_top: float = param(0.9, label="window top", min=0.5, max=1.0, hidden=True,
                              help="the quantile of the bin's photons the window starts below")
    shape_k: float = param(3.0, label="shape", unit="sd", min=1.0, hidden=True,
                           help="robust standard deviations from the bin's typical shape at which one is out")
    heldout: float = param(0.1, label="held out", min=0.0, max=0.5, hidden=True,
                           help="the fraction of molecules kept out of learning, for the held-out check")
    seed: int = param(0, label="seed", hidden=True, help="the random draws' seed")
    # Not in the window yet: a script's, for one channel of a split chip (M2's 665LP movies, October 2026)
    region: Optional[tuple] = param(None, label="region", hidden=True,
                                    help="(x, y, width, height) in the movie's pixels: only molecules inside "
                                         "are used")

    @property
    def per_bin(self) -> int:
        """A z bin's share of *molecules*; a bin with fewer passes its shortfall on (`quotas`)."""
        return max(1, math.ceil(self.molecules / max(1, self.bins)))


# ------------------------------------------------------------------ 1, 2: background and detection

#: the background's median is taken every this many frames, each frame taking the nearest
BACKGROUND_STEP = 7


def temporal_background(frames: np.ndarray, window: int, step: int = BACKGROUND_STEP) -> np.ndarray:
    """[T, Y, X]: each pixel's median over ``window`` frames, taken every ``step`` frames (the windows
    clipped at the block's ends), each frame given the nearest.  A sliding median of every frame cost 4.6 s
    for 200 frames of 336 × 248 pixels, a third of finding the molecules, for a background that changes
    over tens of frames, not one."""
    frames = np.asarray(frames, np.float32)
    T = len(frames)
    half = int(window) // 2
    centres = np.arange(0, T, max(1, int(step)))
    medians = np.stack([np.median(frames[max(0, c - half):min(T, c + half + 1)], axis=0) for c in centres])
    nearest = np.clip(np.rint(np.arange(T) / max(1, int(step))).astype(int), 0, len(centres) - 1)
    return medians[nearest]


def _disc(radius: float) -> np.ndarray:
    r = int(np.ceil(radius))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= radius * radius).astype(np.float32)


@dataclass
class Candidates:
    frame: np.ndarray            # [n] int
    x: np.ndarray                # [n] int, the pixel
    y: np.ndarray
    scale: np.ndarray            # [n] the DoG width that found it, px
    significance: np.ndarray     # [n]

    def __len__(self) -> int:
        return len(self.frame)

    def take(self, index) -> Candidates:
        return Candidates(*(getattr(self, f)[index] for f in self.__dataclass_fields__))


def _disc_sums(image: np.ndarray, f, y, x, radius: float) -> np.ndarray:
    """The sum of ``image`` [T, Y, X] over a disc of ``radius`` about each (f, y, x), the edges repeated:
    only where it is needed.  Convolving the whole block with the disc, twice per width, was 5.8 s of
    14.5 for a block of the Zenodo tubulin movie."""
    _, Y, X = image.shape
    r = int(np.ceil(radius))
    dy, dx = np.nonzero(_disc(radius))
    dy, dx = dy - r, dx - r
    yy = np.clip(np.asarray(y)[:, None] + dy, 0, Y - 1)
    xx = np.clip(np.asarray(x)[:, None] + dx, 0, X - 1)
    return image[np.asarray(f)[:, None], yy, xx].sum(1)


def detect(frames: np.ndarray, background: np.ndarray, settings: InSituSettings,
           read_noise_var: float = 0.0) -> Candidates:
    """Every spot significant above the background at one of the widths (module docstring, 2)."""
    from ..detect import DoGFilter, local_maxima
    signal = np.asarray(frames, np.float32) - background
    found = []
    for s in settings.scales_px:
        f, y, x, _ = local_maxima(DoGFilter(float(s))(signal), threshold=0.0)
        radius = max(1.5, 2.0 * s)
        S = _disc_sums(signal, f, y, x, radius)
        B = _disc_sums(background, f, y, x, radius) + _disc(radius).sum() * read_noise_var
        z = S / np.sqrt(np.maximum(B, 1e-6))
        keep = z >= settings.significance
        found.append((f[keep], x[keep], y[keep], np.full(keep.sum(), float(s)), z[keep]))
    f, x, y, s, z = (np.concatenate(a) for a in zip(*found, strict=True))
    return _strongest(Candidates(f, x, y, s, z))


def _strongest(c: Candidates) -> Candidates:
    """Of candidates of several widths at one place the most significant: within 1.5·max(σ) + 1 pixels."""
    keep = _greedy(c.frame, c.x.astype(float), c.y.astype(float), c.significance,
                   lambda i, j: 1.5 * np.maximum(c.scale[i], c.scale[j]) + 1)
    return c.take(keep)


def _greedy(frame, x, y, score, distance) -> np.ndarray:
    """Indices kept: per frame, highest ``score`` first, each dropping the others closer than
    ``distance(i, others)`` -- with arrays per kept candidate, not a loop over pairs, which cost 1.9 s a
    block of the tubulin movie."""
    keep = []
    order = np.lexsort((-np.asarray(score), frame))
    starts = np.flatnonzero(np.r_[True, np.diff(np.asarray(frame)[order]) != 0])
    for a, b in zip(starts, np.r_[starts[1:], len(order)], strict=True):
        idx = order[a:b]                                   # this frame's, best first
        alive = np.ones(len(idx), bool)
        for k in range(len(idx)):
            if not alive[k]:
                continue
            i = idx[k]
            keep.append(i)
            rest = idx[k + 1:]
            near = np.hypot(x[rest] - x[i], y[rest] - y[i]) < distance(i, rest)
            alive[k + 1:] &= ~near
    return np.sort(np.asarray(keep, int))


# ------------------------------------------------------------------ 3: the first fit

def cut(frames: np.ndarray, frame, x, y, roi_px: int) -> tuple[np.ndarray, np.ndarray]:
    """(regions [n, roi, roi] float32, inside [n] bool): about whole pixels (x, y), those inside the image."""
    _, Y, X = frames.shape
    h = roi_px // 2
    frame, x, y = (np.asarray(a, int) for a in (frame, x, y))
    inside = (x >= h) & (x < X - h) & (y >= h) & (y < Y - h)
    span = np.arange(-h, h + 1)
    f, xi, yi = frame[inside], x[inside], y[inside]
    out = frames[f[:, None, None], (yi[:, None] + span)[:, :, None], (xi[:, None] + span)[:, None, :]]
    return np.ascontiguousarray(out, np.float32), inside


def gaussian(regions: np.ndarray) -> dict:
    """SMAPpy's elliptical Gaussian on each region: x, y (region pixels, centre (roi−1)/2), photons,
    background, sigma_x, sigma_y (px), their CRLBs and the log-likelihood; NaN where it failed."""
    from ..psf import GaussianPSF
    psf = GaussianPSF(sigma=1.5, elliptical=True)
    res = psf.fit(np.ascontiguousarray(regions, np.float32), 50)
    out = {k: np.asarray(v, float) for k, v in psf.unpack(res).items()}
    out["logl"] = np.asarray(res.logl, float)
    bad = ~np.isfinite(np.stack([out[k] for k in ("x_roi", "y_roi", "photons", "sigma_x_pix", "sigma_y_pix")])).all(0)
    bad |= (out["photons"] <= 0) | (out["sigma_x_pix"] <= 0) | (out["sigma_y_pix"] <= 0)
    for v in out.values():
        v[bad] = np.nan
    return out


def measure(frames: np.ndarray, cands: Candidates, settings: InSituSettings) -> tuple[Candidates, dict]:
    """The candidates whose region is inside the image, one per molecule, and their Gaussians in image
    pixels.

    A defocused spot, astigmatic above all, has more than one local maximum, and a secondary one in its
    wing sums the molecule's own photons in its disc, so it passes the significance test: on a simulated
    movie half the detections were such, and each then rejected its own molecule as an overlap.  The
    Gaussian started from either converges on the molecule's centre, so detections of a frame whose
    fitted centres are within ``merge_px`` are one molecule, and the most significant stays."""
    roi_px = settings.roi_px
    regions, inside = cut(frames, cands.frame, cands.x, cands.y, roi_px)
    kept = cands.take(np.flatnonzero(inside))
    g = gaussian(regions)
    h = roi_px // 2
    g["x_px"], g["y_px"] = g["x_roi"] - h + kept.x, g["y_roi"] - h + kept.y
    # A Gaussian wider than a third of its region has spread over the background, not a spot: on a simulated
    # movie a quarter of the detections, σ ~11 px against 2.7 for molecules, each rejecting every molecule
    # within 3σ as its neighbour.  A molecule that defocused would outgrow the region the fit sees anyway.
    wide = np.fmax(g["sigma_x_pix"], g["sigma_y_pix"]) > roi_px / 3
    ok = np.isfinite(g["photons"]) & ~wide
    kept, g = kept.take(np.flatnonzero(ok)), {k: v[ok] for k, v in g.items()}
    one = _one_per_molecule(kept.frame, g["x_px"], g["y_px"], kept.significance, settings.merge_px)
    return kept.take(one), {k: v[one] for k, v in g.items()}


def _one_per_molecule(frame, x, y, significance, merge_px) -> np.ndarray:
    """Indices kept: per frame, most significant first, none within ``merge_px`` of one kept."""
    return _greedy(frame, x, y, significance, lambda i, rest: merge_px)


# ------------------------------------------------------------------ 4, 5: blinks and overlaps

def link(frame, x_px, y_px, precision_px, settings: InSituSettings) -> np.ndarray:
    """[n] blink ids, 0-based: consecutive frames within SMAPpy's precision test (no dark frame)."""
    from ..group import connect
    # The floor: a Gaussian's CRLB (~0.09 px for a bright spot) understates how much a Gaussian fitted to a
    # PSF that is not one scatters (~0.2 px), and the precision test alone split blinks.
    ids = connect(x_px, y_px, frame, dx=settings.link_dx_px, dt=0, precision=precision_px, k=settings.link_k,
                  dx_min=settings.link_dx_min_px)
    return np.unique(ids, return_inverse=True)[1]


def _others(frames_of, det_frame, det_blink, blink_id, det_photons=None, photons=None,
            fraction: float = 0.0) -> np.ndarray:
    """The detections that count as neighbours: on in ``frames_of``, not of this blink, and -- given the
    photons -- at least ``fraction`` of this molecule's own."""
    others = np.isin(det_frame, frames_of) & (det_blink != blink_id)
    if det_photons is not None and photons is not None and fraction > 0:
        others &= np.asarray(det_photons) >= fraction * float(photons)
    return others


def overlaps(centre_xy, frames_of, sigma_px, det_frame, det_xy, det_sigma, det_blink, blink_id,
             settings: InSituSettings, det_photons=None, photons=None) -> bool:
    """Whether another molecule on in ``frames_of`` is closer than r_i + r_j (r = ``overlap_k``·σ of each).

    ``centre_xy`` the position, ``sigma_px`` the width of the one judged; the detections of the movie as
    (frame, xy, width, blink).  Given the photons (``det_photons`` of every detection, ``photons`` this
    molecule's), a neighbour dimmer than ``neighbour_fraction`` of it does not count: on the dense Zenodo
    tubulin movie most rejections were by false detections on the background and by clusters of faint
    ones on a diffuse patch, whose light at the molecule is a few per cent of its own."""
    others = _others(frames_of, det_frame, det_blink, blink_id, det_photons, photons, settings.neighbour_fraction)
    d = np.hypot(det_xy[others, 0] - centre_xy[0], det_xy[others, 1] - centre_xy[1])
    return bool(np.any(d < settings.overlap_k * (sigma_px + det_sigma[others])))


def neighbour_mask(centre_xy, frames_of, det_frame, det_xy, det_sigma, det_blink, blink_id,
                   settings: InSituSettings, det_photons=None, photons=None) -> np.ndarray:
    """[roi, roi] True = used: the region about ``centre_xy`` less every pixel within r_j of another
    molecule on in ``frames_of`` -- one bright enough to count (`overlaps`)."""
    h = settings.roi_px // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    mask = np.ones(yy.shape, bool)
    others = _others(frames_of, det_frame, det_blink, blink_id, det_photons, photons, settings.neighbour_fraction)
    for (nx, ny), s in zip(det_xy[others], det_sigma[others], strict=True):
        mask &= np.hypot(xx - (nx - centre_xy[0]), yy - (ny - centre_xy[1])) >= settings.overlap_k * s
    return mask


@dataclass
class Units:
    """What is fitted, one row each: a frame (the default) or a blink's frames summed (``unit="blink"``),
    its region, its Gaussian and its verdict."""
    frames: list                 # [U] the frames it holds: one, or the blink's
    blink: np.ndarray            # [U] the molecule's blink: a blink's frames are used or held out together
    x_px: np.ndarray             # [U] its position (a blink's: the mean over its frames), image pixels
    y_px: np.ndarray
    centre: np.ndarray           # [U, 2] the region's centre pixel (x, y)
    regions: np.ndarray          # [U, roi, roi] photons
    masks: np.ndarray            # [U, roi, roi] True = used
    gauss: dict                  # the region's Gaussian (`gaussian`)
    reason: list                 # [U] None, or why it is out
    source: np.ndarray = None    # [U] which movie (`InSituROIs.sources`); frames are that movie's


def units(frames: np.ndarray, cands: Candidates, g: dict, settings: InSituSettings) -> Units:
    """Link the detections into blinks, judge each blink's overlaps and photon steps, and cut what is
    fitted: every frame about its own position (``unit="frame"``), or a blink's frames summed about their
    mean position (``"blink"``), each fitted with the Gaussian (module docstring, 4 and 5)."""
    precision = np.hypot(g["x_err_pix"], g["y_err_pix"]) / np.sqrt(2)
    ids = link(cands.frame, g["x_px"], g["y_px"], precision, settings)
    sigma = np.maximum(g["sigma_x_pix"], g["sigma_y_pix"])
    det_xy = np.stack([g["x_px"], g["y_px"]], 1)
    roi, h = settings.roi_px, settings.roi_px // 2
    _, Y, X = frames.shape
    rows_out = []                                  # (frames, blink, x, y, centre, region, mask, reason)
    order = np.argsort(ids, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(ids[order]) != 0])
    for a, b in zip(starts, np.r_[starts[1:], len(order)], strict=True):
        rows = order[a:b]
        rows = rows[np.argsort(cands.frame[rows])]
        fs = cands.frame[rows]
        inner = g["photons"][rows][1:-1]
        step = len(inner) >= 2 and inner.max() > settings.step_ratio * inner.min()
        if settings.unit == "blink":
            parts = [(rows, float(g["x_px"][rows].mean()), float(g["y_px"][rows].mean()))]
        elif settings.unit == "frame":
            parts = [(rows[k:k + 1], float(g["x_px"][r]), float(g["y_px"][r])) for k, r in enumerate(rows)]
        else:
            raise ValueError(f"unit {settings.unit!r}: frame or blink")
        verdicts = []
        for part, x, y in parts:
            c = (int(np.rint(x)), int(np.rint(y)))
            if not (h <= c[0] < X - h and h <= c[1] < Y - h):
                verdicts.append(None)
                continue
            # judged on the whole blink: a molecule that overlaps it in any frame rejects all its frames
            reason = "overlap" if overlaps((x, y), fs, float(sigma[rows].max()), cands.frame, det_xy, sigma, ids,
                                           ids[rows[0]], settings, g["photons"], g["photons"][rows].max()) else None
            mask = neighbour_mask(c, cands.frame[part], cands.frame, det_xy, sigma, ids, ids[rows[0]], settings,
                                  g["photons"], g["photons"][part].max())
            if reason is None and 1 - mask.mean() > settings.max_masked:
                reason = "masked"
            if reason is None and step:
                reason = "photon step"
            region = frames[cands.frame[part], c[1] - h:c[1] + h + 1, c[0] - h:c[0] + h + 1].sum(0)
            verdicts.append((cands.frame[part], ids[rows[0]], x, y, c, region, mask, reason))
        rows_out += [v for v in verdicts if v is not None]
    if rows_out:
        fs, bl, xs, ys, cs, regions, masks, reasons = (list(t) for t in zip(*rows_out, strict=True))
    else:
        fs, bl, xs, ys, cs, regions, masks, reasons = [], [], [], [], [], [], [], []
    regions = np.asarray(regions, np.float32).reshape(-1, roi, roi)
    fitted = gaussian(regions)
    reasons = [r if r is not None or np.isfinite(fitted["photons"][i]) else "no fit" for i, r in enumerate(reasons)]
    # The unit's own fit, with its neighbours masked, is held to `measure`'s rule too: one that ran wide
    # sets the z proxy's range on its own.  On a simulated movie of 168 candidates one unit fitted σx 5 px
    # on macOS and 16 on Windows, which moved the 99th percentile of σx² − σy² from 35 to 116 px² and
    # halved the molecules used.
    wide = np.fmax(fitted["sigma_x_pix"], fitted["sigma_y_pix"]) > roi / 3
    reasons = [r if r is not None or not wide[i] else "wide" for i, r in enumerate(reasons)]
    return Units(fs, np.asarray(bl, int), np.asarray(xs, float), np.asarray(ys, float),
                 np.asarray(cs, int).reshape(-1, 2), regions, np.asarray(masks, bool).reshape(-1, roi, roi), fitted,
                 reasons)


# ------------------------------------------------------------------ 6: selection

def z_proxy(gauss: dict, proxy: str) -> np.ndarray:
    """σx² − σy² (astigmatism: linear in z) or σx² + σy² (size: |z|), px²."""
    sx2, sy2 = gauss["sigma_x_pix"] ** 2, gauss["sigma_y_pix"] ** 2
    if proxy == "astigmatism":
        return sx2 - sy2
    if proxy == "size":
        return sx2 + sy2
    raise ValueError(f"z proxy {proxy!r}: astigmatism or size")


def _robust_z(v: np.ndarray, one_sided: bool = False) -> np.ndarray:
    """|v − median| in robust standard deviations (1.4826·MAD); ``one_sided``: only above the median."""
    mad = 1.4826 * np.median(np.abs(v - np.median(v)))
    d = v - np.median(v)
    return (np.maximum(d, 0) if one_sided else np.abs(d)) / max(mad, 1e-9)


def select(u: Units, settings: InSituSettings, proxy=None, photons=None, shape=None, rejected=None,
           one_sided: bool = False):
    """(state of each unit, its bin [U] (−1: none), the bin edges): the module docstring's 6.

    By default from the units' Gaussians: the z proxy, the photons, and as the shape the widths σx², σy².
    The rounds (`uipsf.workflows.insitu_rounds`) pass the model's instead: its heights, photons and
    deviances, judged ``one_sided``: only a fit worse than the bin's is out.  ``rejected`` [U] (a reason or
    None) adds to the units' own verdicts.  Held out are whole
    molecules: a blink's frames are all held out or none, so that no held-out frame shows a molecule the
    fit has seen."""
    rng = np.random.default_rng(settings.seed)
    n = len(u.reason)
    reasons = list(u.reason) if rejected is None else [a or b for a, b in zip(u.reason, rejected, strict=True)]
    state = ["rejected: " + r if r else "candidate" for r in reasons]
    proxy = z_proxy(u.gauss, settings.proxy) if proxy is None else np.asarray(proxy, float)
    N = u.gauss["photons"] if photons is None else np.asarray(photons, float)
    shape = (np.stack([u.gauss["sigma_x_pix"] ** 2, u.gauss["sigma_y_pix"] ** 2], 1) if shape is None
             else np.asarray(shape, float).reshape(n, -1))
    ok = np.array([st == "candidate" for st in state]) & np.isfinite(proxy) & np.isfinite(N)
    if not ok.any():
        return state, np.full(n, -1), np.array([])
    lo, hi = np.quantile(proxy[ok], [0.01, 0.99])
    edges = np.linspace(lo, hi, settings.bins + 1)
    bins = np.where(ok, np.clip(np.digitize(proxy, edges) - 1, 0, settings.bins - 1), -1)
    bins[ok & ((proxy < lo) | (proxy > hi))] = -1
    for i in np.flatnonzero(ok & (bins < 0)):
        state[i] = "rejected: z range"
    blinks = np.unique(u.blink)
    held = dict(zip(blinks.tolist(), (rng.random(len(blinks)) < settings.heldout).tolist(), strict=True))
    pools = {}
    for k in range(settings.bins):
        idx = np.flatnonzero((bins == k) & np.array([st == "candidate" for st in state]))
        if len(idx) == 0:
            continue
        if len(idx) >= 5:
            odd = np.any([_robust_z(shape[idx, m], one_sided) > settings.shape_k for m in range(shape.shape[1])], 0)
            for i in idx[odd]:
                state[i] = "rejected: shape"
            idx = idx[~odd]
        if len(idx) == 0:
            continue
        top = np.quantile(N[idx], settings.window_top)
        inside = (N[idx] <= top) & (N[idx] >= top / settings.window_factor)
        for i in idx[~inside]:
            state[i] = "rejected: brightness"
        pools[k] = idx[inside]
        for i in pools[k]:
            state[i] = "not drawn"
    take = quotas([len(pools.get(k, ())) for k in range(settings.bins)], settings.molecules)
    for k, pool in pools.items():
        for i in rng.permutation(pool)[:take[k]]:
            state[i] = "held out" if held[int(u.blink[i])] else "used"
    return state, bins, edges


def quotas(pools, total: int) -> list:
    """How many to draw from each bin: ``total`` shared equally, a bin with fewer than its share giving all
    it has and the rest shared among the others (water-filling).  Molecules below the coverslip do not
    exist and the outermost heights are rare in most data: their bins stay emptier and the inner ones make
    up the number, so that what is learnt from is *molecules* wherever the data allow."""
    pools = [int(p) for p in pools]
    take = [0] * len(pools)
    open_, left = [k for k, p in enumerate(pools) if p > 0], int(total)
    while open_ and left > 0:
        share = left / len(open_)
        short = [k for k in open_ if pools[k] <= share]
        if short:                                       # all they have; the rest shared again
            for k in short:
                take[k] = pools[k]
                left -= pools[k]
            open_ = [k for k in open_ if k not in short]
            continue
        base, extra = divmod(left, len(open_))          # every open bin has more than its share
        for j, k in enumerate(open_):
            take[k] = base + (j < extra)
        left = 0
    return take


# ------------------------------------------------------------------ the whole step

@dataclass
class InSituROIs:
    units: Units
    state: list                  # [U] used / held out / not drawn / rejected: <reason>
    bins: np.ndarray             # [U]
    edges: np.ndarray
    candidates: int              # detections in the movies
    pixel_size_nm: tuple
    read_noise_var: float = 0.0
    excess_noise: float = 1.0
    messages: list = field(default_factory=list)
    sources: list = field(default_factory=list)      # per movie: its path and the frames read
    camera: object = None

    def ids(self, state: str) -> np.ndarray:
        return np.flatnonzero(np.array([s == state for s in self.state]))

    def counts(self) -> dict:
        out: dict = {}
        for s in self.state:
            out[s] = out.get(s, 0) + 1
        return out

    def batch(self, state: str = "used", ids=None):
        """(`uipsf.Batch`, its noise) of the units in ``state`` (or ``ids``): their regions, [E, 1, 1, Y, X]
        photons, their masks, the read noise of the K frames each holds, chip positions."""
        import torch

        from uipsf import Batch, ShiftedPoisson
        ids = self.ids(state) if ids is None else np.asarray(ids, int)
        u = self.units
        K = np.array([len(u.frames[i]) for i in ids], float)
        data = torch.tensor(u.regions[ids][:, None, None], dtype=torch.float32)
        mask = torch.tensor(u.masks[ids][:, None, None])
        s = torch.tensor(K * self.read_noise_var, dtype=torch.float32).reshape(-1, 1, 1, 1, 1)
        batch = Batch(data=data, z_nm=torch.zeros(1), s=s, mask=mask,
                      field_xy=torch.tensor(u.centre[ids].astype(float), dtype=torch.float32))
        return batch, ShiftedPoisson(float(self.excess_noise))


def insitu_units(frames: np.ndarray, settings: InSituSettings, read_noise_var: float = 0.0,
                 background=None) -> tuple[Units, int]:
    """One block of consecutive frames [T, Y, X] in photons to its units (steps 1–5) and how many
    detections it had: the background runs within the block."""
    frames = np.asarray(frames, np.float32)
    bg = (temporal_background(frames, settings.background_frames) if background is None
          else np.asarray(background, np.float32))
    cands = detect(frames, bg, settings, read_noise_var)
    kept, g = measure(frames, cands, settings)
    u = units(frames, kept, g, settings)
    if settings.region is not None:
        # outside a region is a reason like any other: detection and the overlap rule still see the
        # whole frame, so a molecule inside is judged against its neighbours outside
        from .core import in_region
        inside = in_region(u.x_px, u.y_px, settings.region)
        u.reason = [r if r is not None or ok else "outside the region" for r, ok in zip(u.reason, inside)]
    return u, len(cands)


def concatenate(parts) -> Units:
    """Units of several blocks and movies as one, from (units, first frame, movie): frames made the movie's,
    blinks numbered apart."""
    fs, bl, xs, ys, cs, regions, masks, gauss, reasons, source = [], [], [], [], [], [], [], [], [], []
    offset = 0
    for u, first, movie in parts:
        fs += [np.asarray(f) + first for f in u.frames]
        bl.append(u.blink + offset)
        offset += int(u.blink.max()) + 1 if len(u.blink) else 0
        xs.append(u.x_px)
        ys.append(u.y_px)
        cs.append(u.centre)
        regions.append(u.regions)
        masks.append(u.masks)
        gauss.append(u.gauss)
        reasons += list(u.reason)
        source.append(np.full(len(u.reason), movie, int) if u.source is None else u.source)
    keys = gauss[0].keys() if gauss else ()
    cat = np.concatenate
    return Units(fs, cat(bl) if bl else np.zeros(0, int), cat(xs) if xs else np.zeros(0),
                 cat(ys) if ys else np.zeros(0), cat(cs) if cs else np.zeros((0, 2), int),
                 cat(regions) if regions else np.zeros((0, 1, 1), np.float32),
                 cat(masks) if masks else np.zeros((0, 1, 1), bool), {k: cat([g[k] for g in gauss]) for k in keys},
                 reasons, cat(source) if source else np.zeros(0, int))


def insitu_rois(frames: np.ndarray, pixel_size_nm, settings: InSituSettings | None = None,
                read_noise_var: float = 0.0, excess_noise: float = 1.0, background=None) -> InSituROIs:
    """A movie in photons [T, Y, X] to chosen units (module docstring): one block, one movie."""
    s = settings or InSituSettings()
    u, n = insitu_units(frames, s, read_noise_var, background)
    return choose(concatenate([(u, 0, 0)]), n, pixel_size_nm, s, read_noise_var, excess_noise)


def choose(u: Units, candidates: int, pixel_size_nm, settings: InSituSettings, read_noise_var: float = 0.0,
           excess_noise: float = 1.0, sources=(), camera=None) -> InSituROIs:
    """The units of every block chosen from at once (step 6)."""
    state, bins, edges = select(u, settings)
    rois = InSituROIs(u, state, bins, edges, candidates, tuple(pixel_size_nm), read_noise_var, excess_noise,
                      sources=list(sources), camera=camera)
    rois.messages.append(f"{candidates} detections, {len(np.unique(u.blink))} blinks, {len(u.reason)} "
                         f"{settings.unit}s: " + ", ".join(f"{v} {k}" for k, v in sorted(rois.counts().items())))
    return rois
