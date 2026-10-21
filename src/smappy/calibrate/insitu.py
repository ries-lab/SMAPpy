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
4. **Blinks** (`link`): SMAPpy's grouping (`group.connect`) with its precision test and no dark frame:
   one molecule's consecutive frames are one emitter.  Its frames are **summed** into one region about
   their mean position: the molecule does not move, so the sum is Poisson with mean (ΣN_k)·I + K·b and
   carries what the frames carry about the PSF, the background's level aside (decided October 2026, plan
   §9 M5, step 4).  The sum is fitted with the Gaussian again.
5. **Overlaps, by the spot's size** (`overlaps`): with r = ``overlap_k``·σ of each spot (3σ), a neighbour
   in any frame of the blink closer than r_i + r_j rejects it; a farther one whose disc reaches into the
   region masks those pixels, and a blink with more than ``max_masked`` of its region masked is rejected.
   A blink whose inner frames' photons step by more than ``step_ratio`` (another molecule switching on or
   off under it) is rejected too; its first and last frames are partial and not judged.
6. **Selection** (`select`): a **z proxy** from the summed spot's widths -- σx² − σy² for an astigmatic
   PSF, linear in z, or σx² + σy² for a 2D one, which says |z| -- split into ``bins`` bins.  Per bin:
   spots whose widths are far from the bin's typical ones (``shape_k`` robust sds) are out -- an
   unresolved pair is wider or elongated for its z; then a random draw of at most ``per_bin`` from a
   **brightness window** ``window_factor`` (3) wide below the bin's ``window_top`` (90th) percentile of
   photons, not the brightest: those are where unresolved pairs collect, and with a Gaussian
   illumination they are the field's centre.  ``heldout`` (10 %) of each bin's draw is held out.

Every blink is a record with its state (*used*, *held out*, *rejected*, *not drawn*) and the reason,
as the beads are (`uipsf.workflows.rejection`).  `InSituROIs.batch` is the `uipsf.Batch` of the used
ones.  torch is imported only there: this module is imported without torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class InSituSettings:
    background_frames: int = 21
    scales_px: tuple = (1.2, 2.0, 3.0)
    significance: float = 6.0          # standard deviations of the background's noise in the disc
    roi_px: int = 15
    merge_px: float = 2.0              # detections whose Gaussians meet within this are one molecule
    overlap_k: float = 3.0
    max_masked: float = 0.25
    link_k: float = 2.5                # SMAPpy's precision test, in standard deviations
    link_dx_px: float = 1.5            # the largest step a blink may make between frames
    link_dx_min_px: float = 0.5        # always linked within this, whatever the precisions say
    step_ratio: float = 2.0
    proxy: str = "astigmatism"         # or "size" for a 2D PSF
    bins: int = 21
    per_bin: int = 100
    window_factor: float = 3.0
    window_top: float = 0.9
    shape_k: float = 3.0
    heldout: float = 0.1
    seed: int = 0


# ------------------------------------------------------------------ 1, 2: background and detection

def temporal_background(frames: np.ndarray, window: int) -> np.ndarray:
    """[T, Y, X]: each pixel's median over ``window`` frames about each frame (nearest at the ends)."""
    from scipy.ndimage import median_filter
    return median_filter(np.asarray(frames, np.float32), size=(int(window), 1, 1), mode="nearest")


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


def detect(frames: np.ndarray, background: np.ndarray, settings: InSituSettings,
           read_noise_var: float = 0.0) -> Candidates:
    """Every spot significant above the background at one of the widths (module docstring, 2)."""
    from scipy.ndimage import convolve

    from ..detect import DoGFilter, local_maxima
    signal = np.asarray(frames, np.float32) - background
    found = []
    for s in settings.scales_px:
        f, y, x, _ = local_maxima(DoGFilter(float(s))(signal), threshold=0.0)
        disc = _disc(max(1.5, 2.0 * s))[None]
        S = convolve(signal, disc, mode="nearest")[f, y, x]
        B = convolve(background, disc, mode="nearest")[f, y, x] + disc.sum() * read_noise_var
        z = S / np.sqrt(np.maximum(B, 1e-6))
        keep = z >= settings.significance
        found.append((f[keep], x[keep], y[keep], np.full(keep.sum(), float(s)), z[keep]))
    f, x, y, s, z = (np.concatenate(a) for a in zip(*found, strict=True))
    return _strongest(Candidates(f, x, y, s, z))


def _strongest(c: Candidates) -> Candidates:
    """Of candidates of several widths at one place the most significant: within 1.5·max(σ) + 1 pixels."""
    keep = np.zeros(len(c), bool)
    order = np.lexsort((-c.significance, c.frame))
    starts = np.flatnonzero(np.r_[True, np.diff(c.frame[order]) != 0])
    for a, b in zip(starts, np.r_[starts[1:], len(order)], strict=True):
        idx = order[a:b]                                   # this frame's, most significant first
        taken = []
        for i in idx:
            if all(np.hypot(c.x[i] - c.x[j], c.y[i] - c.y[j]) >= 1.5 * max(c.scale[i], c.scale[j]) + 1 for j in taken):
                taken.append(i)
        keep[taken] = True
    return c.take(np.flatnonzero(keep))


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
    keep = []
    order = np.lexsort((-significance, frame))
    starts = np.flatnonzero(np.r_[True, np.diff(frame[order]) != 0])
    for a, b in zip(starts, np.r_[starts[1:], len(order)], strict=True):
        taken = []
        for i in order[a:b]:
            if all(np.hypot(x[i] - x[j], y[i] - y[j]) >= merge_px for j in taken):
                taken.append(i)
        keep += taken
    return np.sort(np.asarray(keep, int))


# ------------------------------------------------------------------ 4, 5: blinks and overlaps

def link(frame, x_px, y_px, precision_px, settings: InSituSettings) -> np.ndarray:
    """[n] blink ids, 0-based: consecutive frames within SMAPpy's precision test (no dark frame)."""
    from ..group import connect
    # The floor: a Gaussian's CRLB (~0.09 px for a bright spot) understates how much a Gaussian fitted to a
    # PSF that is not one scatters (~0.2 px), and the precision test alone split blinks.
    ids = connect(x_px, y_px, frame, dx=settings.link_dx_px, dt=0, precision=precision_px, k=settings.link_k,
                  dx_min=settings.link_dx_min_px)
    return np.unique(ids, return_inverse=True)[1]


def overlaps(centre_xy, frames_of, sigma_px, det_frame, det_xy, det_sigma, det_blink, blink_id,
             settings: InSituSettings) -> tuple[str | None, np.ndarray]:
    """One blink's verdict on its neighbours: (reason or None, its region's mask [roi, roi], True = used).

    ``centre_xy`` the region's centre pixel, ``frames_of`` its frames, ``sigma_px`` its width; the
    detections of the movie as (frame, xy, width, blink)."""
    roi = settings.roi_px
    h = roi // 2
    mask = np.ones((roi, roi), bool)
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    r_i = settings.overlap_k * sigma_px
    others = np.isin(det_frame, frames_of) & (det_blink != blink_id)
    for (nx, ny), s in zip(det_xy[others], det_sigma[others], strict=True):
        r_j = settings.overlap_k * s
        dx, dy = nx - centre_xy[0], ny - centre_xy[1]
        if np.hypot(dx, dy) < r_i + r_j:
            return "overlap", mask
        mask &= np.hypot(xx - dx, yy - dy) >= r_j
    if 1 - mask.mean() > settings.max_masked:
        return "masked", mask
    return None, mask


@dataclass
class Blinks:
    """One row per blink: the summed region, its Gaussian and its verdict."""
    frames: list                 # [B] its frames
    x_px: np.ndarray             # [B] mean position over its frames, image pixels
    y_px: np.ndarray
    centre: np.ndarray           # [B, 2] the region's centre pixel (x, y)
    regions: np.ndarray          # [B, roi, roi] photons, summed over its frames
    masks: np.ndarray            # [B, roi, roi] True = used
    gauss: dict                  # the summed region's Gaussian (`gaussian`)
    reason: list                 # [B] None, or why it is out


def blinks(frames: np.ndarray, cands: Candidates, g: dict, settings: InSituSettings) -> Blinks:
    """Link the detections into blinks, sum each blink's frames about its mean position, judge overlaps
    and photon steps, and fit the sums (module docstring, 4 and 5)."""
    precision = np.hypot(g["x_err_pix"], g["y_err_pix"]) / np.sqrt(2)
    ids = link(cands.frame, g["x_px"], g["y_px"], precision, settings)
    sigma = np.maximum(g["sigma_x_pix"], g["sigma_y_pix"])
    det_xy = np.stack([g["x_px"], g["y_px"]], 1)
    roi, h = settings.roi_px, settings.roi_px // 2
    _, Y, X = frames.shape
    out_f, out_x, out_y, centres, regions, masks, reasons = [], [], [], [], [], [], []
    order = np.argsort(ids, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(ids[order]) != 0])
    for a, b in zip(starts, np.r_[starts[1:], len(order)], strict=True):
        rows = order[a:b]
        rows = rows[np.argsort(cands.frame[rows])]
        fs = cands.frame[rows]
        x, y = float(g["x_px"][rows].mean()), float(g["y_px"][rows].mean())
        c = (int(np.rint(x)), int(np.rint(y)))
        if not (h <= c[0] < X - h and h <= c[1] < Y - h):
            continue
        reason, mask = overlaps(c, fs, float(sigma[rows].max()), cands.frame, det_xy, sigma, ids, ids[rows[0]],
                                settings)
        inner = g["photons"][rows][1:-1]
        if reason is None and len(inner) >= 2 and inner.max() > settings.step_ratio * inner.min():
            reason = "photon step"
        region = frames[fs, c[1] - h:c[1] + h + 1, c[0] - h:c[0] + h + 1].sum(0)
        out_f.append(fs)
        out_x.append(x)
        out_y.append(y)
        centres.append(c)
        regions.append(region)
        masks.append(mask)
        reasons.append(reason)
    regions = np.asarray(regions, np.float32).reshape(-1, roi, roi)
    summed = gaussian(regions)
    reasons = [r if r is not None or np.isfinite(summed["photons"][i]) else "no fit" for i, r in enumerate(reasons)]
    return Blinks(out_f, np.asarray(out_x), np.asarray(out_y), np.asarray(centres, int).reshape(-1, 2), regions,
                  np.asarray(masks, bool).reshape(-1, roi, roi), summed, reasons)


# ------------------------------------------------------------------ 6: selection

def z_proxy(gauss: dict, proxy: str) -> np.ndarray:
    """σx² − σy² (astigmatism: linear in z) or σx² + σy² (size: |z|), px²."""
    sx2, sy2 = gauss["sigma_x_pix"] ** 2, gauss["sigma_y_pix"] ** 2
    if proxy == "astigmatism":
        return sx2 - sy2
    if proxy == "size":
        return sx2 + sy2
    raise ValueError(f"z proxy {proxy!r}: astigmatism or size")


def _robust_z(v: np.ndarray) -> np.ndarray:
    mad = 1.4826 * np.median(np.abs(v - np.median(v)))
    return np.abs(v - np.median(v)) / max(mad, 1e-9)


def select(b: Blinks, settings: InSituSettings) -> tuple[list, np.ndarray, np.ndarray]:
    """(state of each blink, its bin [B] (−1: none), the bin edges): the module docstring's 6."""
    rng = np.random.default_rng(settings.seed)
    n = len(b.reason)
    state = ["rejected: " + r if r else "candidate" for r in b.reason]
    proxy = z_proxy(b.gauss, settings.proxy)
    ok = np.array([s == "candidate" for s in state])
    if not ok.any():
        return state, np.full(n, -1), np.array([])
    lo, hi = np.quantile(proxy[ok], [0.01, 0.99])
    edges = np.linspace(lo, hi, settings.bins + 1)
    bins = np.where(ok, np.clip(np.digitize(proxy, edges) - 1, 0, settings.bins - 1), -1)
    bins[ok & ((proxy < lo) | (proxy > hi))] = -1
    for i in np.flatnonzero(ok & (bins < 0)):
        state[i] = "rejected: z range"
    sx, sy, N = b.gauss["sigma_x_pix"], b.gauss["sigma_y_pix"], b.gauss["photons"]
    for k in range(settings.bins):
        idx = np.flatnonzero((bins == k) & np.array([s == "candidate" for s in state]))
        if len(idx) == 0:
            continue
        if len(idx) >= 5:
            odd = (_robust_z(sx[idx] ** 2) > settings.shape_k) | (_robust_z(sy[idx] ** 2) > settings.shape_k)
            for i in idx[odd]:
                state[i] = "rejected: shape"
            idx = idx[~odd]
        if len(idx) == 0:
            continue
        top = np.quantile(N[idx], settings.window_top)
        inside = (N[idx] <= top) & (N[idx] >= top / settings.window_factor)
        for i in idx[~inside]:
            state[i] = "rejected: brightness"
        pool = idx[inside]
        drawn = rng.permutation(pool)[:settings.per_bin]
        for i in pool:
            state[i] = "not drawn"
        n_held = round(settings.heldout * len(drawn))
        for j, i in enumerate(drawn):
            state[i] = "held out" if j < n_held else "used"
    return state, bins, edges


# ------------------------------------------------------------------ the whole step

@dataclass
class InSituROIs:
    blinks: Blinks
    state: list                  # [B] used / held out / not drawn / rejected: <reason>
    bins: np.ndarray             # [B]
    edges: np.ndarray
    candidates: int              # detections in the movie
    pixel_size_nm: tuple
    read_noise_var: float = 0.0
    excess_noise: float = 1.0
    messages: list = field(default_factory=list)

    def ids(self, state: str) -> np.ndarray:
        return np.flatnonzero(np.array([s == state for s in self.state]))

    def counts(self) -> dict:
        out: dict = {}
        for s in self.state:
            out[s] = out.get(s, 0) + 1
        return out

    def batch(self, state: str = "used"):
        """(`uipsf.Batch`, its noise) of the blinks in ``state``: the summed regions, [E, 1, 1, Y, X] photons,
        their masks, the read noise of K summed frames, chip positions."""
        import torch

        from uipsf import Batch, ShiftedPoisson
        ids = self.ids(state)
        b = self.blinks
        K = np.array([len(b.frames[i]) for i in ids], float)
        data = torch.tensor(b.regions[ids][:, None, None], dtype=torch.float32)
        mask = torch.tensor(b.masks[ids][:, None, None])
        s = torch.tensor(K * self.read_noise_var, dtype=torch.float32).reshape(-1, 1, 1, 1, 1)
        batch = Batch(data=data, z_nm=torch.zeros(1), s=s, mask=mask,
                      field_xy=torch.tensor(b.centre[ids].astype(float), dtype=torch.float32))
        return batch, ShiftedPoisson(float(self.excess_noise))


def insitu_rois(frames: np.ndarray, pixel_size_nm, settings: InSituSettings | None = None,
                read_noise_var: float = 0.0, excess_noise: float = 1.0, background=None) -> InSituROIs:
    """A movie in photons [T, Y, X] to chosen blinks (module docstring)."""
    s = settings or InSituSettings()
    frames = np.asarray(frames, np.float32)
    bg = temporal_background(frames, s.background_frames) if background is None else np.asarray(background, np.float32)
    cands = detect(frames, bg, s, read_noise_var)
    kept, g = measure(frames, cands, s)
    b = blinks(frames, kept, g, s)
    state, bins, edges = select(b, s)
    rois = InSituROIs(b, state, bins, edges, len(cands), tuple(pixel_size_nm), read_noise_var, excess_noise)
    rois.messages.append(f"{len(cands)} detections, {len(b.reason)} blinks: " +
                         ", ".join(f"{v} {k}" for k, v in sorted(rois.counts().items())))
    return rois
