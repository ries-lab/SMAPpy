"""Single molecules on the coverslip, imaged through repeated z sweeps, to what uipsf learns a PSF from.

The sample (`docs/uipsf/molecule-stacks.md`): dye molecules stuck to the coverslip in blinking buffer, a z
stack of the stage taken again and again (M5's AF647, 2 November 2026: 41 frames at 50 nm over ±1 µm, 10-15
sweeps per file, one frame per step).  A molecule blinks on for a frame or a few, so each detection is its
own emitter, as in situ's (`insitu`); but its z is known -- the stage's, less the sweep's focus and the
focal plane at its place (`uipsf.model.CoverslipStackGeometry`) -- so what is learnt is a bead model's: a
PSF at the coverslip against defocus.  What is reused from in situ is the container (`InSituROIs`, `Units`),
its table of counts by state and the first fit (`cut`, `gaussian`).  Its detection and selection are not:
they look for spots of unknown z, by their size and a z proxy, where here every frame's defocus is known.
The steps, each a function:

1. **Reading** (`read_file`): NDTiff and OME-TIFF through `smappy.io`, the stage's z per frame from the
   files' own metadata (`stage_z_um`: Micro-Manager's ``ZPositionUm``, or OME's ``PositionZ``), split into
   sweeps where z turns back or jumps (`split_sweeps`): sweeps up, down, or both alternately.
2. **The width curve** (`first_spots`, `width_curve`): a difference of Gaussians and SMAPpy's elliptical
   Gaussian on every frame, each file's focus where σx = σy (an astigmatic PSF) or where the spot is
   smallest (a round one), and the median widths of the brighter half of each z step against defocus,
   pooled over the files.  It says what a molecule should look like in each frame.
3. **Detection** (`detect`): each frame filtered with the spot its defocus should show -- a zero-mean
   elliptical Gaussian of the curve's widths, so a flat background cancels -- and its local maxima above
   ``significance`` robust standard deviations of the filtered frame.  A defocused spot, dim at its peak,
   is found at its own size.  Only inside the illuminated circle (`illumination_radius`): the
   illumination is a circle a little larger than the image's inscribed circle, and the corners are dark.
4. **Selection** (`select`), stringent because free or loosely bound dye images too: the Gaussian centred,
   its widths those of the curve within ``width_tol``, no neighbour within reach with a fifth of its light
   (weaker ones masked, at most ``max_masked`` of the region), its photons from half its defocus bin's
   median up to the bin's bulk (the brightest are overlaps), then at most ``molecules`` drawn evenly over
   100 nm bins of defocus (`insitu.quotas`), so the far defocus, with fewer detections, keeps its share.

torch is imported only in `MoleculeStackROIs.batch`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..plugins import param
from .insitu import InSituROIs, Units, cut, gaussian, quotas

#: *region* on auto, nm of the sample: the widest spots at ±1 µm have σ ≈ 9 px of 105 nm (M5)
REGION_NM = 2600.0
#: the width curve's grid (nm of defocus), its bins' half width at least a step of the stage, and the spots a
#: bin needs: with 100 nm steps a bin of ±50 nm held one frame of each sweep, too few, and the curve
#: interpolated across the gaps rejected every spot near focus for its shape (simulated sweeps)
CURVE_SPAN_NM, CURVE_STEP_NM, CURVE_MIN = 1100.0, 50.0, 5
#: defocus bins for the brightness window (nm) and for drawing evenly (nm)
BRIGHTNESS_BIN_NM, DRAW_BIN_NM = 200.0, 100.0
#: a spot's Gaussian further than this from its region's centre (px) went to a neighbour
MAX_OFF_CENTRE_PX = 1.5
#: the reach of a spot, and the ellipse masked about a weaker neighbour, in σ of the curve's widths
REACH_SIGMA = 2.5
#: a z step bigger than this many median steps starts a sweep (the stage moving back)
JUMP_STEPS = 2.5
PSF_SHAPES = (("auto", "auto"), ("astigmatic", "astigmatic"), ("round", "round (2D)"))


@dataclass
class MoleculeStackSettings:
    """How molecules are found and chosen in z sweeps.  One number to set, *molecules*; under *more* what
    depends on the data, and the molecules' height when learning it goes wrong; the rest hidden, the
    prototype's on M5's AF647 (November 2026)."""
    molecules: int = param(4000, label="molecules", min=50,
                           help="how many spots to learn from at most, drawn evenly over the defocus")
    height_nm: Optional[float] = param(None, label="height", unit="nm", min=0.0, max=200.0, advanced=True,
                                       help="the molecules' height above the glass; auto: learnt")
    psf_shape: str = param("auto", label="PSF", choices=PSF_SHAPES, advanced=True,
                           help="how each file's focus is found: where σx = σy (astigmatic) or the spot is "
                                "smallest (round); auto: from the widths")
    illumination_px: Optional[float] = param(None, label="illumination radius", unit="px", min=5.0,
                                             advanced=True,
                                             help="molecules only inside this circle about the image's "
                                                  "centre; auto: the circle inscribed in the image")
    significance: float = param(5.0, label="significance", min=1.0, advanced=True,
                                help="a detection's peak in the matched filter, in robust standard "
                                     "deviations of the filtered frame")
    roi_px: Optional[int] = param(None, label="region", unit="px", min=7, advanced=True,
                                  help=f"the square region each molecule is fitted in (odd); auto: "
                                       f"{REGION_NM / 1000:g} µm on the sample")
    z_step_nm: Optional[float] = param(None, label="z step", unit="nm", min=1.0, advanced=True,
                                       help="for files that do not record the stage's z: its step; auto: "
                                            "from the metadata")
    sweep_frames: Optional[int] = param(None, label="frames per sweep", min=2, advanced=True,
                                        help="with a typed z step: the frames of one sweep, all upwards")
    max_rounds: int = param(3, label="rounds", min=1, hidden=True,
                            help="learn, check z, drop what fails, learn again: at most this many times")
    width_tol: float = param(0.3, label="width tolerance", min=0.0, hidden=True,
                             help="|log(σ / σ of the curve)| per axis at which a spot is out")
    neighbour_fraction: float = param(0.2, label="neighbour light", min=0.0, max=1.0, hidden=True,
                                      help="a neighbour with this fraction of a spot's photons within reach "
                                           "rejects it; a dimmer one is masked")
    max_masked: float = param(0.25, label="masked at most", min=0.0, max=1.0, hidden=True,
                              help="the fraction of a region a neighbour's mask may cover")
    seed: int = param(0, label="seed", hidden=True, help="the random draws' seed")

    def region(self, pixel_size_nm) -> int:
        """*region* in pixels: as set, else `REGION_NM` of the sample; odd."""
        if self.roi_px is not None:
            return int(self.roi_px) | 1
        return int(round(REGION_NM / float(np.mean(pixel_size_nm)))) | 1


# ------------------------------------------------------------------ 1: reading

@dataclass
class SweepFile:
    """One file: its frames in photons, the stage's z per frame and each frame's sweep."""
    path: str
    frames: np.ndarray           # [T, Y, X] photons
    z_um: np.ndarray             # [T]
    sweep: np.ndarray            # [T] the sweep within the file


def stage_z_um(source, path) -> Optional[np.ndarray]:
    """The stage's z of every frame of ``source`` (µm) from the file's metadata, or None: Micro-Manager's
    ``ZPositionUm`` per frame (an NDTiff's or an MMStack's, collected by `frametags.FrameTags` as the
    frames are read), else an OME-TIFF's ``PositionZ`` per plane."""
    tags = getattr(source, "tags", None)
    table = tags.table() if tags is not None else {}
    for key in table:
        if key.endswith("ZPositionUm"):
            return np.asarray(table[key], float)
    if tags is not None and len(tags) and _first_tag(tags, "ZPositionUm") is not None:
        return np.full(len(tags), _first_tag(tags, "ZPositionUm"))     # z never changed: no sweep
    return _ome_z_um(path)


def _first_tag(tags, name):
    for key, value in getattr(tags, "_first", {}).items():
        if key.endswith(name):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


_UNITS_UM = {"µm": 1.0, "um": 1.0, "nm": 1e-3, "mm": 1e3, "m": 1e6}


def _ome_z_um(path) -> Optional[np.ndarray]:
    """An OME-TIFF's ``PositionZ`` per plane, in µm, in the order of its planes (``TheT``), or None.  A series
    of files (``x.ome.tif``, ``x_1.ome.tif``, ...) is read as `open_stack` reads it, file after file."""
    import xml.etree.ElementTree as ET
    from pathlib import Path

    import tifffile

    from ..io.tiff import _first_of_series, _series_files
    p = Path(path)
    files = sorted(p.glob("*.tif*")) if p.is_dir() else [p]
    if files:
        files = _series_files(_first_of_series(files))
    out = []
    for f in files:
        try:
            with tifffile.TiffFile(f) as tf:
                xml = tf.ome_metadata
        except Exception:                    # not a TIFF tifffile reads: no OME z
            continue
        if not xml:
            continue
        planes = [e for e in ET.fromstring(xml).iter() if e.tag.endswith("Plane") and "PositionZ" in e.attrib]
        if not planes:
            continue
        planes.sort(key=lambda e: int(e.attrib.get("TheT", 0)) * 10 ** 6 + int(e.attrib.get("TheZ", 0)))
        unit = _UNITS_UM.get(planes[0].attrib.get("PositionZUnit", "µm"), 1.0)
        out.append(np.array([float(e.attrib["PositionZ"]) for e in planes]) * unit)
    return np.concatenate(out) if out else None


def split_sweeps(z_um) -> np.ndarray:
    """[T] each frame's sweep: a new one where the stage turns back or jumps by more than `JUMP_STEPS`
    median steps, so sweeps upwards, downwards and both in turn are each a sweep of their own."""
    z = np.asarray(z_um, float)
    if len(z) < 2:
        return np.zeros(len(z), int)
    d = np.diff(z)
    step = float(np.median(np.abs(d[d != 0]))) if np.any(d != 0) else 0.0
    sweep = np.zeros(len(z), int)
    direction = 0.0
    for i, di in enumerate(d):
        new = step > 0 and (abs(di) > JUMP_STEPS * step or (direction and np.sign(di) not in (0, direction)))
        sweep[i + 1] = sweep[i] + int(new)
        direction = 0.0 if new else (direction or float(np.sign(di)))
    return sweep


def read_file(path, camera, settings: MoleculeStackSettings | None = None) -> SweepFile:
    """One file's frames in photons, its stage z and sweeps (module docstring, 1); a ValueError when the
    file records no z and none is typed."""
    from ..camera import to_photons
    from ..frametags import FrameTags
    from ..io.tiff import open_stack
    s = settings or MoleculeStackSettings()
    source = open_stack(path)
    source.tags = FrameTags()
    frames = np.concatenate([b for _, b in source.frames(chunk=200)])
    z = None if s.z_step_nm else stage_z_um(source, path)
    if z is None or len(z) != len(frames) or np.ptp(z) == 0:
        if not s.z_step_nm:
            raise ValueError(f"{path}: the file records no stage z for its frames; type the z step and the "
                             "frames per sweep (more)")
        n = int(s.sweep_frames or len(frames))
        z = (np.arange(len(frames)) % n) * float(s.z_step_nm) / 1000
    return SweepFile(str(path), to_photons(frames, camera), np.asarray(z, float), split_sweeps(z))


# ------------------------------------------------------------------ 2: the width curve

def first_spots(frames: np.ndarray, roi_px: int = 15) -> dict:
    """A difference of Gaussians and SMAPpy's elliptical Gaussian on ``roi_px`` regions of every frame: what
    the width curve is made of (module docstring, 2)."""
    from ..detect import AbsoluteCutoff, DoGFilter, find_candidates
    filt = DoGFilter(1.5)(frames)
    noise = 1.4826 * float(np.median(np.abs(filt - np.median(filt))))
    c = find_candidates(filt, AbsoluteCutoff(6 * noise))
    regions, ok = cut(frames, c.frame, c.x, c.y, roi_px)
    g = gaussian(regions)
    return dict(frame=np.asarray(c.frame)[ok], x=np.asarray(c.x)[ok], y=np.asarray(c.y)[ok], **g)


def _binned_widths(z_um, sx, sy, step_um, min_count: int = 5):
    bins = np.round((z_um - z_um.min()) / step_um).astype(int)
    zs, rx, ry = [], [], []
    for b in np.unique(bins):
        m = bins == b
        if m.sum() >= min_count:
            zs.append(float(np.median(z_um[m])))
            rx.append(float(np.median(sx[m])))
            ry.append(float(np.median(sy[m])))
    return np.array(zs), np.array(rx), np.array(ry)


def psf_shape_of(z_um, sx, sy, step_um) -> str:
    """"astigmatic" when the spots' elongation log(σx/σy) changes by more than 0.5 over the sweep, else
    "round": a cylindrical lens takes it from about −1 to 1 over ±1 µm, a round PSF keeps it near 0."""
    _, rx, ry = _binned_widths(z_um, sx, sy, step_um)
    if len(rx) < 3:
        return "round"
    e = np.log(rx / ry)
    return "astigmatic" if float(np.ptp(e)) > 0.5 else "round"


def focus_of(z_um, sx, sy, shape: str, step_um: float) -> float:
    """The stage z (µm) of focus from one file's spots: where σx = σy (astigmatic), or where σx² + σy² is
    smallest (round), each fitted near it on the spots' median widths per step."""
    zs, rx, ry = _binned_widths(z_um, sx, sy, step_um)
    if len(zs) < 3:
        raise ValueError("too few spots to find the focus")
    if shape == "astigmatic":
        # only between the most elongated steps either way: beyond them the spots outgrow the 15-pixel fit,
        # whose widths come back alike and cross again (simulated sweeps: a focus 2.5 µm off)
        e = np.log(rx / ry)
        lo, hi = sorted((int(np.argmin(e)), int(np.argmax(e))))
        between = np.zeros(len(e), bool)
        between[lo:hi + 1] = True
        near = between & (np.abs(e) < 0.8)
        if near.sum() < 2:
            raise ValueError("the spots are never round: no focus where σx = σy")
        a, b = np.polyfit(zs[near], e[near], 1)
        return float(-b / a)
    w = rx ** 2 + ry ** 2
    near = w < 3 * w.min()
    if near.sum() < 3:
        return float(zs[np.argmin(w)])
    c2, c1, _ = np.polyfit(zs[near], w[near], 2)
    return float(-c1 / (2 * c2)) if c2 > 0 else float(zs[np.argmin(w)])


@dataclass
class WidthCurve:
    focus_um: list               # each file's focus, stage µm
    d_nm: np.ndarray             # the grid of defocus, nm: stage z less the file's focus
    sx: np.ndarray               # median σx (px) at each
    sy: np.ndarray
    shape: str                   # "astigmatic" or "round"

    def at(self, d_nm) -> tuple[np.ndarray, np.ndarray]:
        return np.interp(d_nm, self.d_nm, self.sx), np.interp(d_nm, self.d_nm, self.sy)


def width_curve(spots: list, z_um: list, inside: list, shape: str = "auto") -> WidthCurve:
    """Each file's focus and the pooled median widths (px) against defocus, from the brighter half of each
    file's spots ``inside`` the illumination (module docstring, 2)."""
    step_um = _step_um(z_um)
    keep = []
    for s, z, ins in zip(spots, z_um, inside, strict=True):
        zf = z[s["frame"]]
        ok = np.isfinite(s["sigma_x_pix"]) & ins & _brighter_half(s["photons"], zf, step_um)
        keep.append((zf[ok], s["sigma_x_pix"][ok], s["sigma_y_pix"][ok]))
    if shape == "auto":
        shape = psf_shape_of(*(np.concatenate(v) for v in zip(*keep, strict=True)), step_um)
    focus, d_all, sx_all, sy_all = [], [], [], []
    for z, sx, sy in keep:
        f = focus_of(z, sx, sy, shape, step_um)
        focus.append(f)
        d_all.append((z - f) * 1000)
        sx_all.append(sx)
        sy_all.append(sy)
    d, sx, sy = (np.concatenate(v) for v in (d_all, sx_all, sy_all))
    grid = np.arange(-CURVE_SPAN_NM, CURVE_SPAN_NM + 1, CURVE_STEP_NM)
    half = max(CURVE_STEP_NM, 1.01 * step_um * 1000)
    mx, my = np.full(len(grid), np.nan), np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        m = np.abs(d - g) <= half
        if m.sum() >= CURVE_MIN:
            mx[i], my[i] = np.median(sx[m]), np.median(sy[m])
    good = np.isfinite(mx) & np.isfinite(my)
    if good.sum() < 3:
        raise ValueError("too few spots for the width curve")
    return WidthCurve(focus, grid[good], mx[good], my[good], shape)


def _brighter_half(photons, z_um, step_um) -> np.ndarray:
    """The spots brighter than the median of their own z step: a defocused spot's photons in a 15-pixel
    Gaussian are a fraction of a focused one's, and a median over the whole file left none beyond 100 nm on
    one side of focus (simulated sweeps)."""
    photons = np.asarray(photons, float)
    bins = np.round((np.asarray(z_um) - np.min(z_um)) / step_um).astype(int)
    out = np.zeros(len(photons), bool)
    for b in np.unique(bins):
        m = bins == b
        out[m] = photons[m] > np.nanmedian(photons[m])
    return out


def _step_um(z_um: list) -> float:
    d = np.abs(np.concatenate([np.diff(np.asarray(z, float)) for z in z_um]))
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 0.05


def illumination_radius(shape) -> float:
    """*illumination radius* on auto: the circle inscribed in the image, half its shorter side.  The
    illumination is a circle a little larger than that, with the corners dark (M5: the background falls from
    52 to 42 photons out to 105-110 px of a 216 x 208 image and rises again in the corners, where the
    detections were thrice as dense as inside and false); the prototype's 105 px."""
    return min(shape) / 2


def inside_circle(x, y, shape, radius) -> np.ndarray:
    Y, X = shape
    return np.hypot(np.asarray(x) - (X - 1) / 2, np.asarray(y) - (Y - 1) / 2) <= radius


# ------------------------------------------------------------------ 3: detection

def _template(sx: float, sy: float) -> np.ndarray:
    hx, hy = int(math.ceil(3 * sx)), int(math.ceil(3 * sy))
    yy, xx = np.mgrid[-hy:hy + 1, -hx:hx + 1]
    g = np.exp(-xx ** 2 / (2 * sx ** 2) - yy ** 2 / (2 * sy ** 2))
    return (g - g.mean()) / np.sqrt(((g - g.mean()) ** 2).sum())


def detect(frames: np.ndarray, d_nm: np.ndarray, curve: WidthCurve, radius: float, k: float = 5.0) -> dict:
    """Candidates (frame, x, y) of every frame, filtered with the spot of its defocus ``d_nm`` (module
    docstring, 3), inside ``radius`` of the image's centre."""
    from scipy.signal import fftconvolve

    from ..detect import local_maxima
    sx, sy = curve.at(d_nm)
    out = {"frame": [], "x": [], "y": []}
    for t, frame in enumerate(frames):
        f = fftconvolve(frame - np.median(frame), _template(sx[t], sy[t])[::-1, ::-1], mode="same")
        noise = 1.4826 * float(np.median(np.abs(f - np.median(f))))
        _, y, x, _ = local_maxima(f[None].astype(np.float32), threshold=k * noise)
        keep = inside_circle(x, y, frame.shape, radius)
        out["frame"].append(np.full(keep.sum(), t))
        out["x"].append(x[keep])
        out["y"].append(y[keep])
    return {n: np.concatenate(v).astype(int) for n, v in out.items()}


# ------------------------------------------------------------------ 4: selection

def measure(frames, cands: dict, d_frame: np.ndarray, curve: WidthCurve, roi_px: int) -> dict:
    """``roi_px`` regions about the candidates and SMAPpy's elliptical Gaussian on each, with the widths the
    curve expects at their defocus."""
    regions, ok = cut(frames, cands["frame"], cands["x"], cands["y"], roi_px)
    out = {n: np.asarray(v)[ok] for n, v in cands.items()}
    out["regions"] = regions
    out["gauss"] = gaussian(regions)
    out["d_nm"] = d_frame[out["frame"]]
    out["sx_curve"], out["sy_curve"] = curve.at(out["d_nm"])
    return out


def neighbours(s: dict, roi_px: int, share: float) -> tuple[np.ndarray, np.ndarray]:
    """(rejected [n], masks [n, R, R]) of the spots ``s`` (`measure`): a neighbour in the same frame within
    reach, `REACH_SIGMA` of both spots' curve widths added, with ``share`` of the spot's light rejects it;
    every neighbour in the region is masked, an ellipse of `REACH_SIGMA` of its widths about it."""
    n, h = len(s["frame"]), roi_px // 2
    rejected = np.zeros(n, bool)
    masks = np.ones((n, roi_px, roi_px), bool)
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    g = s["gauss"]
    photons = np.nan_to_num(g["photons"], nan=0.0)
    cx = s["x"] + np.nan_to_num(g["x_roi"] - h)
    cy = s["y"] + np.nan_to_num(g["y_roi"] - h)
    order = np.argsort(s["frame"], kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(s["frame"][order]) != 0])
    for a, b in zip(starts, np.r_[starts[1:], n], strict=True):
        idx = order[a:b]
        if len(idx) < 2:
            continue
        dx = cx[idx][None, :] - s["x"][idx][:, None]           # [i, j]: neighbour j from spot i's region
        dy = cy[idx][None, :] - s["y"][idx][:, None]
        rx = REACH_SIGMA * s["sx_curve"][idx]
        ry = REACH_SIGMA * s["sy_curve"][idx]
        other = ~np.eye(len(idx), dtype=bool)
        near = other & (np.abs(dx) <= h + rx[None, :]) & (np.abs(dy) <= h + ry[None, :])
        reach = np.hypot(dx / (rx[:, None] + rx[None, :]), dy / (ry[:, None] + ry[None, :]))
        bright = photons[idx][None, :] >= share * photons[idx][:, None]
        rejected[idx] |= (near & bright & (reach < 1)).any(1)
        for i, j in zip(*np.nonzero(near), strict=True):
            masks[idx[i]] &= ((xx - dx[i, j]) / rx[j]) ** 2 + ((yy - dy[i, j]) / ry[j]) ** 2 > 1
    return rejected, masks


def select(s: dict, settings: MoleculeStackSettings, roi_px: int) -> tuple[list, np.ndarray]:
    """(each spot's state, its mask [n, R, R]) by the rules of the module docstring (4): "not drawn" (a
    candidate for `draw`), or "rejected: <rule>"."""
    h = roi_px // 2
    g = s["gauss"]
    n = len(s["frame"])
    state = ["candidate"] * n

    def out(mask, reason):
        for i in np.flatnonzero(mask):
            if state[i] == "candidate":
                state[i] = "rejected: " + reason
    fitted = np.isfinite(g["photons"]) & np.isfinite(g["sigma_x_pix"]) & np.isfinite(g["sigma_y_pix"])
    out(~fitted, "fit failed")
    with np.errstate(invalid="ignore"):
        out(~(np.hypot(g["x_roi"] - h, g["y_roi"] - h) < MAX_OFF_CENTRE_PX), "off centre")
        out(~((np.abs(np.log(g["sigma_x_pix"] / s["sx_curve"])) < settings.width_tol)
              & (np.abs(np.log(g["sigma_y_pix"] / s["sy_curve"])) < settings.width_tol)), "shape")
    rejected, masks = neighbours(s, roi_px, settings.neighbour_fraction)
    out(rejected | ((~masks).mean((1, 2)) > settings.max_masked), "neighbour")
    alive = np.array([st == "candidate" for st in state])
    # brightness per bin of defocus: from half the median up to the bulk, median + 3 robust SDs in log
    with np.errstate(divide="ignore", invalid="ignore"):
        logn = np.log(np.where(alive, g["photons"], np.nan))
    bright = np.zeros(n, bool)
    bins = np.round(s["d_nm"] / BRIGHTNESS_BIN_NM).astype(int)
    for b in np.unique(bins):
        m = alive & (bins == b)
        if m.sum() < 10:
            continue
        med = np.median(logn[m])
        mad = 1.4826 * np.median(np.abs(logn[m] - med))
        bright |= m & (logn >= med + np.log(0.5)) & (logn <= med + 3 * mad)
    out(alive & ~bright, "brightness")
    return ["not drawn" if st == "candidate" else st for st in state], masks


def draw(state: list, d_nm, molecules: int, seed: int = 0) -> list:
    """``state`` with at most ``molecules`` of the spots "not drawn" made "used", drawn evenly over
    `DRAW_BIN_NM` bins of defocus (`insitu.quotas`), at random within a bin: over every file at once."""
    state = list(state)
    pool = np.flatnonzero([st == "not drawn" for st in state])
    bins = np.round(np.asarray(d_nm)[pool] / DRAW_BIN_NM).astype(int)
    groups = [pool[bins == b] for b in np.unique(bins)]
    rng = np.random.default_rng(seed)
    for members, take in zip(groups, quotas([len(m) for m in groups], molecules), strict=True):
        for i in rng.permutation(members)[:take]:
            state[i] = "used"
    return state
    return state, masks


# ------------------------------------------------------------------ the whole step

@dataclass
class MoleculeStackROIs(InSituROIs):
    """In situ's container with what a known z adds: each unit's nominal z (SMAPpy's sign, relative to its
    file's focus), its defocus by the stage, its sweep (numbered over all files) and the width curve."""
    z_nm: np.ndarray = None          # [U]
    defocus_nm: np.ndarray = None    # [U] stage z less its file's focus, nm (−z_nm)
    sweep: np.ndarray = None         # [U]
    sweeps: int = 0
    curve: WidthCurve = None
    radius_px: float = float("nan")
    shape: tuple = ()                # the images' (Y, X)
    step_nm: float = 50.0            # the sweeps' step

    def batch(self, state: str = "used", ids=None):
        """(`uipsf.Batch`, its noise) of the units in ``state`` (or ``ids``): their regions, masks, each
        one's nominal z [E, 1], sweep and chip position, and a start from its Gaussian."""
        import torch

        from uipsf import Batch, ShiftedPoisson
        ids = self.ids(state) if ids is None else np.asarray(ids, int)
        u = self.units
        h = u.regions.shape[-1] // 2
        g = {k: np.asarray(v)[ids] for k, v in u.gauss.items()}
        px = np.asarray(self.pixel_size_nm, float)
        start = {"xyz": torch.tensor(np.c_[(np.nan_to_num(g["x_roi"], nan=h) - h) * px[0],
                                           (np.nan_to_num(g["y_roi"], nan=h) - h) * px[1], np.zeros(len(ids))],
                                     dtype=torch.float32),
                 "photons": torch.tensor(np.nan_to_num(g["photons"], nan=1000.0), dtype=torch.float32),
                 "background": torch.tensor(np.maximum(np.nan_to_num(g["background"], nan=1.0), 1.0),
                                            dtype=torch.float32)}
        batch = Batch(data=torch.tensor(u.regions[ids][:, None, None], dtype=torch.float32),
                      z_nm=torch.tensor(self.z_nm[ids][:, None], dtype=torch.float32),
                      s=float(self.read_noise_var), mask=torch.tensor(u.masks[ids][:, None, None]),
                      field_xy=torch.tensor(u.centre[ids].astype(float), dtype=torch.float32),
                      stack_id=torch.tensor(self.sweep[ids], dtype=torch.long), start=start)
        return batch, ShiftedPoisson(float(self.excess_noise))


def molecule_rois(paths, settings: MoleculeStackSettings | None = None, camera=None,
                  progress=None) -> MoleculeStackROIs:
    """The molecules of the z-sweep files at ``paths``, found and chosen (module docstring).  Each file is
    read twice -- once for the width curve, once to cut its molecules -- so that only one is in memory at a
    time.  ``camera`` defaults to the first file's, as a fit would resolve it."""
    from .input import discover_acquisitions
    from .pupil import camera_for, pixel_size_nm
    s = settings or MoleculeStackSettings()
    paths = [str(p) for p in discover_acquisitions(paths)] if paths else []
    if not paths:
        raise ValueError("no files")
    camera = camera if camera is not None else camera_for(paths[0])
    px = pixel_size_nm(camera)
    roi = s.region(px)
    tell = progress or (lambda entry: None)
    spots, zs, shape = [], [], None
    for k, path in enumerate(paths):
        tell({"message": f"the width curve: file {k + 1} of {len(paths)}", "fraction": 0.4 * k / len(paths)})
        f = read_file(path, camera, s)
        shape = f.frames.shape[1:]
        spots.append(first_spots(f.frames))
        zs.append(f.z_um)
    radius = s.illumination_px
    if radius is None:
        radius = illumination_radius(shape)
    curve = width_curve(spots, zs, [inside_circle(p["x"], p["y"], shape, radius) for p in spots], s.psf_shape)
    parts, sweep_base, candidates, sources = [], 0, 0, []
    for k, path in enumerate(paths):
        tell({"message": f"finding molecules: file {k + 1} of {len(paths)}",
              "fraction": 0.4 + 0.6 * k / len(paths)})
        f = read_file(path, camera, s)
        d_frame = (f.z_um - curve.focus_um[k]) * 1000
        cands = detect(f.frames, d_frame, curve, radius, s.significance)
        candidates += len(cands["frame"])
        m = measure(f.frames, cands, d_frame, curve, roi)
        state, masks = select(m, s, roi)
        m.update(state=state, masks=masks, sweep=sweep_base + f.sweep[m["frame"]], file=k)
        sweep_base += int(f.sweep.max()) + 1
        parts.append(m)
        sources.append({"path": path, "frames": len(f.frames), "sweeps": int(f.sweep.max()) + 1})
        del f
    rois = _assemble(parts, roi, px, camera, candidates, sources, curve, radius, shape, sweep_base)
    rois.state = draw(rois.state, rois.defocus_nm, s.molecules, s.seed)
    rois.step_nm = _step_um(zs) * 1000
    counts = ", ".join(f"{v} {k}" for k, v in sorted(rois.counts().items()))
    rois.messages.append(f"{candidates} detections in {sweep_base} sweeps of {len(paths)} files: {counts}")
    return rois


def _assemble(parts, roi, px, camera, candidates, sources, curve, radius, shape, sweeps) -> MoleculeStackROIs:
    h = roi // 2
    cat = np.concatenate
    n = sum(len(p["frame"]) for p in parts)
    gauss = {k: cat([p["gauss"][k] for p in parts]) for k in parts[0]["gauss"]} if parts else {}
    x = cat([p["x"] for p in parts]) if parts else np.zeros(0, int)
    y = cat([p["y"] for p in parts]) if parts else np.zeros(0, int)
    state = [st for p in parts for st in p["state"]]
    units = Units(frames=[[int(f)] for p in parts for f in p["frame"]], blink=np.arange(n),
                  x_px=x + np.nan_to_num(gauss.get("x_roi", np.full(n, h)) - h),
                  y_px=y + np.nan_to_num(gauss.get("y_roi", np.full(n, h)) - h),
                  centre=np.c_[x, y].astype(int), regions=cat([p["regions"] for p in parts]),
                  masks=cat([p["masks"] for p in parts]), gauss=gauss,
                  reason=[st.removeprefix("rejected: ") if st.startswith("rejected") else None for st in state],
                  source=cat([np.full(len(p["frame"]), p["file"], int) for p in parts]))
    d = cat([p["d_nm"] for p in parts]) if parts else np.zeros(0)
    return MoleculeStackROIs(units, state, np.round(d / DRAW_BIN_NM).astype(int), np.array([]), candidates,
                             tuple(px), float(camera.readout_variance), float(camera.excess_noise),
                             sources=sources, camera=camera, z_nm=-d, defocus_nm=d,
                             sweep=cat([p["sweep"] for p in parts]).astype(int), sweeps=int(sweeps), curve=curve,
                             radius_px=float(radius), shape=tuple(shape))


def molecule_widths(rois: MoleculeStackROIs, pixel_nm, settings=None):
    """The used molecules as the elliptical Gaussian calibration reads beads (`elliptical.BeadWidths`), one
    "bead" of one plane each: each region cut to the *fit box* about its centre and fitted with the elliptical
    Gaussian the data will be fitted with, its z in the sample the stage's turned round (``z_nm``, relative to
    its file's focus, where the width curve's spots are round), so no bead needs aligning (``z0_nm`` 0)."""
    from ..psf import GaussianPSF
    from .elliptical import BeadWidths, EllipticalCalibrationSettings
    settings = settings or EllipticalCalibrationSettings()
    used = rois.ids("used")
    R = rois.units.regions.shape[-1]
    box = min(int(settings.fit_box_px) | 1, R)
    a = (R - box) // 2
    regions = np.ascontiguousarray(np.maximum(rois.units.regions[used][:, a:a + box, a:a + box], 0), np.float32)
    psf = GaussianPSF(sigma=float(settings.start_sigma_px), elliptical=True)
    out = psf.unpack(psf.fit(regions))
    px, py = float(pixel_nm[0]), float(pixel_nm[1])

    def col(name, scale=1.0):
        return np.asarray(out[name], np.float64)[:, None] * scale
    return BeadWidths(rois.z_nm[used].astype(np.float64)[:, None], col("sigma_x_pix", px), col("sigma_y_pix", py),
                      col("sigma_x_err_pix", px), col("sigma_y_err_pix", py), col("photons"),
                      z0_nm=np.zeros(len(used)))
