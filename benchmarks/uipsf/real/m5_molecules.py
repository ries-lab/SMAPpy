"""M5's PSF from single AF647 molecules on the coverslip, imaged through repeated z stacks (2 November 2026).

    python benchmarks/uipsf/real/m5_molecules.py FOLDER [...] [--max-spots 4000] [--saf height ring none] \\
        [--pixel-blur] [--chunk-size 256] [--out benchmarks/uipsf/results/m5_molecules.jsonl] [--npz-dir DIR]

Every bead model of M5 carries the bead: a 40 or 100 nm sphere convolved along stage z, its dye in polystyrene
(n 1.59) up to its diameter above the glass, and a supercritical (SAF) ring that in buffer did not fit
(`docs/uipsf/m5-media.md`; 40 nm and 100 nm beads learn different γ and σ, `docs/uipsf/architecture.md`, "The
minimal model").  Molecules stuck to the glass have none of it: points at h ≈ 0, which is what a single-molecule
fit needs and what tests the SAF model in buffer without a bead (the user's decisive experiment).

**The data** (the user's, 2 November 2026): AF647 in blinking buffer, a z stack of 41 frames at 50 nm (±1 µm)
repeated 15 times per file (10 in the first 3D file), one frame per step, about 10 s per stack; 676 nm, PBS,
with and without the cylindrical lens.  A *sweep* here is one z stack, a *file* its repetitions.  Blinking
molecules are seen in a frame or a few, so every detection is its own emitter -- in situ's unit -- with no
linking: its z is not learnt but known, the frame's stage z against the focus.

**The geometry** (`FocalPlane`): a detection in frame k of sweep s at chip position (u, v) is at defocus
z_k − f_s − (a·u + b·v), z_k the stage's nominal z (SMAPpy's sign, `Batch.z_from_objective`), f_s a focus per
sweep (drift between sweeps; within one, about 10 s, the nominal z hold), (a, b) the coverslip's tilt of the
focal plane, shared by all sweeps (one stage position), and its height above the glass h (the fourth coordinate,
through k_z,med: the SAF light decays with it physically), held at `H_NM` by a tight prior -- every dye at one
height.  Each emitter learns x, y, photons and background only.

**Detection** (`detect`): each frame filtered with the spot it should show at its expected defocus, an elliptical
Gaussian whose widths come from the data (`width_curve`: SMAPpy's elliptical Gaussian on a first DoG detection,
its median widths against stage z, the focus of each file where σx = σy (3D) or the width is smallest (2D)).
Inside `CIRCLE_PX` of the image's centre only: the illumination is a circle a little larger than the image.

**Selection** (`select`), stringent because free or loosely bound dye images too: centred, its widths those of
the curve within `WIDTH_TOL`, photons from half the bin's median up to its bulk (the brightest are overlaps), no
neighbour within reach carrying a fifth of its light (weaker ones masked), then uiPSF's rejection after the
first fit (deviance, photons, offset).

**The model** (`model`): the minimal pupil, phase to order 8, the magnitude cos θ^γ (physics only) and the SAF
term as the user chose it (``--saf``): *height*, an effective height of the ring learnt on top of the dye's
(`MinimalPupil`'s exp(−2πκh)); *none*, the dye's own height alone; *ring*, a free factor on the supercritical
light (`RingFactorPupil`).  σ round and learnt, without the pixel's blur under it, as the bead studies'
"minimal0+sigma" (`m5_media_psfs.py`), so the numbers compare.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "synthetic")]
from bead_slices import emit  # noqa: E402

from uipsf.model import InSituGeometry  # noqa: E402
from uipsf.model.pupil import MinimalPupil  # noqa: E402
from uipsf.params import Bounded, Gaussian, ParamSpec  # noqa: E402

N_IMM, N_MED, WAVELENGTH, NA = 1.518, 1.334, 676.0, 1.50
PIXEL_NM = (104.7, 106.4)            # the bead studies' pixel, chosen by likelihood (`m5_media_pixel.jsonl`)
OFFSET_ADU, PHOTONS_PER_ADU = 100.0, 0.24     # M5's Fusion BT (`microscopes.json`)
READ_NOISE_VAR = 0.5                 # photons², an sCMOS's; small against the ~45 photons of background
SWEEP, STEP_NM = 41, 50.0
ORDER = 8
H_NM, H_PRIOR_NM = 5.0, 0.5          # every dye at one height (the user's start); 0.5 nm holds it
FOCUS_PRIOR_NM = 300.0               # each sweep's focus about its file's, from the width curve
TILT_PRIOR = 50.0                    # nm of focus per µm of field: 0.05 rad, as the coverslip's tilt prior
CIRCLE_PX = 105                      # the illumination's circle, radius from the detections' density (≈ 100-130)
ROI_PX = 25                          # 2.6 µm: the widest spots at ±1 µm have σ ≈ 9 px
WIDTH_TOL = 0.3                      # |log(σ/σ_curve)| per axis
NEIGHBOUR_SHARE = 0.2                # a neighbour with a fifth of the light rejects (in situ's rule)
MAX_MASKED = 0.25
Z_REJECT_NM = 100.0                  # the z check's floor: a spot this far off the plane is not on the glass


# ---------------------------------------------------------------------------------------------- loading
def load(folder: str) -> tuple[np.ndarray, np.ndarray]:
    """(frames [T, Y, X] photons, stage z [T] µm) of one NDTiff file."""
    import tifffile
    path = glob.glob(os.path.join(folder, "*.tif"))[0]
    with tifffile.TiffFile(path) as t:
        frames = np.stack([p.asarray() for p in t.pages]).astype(np.float32)
        z = np.array([p.tags[51123].value["ZPositionUm"] for p in t.pages], float)
    return (frames - OFFSET_ADU) * PHOTONS_PER_ADU, z


def inside_circle(x, y, shape, radius=CIRCLE_PX) -> np.ndarray:
    Y, X = shape
    return np.hypot(np.asarray(x) - (X - 1) / 2, np.asarray(y) - (Y - 1) / 2) <= radius


# ----------------------------------------------------------------------------------- the width curve
def first_spots(frames: np.ndarray) -> dict:
    """Round 0: DoG detection and SMAPpy's elliptical Gaussian on 15-pixel regions, every frame."""
    from smappy.calibrate.insitu import cut, gaussian
    from smappy.detect import AbsoluteCutoff, DoGFilter, find_candidates
    filt = DoGFilter(1.5)(frames)
    noise = 1.4826 * float(np.median(np.abs(filt - np.median(filt))))
    c = find_candidates(filt, AbsoluteCutoff(6 * noise))
    regions, ok = cut(frames, c.frame, c.x, c.y, 15)
    g = gaussian(regions)
    return dict(frame=np.asarray(c.frame)[ok], x=np.asarray(c.x)[ok], y=np.asarray(c.y)[ok], **g)


def focus_of(z_um, sx, sy, lens: str) -> float:
    """The stage z (µm) of focus from one file's spots: σx = σy (3D), or the smallest σx² + σy² (2D)."""
    bins = np.round((z_um - z_um.min()) / (STEP_NM / 1000)).astype(int)
    zs, rx, ry = [], [], []
    for b in np.unique(bins):
        m = bins == b
        if m.sum() >= 5:
            zs.append(float(np.median(z_um[m])))
            rx.append(float(np.median(sx[m])))
            ry.append(float(np.median(sy[m])))
    zs, rx, ry = map(np.array, (zs, rx, ry))
    if lens == "3D":
        near = np.abs(np.log(rx / ry)) < 0.8
        a, b = np.polyfit(zs[near], np.log(rx[near] / ry[near]), 1)
        return float(-b / a)
    w = rx ** 2 + ry ** 2
    near = w < 3 * w.min()
    c2, c1, _ = np.polyfit(zs[near], w[near], 2)
    return float(-c1 / (2 * c2))


def width_curve(spots: list[dict], zs: list[np.ndarray], shape, lens: str) -> dict:
    """Each file's focus (stage µm) and the pooled median widths (px) against defocus d = Z − Z_focus (nm), from
    the brighter half of the spots inside the circle."""
    focus, d_all, sx_all, sy_all = [], [], [], []
    for s, z in zip(spots, zs, strict=True):
        z_um = z[s["frame"]]
        ok = (np.isfinite(s["sigma_x_pix"]) & inside_circle(s["x"], s["y"], shape)
              & (s["photons"] > np.nanmedian(s["photons"])))
        f = focus_of(z_um[ok], s["sigma_x_pix"][ok], s["sigma_y_pix"][ok], lens)
        focus.append(f)
        d_all.append((z_um[ok] - f) * 1000)
        sx_all.append(s["sigma_x_pix"][ok])
        sy_all.append(s["sigma_y_pix"][ok])
    d, sx, sy = (np.concatenate(v) for v in (d_all, sx_all, sy_all))
    grid = np.arange(-1100, 1101, 50.0)
    mx, my = [], []
    for g in grid:
        m = np.abs(d - g) <= 50
        mx.append(np.median(sx[m]) if m.sum() >= 10 else np.nan)
        my.append(np.median(sy[m]) if m.sum() >= 10 else np.nan)
    mx, my = np.array(mx), np.array(my)
    good = np.isfinite(mx) & np.isfinite(my)
    return dict(focus_um=focus, d_nm=grid[good], sx=mx[good], sy=my[good])


def widths_at(curve: dict, d_nm) -> tuple[np.ndarray, np.ndarray]:
    return (np.interp(d_nm, curve["d_nm"], curve["sx"]), np.interp(d_nm, curve["d_nm"], curve["sy"]))


# ------------------------------------------------------------------------------------------- detection
def _template(sx: float, sy: float) -> np.ndarray:
    hx, hy = int(math.ceil(3 * sx)), int(math.ceil(3 * sy))
    yy, xx = np.mgrid[-hy:hy + 1, -hx:hx + 1]
    g = np.exp(-xx ** 2 / (2 * sx ** 2) - yy ** 2 / (2 * sy ** 2))
    return (g - g.mean()) / np.sqrt(((g - g.mean()) ** 2).sum())


def detect(frames: np.ndarray, d_nm: np.ndarray, curve: dict, k: float = 5.0) -> dict:
    """Candidates [frame, x, y] of every frame, filtered with the spot of its expected defocus (zero-mean
    template, so a flat background cancels), local maxima above ``k`` robust SDs of the filtered frame."""
    from scipy.signal import fftconvolve
    from smappy.detect import local_maxima
    sx, sy = widths_at(curve, d_nm)
    out = {"frame": [], "x": [], "y": []}
    for t, frame in enumerate(frames):
        f = fftconvolve(frame - np.median(frame), _template(sx[t], sy[t])[::-1, ::-1], mode="same")
        noise = 1.4826 * float(np.median(np.abs(f - np.median(f))))
        _, y, x, _ = local_maxima(f[None].astype(np.float32), threshold=k * noise)
        keep = inside_circle(x, y, frame.shape)
        out["frame"].append(np.full(keep.sum(), t))
        out["x"].append(x[keep])
        out["y"].append(y[keep])
    return {n: np.concatenate(v).astype(int) for n, v in out.items()}


# ------------------------------------------------------------------------------------------- selection
def measure_spots(frames, cands, d_frame, curve) -> dict:
    """25-pixel regions and SMAPpy's elliptical Gaussian on each, with the widths the curve expects."""
    from smappy.calibrate.insitu import cut, gaussian
    regions, ok = cut(frames, cands["frame"], cands["x"], cands["y"], ROI_PX)
    out = {n: v[ok] for n, v in cands.items()}
    out["regions"] = regions
    out.update(gaussian(regions))
    out["d_nm"] = d_frame[out["frame"]]
    out["sx_curve"], out["sy_curve"] = widths_at(curve, out["d_nm"])
    return out


def neighbours(s: dict) -> tuple[np.ndarray, np.ndarray]:
    """(rejected [n], masks [n, R, R]): a neighbour within reach with `NEIGHBOUR_SHARE` of the spot's light
    rejects it; weaker ones are masked, a 2.5 σ ellipse of the curve's widths about them."""
    n, h = len(s["frame"]), ROI_PX // 2
    rejected = np.zeros(n, bool)
    masks = np.ones((n, ROI_PX, ROI_PX), bool)
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    photons = np.nan_to_num(s["photons"], nan=0.0)
    cx = s["x"] + np.nan_to_num(s["x_roi"] - h)
    cy = s["y"] + np.nan_to_num(s["y_roi"] - h)
    for t in np.unique(s["frame"]):
        idx = np.flatnonzero(s["frame"] == t)
        for i in idx:
            for j in idx:
                if i == j:
                    continue
                dx, dy = cx[j] - s["x"][i], cy[j] - s["y"][i]
                rx, ry = 2.5 * s["sx_curve"][j], 2.5 * s["sy_curve"][j]
                if abs(dx) > h + rx or abs(dy) > h + ry:
                    continue
                reach = np.hypot(dx / (2.5 * (s["sx_curve"][i] + s["sx_curve"][j])),
                                 dy / (2.5 * (s["sy_curve"][i] + s["sy_curve"][j])))
                if photons[j] >= NEIGHBOUR_SHARE * photons[i] and reach < 1:
                    rejected[i] = True
                masks[i] &= ((xx - dx) / rx) ** 2 + ((yy - dy) / ry) ** 2 > 1
    return rejected, masks


def select(s: dict) -> tuple[np.ndarray, dict]:
    """The spots to learn from (indices) and how many each rule cut (module docstring)."""
    h = ROI_PX // 2
    n = len(s["frame"])
    reasons = {}
    fitted = np.isfinite(s["photons"]) & np.isfinite(s["sigma_x_pix"])
    reasons["fit failed"] = int((~fitted).sum())
    centred = fitted & (np.hypot(s["x_roi"] - h, s["y_roi"] - h) < 1.5)
    reasons["off centre"] = int((fitted & ~centred).sum())
    shape = centred & (np.abs(np.log(s["sigma_x_pix"] / s["sx_curve"])) < WIDTH_TOL) & \
        (np.abs(np.log(s["sigma_y_pix"] / s["sy_curve"])) < WIDTH_TOL)
    reasons["shape"] = int((centred & ~shape).sum())
    rejected, masks = neighbours(s)
    alone = shape & ~rejected & ((~masks).mean((1, 2)) <= MAX_MASKED)
    reasons["neighbour"] = int((shape & ~alone).sum())
    # brightness per 200 nm bin of defocus: from half the median up to the bulk (median + 3 robust SDs, in log)
    logn = np.log(np.where(alone, s["photons"], np.nan))
    bright = np.zeros(n, bool)
    bins = np.round(s["d_nm"] / 200).astype(int)
    for b in np.unique(bins):
        m = alone & (bins == b)
        if m.sum() < 10:
            continue
        med = np.median(logn[m])
        mad = 1.4826 * np.median(np.abs(logn[m] - med))
        bright |= m & (logn >= med + np.log(0.5)) & (logn <= med + 3 * mad)
    reasons["brightness"] = int((alone & ~bright).sum())
    s["masks"] = masks
    return np.flatnonzero(bright), reasons


def capped(ids: np.ndarray, d_nm: np.ndarray, sweep: np.ndarray, total: int, seed: int = 0) -> np.ndarray:
    """At most ``total`` spots, drawn evenly over 100 nm bins of defocus (so the far defocus, with fewer
    detections, keeps its share), at random within a bin."""
    if len(ids) <= total:
        return ids
    rng = np.random.default_rng(seed)
    bins = np.round(d_nm[ids] / 100).astype(int)
    groups = [ids[bins == b] for b in np.unique(bins)]
    quota, out = total // len(groups), []
    for g in sorted(groups, key=len):
        take = min(len(g), quota)
        out.append(rng.choice(g, take, replace=False))
        total -= take
    rest = np.setdiff1d(ids, np.concatenate(out))
    if total > 0 and len(rest):
        out.append(rng.choice(rest, min(total, len(rest)), replace=False))
    return np.sort(np.concatenate(out))


# ------------------------------------------------------------------------------------------- the model
class FocalPlane(InSituGeometry):
    """Points on a tilted focal plane, a focus per sweep (module docstring): [E, K, 4] as `InSituGeometry`'s,
    x, y, the defocus through the immersion, the height in the sample.  ``Batch.stack_id`` is the sweep,
    ``Batch.field_xy`` the chip position (px); ``Batch.z_nm`` each observation's nominal z [E, 1]."""

    def __init__(self, sweeps: int, focus_nm, centre_px, pixel_nm, h_nm: float = H_NM, h_prior_nm: float = H_PRIOR_NM,
                 fit_plane: bool = True, free_z: bool = False):
        """``free_z``: the z check's geometry (`z_check`): each emitter's z an offset of its defocus from where the
        focal plane puts it, unconstrained, its height held at ``h_nm``."""
        super().__init__(0.0, focus_prior_nm=FOCUS_PRIOR_NM)
        self.sweeps, self.focus0 = int(sweeps), torch.as_tensor(focus_nm, dtype=torch.float32)
        self.centre_px, self.pixel_nm = tuple(float(v) for v in centre_px), tuple(float(v) for v in pixel_nm)
        self.h_nm, self.h_prior_nm, self.fit_plane = float(h_nm), float(h_prior_nm), fit_plane
        self.free_z = free_z

    def specs(self, batch, start) -> list[ParamSpec]:
        dt, dev = batch.data.dtype, batch.data.device
        prior_xyz = Gaussian(torch.tensor([0.0, 0.0, self.h_nm], dtype=dt, device=dev),
                             torch.tensor([1e9, 1e9, self.h_prior_nm], dtype=dt, device=dev))
        f0 = self.focus0.to(dtype=dt, device=dev)
        return [ParamSpec("emitter.xyz", (3,), "emitter", init=start["xyz"], prior=None if self.free_z else prior_xyz),
                ParamSpec("geometry.focus", (self.sweeps,), init=f0, prior=Gaussian(f0, FOCUS_PRIOR_NM)),
                ParamSpec("geometry.plane", (2,), init=0.0, prior=Gaussian(0.0, TILT_PRIOR),
                          fixed=not self.fit_plane)]

    def __call__(self, p, batch) -> torch.Tensor:
        xyz = p["emitter.xyz"]
        E = xyz.shape[0]
        uv = (batch.field_xy.to(xyz) - xyz.new_tensor(self.centre_px)) * xyz.new_tensor(self.pixel_nm) / 1000
        f = p["geometry.focus"][batch.stack_id.to(xyz.device)] + (uv * p["geometry.plane"]).sum(-1)    # [E]
        z_k = batch.z_nm.to(xyz.dtype).reshape(E, -1)                                                   # [E, K]
        K = z_k.shape[1]
        if self.free_z:            # z above the plane's place in the beads' sign, through the immersion
            defocus, h = z_k - f[:, None] + (xyz[:, None, 2] - self.h_nm), torch.full_like(z_k, self.h_nm)
        else:
            defocus, h = z_k - f[:, None], xyz[:, None, 2].expand(E, K)
        return torch.stack([xyz[:, None, 0].expand(E, K), xyz[:, None, 1].expand(E, K), defocus, h], -1)


class RingFactorPupil(MinimalPupil):
    """The minimal pupil with a free factor s on the supercritical light instead of a height: cos θ^γ × (1 or s,
    stepped at the critical angle over half a pupil sample) × the aperture.  s = 1 is a point at the glass."""

    def __init__(self, *args, **kwargs):
        kwargs["saf_damping"] = False
        super().__init__(*args, **kwargs)
        s = self.grid.rho.real * self.grid_na if self.grid.rho.is_complex() else self.grid.rho * self.grid_na
        self.beyond = torch.sigmoid((s - self.n_med) / (self.grid_na / self.grid.size))

    def specs(self):
        return super().specs() + [ParamSpec("pupil.ring_s", (), init=1.0, transform=Bounded(0.0, 3.0))]

    def symmetric(self, p, cos_imm=None, kappa=None, cutoff=None):
        out = super().symmetric(p, cos_imm, kappa, cutoff)
        if cos_imm is not None:          # a profile at given points: no step there
            return out
        return out * (1 + (p["pupil.ring_s"] - 1) * self.beyond)

    def values(self, p) -> dict:
        return dict(super().values(p), ring_s=float(p["pupil.ring_s"]))


def model(saf: str, sweeps: int, focus_nm, shape, device=None, pixel_blur: bool = False):
    from smappy.calibrate.pupil import PSFModelSettings

    from uipsf import Channel, ForwardModel, Optics
    from uipsf.model import Bleaching
    from uipsf.model.build import make_psf, pupil_grid
    optics = Optics(NA=NA, wavelength_nm=WAVELENGTH, n_imm=N_IMM, n_med=N_MED, pixel_size_nm=PIXEL_NM)
    spec = PSFModelSettings(zernike_order=ORDER, magnitude="minimal", blur="round", blur_nm=30.0, pixel_blur=pixel_blur,
                            saf_damping=saf == "height")
    pupil = None
    if saf == "ring":
        grid = pupil_grid(optics, spec, 1100.0, device=device)
        pupil = RingFactorPupil(grid, n_max=ORDER, na=NA)
    psf = make_psf(optics, spec, (ROI_PX, ROI_PX), pupil=pupil, z_max_nm=1100.0, device=device)
    Y, X = shape
    geometry = FocalPlane(sweeps, focus_nm, ((X - 1) / 2, (Y - 1) / 2), PIXEL_NM)
    return ForwardModel([Channel(psf)], geometry=geometry, intensity=Bleaching(order=0))


def batch_of(s: dict, ids, z_nm, sweep, focus_shift_nm):
    """The chosen spots as a `Batch`: [E, 1, 1, R, R] photons, their masks, each one's nominal z (relative to its
    file's focus) and sweep, and a start from the Gaussians."""
    from uipsf import Batch
    h = ROI_PX // 2
    ids = np.asarray(ids)
    E = len(ids)
    x = (s["x_roi"][ids] - h) * PIXEL_NM[0]
    y = (s["y_roi"][ids] - h) * PIXEL_NM[1]
    start = {"xyz": torch.tensor(np.stack([x, y, np.full(E, H_NM)], -1), dtype=torch.float32),
             "photons": torch.tensor(s["photons"][ids], dtype=torch.float32),
             "background": torch.tensor(np.maximum(s["background"][ids], 1.0), dtype=torch.float32)}
    return Batch(data=torch.tensor(s["regions"][ids][:, None, None], dtype=torch.float32),
                 z_nm=torch.tensor(z_nm[ids][:, None], dtype=torch.float32),
                 s=READ_NOISE_VAR, mask=torch.tensor(s["masks"][ids][:, None, None]),
                 field_xy=torch.tensor(np.stack([s["x"][ids], s["y"][ids]], -1), dtype=torch.float32),
                 stack_id=torch.tensor(sweep[ids], dtype=torch.long), start=start)


# ------------------------------------------------------------------------------------------- the z check
def z_check(cal, batch, d_nm, noise, chunk_size: int = 256) -> dict:
    """Each used spot localised again on the learnt model with its z free (`FocalPlane`'s ``free_z``): its z
    error against the focal plane and the CRLB.  The median error per 100 nm of defocus is the z bend, which a
    bead stack measures the same way (`bead_slices.measure`), here with no bead and no stage-z kernel; a spot
    off by more than max(3 CRLB, `Z_REJECT_NM`) is free or stuck above the glass (the user's z filter)."""
    import copy

    from uipsf import ForwardModel
    from uipsf.workflows.scoring import localise
    geometry = copy.copy(cal.model.geometry)
    geometry.free_z = True
    m = ForwardModel(cal.model.channels, geometry=geometry, intensity=cal.model.intensity,
                     background=cal.model.background)
    used = np.asarray(cal.beads.used)
    sub = batch.subset(torch.tensor(used))
    phys = cal.result.params.physical()
    start = {n: v.detach() for n, v in phys.items() if cal.result.params.specs[n].scope == "emitter"}
    fit, _ = localise(m, cal.shared(), sub, noise, start=start, chunk_size=chunk_size)
    dz = fit.params.physical()["emitter.xyz"][:, 2].detach().cpu().double().numpy() - geometry.h_nm
    crlb = fit.std["emitter.xyz"][:, 2].detach().cpu().double().numpy()
    d = np.asarray(d_nm)[used]
    bad = ~np.isfinite(dz) | (np.abs(dz) > np.maximum(3 * crlb, Z_REJECT_NM))
    grid = np.arange(-1000, 1001, 100.0)
    bend = [float(np.median(dz[~bad & (np.abs(d - g) <= 50)])) if (~bad & (np.abs(d - g) <= 50)).sum() >= 10
            else float("nan") for g in grid]
    bend = np.array(bend)

    def rms(r):
        v = bend[(np.abs(grid) <= r) & np.isfinite(bend)]
        return round(float(np.sqrt(np.mean((v - v.mean()) ** 2))), 2)
    scatter = 1.4826 * np.median(np.abs((dz - np.interp(d, grid, np.nan_to_num(bend))) / crlb)[~bad])
    return dict(z_bend_grid_nm=grid.tolist(), z_bend_nm=np.round(bend, 1).tolist(), z_bend_rms_200=rms(200),
                z_bend_rms_600=rms(600), z_scatter_over_crlb=round(float(scatter), 2),
                z_crlb_median_nm=round(float(np.median(crlb)), 2), z_rejected=int(bad.sum()),
                passed=used[~bad])


# ------------------------------------------------------------------------------------------- results
def summary(cal) -> dict:
    p = cal.result.params.physical()
    pupil = cal.model.channels[0].psf.pupil
    phase = p["pupil.phase"].detach().cpu().double().numpy()
    nm = {int(j): round(float(v) * WAVELENGTH / (2 * math.pi), 2)
          for j, v in zip(pupil.phase_modes, phase, strict=True)}
    out = dict(pupil.values(p), sigma_nm=[round(float(v), 2) for v in p["psf.sigma"].detach().cpu().reshape(-1)],
               phase_nm=nm, objective=float(cal.result.objective), used=len(cal.beads.used),
               focus_nm=[round(float(v), 1) for v in p["geometry.focus"].detach().cpu()],
               plane_nm_per_um=[round(float(v), 2) for v in p["geometry.plane"].detach().cpu()])
    out["main_modes_nm"] = {f"Z{j}": nm.get(j) for j in (5, 6, 7, 8, 11, 22)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folders", nargs="+", help="the files' folders of one lens (each holds an NDTiff stack)")
    ap.add_argument("--max-spots", type=int, default=4000)
    ap.add_argument("--saf", nargs="+", default=["height"], choices=["height", "none", "ring"])
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "m5_molecules.jsonl"))
    ap.add_argument("--npz-dir", default=None)
    ap.add_argument("--rounds", type=int, default=2)
    # off, σ is all the blur the image gets, the pixel's p/√12 ≈ 30 nm included (as in the bead runs, m5_media.py);
    # on, the pixel's blur is put under it and σ reads as the blur beyond the pixel
    ap.add_argument("--pixel-blur", action="store_true")
    # spots per chunk on the device: 256 held about 31 GB on the M-series GPU (2 November 2026), into swap
    ap.add_argument("--chunk-size", type=int, default=256)
    args = ap.parse_args()
    from uipsf import ShiftedPoisson, default_device
    from uipsf.workflows.beads import learn_beads
    from uipsf.workflows.rejection import RejectionSettings

    lens = "3D" if "_3D_" in os.path.basename(os.path.normpath(args.folders[0])) else "2D"
    t0 = time.perf_counter()
    frames, zs = zip(*(load(f) for f in args.folders), strict=True)
    shape = frames[0].shape[1:]
    spots0 = [first_spots(f) for f in frames]
    curve = width_curve(spots0, list(zs), shape, lens)
    print(f"{lens}: focus per file (µm) {np.round(curve['focus_um'], 3).tolist()}", flush=True)

    parts, sweep_base = [], 0
    for i, (f, z) in enumerate(zip(frames, zs, strict=True)):
        d_frame = (z - curve["focus_um"][i]) * 1000                      # defocus of each frame, nm
        cands = detect(f, d_frame, curve)
        s = measure_spots(f, cands, d_frame, curve)
        ids, reasons = select(s)
        s["z_nm"] = -d_frame[s["frame"]]                                 # SMAPpy's sign: the emitter below focus
        s["sweep"] = sweep_base + s["frame"] // SWEEP
        sweep_base += int(np.ceil(len(f) / SWEEP))
        print(f"  {os.path.basename(args.folders[i])}: {len(cands['frame'])} detections, {len(s['frame'])} "
              f"regions, kept {len(ids)}; cut {reasons}", flush=True)
        parts.append((s, ids))
    keys = ("regions", "masks", "x", "y", "x_roi", "y_roi", "photons", "background", "z_nm", "sweep", "d_nm")
    s = {k: np.concatenate([p[0][k] for p in parts]) for k in keys}
    offset = np.cumsum([0] + [len(p[0]["x"]) for p in parts])
    ids = np.concatenate([p[1] + o for p, o in zip(parts, offset[:-1], strict=True)])
    ids = capped(ids, s["d_nm"], s["sweep"], args.max_spots)
    print(f"  {len(ids)} spots to learn from, {sweep_base} sweeps", flush=True)

    noise = ShiftedPoisson()
    rejection = RejectionSettings(max_masked=1.0, max_xy_offset_px=2.0)
    for saf in args.saf:
        m = model(saf, sweep_base, np.zeros(sweep_base), shape, device=default_device(), pixel_blur=args.pixel_blur)
        chosen, start = ids, None
        for r in range(1, args.rounds + 1):
            # round 1 learns from the selection; each later round from the spots the last one kept and its z check
            # passed, starting from its model (localised on it, judged, then every mode free)
            t1 = time.perf_counter()
            b = batch_of(s, chosen, s["z_nm"], s["sweep"], None)
            cal = learn_beads(m, b, noise, rejection=rejection, first_order=4, chunk_size=args.chunk_size,
                              max_dz_nm=None, start=start)
            check = z_check(cal, b, s["d_nm"][chosen], noise, args.chunk_size)
            row = dict(script="m5_molecules", lens=lens, saf=saf, round=r, pixel_blur=args.pixel_blur,
                       files=[os.path.basename(f) for f in args.folders], spots=len(chosen),
                       focus_file_um=[round(v, 4) for v in curve["focus_um"]])
            row.update(summary(cal))
            row.update({k: v for k, v in check.items() if k != "passed"})
            row["minutes"] = round((time.perf_counter() - t1) / 60, 2)
            emit(row, args.out)
            if args.npz_dir:
                os.makedirs(args.npz_dir, exist_ok=True)
                tag = f"{lens}_{saf}{'_pb' if args.pixel_blur else ''}_r{r}"
                np.savez(os.path.join(args.npz_dir, f"m5_molecules_{tag}.npz"),
                         **{k: v.detach().cpu().numpy() for k, v in cal.result.params.physical().items()},
                         used=np.asarray(cal.beads.used), ids=chosen)
            chosen, start = chosen[check["passed"]], cal.shared()
            del cal, b
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()                 # the allocator keeps what a round held, round after round
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
