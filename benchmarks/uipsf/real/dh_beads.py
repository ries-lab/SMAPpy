"""Real double-helix beads: is the analytic double helix, placed and with Zernikes beside it, a model of them?

    python benchmarks/uipsf/real/dh_beads.py [--sets dhpsfu moerner] [--out results/dh_beads.jsonl]

S1 of docs/uipsf/scanned-psfs.md: the masks work on simulations (`test_phase_mask.py`); here they meet two real
double-helix plates, neither of which is Prasad's idealised design that `model.mask.DoubleHelix` draws.

**The data** (`~/.cache/uipsf/engineered`, docs/uipsf/data.md):
- *dhpsfu*: DHPSFU (Zhang, Ochirova, Ponjavic, Sci Rep 2025; Zenodo 16357203), ``Calib.tif``: one bead, 127
  planes of 75 × 75 at 40 nm, 207 nm pixels; a Nikon 60× 1.27 NA water objective, a Double Helix Optics mask for
  the far red, an EMCCD (the paper's Microscope 1 description, which this set's README does not contradict).
- *moerner*: Barentine, Balaji, Moerner (bioRxiv 2025; Zenodo 20402153), ``Fig1_Bead_Z_Stack.h5``: one bead, 61
  planes of 40 × 40 at 50 nm, 120 nm pixels, iXon 897, 647 nm excitation, frames averaged.  The objective is not
  stated; it is scanned (``--na``).

**Photons.**  Neither set gives a usable gain: the camera is calibrated from the stack itself, the variance of a
second difference along x against the mean above the offset at low and middle intensities, where the bead's
curvature does not inflate it (`noise_scale`): 7.7 ADU per Poisson-equivalent count above 472 ADU (dhpsfu, the
offset the zero-photon peak of the background's histogram), 4.7 above 352 (moerner, its metadata's offset; the
average of about ten frames).  The likelihood is then Poisson in those counts with dhpsfu's read noise, 2.5
counts².  What is left of a wrong scale is a common factor on every model's deviance, so the comparison stands;
the mean deviance per pixel says how far each model is from the noise.

**The models**, each with the minimal magnitude, a round blur and the bead's size (100 nm, TetraSpeck):
- *Zernike n*: free phase Zernikes up to order n, from a flat start -- the way a PSF without a mask is learnt;
- *DH + Zernike n*: the analytic double helix with its centre, rotation, scale and α free, Zernikes up to n
  beside it, Pavani's design (`model.mask.DoubleHelix`) with 3 or 5 modes, started at four rotations and both
  handednesses; the best kept, then refined with Zernikes up to 8.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

import uipsf
from uipsf.model import MaskSpec
from uipsf.noise import ShiftedPoisson
from uipsf.params import Params
from uipsf.workflows import BeadSettings, PSFModelSettings, bead_model
from uipsf.workflows.beads import fit_beads

DATA = os.path.expanduser(os.environ.get("UIPSF_ENGINEERED", "~/.cache/uipsf/engineered"))


@dataclass
class MaskedModel(PSFModelSettings):
    phase_mask: MaskSpec | None = None


@dataclass
class Dataset:
    name: str
    adu: np.ndarray                 # [K, Y, X]
    step_nm: float
    pixel_nm: float
    offset: float
    na: float
    n_imm: float
    n_med: float
    wavelength_nm: float = 680.0
    bead_nm: float = 100.0
    extra: dict = field(default_factory=dict)


def load(name: str) -> Dataset:
    if name == "dhpsfu":
        import tifffile
        adu = tifffile.imread(os.path.join(DATA, "dh_dhpsfu2025/data/Calib.tif")).astype(np.float64)
        return Dataset(name, adu, 40.0, 207.0, 472.0, na=1.27, n_imm=1.333, n_med=1.333)
    if name == "moerner":
        import h5py
        with h5py.File(os.path.join(DATA, "dh_moerner2026/data/DH_PSF_Fig1/Fig1_Bead_Z_Stack.h5")) as f:
            adu = f["ImageData"][...].astype(np.float64)
        return Dataset(name, adu, 50.0, 120.0, 352.0, na=1.35, n_imm=1.406, n_med=1.333)
    raise ValueError(name)


def noise_scale(adu: np.ndarray, offset: float) -> tuple[float, float]:
    """(ADU per Poisson-equivalent count, read-noise variance in counts²) from the second difference along x,
    whose variance is 6× the pixels' and which a linear gradient leaves alone: the median over the pixels between
    the 30th and 80th percentile of brightness, above the offset (module docstring)."""
    d2 = adu[:, :, 2:] - 2 * adu[:, :, 1:-1] + adu[:, :, :-2]
    m = adu[:, :, 1:-1]
    lo, hi = np.quantile(m, [0.3, 0.8])
    k = (m >= lo) & (m < hi)
    var = np.median(d2[k] ** 2) / 0.4549 / 6               # median of χ²₁ is 0.4549 of its mean
    scale = var / (m[k].mean() - offset)
    floor = np.quantile(m, [0.0, 0.3])
    k0 = (m >= floor[0]) & (m < floor[1])
    read = max(np.median(d2[k0] ** 2) / 0.4549 / 6 - scale * max(m[k0].mean() - offset, 0.0), 0.0)
    return float(scale), float(read / scale ** 2)


def bead_centres(counts: np.ndarray, pixel_nm: float) -> np.ndarray:
    """(y, x) of every bead in the field, [n, 2]: the maximum over z smoothed by 600 nm, so that each bead's two
    lobes, turning about its centre, make one blob, and its local maxima above a fifth of the brightest."""
    from scipy.ndimage import gaussian_filter, maximum_filter
    mp = gaussian_filter(counts.max(0), 600.0 / pixel_nm)
    peaks = (mp == maximum_filter(mp, size=int(round(1500.0 / pixel_nm)) | 1)) & (mp > 0.2 * mp.max())
    return np.argwhere(peaks)


def batch(ds: Dataset, roi: int, sign: float = 1.0, max_dz_nm: float | None = 80.0):
    """The most isolated bead, cut to ``roi`` × ``roi`` about it, in counts, the pixels nearer another bead
    masked out; z from the stack's planes, thinned to ``max_dz_nm``.  The calibration fields are dense (DHPSFU's
    beads lie 5–10 pixels apart): with a neighbour's lobes in the region the fit made up for them with a blur of
    140 nm and γ at its bound."""
    scale, read = noise_scale(ds.adu, ds.offset)
    counts = (ds.adu - ds.offset) / scale
    K, Y, X = counts.shape
    h = roi // 2
    centres = bead_centres(counts, ds.pixel_nm)
    inside = [(c, d) for c, d in ((c, np.sort(np.hypot(*(centres - c).T))[1] if len(centres) > 1 else np.inf)
                                  for c in centres)
              if h <= c[0] < Y - roi + h + 1 and h <= c[1] < X - roi + h + 1]
    (cy, cx), _ = max(inside, key=lambda t: t[1]) if inside else ((Y // 2, X // 2), None)
    cy, cx = int(cy), int(cx)
    cut = counts[:, cy - h:cy - h + roi, cx - h:cx - h + roi]
    yy, xx = np.mgrid[cy - h:cy - h + roi, cx - h:cx - h + roi]
    own = np.hypot(yy - cy, xx - cx)
    others = [np.hypot(yy - c[0], xx - c[1]) for c in centres if (c[0], c[1]) != (cy, cx)]
    keep = np.ones((roi, roi), bool) if not others else own <= np.min(others, 0)
    objective = sign * (np.arange(K) - (K - 1) / 2) * ds.step_nm
    data = torch.as_tensor(cut, dtype=torch.float32)[None, None]
    mask = torch.as_tensor(keep)[None, None, None].expand(1, 1, K, roi, roi).clone()
    b = uipsf.Batch(data=data, z_nm=uipsf.Batch.z_from_objective(torch.as_tensor(objective, dtype=torch.float32)),
                    s=read, mask=mask)
    return (b.thinned(max_dz_nm) if max_dz_nm else b), scale, read


def optics(ds: Dataset, na: float | None = None, n_imm: float | None = None):
    return uipsf.Optics(NA=na or ds.na, wavelength_nm=ds.wavelength_nm, n_imm=n_imm or ds.n_imm, n_med=ds.n_med,
                        pixel_size_nm=(ds.pixel_nm, ds.pixel_nm))


def fit(ds, b, order: int, mask: MaskSpec | None, rotation: float = 0.0, o=None, learn_na: bool = False,
        bead_nm: float | None = None):
    o = o or optics(ds)
    settings = BeadSettings(psf=MaskedModel(zernike_order=order, phase_mask=mask, learn_na=learn_na),
                            bead_diameter_nm=ds.bead_nm if bead_nm is None else bead_nm)
    model = bead_model(o, settings, tuple(b.data.shape[-2:]), z_max_nm=float(b.z_nm.abs().max()), device="cpu")
    values = {"mask.rotation": torch.tensor(rotation)} if mask is not None else None
    start = Params(model.specs(b), b.n_emitters, device="cpu", values=values)
    res = fit_beads(model, b, ShiftedPoisson(), start=start, first_order=None, max_dz_nm=None)
    return model, res


def summary(model, res, b) -> dict:
    p = res.params.physical()
    npix = int(b.mask.expand_as(b.data)[0].sum()) if b.mask is not None else b.data[0].numel()
    out = dict(objective=round(res.objective, 1), deviance_per_pixel=round(2 * res.objective / npix, 3),
               iterations=res.iterations, stop=res.stop_reason)
    if "mask.centre" in p:
        out["mask"] = {k.removeprefix("mask."): (v.tolist() if v.dim() else round(float(v), 4))
                       for k, v in p.items() if k.startswith("mask.")}
        out["mask_crlb"] = {k.removeprefix("mask."): v.tolist() for k, v in res.std.items() if k.startswith("mask.")}
    out["blur_nm"] = [round(float(s), 1) for s in p["psf.sigma"].reshape(-1)]
    if "pupil.gamma" in p:
        out["gamma"] = round(float(p["pupil.gamma"]), 3)
    if "pupil.na" in p:
        out["na"] = round(float(p["pupil.na"]), 3)
    return out


def dh_search(ds, b, order, design="pavani", zones=(3, 5), rotations=4):
    """The double helix started at each zone count, handedness and rotation, order ``order`` beside it: the best."""
    best = None
    for L in zones:
        for hand in (1, -1):
            for i in range(rotations):
                spec = MaskSpec(kind="double helix", design=design, zones=L, charge=hand,
                                free=("centre", "rotation", "scale", "alpha"))
                model, res = fit(ds, b, order, spec, rotation=i * math.pi / rotations)
                tag = dict(design=design, zones=L, hand=hand, rotation_start=round(i * math.pi / rotations, 3))
                print(f"   L {L} hand {hand:+d} rot {i}: {res.objective:12.1f}", flush=True)
                if best is None or res.objective < best[1].objective:
                    best = (model, res, tag, spec)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sets", nargs="+", default=["dhpsfu", "moerner"])
    ap.add_argument("--roi-nm", type=float, default=3500.0, help="the region's side; one bead, no neighbours")
    ap.add_argument("--orders", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--zernike-orders", nargs="+", type=int, default=[8, 12])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "dh_beads.jsonl"))
    args = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    for name in args.sets:
        ds = load(name)
        roi = int(round(args.roi_nm / ds.pixel_nm)) | 1
        b, scale, read = batch(ds, min(roi, *ds.adu.shape[1:]))
        print(f"   {int(b.mask[0, 0, 0].sum())} of {roi * roi} pixels kept, {b.data.shape[2]} planes", flush=True)
        print(f"== {name}: {tuple(b.data.shape)}, {scale:.2f} ADU per count, read noise {read:.2f} counts²")
        rows = []
        for n in args.zernike_orders:
            model, res = fit(ds, b, n, None)
            rows.append(dict(set=name, model=f"Zernike {n}", **summary(model, res, b)))
            print("  ", rows[-1], flush=True)
        model, res, tag, spec = dh_search(ds, b, args.orders[0])
        rows.append(dict(set=name, model=f"DH + Zernike {args.orders[0]}", **tag, **summary(model, res, b)))
        print("  ", rows[-1], flush=True)
        for n in args.orders[1:]:
            rot = float(res.params.physical()["mask.rotation"])
            model, res2 = fit(ds, b, n, spec, rotation=rot)
            rows.append(dict(set=name, model=f"DH + Zernike {n}", **tag, **summary(model, res2, b)))
            print("  ", rows[-1], flush=True)
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps(dict(r, adu_per_count=round(scale, 3), read_counts2=round(read, 3))) + "\n")


if __name__ == "__main__":
    main()
