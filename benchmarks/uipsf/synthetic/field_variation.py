"""Aberrations that vary from bead to bead, learnt as one shared pupil: what does the blur σ absorb?

    python benchmarks/uipsf/synthetic/field_variation.py [--cases scatter astig coma] [--deltas …] \\
        [--seeds 0 1 2] [--truth-sigma zero pixel] [--chunk 1] [--memory 0.25] \\
        [--out benchmarks/uipsf/results/field_variation.jsonl]

The hypothesis (the user's, October 2026): aberrations that differ across the field barely change a PSF in
focus but make defocused ones differ more with |z|.  A pupil shared by every bead can take that up only as a
Gaussian blur σ, which is then too wide in focus (the model's peak too low at z = 0 and −300 nm, as on M5's
real beads in every model) and too narrow far out.  So σ should grow with the variation and with the field,
where the truth has no extra blur.

Truth, M5-like (`real/m5_media.py`): NA 1.50 on a grid of 1.50, oil 1.518 into buffer 1.334, 600 nm, pixels of
104.7 × 106.4 nm, 27 × 27 px, 100 nm beads, 50 000 photons and a background of 19 per pixel (the median edge
of the 600 nm buffer 2D beads), ±1 µm in 100 nm steps.  40 beads uniform over a 360 × 360 px field, each at a
random sub-pixel position and ±100 nm in z.  The base pupil is the free-Zernike phase and magnitude learnt
on 600 nm buffer 2D (`results/m5_media_magnitude.jsonl`, variant "zernike").  The truth's σ is 0 (``zero``)
or a pixel's own blur p/√12 (``pixel``), which the model, sampling at pixel centres, does not have.

Per bead, added to the base phase (nm of path):
- *scatter*: Noll 5, 6, 7, 8 and 11, each drawn independently, normal with rms δ (δ = 0 is the control);
- *astig*: a linear gradient across the field, Noll 5 along x and 6 along y, ±δ at the field's edges;
- *coma*: the same for Noll 7 along x and 8 along y.

Per truth, one shared model (free Zernikes to order 8, phase and magnitude, σx and σy free) is learnt on all
beads and on the central half of the field (the middle 50 % in x and y), and once on all beads with σ held at
the truth's.  Recorded per fit: σ and its excess over the truth (difference and in quadrature), the phase's
bias against the base pupil (nm; Noll 5-8, 11, 22 and the rms over all modes) beside the truth's mean over the
beads used, each slice localised alone on the learnt model (`scoring.slice_fits`): the per-slice median over
beads of x, y and z less their mean (x, y) or a line (z) within ±200 and ±600 nm, rms, and z's slope over
±600; and the model against the average of the used beads' true PSFs, each rendered at the centre of its ROI
with its own pupil (no noise), at the average's peak pixel at z = −300, 0 and +300 nm, in % of that peak, with
the rms of the whole plane's difference.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import time

import numpy as np
import torch

import uipsf
from uipsf.model import ForwardModel
from uipsf.noise import Poisson
from uipsf.optics.system import choose_pupil_size
from uipsf.simulate import parameters, template

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_FILE = os.path.join(HERE, "..", "results", "m5_media_magnitude.jsonl")
BASE_ACQ, BASE_VARIANT = "20261029_600_100nm_mismatch_50ms_2D_1", "zernike"
OPTICS = dict(NA=1.50, wavelength_nm=600.0, n_imm=1.518, n_med=1.334, pixel_size_nm=(104.7, 106.4))
ORDER, ROI, BEAD_NM = 8, (27, 27), 100.0
PHOTONS, BACKGROUND = 50_000.0, 19.0
Z = torch.arange(-1000.0, 1001.0, 100.0)            # the objective's positions
E, FIELD_PX = 40, 360.0
SCATTER = (5, 6, 7, 8, 11)
GRADIENT = {"astig": (5, 6), "coma": (7, 8)}         # (along x, along y)
DELTAS = {"scatter": (0.0, 3.0, 6.0, 10.0), "astig": (5.0, 10.0, 20.0), "coma": (5.0, 10.0, 20.0)}
PLANES = (-300.0, 0.0, 300.0)
REPORTED = (5, 6, 7, 8, 11, 22)


def base_pupil() -> tuple[dict, dict]:
    """The base phase (nm of path) and magnitude by Noll index."""
    for line in open(BASE_FILE):
        r = json.loads(line)
        if r["acquisition"] == BASE_ACQ and r["variant"] == BASE_VARIANT:
            return ({int(j): float(v) for j, v in r["phase_nm_path"].items()},
                    {int(j): float(v) for j, v in r["magnitude"].items()})
    raise LookupError(f"{BASE_ACQ} / {BASE_VARIANT} not in {BASE_FILE}")


def model(device, sigma_nm=(30.0, 30.0), fit_sigma=True) -> ForwardModel:
    optics = uipsf.Optics(**OPTICS)
    grid = uipsf.PupilGrid.build(optics, choose_pupil_size(optics, float(Z.abs().max()) + 100),
                                 device=device)
    pupil = uipsf.ZernikePupil(grid, n_max=ORDER)
    psf = uipsf.VectorPSF(grid, pupil, ROI, bead_radius_nm=BEAD_NM / 2, sigma_nm=tuple(sigma_nm),
                          fit_sigma=fit_sigma)
    return ForwardModel(psf)


def rad(nm):
    return np.asarray(nm, float) * 2 * math.pi / OPTICS["wavelength_nm"]


def variation(case: str, delta: float, uv: np.ndarray, rng) -> np.ndarray:
    """[E, 41] nm added to each bead's phase, in `ZernikePupil`'s phase modes (5 … 45)."""
    out = np.zeros((len(uv), 41))
    if case == "scatter":
        for j in SCATTER:
            out[:, j - 5] = rng.normal(0.0, delta, len(uv))
    else:
        jx, jy = GRADIENT[case]
        out[:, jx - 5] = delta * uv[:, 0]
        out[:, jy - 5] = delta * uv[:, 1]
    return out


def truth_sigma(kind: str) -> tuple[float, float]:
    return (0.0, 0.0) if kind == "zero" else tuple(p / math.sqrt(12) for p in OPTICS["pixel_size_nm"])


def render(m, values: dict, per_bead_phase, xyz, z, chunk: int) -> torch.Tensor:
    """[E, K, Y, X] expected photons of ``m`` at ``xyz`` on the planes ``z``, one bead's pupil each if
    ``per_bead_phase`` [E, n] (rad) is given, in chunks of beads; on the CPU."""
    dev = m.channels[0].psf.fields.device
    out = []
    n = len(xyz)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        b0 = template(b - a, uipsf.Batch.z_from_objective(z), ROI, device=dev)
        v = dict(values, **{"emitter.xyz": xyz[a:b]})
        v["emitter.photons"] = values["emitter.photons"][a:b]
        v["emitter.background"] = values["emitter.background"][a:b]
        p = parameters(m, b0, v, device=dev).physical()
        if per_bead_phase is not None:
            p["pupil.phase"] = torch.as_tensor(per_bead_phase[a:b], dtype=torch.float32, device=dev)
        with torch.no_grad():
            mu = m(p, b0)
        out.append(mu.cpu().clone().double()[:, 0])          # copy, then cast (MPS, `uipsf.simulate`)
    return torch.cat(out)


def simulate(case, delta, seed, sigma_kind, device, chunk):
    """The batch, the beads' field positions (u, v in −1 … 1) and their phases (nm, [E, 41])."""
    phase_nm, magnitude = base_pupil()
    rng = np.random.default_rng(seed)
    uv = rng.uniform(-1.0, 1.0, (E, 2))
    xyz = np.column_stack([rng.uniform(-0.5, 0.5, (E, 2)) * np.array(OPTICS["pixel_size_nm"]),
                           rng.uniform(-100.0, 100.0, E)])
    base = np.array([phase_nm.get(j, 0.0) for j in range(5, 46)])
    per_bead = base[None] + variation(case, delta, uv, rng)
    m = model(device, truth_sigma(sigma_kind), fit_sigma=False)
    mags = torch.tensor([magnitude.get(j, 0.0) for j in m.channels[0].psf.pupil.magnitude_modes])
    values = {"pupil.magnitude": mags, "psf.sigma": torch.tensor(truth_sigma(sigma_kind)),
              "emitter.photons": torch.full((E,), PHOTONS), "emitter.background": torch.full((E,), BACKGROUND)}
    mu = render(m, values, rad(per_bead), torch.as_tensor(xyz, dtype=torch.float32), Z, chunk)
    data = torch.poisson(mu, generator=torch.Generator().manual_seed(seed + 1000)).float()[:, None]
    batch = uipsf.Batch(data=data, z_nm=uipsf.Batch.z_from_objective(Z))
    return batch, uv, per_bead, mags, xyz


def bends(m, res, batch) -> dict:
    """Each slice localised alone on the learnt model: the median over beads per slice of found − expected,
    x and y less their mean and z less a line, within ±200 and ±600 nm, rms (nm); z's slope over ±600.  A slice
    counts where at least six beads (or all, if fewer) were localised."""
    from uipsf.workflows.scoring import slice_fits
    found, crlb, expected = slice_fits(m, res.params.physical(), batch, Poisson(), res.params)
    z0 = expected[..., 2]
    stage = np.median(z0 - z0.mean(1, keepdims=True), 0)
    ok = np.isfinite(found).all(-1) & (crlb[..., 2] < 100.0)
    out = {}
    for w in (200, 600):
        near = np.abs(stage) <= w + 1e-6
        x, y = z0[:, near][ok[:, near]], found[..., 2][:, near][ok[:, near]]
        slope, offset = np.polyfit(x, y, 1)
        r = found - expected
        r[..., 2] = found[..., 2] - (slope * z0 + offset)
        need = min(6, len(ok))                       # a central field may hold fewer than six beads
        per = np.array([[np.median(r[ok[:, k], k, a]) if ok[:, k].sum() >= need else np.nan for a in range(3)]
                        for k in np.flatnonzero(near)])
        per[:, :2] -= np.nanmean(per[:, :2], 0)
        out[f"rms_{w}_nm"] = [round(float(np.sqrt(np.nanmean(per[:, a] ** 2))), 2) for a in range(3)]
        if w == 600:
            out["z_slope"] = round(float(slope), 4)
    return out


def peak_residual(m, res, per_bead, mags, sigma_kind, used, device, chunk) -> dict:
    """The learnt model against the average of the used beads' true PSFs, all centred, at `PLANES`: the
    residual at the average's peak pixel and the plane's rms, in % of that peak."""
    z = torch.as_tensor(PLANES)
    n = len(used)
    unit = {"emitter.photons": torch.ones(n), "emitter.background": torch.zeros(n)}
    centre = torch.zeros(n, 3)
    tm = model(device, truth_sigma(sigma_kind), fit_sigma=False)
    truth = render(tm, dict(unit, **{"pupil.magnitude": mags, "psf.sigma": torch.tensor(truth_sigma(sigma_kind))}),
                   rad(per_bead[used]), centre, Z, chunk).mean(0)
    shared = {k: v.detach().cpu() for k, v in res.params.physical().items() if not k.startswith("emitter.")}
    learnt = render(m, dict(unit, **shared), None, centre[:1], Z, 1)[0]
    out = {}
    for zp in z:
        k = int(torch.argmin((-Z - zp).abs()))       # the bead's z is minus the objective's
        a, b = truth[k].numpy(), learnt[k].numpy()
        i = np.unravel_index(np.argmax(a), a.shape)
        top = a[i]
        out[f"{float(zp):+.0f}"] = dict(peak_pct=round(float(100 * (b[i] - top) / top), 2),
                                           rms_pct=round(float(100 * np.sqrt(np.mean((b - a) ** 2)) / top), 3))
    return out


def phase_bias(res, m, per_bead, used) -> dict:
    lam = OPTICS["wavelength_nm"]
    phase_nm, _ = base_pupil()
    modes = m.channels[0].psf.pupil.phase_modes
    learnt = res.params.physical()["pupil.phase"].detach().cpu().double().numpy() * lam / (2 * math.pi)
    base = np.array([phase_nm.get(j, 0.0) for j in modes])
    d = learnt - base
    mean_truth = per_bead[used].mean(0) - base
    return dict(rms_nm=round(float(np.sqrt(np.mean(d ** 2))), 2),
                max_nm=round(float(d[np.argmax(np.abs(d))]), 2), max_noll=int(modes[int(np.argmax(np.abs(d)))]),
                modes_nm={j: round(float(d[modes.index(j)]), 2) for j in REPORTED},
                truth_mean_nm={j: round(float(mean_truth[modes.index(j)]), 2) for j in SCATTER})


def fit(batch, used, sigma_kind, held, device, chunk):
    from uipsf.workflows import fit_beads
    s = truth_sigma(sigma_kind)
    m = model(device, s if held else (30.0, 30.0), fit_sigma=not held)
    sub = batch.subset(torch.as_tensor(used))
    res = fit_beads(m, sub, Poisson(), chunk_size=chunk)
    return m, res, sub


def run(case, delta, seed, sigma_kind, device, chunk) -> list:
    batch, uv, per_bead, mags, _ = simulate(case, delta, seed, sigma_kind, device, chunk)
    central = np.flatnonzero((np.abs(uv) <= 0.5).all(1))
    every = np.arange(E)
    rows = []
    for field, used, held in (("all", every, False), ("central", central, False), ("all", every, True)):
        t = time.perf_counter()
        m, res, sub = fit(batch, used, sigma_kind, held, device, chunk)
        st = truth_sigma(sigma_kind)
        sl = res.params.physical()["psf.sigma"].detach().cpu().double().numpy()
        sl = np.broadcast_to(sl, (2,))
        row = dict(case=case, delta_nm=delta, seed=seed, truth_sigma=sigma_kind, field=field, sigma_held=held,
                   beads=int(len(used)), field_px=FIELD_PX if field == "all" else FIELD_PX / 2,
                   converged=bool(res.converged), objective=round(float(res.objective), 1),
                   iterations=res.iterations, truth_sigma_nm=[round(v, 2) for v in st],
                   sigma_nm=[round(float(v), 2) for v in sl],
                   sigma_excess_nm=[round(float(a - b), 2) for a, b in zip(sl, st, strict=True)],
                   sigma_excess_quad_nm=[round(float(np.sign(a * a - b * b) * math.sqrt(abs(a * a - b * b))), 2)
                                         for a, b in zip(sl, st, strict=True)],
                   phase=phase_bias(res, m, per_bead, used), bends=bends(m, res, sub),
                   peak=peak_residual(m, res, per_bead, mags, sigma_kind, used, device, chunk))
        row["seconds"] = round(time.perf_counter() - t, 1)
        rows.append(json.loads(json.dumps(row, default=float)))
        print(json.dumps(rows[-1]), flush=True)
    return rows


def done(path) -> set:
    if not path or not os.path.exists(path):
        return set()
    # a truth is done when all three of its fits are written: one stopped halfway is run again
    from collections import Counter
    n = Counter((r["case"], r["delta_nm"], r["seed"], r["truth_sigma"]) for r in map(json.loads, open(path)))
    return {k for k, v in n.items() if v == 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["scatter", "astig", "coma"])
    ap.add_argument("--deltas", type=float, nargs="+", default=None, help="instead of each case's own")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="the gradients take the first only")
    ap.add_argument("--truth-sigma", nargs="+", default=["zero", "pixel"], choices=["zero", "pixel"])
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--memory", type=float, default=0.25)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(args.memory)
    import uipsf.workflows.scoring as scoring
    scoring.localise = functools.partial(scoring.localise, chunk_size=len(Z) * args.chunk)
    skip = done(args.out)
    for sigma_kind in args.truth_sigma:
        for case in args.cases:
            for delta in args.deltas or DELTAS[case]:
                for seed in args.seeds if case == "scatter" else args.seeds[:1]:
                    if (case, delta, seed, sigma_kind) in skip:
                        continue
                    for row in run(case, delta, seed, sigma_kind, device, args.chunk):
                        if args.out:
                            with open(args.out, "a") as f:
                                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
