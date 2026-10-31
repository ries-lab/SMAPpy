"""A minimal M5 model with the objective's part held at a consensus of the conditions, and what it costs.

    python benchmarks/uipsf/real/m5_consensus.py ACQUISITION [...] [--forms height "bead s"] \\
        [--consensus-forms height] [--asym-modes 2 3 5 6 8 16] [--centre 0.5] [--dz 100] [--chunk 1] [--memory 0.5] \\
        [--out benchmarks/uipsf/results/m5_consensus.jsonl]

The user's minimal model (31 October 2026), one pupil grid at NA 1.51 with the aperture held at the nominal
NA 1.50 (a hard cutoff at ρ 1.50/1.51: the Studio found the grid's own edge at 1.50 cutting light the data
wanted, `docs/uipsf/m5-media.md`):

- magnitude cos θ_imm^γ, symmetric, times the supercritical ring in buffer, one free parameter in one of two
  *forms*: ``height``, exp(−2πκh), h ≥ 0, the physical one (a point at the glass is h = 0, which single
  molecules are), or ``bead s``, the 100 nm sphere's exact ring times a free factor s ≤ 1.5 (`EdgePupil`);
  M5's beads in buffer seemed to want more light there than the sphere gives, which h cannot (it stops at 0)
  and s can.  In matched oil there is no ring, and both forms are one model;
- the blur held at the pixel's own, σ = p/√12 (learning on pixel centres puts the pixel's width into σ), or
  learnt (``--blurs``): the first 40 nm beads wanted σ 40 / 32 nm beyond the pixel, and held at the pixel's
  the z bend doubled (4 → 9 nm), so both are run, each with its own consensus; and held at 0, the user's
  question whether no blur at all removes the core every model misses near focus (`m5_media_psfs.py`);
- the phase: every Zernike to order 8, free per condition;
- the objective's level: γ and a few m ≠ 0 magnitude modes (``--asym-modes``; 2, 3, 5, 6, 8, 16 are the
  ones above 0.01 in every M5 run), the same for every condition.

Only beads in the central part of the chip (``--centre``: the middle half of its width and height by
default), to keep field-dependent aberrations out.  Per condition, all on the same beads and slices, so the
objectives compare in nats:

1. *references*: free Zernike magnitude with σ free, and with σ held as here;
2. *free*: the minimal model with γ and the asymmetric modes learnt (each form in buffer);
3. *consensus*: γ and the asymmetric modes held at the other seven conditions' median, and at their mean
   weighted by the inverse CRLB variance (leave one out, so no condition votes for its own value); only the
   phase and the ring learnt.

The acquisitions are `m5_media.acquisition`'s: the 100 nm beads of 2026-10-29 (both colours) and the 40 nm
beads of 2026-10-31 (600 nm only, each condition twice, at 50 and 100 nm steps), each with its own bead
diameter in the model and in the ring.  The consensus pools them all: γ and the asymmetric modes are the
objective's, whatever the bead.

Restartable: a row already in ``--out`` is not run again.  `bead_slices.measure` gives each fit's bends.
"""
from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
from bead_slices import emit, measure  # noqa: E402,I001
from edge_magnitude import use_pupil  # noqa: E402
from m5_media import N_IMM, N_MED, ORDER, WAVELENGTH, acquisition, select, subset  # noqa: E402
from radial_magnitude import EdgePupil  # noqa: E402

PIXEL = (104.7, 106.4)
NA, GRID_NA = 1.50, 1.51
#: a condition with fewer central beads than this does not vote in the consensus
MIN_BEADS = 6
ESTIMATORS = ("median", "weighted", "joint")


def central(rois, keep, fraction: float, shape) -> list:
    """The kept beads whose centre lies in the middle ``fraction`` of the image's width and height."""
    H, W = shape
    lo_x, hi_x = W * (1 - fraction) / 2, W * (1 + fraction) / 2
    lo_y, hi_y = H * (1 - fraction) / 2, H * (1 + fraction) / 2
    return [i for i in keep if lo_x <= rois.records[i]["x_px"] <= hi_x and lo_y <= rois.records[i]["y_px"] <= hi_y]


def minimal(form: str, ring: bool, modes, hold: dict, held_magnitude: dict, bead_nm: float):
    """Make `bead_model` build the minimal pupil (module docstring)."""
    import uipsf.workflows.calibrate as calibrate
    from uipsf.model.pupil import ZernikePupil

    class Build(ZernikePupil):
        # a class, not a function: the module also asks isinstance(pupil, ZernikePupil)
        def __new__(cls, grid, n_max, **kwargs):
            return EdgePupil(grid, n_max, centre="cos", edge="hard", asymmetric=True, asym_modes=tuple(modes),
                             saf=ring, ring="bead" if form == "bead s" else "height",
                             saf_step=ring and form == "bead s",
                             held_magnitude=held_magnitude, hold={"pupil.edge_rho": NA / GRID_NA} | hold,
                             bead_nm=bead_nm)
    calibrate.ZernikePupil = Build
    return "free"


def summary(run, cond) -> dict:
    """What a fit learnt, with the CRLB of each learnt number."""
    cal = run.cal
    p, std = cal.result.params.physical(), cal.result.std
    pupil = cal.model.channels[0].psf.pupil
    lam = WAVELENGTH[cond["colour"]]

    def values(name):
        return p[name].detach().cpu().double().numpy().reshape(-1)

    def errors(name):
        return std[name].detach().cpu().double().numpy().reshape(-1) if name in std else None

    row = dict(used=len(cal.beads.used), objective=round(float(cal.result.objective), 1),
               iterations=cal.result.iterations, stop=cal.result.stop_reason,
               sigma_nm=values("psf.sigma").round(2).tolist(),
               phase_nm_path={int(j): round(float(v) * lam / (2 * math.pi), 2)
                              for j, v in zip(pupil.phase_modes, values("pupil.phase"), strict=True)},
               magnitude={int(j): round(float(v), 4) for j, v in pupil.coefficients(p, "magnitude").items()},
               atlas_mean_deviance=round(float(cal.atlas.mean_deviance), 3))
    e = errors("pupil.magnitude")
    row["magnitude_crlb"] = ({int(j): round(float(v), 5) for j, v in zip(pupil.magnitude_modes, e, strict=True)}
                             if e is not None else {})
    e = errors("pupil.phase")
    row["phase_crlb_nm"] = ({int(j): round(float(v) * lam / (2 * math.pi), 3)
                             for j, v in zip(pupil.phase_modes, e, strict=True)} if e is not None else {})
    if isinstance(pupil, EdgePupil):
        row["edge"] = pupil.edge_values(p)
        row["edge_crlb"] = {k.split(".", 1)[1]: round(float(errors(k)[0]), 5) for k in
                            ("pupil.centre_g", "pupil.saf_h", "pupil.saf_s") if errors(k) is not None}
        row["profile"] = pupil.edge_profile(p)
        # the covariance of γ and the m ≠ 0 modes, every other parameter profiled out (the Schur complement):
        # what a joint fit over the conditions would weigh each by (`consensus`, "joint")
        index, names = [], []
        for n, a, b in cal.result.params.layout("shared"):
            if n == "pupil.centre_g":
                index.append(a)
                names.append("g")
            elif n == "pupil.magnitude":
                index += list(range(a, b))
                names += [str(j) for j in pupil.magnitude_modes]
        if index:
            c = cal.result.covariance_shared.detach().cpu().double().numpy()[np.ix_(index, index)]
            row["objective_cov"] = dict(names=names, cov=c.tolist())
    return row


def consensus(rows: list, form: str, estimator: str, leave_out: str, modes) -> tuple[dict, dict]:
    """γ and the asymmetric modes from the other conditions' *free* fits of ``form`` (oil's, with no ring,
    count for both forms): their median; their mean weighted by 1/CRLB², each number alone; or *joint*, the
    generalised least squares of all of them together, θ = (Σ C_i⁻¹)⁻¹ Σ C_i⁻¹ θ_i with C_i the fit's
    covariance of them, every other parameter profiled out -- what one fit of the conditions together, with
    these shared and the rest each condition's own, gives where each condition's likelihood is quadratic
    about its own optimum."""
    pool = [r for r in rows if r["stage"] == "free" and r["form"] in (form, "none") and r["acquisition"] != leave_out
            and r["beads"]["central"] >= MIN_BEADS]

    def combine(v, e):
        v, e = np.asarray(v, float), np.asarray(e, float)
        if estimator == "median":
            return float(np.median(v))
        w = 1 / np.maximum(e, 1e-9) ** 2
        return float(np.sum(w * v) / np.sum(w))

    if estimator == "joint":
        names = ["g"] + [str(j) for j in modes]
        A, b = np.zeros((len(names), len(names))), np.zeros(len(names))
        for r in pool:
            c = r["objective_cov"]
            at = [c["names"].index(n) for n in names]
            F = np.linalg.pinv(np.asarray(c["cov"])[np.ix_(at, at)])
            v = np.array([r["edge"]["centre_g"]] + [r["magnitude"][str(j)] for j in modes])
            A, b = A + F, b + F @ v
        theta = np.linalg.solve(A, b)
        return {"pupil.centre_g": float(theta[0])}, {int(j): float(t) for j, t in zip(modes, theta[1:], strict=True)}
    gamma = combine([r["edge"]["centre_g"] for r in pool], [r["edge_crlb"].get("centre_g", np.inf) for r in pool])
    held = {int(j): combine([r["magnitude"][str(j)] for r in pool],
                            [r["magnitude_crlb"].get(str(j), np.inf) for r in pool]) for j in modes}
    return {"pupil.centre_g": gamma}, held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acquisitions", nargs="+")
    ap.add_argument("--forms", nargs="+", default=["height", "bead s"], choices=("height", "bead s"))
    ap.add_argument("--consensus-forms", nargs="+", default=["height"], choices=("height", "bead s"),
                    help="the ring forms held at the consensus; the free fits run every --forms")
    ap.add_argument("--blurs", nargs="+", default=["fixed", "free", "zero"], choices=("fixed", "zero", "round", "free"),
                    help="the minimal model's blur: held at the pixel's, held at 0, one σ, or σx and σy")
    ap.add_argument("--asym-modes", type=int, nargs="+", default=[2, 3, 5, 6, 8, 16])
    ap.add_argument("--centre", type=float, default=0.5, help="the central fraction of the width and height")
    ap.add_argument("--z-range", type=float, default=600.0)
    ap.add_argument("--window", type=float, default=1000.0)
    ap.add_argument("--dz", type=float, default=100.0)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--memory", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "m5_consensus.jsonl"))
    args = ap.parse_args()
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    import tifffile

    import uipsf.workflows.beads as beads
    import uipsf.workflows.calibrate as calibrate
    import uipsf.workflows.scoring as scoring
    beads.MAX_DZ_NM = args.dz
    calibrate.learn_beads = functools.partial(calibrate.learn_beads, chunk_size=args.chunk, max_dz_nm=args.dz)
    scoring.localise = functools.partial(scoring.localise,
                                         chunk_size=int(2 * args.window // args.dz + 1) * args.chunk)
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, RunSettings, bead_rois, camera_for, run_calibration

    done = [json.loads(line) for line in open(args.out)] if os.path.exists(args.out) else []
    key = lambda r: (r["acquisition"], r["stage"], r["form"], r.get("estimator"), r.get("blur"))  # noqa: E731
    have = {key(r) for r in done}
    sigma = float(np.mean(PIXEL)) / math.sqrt(12)                     # the pixel's own blur, both axes
    cache = {}

    def prepared(path):
        if path not in cache:
            _, _, files, _ = acquisition(path)
            rois = bead_rois(files, camera_for(files[0]), dual=False)
            keep, focus, cut = select(rois, acquisition(path)[0])
            shape = tifffile.TiffFile(files[0]).pages[0].shape[-2:]
            inner = central(rois, keep, args.centre, shape)
            cut = dict(cut, central=len(inner), image_px=list(shape))
            cache[path] = (files, subset(rois, inner, focus, args.window), cut)
        return cache[path]

    def fit(path, stage, form, estimator, magnitude, blur, extra):
        name, cond, _, bead_nm = acquisition(path)
        if (name, stage, form, estimator, blur) in have:
            return
        files, rois, cut = prepared(path)
        if cut["central"] < 2:
            print(name, "has", cut["central"], "central beads: skipped", flush=True)
            return
        t0 = time.perf_counter()
        optics = OpticsSettings(NA=GRID_NA, wavelength_nm=WAVELENGTH[cond["colour"]], n_immersion=N_IMM,
                                n_medium=N_MED[cond["medium"]], pixel_x_nm=PIXEL[0], pixel_y_nm=PIXEL[1])
        model = BeadSettings(zernike_order=ORDER, magnitude=magnitude, blur="fixed" if blur == "zero" else blur,
                             blur_nm={"fixed": sigma, "zero": 0.0}.get(blur, 30.0), bead_diameter_nm=bead_nm)
        run = run_calibration(files, False, optics, model, run=RunSettings(save_model=False), rois=rois,
                              decisions={i: "include" for i in range(len(rois.records))})
        row = dict(acquisition=name, **cond, stage=stage, form=form, estimator=estimator, blur=blur, beads=cut,
                   grid_na=GRID_NA, na=NA, window_nm=args.window, dz_nm=args.dz, centre=args.centre,
                   asym_modes=args.asym_modes, **extra)
        row.update(summary(run, cond))
        row.update(measure(run, args.z_range, centre_nm=[rois.records[i]["focus_nm"] for i in run.cal.beads.used]))
        row["minutes"] = round((time.perf_counter() - t0) / 60, 2)
        emit(row, args.out)
        done.append(json.loads(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))))
        have.add((name, stage, form, estimator, blur))
        print(name, stage, form, estimator or "", blur, f"{row['minutes']} min", flush=True)

    # 1 and 2: the references and the free minimal model, for every condition before any consensus
    for path in args.acquisitions:
        _, cond, _, bead_nm = acquisition(path)
        ring = cond["medium"] == "mismatch"
        fit(path, "zernike", "none", None, use_pupil("zernike"), "free", {})
        fit(path, "zernike fixed σ", "none", None, use_pupil("zernike"), "fixed", {})
        for blur in args.blurs:
            for form in (args.forms if ring else ["none"]):
                fit(path, "free", form, None, minimal(form, ring, args.asym_modes, {}, {}, bead_nm), blur, {})
    # 3: γ and the asymmetric modes held at the others' consensus
    for path in args.acquisitions:
        name, cond, _, bead_nm = acquisition(path)
        ring = cond["medium"] == "mismatch"
        for blur, form, estimator in itertools.product(args.blurs, args.consensus_forms, ESTIMATORS):
            pool = [r for r in done if r.get("blur") == blur]
            if not [r for r in pool if r["stage"] == "free" and r["acquisition"] != name]:
                continue                                        # nothing else to agree with
            hold, held = consensus(pool, form, estimator, name, args.asym_modes)
            fit(path, "consensus", form, estimator, minimal(form, ring, args.asym_modes, hold, held, bead_nm), blur,
                dict(held_gamma=round(hold["pupil.centre_g"], 4),
                     held_magnitude={j: round(v, 4) for j, v in held.items()}))


if __name__ == "__main__":
    main()
