"""Which of the pupil model's choices can get fixed defaults (plan §7.5 "Settings to test", §10 items 1-6).

    python benchmarks/uipsf/real/model_choices.py zenodo_1ch --device mps \\
        --out benchmarks/uipsf/results/model_choices.jsonl
    python benchmarks/uipsf/real/model_choices.py zenodo_2ch --device mps --variants base,order8,mag_fixed
    python benchmarks/uipsf/real/model_choices.py sim --device mps
    python benchmarks/uipsf/real/model_choices.py "/data/scope3/beads/*.tif" --NA 1.5 --wavelength 670 \\
        --n-imm 1.518 --n-med 1.33 [--dual] [--bead 40]      # a microscope of one's own: one argument

Every variant is one bead calibration of the same ROIs, cut once by SMAPpy's adapter
(`smappy.calibrate.pupil.bead_rois`, the window's own), with one choice changed from ``base`` (the
form's defaults: `BeadSettings()`, order 6, free magnitude, free blur, the z scale fixed at 1,
bleaching per bead, the first stage to order 4, slices thinned to 50 nm).  Each is judged by:

- **the training objective** (½ deviance + priors, nats) on *the same beads*: ``base`` runs the
  rejection, and every other variant gets its states as decisions (``include`` / ``exclude``), so a
  difference is the model's, not a different bead set.  More parameters always lower it; it is what
  a likelihood-ratio test reads (2Δ against χ² with the added parameters' count);
- **the held-out deviance**: 20 % of the beads, spread over the field (`rejection.held_out`, the same
  beads for every variant), localised on the learnt pupil with every slice
  (`diagnostics.heldout_report`, recomputed here on the unthinned stacks so that ``max_dz`` variants are
  judged on the same pixels); in nats over all their pixels, and its z bias (the slice fits against
  the stack fit) as the report gives it;
- **Spline 3D on the exported spline** (`pupil.fit_quality`): every slice of every bead fitted by
  SMAPpy's fitter with the spline the run would save; per slice the median over beads of the centred
  z error (each bead's own offset removed) is the calibration's z distortion.  Reported over the
  central slices (|z| ≤ 60 % of the range, `HeldOutReport.central`) for the held-out beads and the
  used ones: its RMS, its slope (a z scale: 0 when the spline's z is the stage's), and the spread
  over beads.
- on simulated beads, also the phase against the truth (rad RMS, the modes of n ≥ 2 but defocus).

The *photons per slice* variants (`PerSliceFactor`, `PerObservation`) also free the held-out beads'
photons per slice, so their held-out deviance is lower by construction: compare them on the z bias.

Variants named ``*_from_base`` (or with ``start_from``) start at that earlier variant's solution of the same
run (`start_values`), past stage 1: freeing terms in stages (plan §10, 4).  The drift variants are uipsf's
shared drift per slice and `BeadDrift`, a drift per bead and slice.  On a Mac shared with other work,
``--chunk 8 --mps-fraction 0.3`` under ``nice`` keeps it near 10 GB (2ch at order 8 needed 0.45).

Data: ``zenodo_1ch`` / ``zenodo_2ch`` (`$UIPSF_DATA`, `zenodo_camera.py`: 4.7 e⁻/ADU, 21 e⁻ read noise,
116 × 127 nm; NA 1.43, n_imm 1.516, beads in water 1.33 (confirmed for 1ch, plan §10, 2), 40 nm beads,
600 nm for 1ch and 680 nm for 2ch as v1 ran them); ``sim``: 36 astigmatic beads in three stacks drawn
by uipsf's model with a known pupil, a radial magnitude (Noll 4 = −0.15, so *fixed* is wrong), σ = 30
nm, 40 nm beads, Zenodo's camera and sampling; or a glob of the user's stacks with the optics given and
the camera from SMAPpy's database.  One JSON line per variant; ``--summary`` prints the table of a
results file.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import time
from dataclasses import replace

import numpy as np
import torch
from smappy.calibrate.core import CalibrationSettings
from smappy.calibrate.dual import DualColorSettings
from smappy.calibrate.pupil import BeadSettings, PupilRun, bead_rois, fit_quality
from zenodo_camera import CAMERA

import uipsf
from uipsf.diagnostics import heldout_report
from uipsf.model import BeadStackGeometry
from uipsf.model.intensity import Bleaching, PerObservation, PerSliceFactor
from uipsf.params import Gaussian, ParamSpec
from uipsf.workflows import bead_model, learn_beads
from uipsf.workflows.calibrate import start_values
from uipsf.workflows.rejection import RejectionSettings

HELDOUT = 0.2
ROI = 25


class BeadDrift(BeadStackGeometry):
    """A lateral drift per bead and slice, δxy[e, k], with uipsf's 5 nm prior: what any motion between
    frames could take from the blur.  uipsf's own drift is one per slice shared by every bead, but Zenodo's
    beads come from ~18 acquisitions with their own drifts; one per stack (1440 shared parameters for 1ch)
    is more than MPS takes in one Jacobian, and per bead is solved bead by bead.  It is more freedom than
    per stack, so it bounds what a drift can explain; held-out beads get their own too."""

    def specs(self, batch, start):
        K = batch.data.shape[2]
        return super().specs(batch, start) + [
            ParamSpec("emitter.drift", (K, 2), "emitter", init=0.0, prior=Gaussian(0.0, self.drift_prior_nm))]

    def __call__(self, p, batch):
        xyz = super().__call__(p, batch)
        d = p["emitter.drift"]                                                        # [E, K, 2]
        return xyz + torch.cat([d, torch.zeros_like(d[..., :1])], -1)


# name: what changes from base.  "settings": BeadSettings fields; "optics": Optics fields;
# "geometry" / "intensity": model components; "learn": learn_beads arguments.
VARIANTS = {
    "base": {},
    "order8": dict(settings=dict(zernike_order=8)),
    "mag_symmetric": dict(settings=dict(magnitude="symmetric")),
    "mag_fixed": dict(settings=dict(magnitude="fixed")),
    "order8_mag_symmetric": dict(settings=dict(zernike_order=8, magnitude="symmetric")),
    "blur_fixed_0": dict(settings=dict(blur="fixed", blur_nm=0.0)),
    "blur_fixed_20": dict(settings=dict(blur="fixed", blur_nm=20.0)),
    "blur_fixed_40": dict(settings=dict(blur="fixed", blur_nm=40.0)),
    "z_scale_free": dict(geometry=dict(z_scale=True)),
    "wl_plus10": dict(optics=dict(dwavelength=10.0)),
    "wl_plus20": dict(optics=dict(dwavelength=20.0)),
    "wl_plus30": dict(optics=dict(dwavelength=30.0)),
    "NA_minus007": dict(optics=dict(dNA=-0.007)),
    "NA_minus013": dict(optics=dict(dNA=-0.013)),
    "NA_minus020": dict(optics=dict(dNA=-0.020)),
    "photons_slice_factor": dict(intensity="slice_factor"),
    "photons_bleach2": dict(intensity="bleach2"),
    "photons_per_slice": dict(intensity="per_slice"),
    "order8_bleach2": dict(settings=dict(zernike_order=8), intensity="bleach2"),
    "first_order_none": dict(learn=dict(first_order=None)),
    "first_order_2": dict(learn=dict(first_order=2)),
    "max_dz_100": dict(learn=dict(max_dz_nm=100.0)),
    "max_dz_200": dict(learn=dict(max_dz_nm=200.0)),
    # does the magnitude we do not understand make the z scale?  (the z scale freed with each magnitude)
    "z_scale_free_mag_fixed": dict(settings=dict(magnitude="fixed"), geometry=dict(z_scale=True)),
    "z_scale_free_mag_symmetric": dict(settings=dict(magnitude="symmetric"), geometry=dict(z_scale=True)),
    "order8_z_scale_free": dict(settings=dict(zernike_order=8), geometry=dict(z_scale=True)),
    # a richer model started from base's fit (start_values), past stage 1: plan §10, 4's cheapest remedy
    "z_scale_free_from_base": dict(geometry=dict(z_scale=True), from_base=True),
    "order8_from_base": dict(settings=dict(zernike_order=8), from_base=True),
    "order8_bleach2_from_base": dict(settings=dict(zernike_order=8), intensity="bleach2", from_base=True),
    "order8_z_scale_free_from_order8": dict(settings=dict(zernike_order=8), geometry=dict(z_scale=True),
                                            start_from="order8"),
    "drift_shared_from_base": dict(geometry=dict(drift=True), from_base=True),
    "drift_bead_from_base": dict(geometry="bead", from_base=True),
    # does a drift take the blur?  uipsf's shared one, and one per bead
    "drift_shared": dict(geometry=dict(drift=True)),
    "drift_bead": dict(geometry="bead"),
    "drift_bead_blur_fixed_20": dict(geometry="bead", settings=dict(blur="fixed", blur_nm=20.0)),
}
DEFAULT_ORDER = list(VARIANTS)


# ------------------------------------------------------------------ data

def zenodo(name, data, device):
    files = sorted(glob.glob(os.path.join(data, f"{name}_40nm_bead", "*", "*.ome.tif")))
    if not files:
        raise SystemExit(f"no {name} stacks under {data} (scripts/uipsf/fetch_zenodo.py)")
    dual = name == "2ch"
    settings = DualColorSettings(roi_size=ROI) if dual else CalibrationSettings(roi_size=ROI)
    rois = bead_rois(files, CAMERA, dual=dual, settings=settings)
    wl = 680.0 if dual else 600.0
    optics = [uipsf.Optics(NA=1.43, wavelength_nm=wl, n_imm=1.516, n_med=1.33, pixel_size_nm=rois.pixel_size_nm)
              for _ in range(2 if dual else 1)]
    return rois, optics, 40.0, None


SIM_PHASE = {5: 0.8, 6: -0.2, 7: 0.15, 8: 0.1, 11: 0.3, 12: 0.1, 13: -0.08, 16: 0.06}   # rad
SIM_MAGNITUDE = {4: -0.15}
SIM_SIGMA_NM = 30.0


def simulated(seed=0, photons=20000.0, background=40.0):
    """Three ADU stacks of twelve beads each, drawn by uipsf's model with a known pupil; read back by the
    adapter, as `tests/uipsf_tests/test_pupil_spline.py` does at a smaller scale."""
    from smappy.calibrate.input import BeadStack

    from uipsf.simulate import parameters, template
    optics = uipsf.Optics(NA=1.43, wavelength_nm=600, n_imm=1.516, n_med=1.33, pixel_size_nm=(116.0, 127.0))
    truth_settings = BeadSettings(zernike_order=6, magnitude="symmetric", blur="fixed", blur_nm=SIM_SIGMA_NM,
                                  bead_diameter_nm=40.0)
    model = bead_model(optics, truth_settings, (ROI, ROI), z_max_nm=1100, device="cpu")
    pupil = model.psf.pupil
    phase = torch.zeros(len(pupil.phase_modes))
    for j, v in SIM_PHASE.items():
        phase[pupil.phase_modes.index(j)] = v
    magnitude = torch.zeros(len(pupil.magnitude_modes))
    for j, v in SIM_MAGNITUDE.items():
        magnitude[pupil.magnitude_modes.index(j)] = v
    shared = {"pupil.phase": phase, "pupil.magnitude": magnitude,
              "psf.sigma": torch.tensor([SIM_SIGMA_NM, SIM_SIGMA_NM])}
    rng = np.random.default_rng(seed)
    objective = np.arange(42) * 50.0                  # 42 slices: the first is dropped, as on Zenodo
    b0 = template(12, uipsf.Batch.z_from_objective(torch.tensor(objective - objective.mean(), dtype=torch.float32)),
                  (ROI, ROI), device="cpu")
    stacks, h = [], ROI // 2
    for s in range(3):
        image = np.full((len(objective), 160, 200), background)
        xyz = np.c_[rng.uniform(-50, 50, (12, 2)), rng.uniform(-100, 100, 12)]
        values = dict(shared, **{"emitter.xyz": torch.tensor(xyz, dtype=torch.float32),
                                 "emitter.photons": torch.tensor(photons * rng.uniform(0.7, 1.3, 12),
                                                                 dtype=torch.float32),
                                 "emitter.background": torch.zeros(12)})
        with torch.no_grad():
            mu = model(parameters(model, b0, values, device="cpu").physical(), b0)[:, 0].double().numpy()
        for i in range(12):
            x, y = 30 + 45 * (i % 4), 30 + 50 * (i // 4)
            image[:, y - h:y + h + 1, x - h:x + h + 1] += mu[i]
        electrons = rng.poisson(image) + rng.normal(0, CAMERA.read_noise_e, image.shape)
        adu = np.round(electrons / CAMERA.conversion + CAMERA.offset).clip(0, 65535).astype(np.uint16)
        stacks.append(BeadStack(adu, objective, f"sim{s}", {}, (0, 0, 200, 160)))
    rois = bead_rois(stacks, CAMERA, settings=CalibrationSettings(roi_size=ROI, min_distance_px=20))
    truth = dict(phase=SIM_PHASE, magnitude=SIM_MAGNITUDE, sigma_nm=SIM_SIGMA_NM)
    return rois, [replace(optics, pixel_size_nm=rois.pixel_size_nm)], 40.0, truth


def user(pattern, args):
    files = sorted(glob.glob(os.path.expanduser(pattern)))
    if not files:
        raise SystemExit(f"no files match {pattern}")
    if not (args.NA and args.wavelength):
        raise SystemExit("give --NA and --wavelength for a dataset of one's own")
    settings = DualColorSettings(roi_size=ROI) if args.dual else CalibrationSettings(roi_size=ROI)
    rois = bead_rois(files, None, dual=args.dual, settings=settings)
    waves = [args.wavelength, args.wavelength2 or args.wavelength][:2 if args.dual else 1]
    optics = [uipsf.Optics(NA=args.NA, wavelength_nm=w, n_imm=args.n_imm, n_med=args.n_med,
                           pixel_size_nm=rois.pixel_size_nm) for w in waves]
    return rois, optics, args.bead, None


# ------------------------------------------------------------------ one variant

def build(name, rois, optics, bead_nm, device):
    spec = VARIANTS[name]
    settings = BeadSettings(bead_diameter_nm=bead_nm, heldout=HELDOUT, **spec.get("settings", {}))
    o = spec.get("optics", {})
    optics = [replace(op, NA=op.NA + o.get("dNA", 0.0), wavelength_nm=op.wavelength_nm + o.get("dwavelength", 0.0))
              for op in optics]
    batch = rois.batch
    transforms = [None, rois.transform] if len(optics) == 2 else None
    model = bead_model(optics, settings, tuple(batch.data.shape[-2:]), transforms,
                       float(batch.z_nm.abs().max()), device=device)
    if spec.get("geometry") == "bead":
        model.geometry = BeadDrift()
    elif "geometry" in spec:
        model.geometry = BeadStackGeometry(**spec["geometry"])
    intensity = spec.get("intensity")
    if intensity == "slice_factor":
        model.intensity = PerSliceFactor()
    elif intensity == "bleach2":
        model.intensity = Bleaching(order=2)
    elif intensity == "per_slice":
        model.intensity = PerObservation()
    return model, settings, optics


def central_curve(fq, beads, key=""):
    """The z distortion over the central slices for ``beads``: per slice the median of the centred error."""
    if not len(beads):
        return {}
    bead_id, frame = fq["bead_id"], fq["frame"]
    nominal, err = fq["nominal_z_nm"], fq[f"centred_error_nm{key}"]
    K = int(frame.max()) + 1
    sel = np.isin(bead_id, beads)
    curve, spread, z = np.full(K, np.nan), np.full(K, np.nan), np.full(K, np.nan)
    for k in range(K):
        m = sel & (frame == k)
        e = err[m][np.isfinite(err[m])]
        z[k] = np.median(nominal[m])
        if len(e):
            curve[k] = np.median(e)
            spread[k] = 1.4826 * np.median(np.abs(e - np.median(e)))
    c = np.abs(z) <= 0.6 * np.nanmax(np.abs(z))
    ok = c & np.isfinite(curve)
    slope = float(np.polyfit(z[ok], curve[ok], 1)[0]) if ok.sum() > 2 else np.nan
    return dict(z_rms_nm=float(np.sqrt(np.nanmean(curve[c] ** 2))), slope=slope,
                spread_nm=float(np.nanmedian(spread[c])),
                failed=float(np.mean(~fq[f"fit_valid{key}"][sel] | fq[f"at_z_boundary{key}"][sel])),
                curve_nm=np.round(curve, 1))


def phase_error(cal, truth) -> dict:
    """Phase and magnitude against a simulation's truth, over every mode either has (n ≥ 2, no defocus)."""
    p, modes = cal.shared(), cal.model.psf.pupil.phase_modes
    phase = p["pupil.phase"].detach().cpu().double().numpy()
    err = [phase[i] - truth["phase"].get(j, 0.0) for i, j in enumerate(modes) if j > 4]
    out = dict(phase_rms_err_rad=float(np.sqrt(np.mean(np.square(err)))))
    mm = cal.model.psf.pupil.magnitude_modes
    mag = p.get("pupil.magnitude")
    got = {j: float(mag[i]) for i, j in enumerate(mm)} if mag is not None and len(mm) else {}
    out["magnitude_noll4"] = got.get(4, 0.0)
    return out


def run_variant(name, rois, optics, bead_nm, device, decisions, truth, fq_on=True, done=None, chunk=None):
    """``done``: the calibrations of this run so far, by variant, for a variant started from one of them.
    ``chunk``: emitters computed at once (memory only, the sums do not change)."""
    model, settings, optics_used = build(name, rois, optics, bead_nm, device)
    learn = dict(VARIANTS[name].get("learn", {}), chunk_size=chunk)
    start = "base" if VARIANTS[name].get("from_base") else VARIANTS[name].get("start_from")
    if start:
        if start not in (done or {}):
            raise ValueError(f"needs {start} earlier in the same run")
        learn["start"] = start_values(model, done[start])
    t0 = time.perf_counter()
    cal = learn_beads(model, rois.batch, rois.noise, decisions=decisions,
                      rejection=RejectionSettings(k=settings.rejection_k), heldout=settings.heldout,
                      geometry=rois.geometry, **learn)
    cal.options = settings
    seconds = time.perf_counter() - t0
    held = cal.beads.ids("held out")
    hb = rois.batch.subset(torch.tensor(held))
    # per-slice stage terms (fixed at 0 by default) belong to the fit's own, perhaps thinned, slices
    shapes = {s.name: tuple(s.shape) for s in model.specs(hb) if s.scope == "shared"}
    shared = {n: v for n, v in cal.shared().items() if shapes.get(n) == tuple(v.shape)}
    rep = heldout_report(model, shared, hb, rois.noise, held)
    mask = hb.mask
    n_pix = int(np.prod(hb.data.shape)) if mask is None else int(mask.expand_as(hb.data).sum())
    p = cal.result.params.physical()
    std = cal.result.std
    row = dict(variant=name, change=VARIANTS[name], seconds=round(seconds, 1),
               used=len(cal.beads.used), rejected=len(cal.beads.ids("rejected")) + len(cal.beads.ids("excluded")),
               held_out=len(held), stop=cal.result.stop_reason, iterations=cal.result.iterations,
               objective=float(cal.result.objective),
               free_shared=int(sum(int(np.prod(v.shape)) for n, v in std.items() if not n.startswith("emitter"))),
               heldout_deviance_nats=float(rep.deviance) * n_pix / 2, heldout_pixels=n_pix,
               heldout=rep.summary(), optics=[dict(NA=o.NA, wavelength_nm=o.wavelength_nm) for o in optics_used])
    for c in range(model.n_channels):
        pre = "" if c == 0 else f"ch{c}."
        row[f"{pre}sigma_nm"] = p[pre + "psf.sigma"].detach().cpu().tolist()
        modes = model.channels[c].psf.pupil.phase_modes
        row[f"{pre}phase"] = dict(zip(modes, p[pre + "pupil.phase"].detach().cpu().tolist(), strict=True))
        if pre + "pupil.phase" in std:
            row[f"{pre}phase_crlb"] = dict(zip(modes, std[pre + "pupil.phase"].cpu().tolist(), strict=True))
        mm = model.channels[c].psf.pupil.magnitude_modes
        if mm:
            row[f"{pre}magnitude"] = dict(zip(mm, p[pre + "pupil.magnitude"].detach().cpu().tolist(), strict=True))
    if "geometry.z_scale" in p:
        row["z_scale"] = float(p["geometry.z_scale"])
    drifts = {"drift": p["geometry.drift"] if model.geometry.drift else None,
              "bead_drift": p.get("emitter.drift")}
    for key, v in drifts.items():
        if v is not None:            # RMS over slices (and stacks) of the lateral drift, x and y
            v = v.detach().cpu().double()[..., :2]
            row[f"{key}_rms_nm"] = v.reshape(-1, 2).pow(2).mean(0).sqrt().tolist()
    if truth is not None:
        row.update(phase_error(cal, truth))
    if fq_on:
        t1 = time.perf_counter()
        try:
            fq = fit_quality(PupilRun(rois, cal, dual=len(optics) == 2))
            for c in range(len(optics)):
                key = "" if c == 0 else "_secondary"
                row[f"spline{key}_heldout"] = central_curve(fq, held, key)
                row[f"spline{key}_used"] = central_curve(fq, cal.beads.used, key)
        except Exception as error:          # the dual path is being built elsewhere: record, go on
            row["spline_error"] = repr(error)
        row["spline_seconds"] = round(time.perf_counter() - t1, 1)
    return row, cal


def emit(row, out):
    row = json.loads(json.dumps(row, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))
    print(json.dumps({k: v for k, v in row.items() if k not in ("phase", "phase_crlb", "ch1.phase",
                                                                  "ch1.phase_crlb")}), flush=True)
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")


# ------------------------------------------------------------------ the table

def phase_shift(row, base) -> tuple[float, float]:
    """RMS over the modes both have of phase − base's phase, in rad and in base's CRLBs, main channel:
    whether a choice moves the aberrations (a magnitude that absorbs phase, plan §10, 2)."""
    if not base or "phase_crlb" not in base:
        return np.nan, np.nan
    modes = [j for j in base["phase"] if j in row["phase"]]
    d = np.array([row["phase"][j] - base["phase"][j] for j in modes])
    crlb = np.array([base["phase_crlb"][j] for j in modes])
    return float(np.sqrt(np.mean(d ** 2))), float(np.sqrt(np.mean((d / crlb) ** 2)))


def summary(path, dataset=None):
    rows = [json.loads(line) for line in open(path)]
    rows = [r for r in rows if r.get("variant") and (dataset is None or r["dataset"] == dataset)]
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        group = [r for r in rows if r["dataset"] == ds]
        bases = {r.get("run", ""): r for r in group if r["variant"] == "base"}
        print(f"\n{ds}")
        print(f"{'variant':22s} {'Δobj':>8s} {'Δheld':>8s} {'held z':>7s} {'spl held':>8s} {'slope':>7s} "
              f"{'spl used':>8s} {'spread':>6s} {'σ nm':>11s} {'phase':>6s} {'dφ rad':>6s} {'dφ/σ':>6s} "
              f"{'s':>6s} {'sec':>5s}")
        for r in group:
            if "error" in r:
                print(f"{r['variant']:22s} failed: {r['error'][:90]}")
                continue
            base = bases.get(r.get("run", ""))
            # a thinned fit sees fewer pixels: its objective is not comparable, its held-out deviance is
            thinned = "max_dz_nm" in r["change"].get("learn", {})
            d_obj = r["objective"] - base["objective"] if base and not thinned else np.nan
            d_held = r["heldout_deviance_nats"] - base["heldout_deviance_nats"] if base else np.nan
            sh, su = r.get("spline_heldout", {}), r.get("spline_used", {})
            ph = r.get("phase_rms_err_rad")
            shift = phase_shift(r, base)
            print(f"{r['variant']:22s} {d_obj:8.1f} {d_held:8.1f} {r['heldout']['z_bias_rms_nm']:7.1f} "
                  f"{sh.get('z_rms_nm', np.nan):8.1f} {sh.get('slope', np.nan):7.3f} {su.get('z_rms_nm', np.nan):8.1f} "
                  f"{su.get('spread_nm', np.nan):6.1f} {str(np.round(r['sigma_nm'], 0).tolist()):>11s} "
                  f"{'' if ph is None else f'{ph:.3f}':>6s} {shift[0]:6.3f} {shift[1]:6.1f} "
                  f"{r.get('z_scale', 1.0):6.3f} "
                  f"{r['seconds']:5.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default="zenodo_1ch",
                    help="zenodo_1ch, zenodo_2ch, sim, or a glob of one's own bead stacks")
    ap.add_argument("--variants", default=",".join(DEFAULT_ORDER))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--data", default=os.environ.get("UIPSF_DATA", os.path.expanduser("~/.cache/uipsf/zenodo")))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0, help="sim: the draw")
    ap.add_argument("--no-spline", action="store_true", help="skip Spline 3D on the exported spline")
    ap.add_argument("--fresh-beads", action="store_true",
                    help="let every variant run its own rejection instead of base's bead states")
    ap.add_argument("--summary", default=None, help="print the table of a results file and stop")
    ap.add_argument("--chunk", type=int, default=None, help="emitters at once in the solver (memory)")
    ap.add_argument("--mps-fraction", type=float, default=None,
                    help="cap the MPS allocator at this fraction of the recommended memory, so a shared Mac "
                         "stays usable (torch.mps.set_per_process_memory_fraction)")
    for flag, kind in (("--NA", float), ("--wavelength", float), ("--wavelength2", float)):
        ap.add_argument(flag, type=kind, default=None)
    ap.add_argument("--n-imm", type=float, default=1.518)
    ap.add_argument("--n-med", type=float, default=1.33)
    ap.add_argument("--bead", type=float, default=40.0, help="bead diameter, nm")
    ap.add_argument("--dual", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(args.summary)
        return
    if args.mps_fraction and args.device == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_fraction)
    t0 = time.perf_counter()
    if args.dataset in ("zenodo_1ch", "zenodo_2ch"):
        rois, optics, bead_nm, truth = zenodo(args.dataset.split("_")[1], args.data, args.device)
        label = args.dataset
    elif args.dataset == "sim":
        rois, optics, bead_nm, truth = simulated(args.seed)
        label = f"sim{args.seed}"
    else:
        rois, optics, bead_nm, truth = user(args.dataset, args)
        label = args.dataset
    print(json.dumps(dict(dataset=label, beads=rois.batch.n_emitters, slices=int(rois.batch.data.shape[2]),
                          z_step_nm=float(rois.batch.z_step_nm), channels=len(optics),
                          seconds=round(time.perf_counter() - t0, 1))), flush=True)
    names = args.variants.split(",")
    run = time.strftime("%Y-%m-%dT%H:%M:%S")       # each invocation's variants are compared with its own base
    if "base" in names:
        names = ["base"] + [n for n in names if n != "base"]
    decisions, done = None, {}
    for name in names:
        try:
            row, cal = run_variant(name, rois, optics, bead_nm, args.device, decisions, truth, not args.no_spline,
                                   done, args.chunk)
        except Exception as error:          # a choice the workflow cannot run is a finding too
            emit(dict(variant=name, change=VARIANTS[name], error=repr(error), dataset=label, run=run), args.out)
            continue
        row.update(dataset=label, run=run, device=args.device, machine=platform.machine(), torch=torch.__version__)
        emit(row, args.out)
        done[name] = cal
        if name == "base" and not args.fresh_beads:
            decisions = {r.bead: ("include" if r.state == "used" else "exclude")
                         for r in cal.beads.rows if r.state != "held out"}


if __name__ == "__main__":
    main()
