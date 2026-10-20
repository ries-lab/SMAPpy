"""How much a better start could gain a bead calibration: wrong minima avoided, and time (plan §10, 5).

    python benchmarks/uipsf/real/starts.py sim --seeds 0-4 --device mps --out benchmarks/uipsf/results/starts.jsonl
    python benchmarks/uipsf/real/starts.py zenodo_1ch zenodo_2ch --device mps --out ...
    python benchmarks/uipsf/real/starts.py m2 --device mps --out ...          # the lab's M2, one from another
    python benchmarks/uipsf/real/starts.py sim2 --seeds 0-4 --out ...         # two simulated channels, CPU
    python benchmarks/uipsf/real/starts.py --summary benchmarks/uipsf/results/starts.jsonl

Before building a start from the data (an astigmatic Gaussian fit, a transform from per-slice
positions) or from a previous calibration, this asks what an *ideal* start buys, by giving the fit
pieces of its own answer: the cold fit (``cold``, today's start: the pupil at zero, each bead from
its brightest slice, the transform from the pairing's centroids) is run first, then each start is
built from it or from the truth:

- ``final``: every shared value of the cold fit (`start_values`, past stage 1, the beads localised
  on it): the best a previous model of the same microscope can do;
- ``astig``: only the cold fit's (or the truth's) Noll 5 and 6, past stage 1 (``astig_skip``) or
  into stage 1 (``astig_stage1``): the best an astigmatic Gaussian start can do for the pupil;
- ``transform``: two channels, the pupils at zero, only the final (or true) transform, the default
  stages: the best a better transform start can do; ``shift<px>`` the pairing's transform moved by
  that many pixels in x and y, the other direction;
- ``m2``: each of the lab's M2 acquisitions started from another's calibration, replacing stage 1
  (``prev:<name>``) or into it (``prev_stage1:<name>``, ``--prev stage1``): the realistic ``final``; the
  665LP acquisition against the 676/37 ones is the filter change;
- ``gauss``, ``gauss_pos``, ``gauss_T``, ``gauss_fit``: the start from elliptical Gaussians
  (`uipsf.model.astig.gaussian_start`, `gauss_calibrate`): all of it, the beads and transform only, the
  transform only, all of it with the angled Gaussian.

Each start runs on the cold fit's bead states (decisions), so the objectives compare.  Reported: the
objective's difference to the best of the dataset's starts (a wrong minimum shows as tens of nats or
more), the seconds (and those before stage 2: stage 1 or the localisation), the iterations per stage,
the phase against the cold fit's (and the truth's) and, for two channels, the transform's error where
the beads are.  ``sim``: `model_choices.simulated`, through SMAPpy's adapter; ``sim2``:
`uipsf_tests.conftest.two_channel_beads`, biplane and ratiometric (mirrored), with the transform from
the centroids as `test_multichannel.py` makes it.  Each start's model is built anew; on a shared Mac
run under ``nice`` with ``--mps-fraction 0.3``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tests"))

import uipsf  # noqa: E402
from uipsf.device import on_model_device  # noqa: E402
from uipsf.model.channels import map_points, unpack_transform  # noqa: E402
from uipsf.noise import ShiftedPoisson  # noqa: E402
from uipsf.workflows import bead_model, learn_beads  # noqa: E402
from uipsf.workflows.beads import MAX_DZ_NM  # noqa: E402
from uipsf.workflows.calibrate import start_values  # noqa: E402

LAB = os.path.expanduser("~/.cache/uipsf/lab_beads/M2")
ASTIG = (5, 6)


# ------------------------------------------------------------------ one calibration, timed by stage

class Clock:
    """The time of each solver iteration, by stage: the seconds before stage 2 are stage 1 or the
    localisation on a start pupil."""

    def __init__(self):
        self.t0, self.first2 = time.perf_counter(), None

    def __call__(self, entry):
        if entry.get("stage") == 2 and self.first2 is None:
            self.first2 = time.perf_counter() - self.t0


def calibrate(model, batch, noise, decisions=None, start=None, stage1=None, chunk=None, geometry=None):
    """`learn_beads` from ``start`` (shared values: stage 1 skipped) or with ``stage1`` (shared values
    stage 1 starts from, `learn_beads(start_stage1=True)`).  (Calibration, seconds, seconds before
    stage 2)."""
    clock = Clock()
    noise = noise if noise is not None else ShiftedPoisson()
    cal = learn_beads(model, batch, noise, decisions=decisions, start=start if stage1 is None else stage1,
                      start_stage1=stage1 is not None, progress=clock, chunk_size=chunk, geometry=geometry)
    return cal, time.perf_counter() - clock.t0, clock.first2


def gauss_calibrate(start, model, batch, noise, decisions, **kw):
    """A calibration from the Gaussian start (`uipsf.model.astig.gaussian_start`): ``gauss`` all of it
    (the beads' z and x, y as `Batch.start`, the astigmatism into stage 1, the transform), ``gauss_pos``
    the beads and the transform only (the pupil at zero), ``gauss_T`` the transform only, ``gauss_fit``
    as ``gauss`` with the angled Gaussian."""
    from dataclasses import replace

    from uipsf.model.astig import gaussian_start
    method = "fit" if start == "gauss_fit" else "gauss_xy"
    t0 = time.perf_counter()
    g = gaussian_start(model, on_model_device(batch, model).thinned(MAX_DZ_NM), method=method)
    t_start = time.perf_counter() - t0
    if model.n_channels > 1:
        for c, H in g["transform"].items():
            model.channels[c].transform = H
    if start != "gauss_T":
        batch = replace(batch, start=g["emitters"])
    stage1 = g["shared"] if start in ("gauss", "gauss_fit") else None
    cal, s, b2 = calibrate(model, batch, noise, decisions, stage1=stage1, **kw)
    cal.gauss = g
    cal.gauss_seconds = t_start
    return cal, s + t_start, (b2 or 0) + t_start


def stages(cal) -> dict:
    h = cal.result.history
    out = {f"iter{s}": sum(1 for e in h if e.get("stage") == s and e["iteration"] > 0) for s in (1, 2)}
    out["iter1"] += getattr(cal, "pre_iter1", 0)
    return out


def phase_of(cal, c=0) -> dict:
    pre = "" if c == 0 else f"ch{c}."
    modes = cal.model.channels[c].psf.pupil.phase_modes
    v = cal.shared()[pre + "pupil.phase"].detach().cpu().double().numpy()
    return dict(zip(modes, v, strict=True))


def phase_rms(a: dict, b: dict) -> float:
    modes = [j for j in a if j > 4]
    return float(np.sqrt(np.mean([(a[j] - b.get(j, 0.0)) ** 2 for j in modes])))


def astig_values(model, phases: list[dict]) -> dict:
    """Shared start values: each channel's phase zero but Noll 5 and 6, from ``phases`` (one per
    channel); nothing else, so the transform, share and offset keep the model's own start."""
    out = {}
    for c, ch in enumerate(model.channels):
        pre = "" if c == 0 else f"ch{c}."
        modes = ch.psf.pupil.phase_modes
        out[pre + "pupil.phase"] = torch.tensor([phases[c].get(j, 0.0) if j in ASTIG else 0.0 for j in modes])
    return out


def transform_error_px(H, H_ref, main_xy) -> float:
    """The largest distance (px) between where H and H_ref put the partners of the main ROIs' centres."""
    a = map_points(torch.linalg.inv(torch.as_tensor(H, dtype=torch.float64)), main_xy)
    b = map_points(torch.linalg.inv(torch.as_tensor(H_ref, dtype=torch.float64)), main_xy)
    return float((a - b).norm(dim=-1).max())


def shifted(H, px):
    return np.array([[1.0, 0, px], [0, 1.0, -px], [0, 0, 1.0]]) @ np.asarray(H, float)


def h_of(cal):
    return unpack_transform(cal.shared()["ch1.transform"].detach().cpu().double()).numpy()


# ------------------------------------------------------------------ datasets through SMAPpy's adapter

def adapter_dataset(name, seed, args):
    import model_choices as mc
    if name == "sim":
        rois, optics, bead, truth = mc.simulated(seed)
        return rois, optics, bead, truth, f"sim{seed}"
    rois, optics, bead, truth = mc.zenodo(name.split("_")[1], args.data, args.device)
    return rois, optics, bead, truth, name


def settings_for(bead_nm, order=6):
    from smappy.calibrate.pupil import BeadSettings
    return BeadSettings(bead_diameter_nm=bead_nm, zernike_order=order)


def run_adapter(name, seed, args, emit):
    rois, optics, bead, truth, label = adapter_dataset(name, seed, args)
    dual = len(optics) == 2
    settings = settings_for(bead, args.order)
    batch, noise = rois.batch, rois.noise
    z_max = float(batch.z_nm.abs().max())
    roi = tuple(batch.data.shape[-2:])
    main_xy = batch.roi_xy[:, 0].double() if dual else None

    def model(transform=None):
        return bead_model(optics, settings, roi, [None, transform] if dual else None, z_max, device=args.device)

    rows = []

    def record(start, cal, seconds, before2, **extra):
        row = dict(dataset=label, start=start, seconds=round(seconds, 1),
                   before_stage2_s=None if before2 is None else round(before2, 1),
                   objective=float(cal.result.objective), used=len(cal.beads.used),
                   stop=cal.result.stop_reason, **stages(cal), **extra)
        z = cal.result.params.physical()["emitter.xyz"][:, 2].detach().cpu().double()
        row["bead_z_median_nm"] = float(z.median())
        if rows:
            z0 = rows[0][1].result.params.physical()["emitter.xyz"][:, 2].detach().cpu().double()
            row["bead_z_vs_cold_nm"] = [float((z - z0).median()), float((z - z0).std())]
            row["phase_vs_cold_rad"] = phase_rms(phase_of(cal), phase_of(rows[0][1]))
        if truth is not None:
            row["phase_vs_truth_rad"] = phase_rms(phase_of(cal), truth["phase"])
        if dual:
            row["ch1_phase_vs_cold_rad"] = (phase_rms(phase_of(cal, 1), phase_of(rows[0][1], 1)) if rows else 0.0)
            row["transform_vs_cold_px"] = transform_error_px(h_of(cal), h_of(rows[0][1]), main_xy) if rows else 0.0
        if getattr(cal, "gauss", None) is not None:
            row["gauss_seconds"] = round(cal.gauss_seconds, 1)
            row["gauss_astig"] = [[round(r["a5"], 3), round(r["a6"], 3)] for r in cal.gauss["per_channel"]]
            if dual:
                row["gauss_transform_vs_cold_px"] = transform_error_px(cal.gauss["transform"][1], h_of(rows[0][1]),
                                                                       main_xy)
        rows.append((row, cal))
        emit(row)

    cal, s, b2 = calibrate(model(rois.transform if dual else None), batch, noise, chunk=args.chunk,
                           geometry=rois.geometry)
    record("cold", cal, s, b2)
    cold = cal
    decisions = {r.bead: ("include" if r.state == "used" else "exclude") for r in cold.beads.rows}
    for start in args.starts:
        m = model(rois.transform if dual else None)
        try:
            if start == "final":
                cal, s, b2 = calibrate(m, batch, noise, decisions, start=start_values(m, cold), chunk=args.chunk,
                                       geometry=rois.geometry)
            elif start.startswith("astig"):
                ph = [phase_of(cold, c) for c in range(m.n_channels)]
                if start.endswith("truth"):
                    if truth is None:
                        continue
                    ph = [truth["phase"]]
                v = astig_values(m, ph)
                kw = dict(stage1=v) if "stage1" in start else dict(start=v)
                cal, s, b2 = calibrate(m, batch, noise, decisions, chunk=args.chunk, geometry=rois.geometry, **kw)
            elif start == "transform" and dual:
                m = model(h_of(cold))
                cal, s, b2 = calibrate(m, batch, noise, decisions, chunk=args.chunk, geometry=rois.geometry)
            elif start.startswith("gauss"):
                m = model(rois.transform if dual else None)
                cal, s, b2 = gauss_calibrate(start, m, batch, noise, decisions, chunk=args.chunk,
                                             geometry=rois.geometry)
            elif start.startswith("shift") and dual:
                m = model(shifted(rois.transform, float(start[5:])))
                cal, s, b2 = calibrate(m, batch, noise, decisions, chunk=args.chunk, geometry=rois.geometry)
            else:
                continue
        except Exception as error:                  # a start the fit cannot take is a finding too
            emit(dict(dataset=label, start=start, error=repr(error)))
            continue
        record(start, cal, s, b2)
    return rows


# ------------------------------------------------------------------ the lab's M2: one acquisition from another

def m2_acquisitions():
    return sorted(d for d in glob.glob(os.path.join(LAB, "*")) if os.path.isdir(d))


def run_m2(args, emit):
    """Each M2 acquisition cold, then from each other one's calibration.  The optics are lab_beads.py's
    (676 nm, NA 1.43, 100 nm beads), the pixel 116 × 127 nm before July 2026 and 127 × 116 after."""
    from smappy.calibrate.pupil import OpticsSettings, bead_rois
    acqs = m2_acquisitions()
    if args.acquisitions:
        acqs = [a for a in acqs if any(k in os.path.basename(a) for k in args.acquisitions)]
    cold = {}
    for acq in acqs:
        name = os.path.basename(acq)
        pixel = (116.0, 127.0) if name.startswith("2606") else (127.0, 116.0)
        files = sorted(glob.glob(os.path.join(acq, "*", "*.ome.tif")))
        rois = bead_rois(files, None)
        optics = OpticsSettings(NA=1.43, wavelength_nm=676.0, n_immersion=1.516, n_medium=1.33,
                                pixel_x_nm=pixel[0], pixel_y_nm=pixel[1]).optics(rois.camera, 1)
        settings = settings_for(100.0, args.order)
        roi = tuple(rois.batch.data.shape[-2:])
        z_max = float(rois.batch.z_nm.abs().max())
        m = bead_model(optics, settings, roi, None, z_max, device=args.device)
        cal, s, b2 = calibrate(m, rois.batch, rois.noise, chunk=args.chunk)
        emit(dict(dataset=name, start="cold", seconds=round(s, 1), before_stage2_s=b2 and round(b2, 1),
                  objective=float(cal.result.objective), used=len(cal.beads.used), stop=cal.result.stop_reason,
                  **stages(cal)))
        cold[name] = (cal, rois, optics, settings)
    for name, (cal0, rois, optics, settings) in cold.items():
        decisions = {r.bead: ("include" if r.state == "used" else "exclude") for r in cal0.beads.rows}
        roi = tuple(rois.batch.data.shape[-2:])
        z_max = float(rois.batch.z_nm.abs().max())
        for other, (prev, *_) in cold.items():
            if other == name:
                continue
            for how in args.prev:
                m = bead_model(optics, settings, roi, None, z_max, device=args.device)
                # "skip": the previous model replaces stage 1 (learn_beads' start); "stage1": it starts stage 1,
                # whose order-4 truncation keeps its low modes and frees them with the beads
                kw = dict(start=start_values(m, prev)) if how == "skip" else dict(stage1=start_values(m, prev))
                cal, s, b2 = calibrate(m, rois.batch, rois.noise, decisions, chunk=args.chunk, **kw)
                tag = "prev" if how == "skip" else "prev_stage1"
                emit(dict(dataset=name, start=f"{tag}:{other}", seconds=round(s, 1),
                          before_stage2_s=b2 and round(b2, 1), objective=float(cal.result.objective),
                          used=len(cal.beads.used), stop=cal.result.stop_reason, **stages(cal),
                          phase_vs_cold_rad=phase_rms(phase_of(cal), phase_of(cal0)),
                          prev_vs_cold_rad=phase_rms(phase_of(prev), phase_of(cal0))))


# ------------------------------------------------------------------ two simulated channels

def run_sim2(seed, args, emit):
    from uipsf_tests.conftest import two_channel_beads
    from uipsf_tests.test_multichannel import _from_centroids
    optics = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))
    for kind, mirrored in (("biplane", False), ("ratiometric", True)):
        def make(kind=kind, mirrored=mirrored):
            m, truth, batch = two_channel_beads(optics, n=12, kind=kind, dtype=torch.float32, P=48, roi=(17, 17),
                                                K=21, z_range=1000.0, n_max=6, photons=args.photons,
                                                mirrored=mirrored, seed=seed)
            m.channels[1].share = m.channels[1].dz_nm = None
            return m, truth, batch
        m, truth, batch = make()
        tp = truth.physical()
        H_true = unpack_transform(tp["ch1.transform"].double()).numpy()
        H_cent = _from_centroids(batch, optics.pixel_size_nm)
        main = batch.roi_xy[:, 0].double()
        truth_phase = [dict(zip(ch.psf.pupil.phase_modes, tp[("" if c == 0 else "ch1.") + "pupil.phase"].tolist(),
                                strict=True)) for c, ch in enumerate(m.channels)]
        label = f"sim2_{kind}{seed}"
        starts = {"cold": dict(H=H_cent), "truth_transform": dict(H=H_true),
                  "truth_all": dict(H=H_cent, start=True),
                  "astig_truth_stage1": dict(H=H_cent, astig=True)}
        for px in args.shifts:
            starts[f"shift{px:g}"] = dict(H=shifted(H_cent, px))
        for g in ("gauss", "gauss_T", "gauss_fit"):
            starts[g] = dict(H=H_cent, gauss=g)
        if args.sim2_starts:
            starts = {k: v for k, v in starts.items() if k == "cold" or k in args.sim2_starts}
        cold = None
        for name, how in starts.items():
            m, _, _ = make()
            m.channels[1].transform = how["H"]
            kw = {}
            if how.get("start"):
                kw["start"] = {n: v.detach().cpu() for n, v in tp.items() if truth.specs[n].scope == "shared"}
            if how.get("astig"):
                kw["stage1"] = astig_values(m, truth_phase)
            decisions = None if cold is None else {
                r.bead: ("include" if r.state == "used" else "exclude") for r in cold.beads.rows}
            if how.get("gauss"):
                cal, s, b2 = gauss_calibrate(how["gauss"], m, batch, None, decisions)
                how = dict(how, H=cal.gauss["transform"][1])
            else:
                cal, s, b2 = calibrate(m, batch, None, decisions, **kw)
            if cold is None:
                cold = cal
            fp = cal.result.params.physical()
            emit(dict(dataset=label, start=name, seconds=round(s, 1), before_stage2_s=b2 and round(b2, 1),
                      objective=float(cal.result.objective), used=len(cal.beads.used),
                      stop=cal.result.stop_reason, **stages(cal),
                      start_transform_px=transform_error_px(how["H"], H_true, main),
                      transform_px=transform_error_px(h_of(cal), H_true, main),
                      phase_vs_truth_rad=phase_rms(phase_of(cal), truth_phase[0]),
                      ch1_phase_vs_truth_rad=phase_rms(phase_of(cal, 1), truth_phase[1]),
                      dz_err_nm=float(fp["ch1.dz"] - tp["ch1.dz"]),
                      share_err=float(fp["ch1.share"] - tp["ch1.share"])))


# ------------------------------------------------------------------ the table

def summary(path):
    rows = [json.loads(line) for line in open(path)]
    by = {}
    for r in rows:
        by.setdefault(r["dataset"], []).append(r)
    print(f"{'dataset':28s} {'start':26s} {'Δobj':>9s} {'sec':>6s} {'pre2':>5s} {'it1':>4s} {'it2':>4s} "
          f"{'φ cold':>7s} {'φ true':>7s} {'H px':>6s}")
    for ds, group in by.items():
        ok = [r for r in group if "objective" in r]
        best = min(r["objective"] for r in ok) if ok else np.nan
        for r in group:
            if "error" in r:
                print(f"{ds:28s} {r['start']:26s} failed: {r['error'][:60]}")
                continue
            H = r.get("transform_px", r.get("transform_vs_cold_px"))
            print(f"{ds:28s} {r['start']:26s} {r['objective'] - best:9.1f} {r['seconds']:6.1f} "
                  f"{r.get('before_stage2_s') or 0:5.1f} {r.get('iter1', 0):4d} {r.get('iter2', 0):4d} "
                  f"{r.get('phase_vs_cold_rad', np.nan):7.3f} {r.get('phase_vs_truth_rad', np.nan):7.3f} "
                  f"{np.nan if H is None else H:6.2f}")


def seeds(text):
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in text.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", default=["sim"])
    ap.add_argument("--starts", default="final,astig_skip,astig_stage1,astig_stage1_truth,transform,shift1.5,shift3")
    ap.add_argument("--seeds", type=seeds, default=[0])
    ap.add_argument("--shifts", type=lambda t: [float(v) for v in t.split(",")], default=[1.5, 3.0])
    ap.add_argument("--sim2-starts", type=lambda t: t.split(","), default=None,
                    help="sim2: only these starts (and cold)")
    ap.add_argument("--photons", type=float, default=20000.0, help="sim2: photons per bead")
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--prev", type=lambda t: t.split(","), default=["skip"],
                    help="m2: how a previous model starts: skip (stage 1), stage1 (into it), or both")
    ap.add_argument("--acquisitions", nargs="*", default=None, help="m2: names (parts of) to use")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--data", default=os.environ.get("UIPSF_DATA", os.path.expanduser("~/.cache/uipsf/zenodo")))
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--mps-fraction", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()
    if args.summary:
        summary(args.summary)
        return
    args.starts = args.starts.split(",")
    if args.mps_fraction and args.device == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_fraction)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    def emit(row):
        row = json.loads(json.dumps(dict(row, run=stamp, device=args.device),
                                    default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))
        print(json.dumps(row), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")

    for name in args.datasets:
        if name == "m2":
            run_m2(args, emit)
        elif name == "sim2":
            for s in args.seeds:
                run_sim2(s, args, emit)
        else:
            for s in (args.seeds if name == "sim" else [0]):
                run_adapter(name, s, args, emit)


if __name__ == "__main__":
    main()
