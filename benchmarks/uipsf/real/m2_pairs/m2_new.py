"""M2, 12 Nov 2026: beads vs molecules on the coverslip (NBX4, AB), minimal pupil model.

    python m2_new.py beads|nbx4|ab OUTDIR [--files i,j]
"""
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

# the data are not in git: M2_DATA names the folder that holds 261112_new's sets
ROOT = os.environ.get("M2_DATA", os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Vienna/Data/"
                                                     "smappy test/261112_new"))
SETS = {
    "beads": "261112_MR_Intercalib2_Beads_1_700/*/*.ome.tif",
    "nbx4": "261112_MR_Intercalib2_NBX4_50ms_coverslip/*/*.ome.tif",
    "ab": "261112_MR_Intercalib2_AB_50ms_coverslip/*/*.ome.tif",
    "insitu": "261112_MR_Intercalib2_BGAAF647_50ms_30i_nucleus_1/*_MMStack_Pos0.ome.tif",
}
WAVELENGTH = float(os.environ.get("WL", 690.0))
PIXEL = (127.0, 116.0)      # iXon on M2's astigmatic path (docs/uipsf/microscopes.md)


def nm_of(pupil, values):
    from uipsf.optics.zernike import noll_to_nm
    return {f"Z{int(j)}{noll_to_nm(int(j))}": round(float(v) * WAVELENGTH / (2 * math.pi), 2)
            for j, v in zip(pupil.phase_modes, values, strict=True)}


def main():
    which, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(0.6)
    import functools

    import uipsf.workflows.calibrate as _c
    _c.learn_beads = functools.partial(_c.learn_beads, chunk_size=int(os.environ.get("CHUNK", 16)))
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, PSFModelSettings, RunSettings
    if os.environ.get("GAMMA") is not None and os.environ.get("GAMMA") != "":
        # γ held at a value (no setting holds it for a molecule stack): every pupil built holds it
        import uipsf.model.build as build
        _make = build.make_pupil

        def make_pupil(optics, spec, held=None, **kw):
            held = held or build.Held()
            if held.gamma is None:
                held = build.Held(phase=held.phase, magnitude=held.magnitude, gamma=float(os.environ["GAMMA"]),
                                  na=held.na, source="typed")
            return _make(optics, spec, held, **kw)
        build.make_pupil = make_pupil
    if os.environ.get("NMED_RANGE"):
        # the sample's index learnt within a range (a fixed, permeabilised cell in buffer: a little above water)
        import uipsf.model.psf as psf_mod
        from uipsf.params import Bounded, ParamSpec
        lo_n, hi_n = (float(v) for v in os.environ["NMED_RANGE"].split(","))
        psf_mod.n_med_spec = lambda n_typed: ParamSpec("optics.n_med", (), init=float(n_typed),
                                                       transform=Bounded(lo_n, hi_n))
    if os.environ.get("SOFT"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import soft_select  # noqa: F401  -- patches the selection
    if os.environ.get("H_START"):
        # the height's start (the molecule run drops a start model's geometry): put back where learning begins
        import uipsf.workflows.molecule_stack as ms
        _learn = ms.learn_molecule_stack

        def learn(model, batch, defocus_nm, noise=None, settings=None, start=None, **kw):
            start = dict(start or {})
            start["geometry.height"] = torch.tensor(float(os.environ["H_START"]))
            print("start:", {k: (v.reshape(-1)[:3].tolist() if hasattr(v, "reshape") else v) for k, v in start.items()},
                  flush=True)
            return _learn(model, batch, defocus_nm, noise, settings, start=start, **kw)
        ms.learn_molecule_stack = learn
    if os.environ.get("FLIP_Z"):
        # every stage z read as −z (z_direction.py, the stage's direction): the bead planes in reverse order,
        # the sweeps' z = +(stage z − focus); the stack learns what it would if ZPositionUm ran the other way
        import smappy.calibrate.input as cal_input
        import smappy.calibrate.molecule_stack as ms_mod
        _assemble, _stage_z = cal_input._assemble, ms_mod.stage_z_um

        def assemble(records, summary, source, dz_nm):
            flipped = []
            for image, axes, md in records:
                if md.get("ZPositionUm") is not None:
                    md = dict(md, ZPositionUm=-float(md["ZPositionUm"]))
                flipped.append((image, axes, md))
            return _assemble(flipped, summary, source, dz_nm)

        def stage_z_um(source, path):
            z = _stage_z(source, path)
            return None if z is None else -z
        cal_input._assemble = assemble
        ms_mod.stage_z_um = stage_z_um
    files = sorted(glob.glob(os.path.join(ROOT, SETS[which])))
    if "--files" in sys.argv:
        keep = [int(i) for i in sys.argv[sys.argv.index("--files") + 1].split(",")]
        files = [f for i, f in enumerate(files) if i + 1 in keep]
    picked = "_" + sys.argv[sys.argv.index("--files") + 1].replace(",", "") if "--files" in sys.argv else ""
    tag = which + os.environ.get("TAG", "") + picked
    print(tag, [os.path.basename(f) for f in files], flush=True)
    n_med = float(os.environ.get("NMED", 1.33))
    optics = OpticsSettings(NA=1.43, wavelength_nm=WAVELENGTH, n_immersion=1.516, n_medium=n_med,
                            pixel_x_nm=PIXEL[0], pixel_y_nm=PIXEL[1])
    env = os.environ.get
    # minimal magnitude, round blur: the defaults
    psf = PSFModelSettings(zernike_order=int(env("ORDER", 8)), learn_na=bool(env("LEARN_NA")),
                           learn_n_med=bool(env("LEARN_NMED")), blur="fixed" if env("BLUR_NM") else "round",
                           blur_nm=float(env("BLUR_NM", 20.0)))
    model = BeadSettings(psf=psf, fit_tilt=bool(env("TILT")))
    if env("REJECT_K"):
        model.rejection_k = float(env("REJECT_K"))        # large: every bead kept, the same set both ways
    save_to = os.path.join(out, f"{tag}_3dcal.h5")
    t0 = time.perf_counter()
    row = dict(set=tag, files=len(files), wavelength_nm=WAVELENGTH, pixel_nm=PIXEL)
    if which == "beads":
        from smappy.calibrate.pupil import fit_quality, run_calibration
        model.bead_diameter_nm = 100.0
        model.drift = os.environ.get("DRIFT", "linear")
        run = run_calibration(files, False, optics, model, run=RunSettings(save_model=True), save_to=save_to,
                              progress=lambda *a: print(*a, flush=True))
        cal = run.cal
        q = fit_quality(run)
        err = q["centred_error_nm"]
        used = np.isin(q["bead_id"], np.asarray(cal.beads.used))
        z = q["nominal_z_nm"]
        grid = np.arange(-1000, 1001, 100.0)
        bend = [float(np.nanmedian(err[used & (np.abs(z - g) < 50)])) if np.any(used & (np.abs(z - g) < 50))
                else None for g in grid]
        row.update(beads_used=len(cal.beads.used), beads_found=int(run.rois.batch.n_emitters)
                   if hasattr(run.rois.batch, "n_emitters") else None,
                   z_bend=dict(zip(grid.tolist(), bend, strict=False)))
    elif which == "insitu":
        from smappy.calibrate.insitu_run import InSituModelSettings, run_insitu
        start = os.environ.get("START", "")
        start_name = os.path.basename(start).split("_3dcal")[0] if start else "nostart"
        tag = "insitu_" + env("HOLD", "start") + env("TAG", "") + "_" + start_name
        save_to = os.path.join(out, f"{tag}_3dcal.h5")
        row["set"] = tag
        insitu = InSituModelSettings(hold=env("HOLD", "start"), focus_fit=env("FOCUS_FIT", "learnt"))
        res = run_insitu(files, optics, model, insitu,
                         run=RunSettings(save_model=True, start_from=start), save_to=save_to,
                         progress=lambda *a: print(*a, flush=True))
        cal = res.cal
        params = getattr(res.calibration, "parameters", {}) or {}
        row["coverslip_tilt_deg"] = params.get("coverslip_tilt_deg")
        row.update(focus_estimate_nm=res.focus_estimate, messages=res.messages + list(res.rois.messages),
                   molecules_used=int(len(res.rois.ids("used"))))
        for k in ("geometry.focus", "pupil.gamma", "psf.n_med", "optics.n_med", "n_med"):
            v = cal.result.params.physical().get(k)
            if v is not None:
                row[k] = np.round(v.detach().cpu().double().numpy().reshape(-1), 2).tolist()
    else:
        from smappy.calibrate.molecule_stack import MoleculeStackSettings
        from smappy.calibrate.molecule_stack_run import run_molecule_stack
        # HEIGHT unset: learnt (since 58f38c76 the setting's default holds 5 nm; these runs learnt it)
        stack = MoleculeStackSettings(molecules=int(env("MOLECULES", 4000)),
                                      illumination_px=float(env("RADIUS")) if env("RADIUS") else None,
                                      significance=float(env("SIG", 5.0)),
                                      height_nm=float(env("HEIGHT")) if env("HEIGHT") else None)
        res = run_molecule_stack(files, optics, model, stack,
                                 run=RunSettings(save_model=True, start_from=env("START", "")),
                                 save_to=save_to, progress=lambda *a: print(*a, flush=True))
        cal = res.cal
        ch = res.check
        row.update(molecules_used=int(len(res.used())), height_nm=res.height_nm(),
                   height_crlb=(float(cal.result.std["geometry.height"]) if "geometry.height" in cal.result.std
                                else None),
                   bend_rms_600=ch.bend_rms(600.0), bend_rms_1000=ch.bend_rms(1000.0), scatter=float(ch.scatter),
                   z_bend=dict(zip(ch.grid_nm.tolist(), [None if not np.isfinite(b) else round(float(b), 1)
                                                         for b in ch.bend_nm], strict=False)))
        try:
            np.savez(os.path.join(out, f"{tag}_zcheck.npz"), dz=ch.dz_nm if hasattr(ch, "dz_nm") else ch[0],
                     grid=ch.grid_nm, bend=ch.bend_nm)
        except Exception as e:          # noqa: BLE001
            print("zcheck save:", e)
    p, std = cal.result.params.physical(), cal.result.std
    pupil = cal.model.psf.pupil
    row["phase_nm"] = nm_of(pupil, p["pupil.phase"].detach().cpu().reshape(-1)[:len(pupil.phase_modes)])
    for k in ("pupil.gamma", "pupil.na", "psf.sigma"):
        if k in p:
            row[k] = np.round(p[k].detach().cpu().double().numpy().reshape(-1), 4).tolist()
            if k in std:
                row[k + "_crlb"] = np.round(std[k].detach().cpu().double().numpy().reshape(-1), 4).tolist()
    row["objective"] = float(cal.result.objective)          # ½ deviance + priors: FLIP_Z against not
    row["n_emitters"] = int(cal.result.params.n_emitters)
    row["flip_z"] = bool(os.environ.get("FLIP_Z"))
    row["params"] = sorted(p.keys())
    row["minutes"] = round((time.perf_counter() - t0) / 60, 2)
    row["saved"] = save_to
    print(json.dumps(row, default=str), flush=True)
    with open(os.path.join(out, "summary.jsonl"), "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


if __name__ == "__main__":
    main()
