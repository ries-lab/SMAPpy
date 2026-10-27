"""4Pi PSFs learnt from stacks at one fixed phase: does each mirror step of `4pi_40nm_bead` give the same PSF?

    python benchmarks/uipsf/real/zenodo_4pi_single_step.py [--device mps] [--out DIR]

A 4Pi whose deformable mirrors are not stepped takes each bead stack at one cavity phase, different from
bead to bead.  The example beads were taken at three mirror steps, so each step alone is such a data set:
18 beads × 21 slices at one phase each.  One model is learnt from each (`zenodo_4pi_beads.py`'s settings:
v1's ROIs and optics, Zernikes to n = 6, a free photon count per observation), and the three are compared
with each other and with the model learnt from all three steps (``--reference``, that script's output):
- **the parameters:** each pupil's Zernikes (in the labelling of the arms nearest the reference,
  `uipsf.model.fourpi.swapped`), the misalignment, the channels' offsets, coherences and balances;
- **the PSFs:** each channel's incoherent image I and interference Q (`workflows.fourpi.sample_iab`) over
  ±500 nm, as the relative difference ‖X₁ − X₂‖/‖X₂‖, Q up to one common phase (the beads' phases absorb
  it);
- **the localisations:** every bead, step and slice fitted by `Global4PiPSF` with each model's export: the
  phase and z scatter about a line through the stage, the mirror steps the phase measures (a model learnt
  at one step must measure the other two), and z and phase against the reference model's, per
  localisation.
Prints JSON lines; the calibrations are written to ``--out``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from smappy import fourpi
from smappy.psf import Global4PiPSF
from zenodo_4pi_beads import CROP, tracks, v1_batch

import uipsf
from uipsf.model import PerObservation
from uipsf.model.fourpi import FourPiModel, swapped
from uipsf.workflows.fourpi import calibration, learn_fourpi, sample_iab

Z_NM = np.linspace(-500, 500, 21)


def model_for(grid, shape, n_phases, n_max):
    return FourPiModel(grid, shape, n_phases=n_phases, n_max=n_max, steps_rad=[2, 0, -2][:n_phases],
                       intensity=PerObservation())


def labelled_like(model, shared, reference):
    """``shared`` or its arms swapped, whichever pupils are nearer the reference's."""
    other = swapped(model, shared)

    def distance(s):
        return sum(float((torch.as_tensor(s[n]) - torch.as_tensor(reference[n])).abs().sum())
                   for n in ("pupil.phase", "lo.pupil.phase"))
    return shared if distance(shared) <= distance(other) else other


def psf_difference(a, b):
    """(I, Q) relative differences of two (I, Q) samples, Q up to one common phase."""
    Ia, Qa = a
    Ib, Qb = b
    rot = np.vdot(Qa, Qb)
    rot = rot / abs(rot)
    return (float(np.linalg.norm(Ia - Ib) / np.linalg.norm(Ib)),
            float(np.linalg.norm(Qa * rot - Qb) / np.linalg.norm(Qb)))


def localisations(cal, data):
    C, E, M, J, Y, X = data.shape
    o = (Y - CROP) // 2
    rois = np.moveaxis(data[..., o:o + CROP, o:o + CROP], 0, 3).reshape(-1, C, CROP, CROP)
    fit = Global4PiPSF(cal)
    v = fit.unpack(fit.fit(rois.astype(np.float32)))
    return v["z_nm"], v["phase_rad"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--out", default=str(Path.home() / ".cache/uipsf/m5"))
    ap.add_argument("--reference", default=None, help="the three-step model's shared parameters (.json); "
                                                         "default <out>/v2_4pi_nmax<n>_obs.json")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    full, data = v1_batch()
    C, E, M, J, Y, X = data.shape
    optics = uipsf.Optics(NA=1.35, wavelength_nm=600, n_imm=1.406, n_med=1.406, pixel_size_nm=(129, 129))
    grid = uipsf.PupilGrid.build(optics, 64, device=a.device)
    ref_path = Path(a.reference) if a.reference else out / f"v2_4pi_nmax{a.n_max}_obs.json"
    reference = {k: torch.tensor(v, dtype=torch.float32, device=grid.rho.device)
                 for k, v in json.loads(ref_path.read_text()).items()}
    ref_model = model_for(grid, (Y, X), M, a.n_max)
    ref_cal = calibration((ref_model, reference), z_range_nm=600, dz_nm=25, roi_px=21)
    ref_psf = sample_iab(ref_model, reference, Z_NM)
    ref_z, ref_phase = localisations(ref_cal, data)
    models = {"all steps": (ref_model, reference, ref_cal)}
    for m in range(M):
        idx = torch.arange(m, J * M, M)
        batch = uipsf.Batch(data=full.data[:, :, idx], z_nm=full.z_nm[idx], s=full.s, roi_xy=full.roi_xy)
        model = model_for(grid, (Y, X), 1, a.n_max)
        t = time.time()
        fit = learn_fourpi(model, batch)
        seconds = time.time() - t
        shared = labelled_like(model, fit.shared(), reference)
        cal = calibration((model, shared), z_range_nm=600, dz_nm=25, roi_px=21)
        fourpi.save_fourpi_calibration(out / f"v2_4pi_step{m}.h5", cal, overwrite=True)
        (out / f"v2_4pi_step{m}.json").write_text(json.dumps({k: v.detach().cpu().double().numpy().tolist()
                                                              for k, v in shared.items()}))
        models[f"step {m}"] = (model, shared, cal)
        print(json.dumps({"step": m, "seconds": seconds, "iterations": len(fit.result.history),
                          "stop": fit.result.stop_reason,
                          "mean_deviance": float(fit.result.objective) * 2 / batch.data.numel(),
                          "start_deltas": fit.start["deltas"].tolist(), "start_period_nm": fit.start["period_nm"]}),
              flush=True)
    psfs = {name: sample_iab(model, shared, Z_NM) for name, (model, shared, _) in models.items()}
    names = list(models)
    for name in names:
        model, shared, cal = models[name]
        z, phase = localisations(cal, data)
        dz = z - ref_z
        dphase = np.angle(np.exp(1j * (phase - ref_phase)))
        dphase = np.angle(np.exp(1j * (dphase - np.angle(np.mean(np.exp(1j * dphase))))))
        row = {"model": name, "period_nm": cal.period_nm,
               "zernikes": {n: np.round(torch.as_tensor(shared[n]).cpu().numpy(), 3).tolist()
                            for n in ("pupil.phase", "lo.pupil.phase")},
               "zernike_rms_vs_all_rad": float(np.sqrt(np.mean(np.concatenate([
                   (torch.as_tensor(shared[n]) - torch.as_tensor(reference[n])).cpu().numpy() ** 2
                   for n in ("pupil.phase", "lo.pupil.phase")])))),
               "params": {k: v for k, v in cal.parameters.items() if k != "method"},
               "psf_vs_all": psf_difference(psfs[name], ref_psf),
               "psf_vs_steps": {other: psf_difference(psfs[name], psfs[other]) for other in names
                                if other.startswith("step") and other != name},
               "tracks": tracks(cal, data),
               "vs_all": {"z_offset_nm": float(np.median(dz)), "z_rms_nm": float(np.std(dz)),
                          "phase_rms_rad": float(np.std(dphase))}}
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
