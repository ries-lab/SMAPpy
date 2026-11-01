"""Field dependence on Zenodo `1ch_FD_bead`: a pupil learnt with and without *field dependence*, and what it
changes for localisation.  `docs/uipsf/field-dependence.md`, §9.

    python scripts/uipsf/fetch_zenodo.py 1ch_FD_bead/ results/1ch_FD_bead/
    python benchmarks/uipsf/real/zenodo_fd_beads.py detect
    python benchmarks/uipsf/real/zenodo_fd_beads.py fit constant --beads 60
    python benchmarks/uipsf/real/zenodo_fd_beads.py fit field --beads 60
    python benchmarks/uipsf/real/zenodo_fd_beads.py localise

Two stacks of astigmatic beads over a 1608 × 1608 Prime 95B chip, 10 nm steps over 2 µm.  The optics are v1's,
from its result file's ``params``: NA 1.5, 680 nm, immersion 1.515, medium 1.406, 110 nm pixels.  The camera's
conversion is not in the files (Micro-Manager records the offset, 100, and a 62 000 e⁻ full well): 0.47 e⁻/ADU,
v1's.  The residuals' photon transfer (`noise_transfer`) does not pin it: 2.4 at 1 e⁻/ADU without the field, then
1.08 and 0.62 with it on two sets of beads.
Read noise 1.6 e⁻.

*detect* finds the beads once (the PSF calibration window's `bead_rois`) and keeps them in ``--work``.  *fit*
learns from ``--beads`` of them, spread over the chip (the window's `calibrate_beads` path, run through
`learn_beads` in chunks of 8 beads: 60 at once needed more than 15 GB), and writes a JSON line: beads used,
time, deviance, blur, the field report and the spline grid.  *localise* fits every slice within ±600 nm of
every bead (the ones not learnt from included) with the field's node splines, the field model's centre spline
alone and the model learnt without the field, and prints the z error per bead (rms over the slices, each bead's
own height removed) by distance from the chip's centre.  Results of the November 2026 run:
`results/zenodo_fd.json`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import time

import numpy as np

DATA = os.environ.get("UIPSF_DATA", os.path.expanduser("~/.cache/uipsf/zenodo"))
CONVERSION, OFFSET, READ_NOISE_E, PIXEL_UM = 0.47, 100.0, 1.6, 0.110
OPTICS = dict(NA=1.5, wavelength_nm=680.0, n_imm=1.515, n_med=1.406, pixel_size_nm=(110.0, 110.0))
CHIP_CENTRE = (804.0, 804.0)


def detect(work):
    from smappy.calibrate.core import CalibrationSettings
    from smappy.calibrate.pupil import bead_rois
    from smappy.metadata import CameraMetadata
    paths = sorted(glob.glob(os.path.join(DATA, "1ch_FD_bead", "*", "*.ome.tif")))
    camera = CameraMetadata(conversion=CONVERSION, offset=OFFSET, pixelsize_um=PIXEL_UM, em_on=False,
                            read_noise_e=READ_NOISE_E, roi=(0, 0, 1608, 1608))
    t = time.time()
    rois = bead_rois(paths, camera, settings=CalibrationSettings(dz_nm=10.0))
    print(f"{rois.batch.n_emitters} beads in {time.time() - t:.0f} s; {rois.messages}")
    with open(os.path.join(work, "rois.pkl"), "wb") as f:
        pickle.dump(rois, f)


def spread(xy, n):
    """``n`` beads spread over the chip: the one nearest each point of a regular grid over the beads' span.
    (Every k-th bead in order of x, the first version, left out the chip's right side.)"""
    side = int(np.ceil(np.sqrt(n)))
    lo, hi = xy.min(0), xy.max(0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], side), np.linspace(lo[1], hi[1], side))
    chosen = []
    for p in np.c_[gx.ravel(), gy.ravel()]:
        d = np.hypot(*(xy - p).T)
        d[chosen] = np.inf
        chosen.append(int(np.argmin(d)))
    return np.sort(chosen[:n])


def fit(work, kind, n_beads):
    import torch

    import uipsf
    from smappy.calibrate.pupil import BeadSettings, field_centre, spline_calibration
    from uipsf.diagnostics import field_report, noise_transfer
    from uipsf.workflows import learn_beads
    from uipsf.workflows.beads import MAX_DZ_NM
    from uipsf.workflows.calibrate import bead_model
    from uipsf.workflows.rejection import RejectionSettings
    with open(os.path.join(work, "rois.pkl"), "rb") as f:
        rois = pickle.load(f)
    batch = rois.batch
    if n_beads:
        batch = batch.subset(torch.as_tensor(spread(batch.field_xy.numpy(), n_beads)))
    settings = BeadSettings(field_dependence=kind == "field")
    model = bead_model([uipsf.Optics(**OPTICS)], settings, tuple(batch.data.shape[-2:]), None,
                       float(batch.z_nm.abs().max()), field_centre_px=field_centre(batch))
    t0 = time.time()
    cal = learn_beads(model, batch, rois.noise, rejection=RejectionSettings(k=settings.rejection_k), chunk_size=8,
                      progress=lambda e: print(f"  [{time.time() - t0:5.0f} s] {e.get('message', '')}", flush=True)
                      if isinstance(e, dict) and e.get("message") else None)
    cal.options = settings
    minutes = (time.time() - t0) / 60
    sub = batch.subset(torch.as_tensor(cal.beads.used)).thinned(MAX_DZ_NM)
    phys = cal.result.params.physical()
    with torch.no_grad():
        mu = torch.cat([cal.model({k: v[i:i + 8] if cal.result.params.specs[k].scope == "emitter" else v
                                   for k, v in phys.items()}, sub.subset(torch.arange(i, min(i + 8, sub.n_emitters))))
                        for i in range(0, sub.n_emitters, 8)])
    nt = noise_transfer(sub.data, mu)
    shared = cal.shared()
    out = dict(kind=kind, beads=len(cal.beads.used), of=batch.n_emitters, minutes=round(minutes, 1),
               objective=float(cal.result.objective), deviance_per_pixel=float(cal.atlas.mean_deviance),
               sigma_nm=[round(float(v), 1) for v in shared["psf.sigma"].reshape(-1)],
               noise_slope=round(nt.slope, 3))
    rep = field_report(cal)
    if rep is not None:
        spline = spline_calibration(cal, 20.0, 900.0, 27)
        layout = map(str, cal.model.channels[0].field.layout)
        values = (round(float(v), 4) for v in shared["field.phase"])
        out.update(field=rep.summary(), coefficients=dict(zip(layout, values, strict=True)),
                   centre_px=list(cal.model.channels[0].field.centre_px),
                   grid=None if spline.field_splines is None else list(spline.field_splines.shape))
    print(json.dumps(out))
    with open(os.path.join(work, f"cal_{kind}.pkl"), "wb") as f:
        pickle.dump(cal, f)


def localise(work):
    from smappy.calibrate.pupil import spline_calibration
    from smappy.io.calibration import SplineCalibration
    from smappy.psf import SplinePSF
    load = lambda name: pickle.load(open(os.path.join(work, name), "rb"))   # noqa: E731
    rois, field, plain = load("rois.pkl"), load("cal_field.pkl"), load("cal_constant.pkl")
    spline = spline_calibration(field, 20.0, 900.0, 27)
    splines = {"field (node splines)": (spline, True),
               "field model's centre spline": (SplineCalibration(spline.coeff, spline.dz, spline.z0, x0=spline.x0),
                                               False),
               "learnt without field": (spline_calibration(plain, 20.0, 900.0, 27), False)}
    b = rois.batch
    data, z = b.data[:, 0].numpy(), b.z_nm.numpy().reshape(-1)
    keep = np.abs(z) <= 600
    xy = b.field_xy.numpy()
    r = np.hypot(*(xy - CHIP_CENTRE).T) * PIXEL_UM
    rois_ = np.ascontiguousarray(data[:, keep][:, :, 7:20, 7:20].reshape(-1, 13, 13), dtype=np.float32)
    held = ~np.isin(np.arange(len(xy)), field.beads.used)
    print(f"{len(xy)} beads, {keep.sum()} slices each; z error rms per bead, median (nm)")
    for name, (cal, blended) in splines.items():
        m = SplinePSF(cal)
        kw = dict(chip_xy=np.repeat(xy, keep.sum(), 0)) if blended else {}
        e = m.unpack(m.fit(rois_, 50, 0, **kw))["z_nm"].reshape(len(xy), -1) - z[keep]
        rms = np.sqrt(np.nanmean((e - np.nanmedian(e, 1, keepdims=True)) ** 2, 1))
        bands = "; ".join(f"{a}-{c} µm {np.nanmedian(rms[(r >= a) & (r < c)]):.0f}"
                          for a, c in ((0, 40), (40, 70), (70, 120)))
        print(f"  {name:28s} all {np.nanmedian(rms):5.1f}, not learnt from {np.nanmedian(rms[held]):5.1f}; {bands}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("detect", "fit", "localise"))
    ap.add_argument("kind", nargs="?", choices=("constant", "field"), default="field")
    ap.add_argument("--beads", type=int, default=60)
    ap.add_argument("--work", default=os.path.join(DATA, "work_1ch_FD_bead"))
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    {"detect": lambda: detect(args.work), "fit": lambda: fit(args.work, args.kind, args.beads),
     "localise": lambda: localise(args.work)}[args.step]()


if __name__ == "__main__":
    main()
