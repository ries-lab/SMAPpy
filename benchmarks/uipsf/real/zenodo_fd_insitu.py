"""Field dependence in situ on Zenodo `1ch_FD_smlm_nup96`, against the field the beads of `1ch_FD_bead` gave
(`docs/uipsf/field-dependence.md`, §9 and §12.3).

    python benchmarks/uipsf/real/zenodo_fd_insitu.py detect
    python benchmarks/uipsf/real/zenodo_fd_insitu.py fit field
    python benchmarks/uipsf/real/zenodo_fd_insitu.py fit constant
    python benchmarks/uipsf/real/zenodo_fd_insitu.py compare

The movie: Nup96-SNAP-AF647, astigmatic, 800 frames over the same 1608 × 1608 Prime 95B chip as the beads.  The
optics and the start are v1's from its own in situ result on these data (``psfmodel_insitu_FD_single.h5``):
NA 1.5, 680 nm, immersion 1.515, sample 1.406, 110 nm pixels, gain 0.47 e⁻/ADU, offset 100, the focus 0.8 µm
above the coverslip.  v1 learnt 0.70 rad of Noll 6 at the chip's centre, 76 nm, but in v2's convention the
data want the other sign: started at +76 nm the twin test prefers the mirrored pupil by 116 nats, started at
−76 nm it agrees, and only then does the field match the beads' (`results/zenodo_fd_insitu.jsonl`, §12.3).
No bead model of this microscope in v2 form: γ is held at the nominal value, with the run's warning.
Learnt on gpuvm's H100 (`--device cuda`): 8-10 minutes a run; on the Mac's GPU round 1 grew to a 33.5 GB
footprint and thrashed, on its CPU round 1 alone took over an hour.

*detect* finds the molecules once (`insitu_run.movie_rois`, 21 px regions as v1's) and keeps them in ``--work``.
*fit* runs `insitu_run.run_insitu` as the window does, with or without *field dependence* (which redraws the
molecules evenly over the field, `insitu.spread_draw`), and writes a JSON line.  *compare* evaluates each mode's
change over the chip -- the field's value less its value at the chip's centre -- for the in situ field, the
beads' (`results/zenodo_fd.json`, learnt from 51 beads) and v1's in situ map (its ``zernike_map``, 40 × 40 over
the chip), and prints the rms of each and of their differences over the part of the chip all three cover.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time

import numpy as np

DATA = os.environ.get("UIPSF_DATA", os.path.expanduser("~/.cache/uipsf/zenodo"))
MOVIE = os.path.join(DATA, "1ch_FD_smlm_nup96",
                     "NUP96_SNP647_astigmatism_3D_1608_10ms_hama_mm_1800mW_3_MMStack_Default.ome.tif")
V1 = os.path.join(DATA, "results", "1ch_FD_smlm_nup96", "psfmodel_insitu_FD_single.h5")
BEADS = os.path.join(os.path.dirname(__file__), "..", "results", "zenodo_fd.json")
CONVERSION, OFFSET, READ_NOISE_E, PIXEL_UM = 0.47, 100.0, 1.6, 0.110
WAVELENGTH_NM, FOCUS_NM, ASTIGMATISM_NM = 680.0, 800.0, 0.70 * 680.0 / (2 * math.pi)
CHIP_CENTRE = (803.5, 803.5)
ROI_PX, MOLECULES = 21, 3000


def camera():
    from smappy.metadata import CameraMetadata
    return CameraMetadata(conversion=CONVERSION, offset=OFFSET, pixelsize_um=PIXEL_UM, em_on=False,
                          read_noise_e=READ_NOISE_E, roi=(0, 0, 1608, 1608))


def optics():
    from smappy.calibrate.pupil import OpticsSettings
    return OpticsSettings(NA=1.5, wavelength_nm=WAVELENGTH_NM, n_immersion=1.515, n_medium=1.406, pixel_x_nm=110.0,
                          pixel_y_nm=110.0)


def selection(field: bool):
    from smappy.calibrate.insitu import InSituSettings
    # blocks of 50 frames: 200 of the 1608 x 1608 chip held 20 GB while reading (31.6 GB at its peak)
    return InSituSettings(roi_px=ROI_PX, molecules=MOLECULES, field_tiles=4 if field else 1, block_frames=50)


def detect(work):
    from smappy.calibrate.insitu_run import movie_rois
    t = time.time()
    rois = movie_rois([MOVIE], selection(False), camera(), progress=lambda e: print(e.get("message", ""),
                                                                                    flush=True))
    print(f"{len(rois.state)} units, {rois.counts()} in {time.time() - t:.0f} s")
    with open(os.path.join(work, "rois.pkl"), "wb") as f:
        pickle.dump(rois, f)


def fit(work, kind, rounds, device, astigmatism_nm=ASTIGMATISM_NM):
    from smappy.calibrate.insitu_run import InSituModelSettings, run_insitu
    from smappy.calibrate.pupil import BeadSettings, PSFModelSettings, RunSettings

    from uipsf.diagnostics import field_report
    with open(os.path.join(work, "rois.pkl"), "rb") as f:
        rois = pickle.load(f)
    field = kind == "field"
    model = BeadSettings(psf=PSFModelSettings(zernike_order=6), field_dependence=field)
    insitu = InSituModelSettings(focus_nm=FOCUS_NM, astigmatism_nm=astigmatism_nm, max_rounds=rounds)
    t = time.time()
    def tell(entry):
        if "message" in entry:
            print(entry["message"], flush=True)
    result = run_insitu(None, optics(), model, insitu, detection=selection(field), run=RunSettings(save_model=False),
                        rois=rois, progress=tell, device=device)
    minutes = (time.time() - t) / 60
    cal = result.cal
    p = {k: v.detach().cpu().double().numpy() for k, v in cal.shared().items()}
    modes = result.model.psf.pupil.phase_modes
    out = dict(kind=kind, start_astigmatism_nm=round(astigmatism_nm, 1), minutes=round(minutes, 1),
               used=len(cal.beads.used), rounds=len(result.run.rounds),
               focus_nm=round(float(p["geometry.focus"]), 1), sigma_nm=p["psf.sigma"].round(1).tolist(),
               phase=dict(zip([int(j) for j in modes], p["pupil.phase"].round(4).tolist(), strict=True)),
               twin=None if result.run.final.twin is None else result.run.final.twin.verdict,
               messages=result.messages)
    if field:
        fp = cal.model.channels[0].field
        out.update(centre_px=list(fp.centre_px), report=field_report(cal).summary(),
                   coefficients={str(t): round(float(v), 4) for t, v in zip(fp.layout, p["field.phase"], strict=True)})
    print(json.dumps(out, indent=1))
    with open(os.path.join(work, "insitu_fd.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")


def _polynomial(centre, coefficients):
    from uipsf.model import FieldPolynomial
    terms = sorted({(j, 2 if j in (5, 6) else 1) for j in (int(k.strip("()").split(",")[0]) for k in coefficients)})
    fp = FieldPolynomial(centre, (110.0, 110.0), terms=terms)
    values = np.array([coefficients.get(str(t), 0.0) for t in fp.layout])
    return fp, values


def _changes(centre, coefficients, xy):
    """{Noll: [N] the mode's value at chip positions ``xy`` less its value at the chip's centre}."""
    import torch
    fp, c = _polynomial(centre, coefficients)
    pts = torch.as_tensor(np.r_[xy, [CHIP_CENTRE]], dtype=torch.float64)
    terms = fp.basis(pts).numpy() * c
    out = {}
    for j, _ in fp.terms:
        rows = [i for i, (m, _, _) in enumerate(fp.layout) if m == j]
        v = terms[:, rows].sum(1)
        out[j] = v[:-1] - v[-1]
    return out


def _v1_changes(xy):
    """v1's in situ phase map (rad) at ``xy``, less its value at the chip's centre: bilinear on its 40 × 40 grid."""
    import h5py
    from scipy.ndimage import map_coordinates
    with h5py.File(V1, "r") as f:
        maps = f["res/zernike_map"][1]                   # [28, 40, 40], Noll 1.. on a grid over the chip
    step = 1608 / maps.shape[-1]
    out = {}
    def at(m, p):
        return map_coordinates(m, [(p[:, 1] / step) - 0.5, (p[:, 0] / step) - 0.5], order=1, mode="nearest")
    for j in (5, 6, 7, 8):
        out[j] = at(maps[j - 1], xy) - at(maps[j - 1], np.array([CHIP_CENTRE]))[0]
    return out


def compare(work):
    with open(BEADS) as f:
        beads = json.load(f)["field"]
    runs = [json.loads(line) for line in open(os.path.join(work, "insitu_fd.jsonl"))]
    fields = [r for r in runs if r["kind"] == "field"]
    for r in fields:
        print(f"start {r.get('start_astigmatism_nm', ASTIGMATISM_NM):+.0f} nm: twin {r['twin']}; "
              + "; ".join(m for m in r["messages"] if "sign" in m or "mirrored" in m))
    insitu = fields[-1]
    g = np.linspace(150, 1450, 27)
    xy = np.array([[x, y] for x in g for y in g])
    bead = _changes(beads["centre_px"], beads["coefficients"], xy)
    situ = _changes(insitu["centre_px"], insitu["coefficients"], xy)
    v1 = _v1_changes(xy)
    print(f"in situ: {insitu['report']}")
    print("each mode's change over the chip (rad, rms over 150-1450 px); differences rms")
    print("Noll  beads  in-situ  v1-in-situ | situ-beads  v1-beads  situ-v1")
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))   # noqa: E731
    for j in (5, 6, 7, 8):
        print(f"{j:4d}  {rms(bead[j]):5.3f}  {rms(situ[j]):7.3f}  {rms(v1[j]):10.3f} | {rms(situ[j] - bead[j]):10.3f}"
              f"  {rms(v1[j] - bead[j]):8.3f}  {rms(situ[j] - v1[j]):7.3f}")
    corners = np.array([[150, 150], [1450, 150], [150, 1450], [1450, 1450]], float)
    for name, f in (("beads", lambda: _changes(beads["centre_px"], beads["coefficients"], corners)),
                    ("in situ", lambda: _changes(insitu["centre_px"], insitu["coefficients"], corners)),
                    ("v1 in situ", lambda: _v1_changes(corners))):
        c = f()
        print(f"{name:10s} corners, Noll 5: {np.round(c[5], 2)}  Noll 6: {np.round(c[6], 2)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("detect", "fit", "compare"))
    ap.add_argument("kind", nargs="?", choices=("field", "constant"), default="field")
    ap.add_argument("--rounds", type=int, default=3)
    # the CPU by default: on Apple's GPU round 1 of 2713 molecules grew to a 33.5 GB footprint and thrashed
    ap.add_argument("--device", default="cpu")
    # the start's sign: v1's (+), or the other (-), which v2's twin test prefers on these data (§12.3)
    ap.add_argument("--astigmatism", type=float, default=ASTIGMATISM_NM)
    ap.add_argument("--work", default=os.path.join(DATA, "results", "1ch_FD_smlm_nup96", "v2_fd"))
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    {"detect": lambda: detect(a.work), "fit": lambda: fit(a.work, a.kind, a.rounds, a.device, a.astigmatism),
     "compare": lambda: compare(a.work)}[a.step]()
