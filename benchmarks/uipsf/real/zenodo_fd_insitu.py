"""Field dependence in situ on Zenodo `1ch_FD_smlm_nup96`, against the field the beads of `1ch_FD_bead` gave
(`docs/uipsf/field-dependence.md`, §9 and §13.3).

    python benchmarks/uipsf/real/zenodo_fd_insitu.py detect
    python benchmarks/uipsf/real/zenodo_fd_insitu.py fit field
    python benchmarks/uipsf/real/zenodo_fd_insitu.py fit constant
    python benchmarks/uipsf/real/zenodo_fd_insitu.py compare

The movie: Nup96-SNAP-AF647, astigmatic, 800 frames over the same 1608 × 1608 Prime 95B chip as the beads.  The
optics and the start are v1's from its own in situ result on these data (``psfmodel_insitu_FD_single.h5``):
NA 1.5, 680 nm, immersion 1.515, sample 1.406, 110 nm pixels, gain 0.47 e⁻/ADU, offset 100, the focus 0.8 µm
above the coverslip.  v1 learnt 0.70 rad of Noll 6 at the chip's centre, 76 nm, but in v2's convention the
data want the other sign: started at +76 nm the twin test prefers the mirrored pupil by 116 nats, started at
−76 nm it agrees, and only then does the field match the beads' (`results/zenodo_fd_insitu.jsonl`, §13.3).
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


def fit(work, kind, rounds, device, astigmatism_nm=ASTIGMATISM_NM, save=False):
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
    # with --save the spline (and the field's node splines) beside the work: what *localise* fits with
    result = run_insitu(None, optics(), model, insitu, detection=selection(field), run=RunSettings(save_model=save),
                        rois=rois, progress=tell, device=device,
                        save_to=os.path.join(work, f"insitu_{kind}_3dcal.h5") if save else None)
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


#: how each localisation of the movie is fitted (*localise*): the field's node splines; the same model's
#: centre spline alone; that fit with *Field correction* subtracted; the model learnt without the field
WAYS = ("nodes", "centre", "corrected", "constant")


def _spline(work, way):
    import copy

    from smappy.io.calibration import load_spline_calibration
    cal = load_spline_calibration(os.path.join(work, f"insitu_{'constant' if way == 'constant' else 'field'}_3dcal.h5"))
    if way in ("centre", "corrected"):
        cal = copy.copy(cal)
        cal.field_splines = None
    return cal


def localise(work, ways, frames, threads):
    """Every frame (or the first ``frames``) fitted with Spline 3D on each way's spline, as the fitter does:
    the dynamic cutoff, a 13 px box, each spot's chip position for the node splines.  Kept as
    ``locs_<way>.npz``; *corrected* is the *centre* fit with `field_correction` applied, no fit of its own."""
    from smappy.field_correction import corrected, kept, z_range_of
    from smappy.io.calibration import load_spline_calibration
    from smappy.io.tiff import open_stack
    from smappy.pipeline import FitSettings, fit_stack
    from smappy.plugins.fit import DetectionSettings
    from smappy.psf import SplinePSF
    src = open_stack(MOVIE)
    stop = src.n_frames if frames is None else min(frames, src.n_frames)
    for way in ways:
        out = os.path.join(work, f"locs_{way}.npz")
        if way == "corrected":
            d = dict(np.load(os.path.join(work, "locs_centre.npz")))
            field = load_spline_calibration(os.path.join(work, "insitu_field_3dcal.h5"))
            t = time.time()
            correction = kept(field, float(np.median(d["photons"])), float(np.median(d["background"])), 13,
                              n_threads=threads, z_range=z_range_of(d["z_nm"]))
            d["x_nm"], d["y_nm"], d["z_nm"], outside = corrected(d["x_nm"], d["y_nm"], d["z_nm"], correction,
                                                                 (110.0, 110.0))
            print(f"corrected: {outside.mean():.1%} outside what was measured, {time.time() - t:.0f} s; "
                  f"{json.dumps(correction.summary())}", flush=True)
            np.savez(out, **d)
            continue
        model = SplinePSF(_spline(work, way))
        t = time.time()
        locs, engine = fit_stack(src.frames(chunk=20, stop=stop), camera(), DetectionSettings().finder(threads),
                                 model, FitSettings(roisize=13, n_threads=threads), read_ahead=1)
        keep = ("frame", "x_nm", "y_nm", "z_nm", "photons", "background", "xy_err_nm", "z_err_nm", "logl",
                "peak_x_pix", "peak_y_pix")
        np.savez(out, **{k: np.asarray(locs[k]) for k in keep if k in locs})
        print(f"{way}: {len(locs)} localisations in {(time.time() - t) / 60:.1f} min", flush=True)


#: the pores (*npc*).  The movie is a calibration movie, 800 frames: after the precision cut 20 000
#: localisations over the chip, 10-30 per pore, too few to measure one pore.  So the pores are pooled per band
#: of distance from the chip's centre: each pore's localisations by DBSCAN (35 nm, 4 neighbours, 8-200 of
#: them, at most 150 nm across), its z less its own median, and the two rings fitted to the band's pooled z as
#: two Gaussians of equal weight and width, ``separation`` apart -- Nup96's two rings are ~50 nm apart in z
#: (Thevathasan et al. 2019).  A z scale wrong at the edges of the field shows as a separation that changes
#: from band to band.
NPC_EPS_NM, NPC_K, NPC_MIN, NPC_MAX, NPC_WIDTH_NM = 35.0, 4, 8, 200, 150.0
BANDS_UM = ((0, 30), (30, 55), (55, 80), (80, 125))


def _two_rings(dz):
    """(separation, its standard error, width) nm: two Gaussians of equal weight and width at ±s/2 about a
    common offset, fitted to ``dz`` by maximum likelihood."""
    from scipy.optimize import minimize

    def nll(p):
        m, s, w = p[0], abs(p[1]), abs(p[2]) + 1e-3
        a = np.exp(-0.5 * ((dz - m - s / 2) / w) ** 2) + np.exp(-0.5 * ((dz - m + s / 2) / w) ** 2)
        return -np.sum(np.log(a / (2 * w * np.sqrt(2 * np.pi)) + 1e-300))
    best = min((minimize(nll, [0.0, s0, 20.0], method="Nelder-Mead", options=dict(xatol=0.01, fatol=1e-4,
                                                                                  maxiter=4000))
                for s0 in (20.0, 50.0, 80.0)), key=lambda r: r.fun)
    m, s, w = best.x[0], abs(best.x[1]), abs(best.x[2])
    h = 0.5                                                   # the curvature in s, numerically
    curv = (nll([m, s + h, w]) - 2 * best.fun + nll([m, s - h, w])) / h ** 2
    return float(s), float(1 / np.sqrt(curv)) if curv > 0 else float("nan"), float(w)


def npc(work, ways):
    """Each way's pooled ring separation per band (module notes above `NPC_EPS_NM`), and for the fits of the
    same detections the log-likelihood each spot gained with the node splines over the centre spline, per band."""
    from smappy.plugins.clusters import dbscan
    results = {}
    data = {way: np.load(os.path.join(work, f"locs_{way}.npz")) for way in ways}
    for way, d in data.items():
        ok = (d["xy_err_nm"] < 15) & (d["z_err_nm"] < 40) & (d["photons"] > 1000) & np.isfinite(d["z_nm"])
        x, y, z = d["x_nm"][ok], d["y_nm"][ok], d["z_nm"][ok]
        label = dbscan(np.c_[x, y], NPC_EPS_NM, NPC_K)
        far, dz = [], []
        for c in range(1, int(label.max()) + 1):
            m = np.flatnonzero(label == c)
            if not NPC_MIN <= len(m) <= NPC_MAX:
                continue
            if max(np.ptp(x[m]), np.ptp(y[m])) > NPC_WIDTH_NM:
                continue
            r = np.hypot(np.mean(x[m]) / 110.0 - CHIP_CENTRE[0], np.mean(y[m]) / 110.0 - CHIP_CENTRE[1]) * PIXEL_UM
            far.append(np.full(len(m), r))
            dz.append(z[m] - np.median(z[m]))
        far, dz = np.concatenate(far), np.concatenate(dz)
        out = {}
        for a, b in BANDS_UM:
            m = (far >= a) & (far < b) & (np.abs(dz) < 150)
            sep, err, width = _two_rings(dz[m])
            out[f"{a}-{b} um"] = dict(localisations=int(m.sum()), separation_nm=round(sep, 1),
                                      error_nm=round(err, 1), width_nm=round(width, 1))
        results[way] = out
        print(way, json.dumps(out), flush=True)
    if "nodes" in data and "centre" in data and "logl" in data["nodes"]:
        a, b = data["nodes"], data["centre"]
        if len(a["logl"]) == len(b["logl"]) and np.array_equal(a["frame"], b["frame"]):
            gain = a["logl"] - b["logl"]
            r = np.hypot(a["x_nm"] / 110.0 - CHIP_CENTRE[0], a["y_nm"] / 110.0 - CHIP_CENTRE[1]) * PIXEL_UM
            bright = a["photons"] > 1000
            out = {}
            for lo, hi in BANDS_UM:
                m = bright & (r >= lo) & (r < hi) & np.isfinite(gain)
                out[f"{lo}-{hi} um"] = dict(spots=int(m.sum()), median_gain_nats=round(float(np.median(gain[m])), 2),
                                            better=round(float(np.mean(gain[m] > 0)), 3))
            results["nodes over centre, log-likelihood"] = out
            print("nodes over centre:", json.dumps(out), flush=True)
    with open(os.path.join(work, "npc.json"), "w") as f:
        json.dump(results, f, indent=1)


BEAD_WORK = os.path.join(DATA, "work_1ch_FD_bead")


def beads(work):
    """The in situ models' accuracy on beads of the same microscope: `zenodo_fd_beads.py detect`'s beads, every
    slice within ±600 nm fitted with each in situ model drawn as a spline for an emitter at the glass
    (`pupil.spline_calibration`: the node splines with it), the z error per bead after its own height, rms
    over its slices, median per band of distance from the chip's centre -- §9's test, the beads' own model
    there.  An in situ model of the wrong sign would put every bead's z mirrored, its error hundreds of nm."""
    from smappy.calibrate.pupil import spline_calibration
    from smappy.io.calibration import SplineCalibration
    from smappy.psf import SplinePSF

    import uipsf
    with open(os.path.join(BEAD_WORK, "rois.pkl"), "rb") as f:
        rois = pickle.load(f)
    field = uipsf.load_model(os.path.join(work, "insitu_field_3dcal.h5"), device="cpu")
    plain = uipsf.load_model(os.path.join(work, "insitu_constant_3dcal.h5"), device="cpu")
    spline = spline_calibration(field, 20.0, 900.0, 27)
    splines = {"in situ, field (node splines)": (spline, True),
               "in situ, the field model's centre": (SplineCalibration(spline.coeff, spline.dz, spline.z0,
                                                                      x0=spline.x0), False),
               "in situ, learnt without field": (spline_calibration(plain, 20.0, 900.0, 27), False)}
    b = rois.batch
    data, z = b.data[:, 0].numpy(), b.z_nm.numpy().reshape(-1)
    keep = np.abs(z) <= 600
    xy = b.field_xy.numpy()
    r = np.hypot(*(xy - CHIP_CENTRE).T) * PIXEL_UM
    rois_ = np.ascontiguousarray(data[:, keep][:, :, 7:20, 7:20].reshape(-1, 13, 13), dtype=np.float32)
    out = {"beads": int(len(xy)), "slices": int(keep.sum()), "grid": list(spline.field_splines.shape)}
    print(f"{len(xy)} beads, {keep.sum()} slices each; z error rms per bead, median (nm); slope of z against "
          f"the stage")
    for name, (cal, blended) in splines.items():
        m = SplinePSF(cal)
        kw = dict(chip_xy=np.repeat(xy, keep.sum(), 0)) if blended else {}
        fitted = m.unpack(m.fit(rois_, 50, 0, **kw))["z_nm"].reshape(len(xy), -1)
        e = fitted - z[keep]
        rms = np.sqrt(np.nanmean((e - np.nanmedian(e, 1, keepdims=True)) ** 2, 1))
        slope = float(np.nanmedian([np.polyfit(z[keep][np.isfinite(f)], f[np.isfinite(f)], 1)[0] for f in fitted]))
        row = {"all": round(float(np.nanmedian(rms)), 1), "slope": round(slope, 3)}
        row.update({f"{a}-{c} um": round(float(np.nanmedian(rms[(r >= a) & (r < c)])), 1)
                    for a, c in ((0, 40), (40, 70), (70, 120))})
        out[name] = row
        print(f"  {name:36s} {json.dumps(row)}", flush=True)
    with open(os.path.join(work, "beads.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("detect", "fit", "compare", "localise", "npc", "beads"))
    ap.add_argument("kind", nargs="?", choices=("field", "constant"), default="field")
    ap.add_argument("--rounds", type=int, default=3)
    # the CPU by default: on Apple's GPU round 1 of 2713 molecules grew to a 33.5 GB footprint and thrashed
    ap.add_argument("--device", default="cpu")
    # the start's sign: v1's (+), or the other (-), which v2's twin test prefers on these data (§13.3)
    ap.add_argument("--astigmatism", type=float, default=ASTIGMATISM_NM)
    ap.add_argument("--save", action="store_true", help="fit: save the spline as insitu_<kind>_3dcal.h5")
    ap.add_argument("--ways", nargs="+", default=list(WAYS), choices=WAYS)
    ap.add_argument("--frames", type=int, default=None, help="localise: only the first frames")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--work", default=os.path.join(DATA, "results", "1ch_FD_smlm_nup96", "v2_fd"))
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    {"detect": lambda: detect(a.work), "fit": lambda: fit(a.work, a.kind, a.rounds, a.device, a.astigmatism, a.save),
     "compare": lambda: compare(a.work), "localise": lambda: localise(a.work, a.ways, a.frames, a.threads),
     "npc": lambda: npc(a.work, a.ways), "beads": lambda: beads(a.work)}[a.step]()
