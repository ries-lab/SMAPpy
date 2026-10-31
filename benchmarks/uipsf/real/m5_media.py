"""M5's beads of 2026-10-29: two colours, index-matched or not, with and without the cylindrical lens.

    python benchmarks/uipsf/real/m5_media.py ACQUISITION [...] [--variants zernike "cos bead na asym"] \\
        [--pixel 104.7 106.4] [--window 1000] [--dz 50] [--grid-na 1.50] [--chunk 1] \\
        [--out benchmarks/uipsf/results/m5_media.jsonl]

One microscope, one day, eight conditions that each change one thing: the emission (600 nm, TetraSpeck's
orange dye under 561 nm through an RFP filter; 676 nm, the far-red dye under 640 nm), the medium (beads dried
on the coverslip and covered with immersion oil, n 1.518, or in PBS, 1.334) and the 3D lens (in or out).
What physics says must agree between them -- the objective's magnitude, the phase in nm of path, σ -- is
the microscope; what moves stands in for something else (the user's study A, October 2026).  Matched
medium has no supercritical light at all (n_med > NA), so its magnitude is the objective's alone and
separates the effective NA from the ring's transmission, which M2's beads in water could not
(`docs/uipsf/architecture.md`, "On real beads, and the supercritical ring as its own term").

The beads (the user's data check): the oil samples carry hundreds of faint objects, 10-50x dimmer than a
bead and spread up to a micrometre in focus, so only beads at least `BRIGHT` of the brightest 20's median
are kept; and the focus lock drifted in oil, up to 1.2 µm from the stack's centre, so only beads whose
focus lies within `FOCUS_NM` of it, which leaves each at least a micrometre on both sides.  Every kept
bead is included (no rejection), so variants of one acquisition compare in nats.

Variants (`edge_magnitude.use_pupil`): *zernike*, every magnitude mode to order 8; *cos bead na asym*, the
bead physics: cos θ_imm^γ, a 100 nm sphere's supercritical ring (none when matched) and a learnt NA cutoff,
with the m ≠ 0 Zernikes.  The grid's NA (``--grid-na``) is by default `GRID_NA`, the nominal one, so the
cutoff can only find an effective NA below it: at 1.51 the pupil needs 208 samples for ±1 µm, which a 32 GB
Mac's GPU does not fit even one bead at a time (1.50: 144).
"""
from __future__ import annotations

import argparse
import functools
import glob
import math
import os
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "synthetic")]
from bead_slices import emit, measure  # noqa: E402
from edge_magnitude import use_pupil  # noqa: E402
from radial_magnitude import EdgePupil  # noqa: E402

N_IMM, BEAD_NM, ORDER = 1.518, 100.0, 8
N_MED = {"match": 1.518, "mismatch": 1.334}
WAVELENGTH = {"600": 600.0, "676": 676.0}
GRID_NA = 1.50
BRIGHT, FOCUS_NM, WINDOW_NM, APERTURE_PX = 0.2, 1000.0, 1000.0, 5


def condition(acquisition: str) -> dict:
    """Colour, medium and lens from the folder's name, e.g. 20261029_600_100nm_mismatch_50ms_3D_1."""
    m = re.search(r"_(600|676)_.*_(match|mismatch)_.*_(2D|3D)_", os.path.basename(os.path.normpath(acquisition)))
    if not m:
        raise ValueError(f"cannot read colour, medium and lens from {acquisition}")
    return dict(colour=m[1], medium=m[2], lens=m[3])


def select(rois) -> tuple[list, dict]:
    """The bright beads in focus near the stack's centre: indices and what was cut."""
    data = rois.batch.data[:, 0].double().numpy()                      # [E, K, Y, X] photons
    z = rois.batch.z_nm.numpy()
    z = z if z.ndim == 1 else z[0]
    E, K, Y, X = data.shape
    edge = np.concatenate([data[..., 0, :], data[..., -1, :], data[..., :, 0], data[..., :, -1]], -1)
    s = data - np.median(edge, -1)[..., None, None]
    yy, xx = np.mgrid[:Y, :X]
    aperture = (yy - Y // 2) ** 2 + (xx - X // 2) ** 2 <= APERTURE_PX ** 2
    core = s[..., Y // 2 - 1:Y // 2 + 2, X // 2 - 1:X // 2 + 2].sum((-1, -2))
    focus = np.array([np.convolve(c, np.ones(3) / 3, "same").argmax() for c in core])
    photons = (s * aperture).sum((-1, -2))[np.arange(E), focus]
    bright = photons >= BRIGHT * np.median(np.sort(photons)[-20:])
    near = np.abs(z[focus] - z.mean()) <= FOCUS_NM
    keep = [int(i) for i in np.flatnonzero(bright & near)]
    return keep, focus, dict(found=E, dim=int((~bright).sum()), off_focus=int((bright & ~near).sum()), kept=len(keep),
                      photons_median=round(float(np.median(photons[keep])), 0))


def subset(rois, keep, focus, window_nm=WINDOW_NM):
    """The kept beads, each on its own window of ±``window_nm`` about its focus (at most the whole stack, then
    about its centre).  The stacks span ±2 µm, and the pupil's sampling grows with the defocus it must hold
    (`choose_pupil_size` at NA 1.50: 144 samples for ±1 µm, 240 for ±2 µm), so the window is a question of
    memory as well as of what the learning sees (the user's: does the whole stack learn better or worse?)."""
    from dataclasses import replace
    b = rois.batch.subset(torch.tensor(keep))
    z = b.z_nm if b.z_nm.dim() == 1 else None
    K = b.data.shape[2]
    half = min(int(round(window_nm / float(b.z_step_nm))), (K - 1) // 2)      # the whole stack at most
    # a focus at the edge of `FOCUS_NM` may leave its window a slice short: the window slides inwards
    first = [min(max(int(focus[i]) - half, 0), K - 2 * half - 1) for i in keep]
    index = torch.stack([torch.arange(f, f + 2 * half + 1) for f in first])            # [E, K']
    rows = torch.arange(len(keep))[:, None]
    rois.batch = replace(b, data=b.data.transpose(1, 2)[rows, index].transpose(1, 2),
                         # each window about its own middle: the pupil is sized by the batch's largest |z|
                         # (`calibrate_beads`), and every bead's z is learnt anyway
                         mask=b.mask.transpose(1, 2)[rows, index].transpose(1, 2),
                         z_nm=z[index] - z[index[:, half:half + 1]])
    rois.records = [dict(rois.records[i], id=n) for n, i in enumerate(keep)]
    return rois


def summary(run, variant, pixel, cond, grid_na) -> dict:
    from uipsf.optics.zernike import noll_to_nm
    cal = run.cal
    p, std = cal.result.params.physical(), cal.result.std
    pupil = cal.model.channels[0].psf.pupil
    lam = WAVELENGTH[cond["colour"]]
    sigma = p["psf.sigma"].detach().cpu().double().numpy()
    phase = p["pupil.phase"].detach().cpu().double().numpy()
    mag = p["pupil.magnitude"].detach().cpu().double().numpy()
    nm = [noll_to_nm(j) for j in pupil.magnitude_modes]
    radial = [v for v, (_, m) in zip(mag, nm, strict=True) if m == 0]
    asym = [v for v, (_, m) in zip(mag, nm, strict=True) if m != 0]
    row = dict(variant=variant, pixel_nm=list(pixel), used=len(cal.beads.used),
               objective=round(float(cal.result.objective), 1), iterations=cal.result.iterations,
               stop=cal.result.stop_reason, sigma_nm=sigma.round(2).tolist(),
               sigma_crlb_nm=std["psf.sigma"].cpu().double().numpy().round(2).tolist(),
               sigma_optics_nm=[round(math.sqrt(max(v * v - q * q / 12, 0.0)), 1)
                                for v, q in zip(sigma, pixel, strict=True)],
               phase_modes=list(pupil.phase_modes),
               # a path difference is what two colours share: rad × λ / 2π
               phase_nm_path={int(j): round(float(v) * lam / (2 * math.pi), 2)
                              for j, v in zip(pupil.phase_modes, phase, strict=True)},
               phase_crlb_rad=std["pupil.phase"].cpu().double().numpy().round(4).tolist(),
               magnitude={int(j): round(float(v), 4) for j, v in zip(pupil.magnitude_modes, mag, strict=True)},
               magnitude_radial_rms=round(float(np.sqrt(np.sum(np.square(radial)))), 4),
               magnitude_asym_rms=round(float(np.sqrt(np.sum(np.square(asym)))), 4),
               atlas_mean_deviance=round(float(cal.atlas.mean_deviance), 3))
    if isinstance(pupil, EdgePupil):
        row.update(edge=pupil.edge_values(p), profile=pupil.edge_profile(p))
        if "edge_rho" in row["edge"]:
            row["effective_na"] = round(row["edge"]["edge_rho"] * grid_na, 4)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acquisitions", nargs="+")
    ap.add_argument("--variants", nargs="+", default=["zernike", "cos bead na asym"])
    ap.add_argument("--pixel", type=float, nargs=2, action="append", default=None, metavar=("X", "Y"),
                    help="x y on the sample, nm; several to compare by likelihood")
    ap.add_argument("--z-range", type=float, default=600.0)
    ap.add_argument("--window", type=float, default=WINDOW_NM,
                    help="each bead's slices, ± this about its focus; 2000: the whole stack, about its centre")
    ap.add_argument("--grid-na", type=float, default=GRID_NA, help="the pupil grid's edge, ρ = 1")
    ap.add_argument("--dz", type=float, default=50.0, help="the slices' largest spacing in learning, nm")
    ap.add_argument("--chunk", type=int, default=4, help="emitters per solver chunk: M5's 128-sample pupil")
    ap.add_argument("--memory", type=float, default=0.5, help="MPS memory fraction")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    import uipsf.workflows.beads as beads
    import uipsf.workflows.calibrate as calibrate
    import uipsf.workflows.scoring as scoring
    # the slices learning keeps: every n-th, n·Δz ≤ --dz; `measure` reads the module's constant, so both
    beads.MAX_DZ_NM = args.dz
    calibrate.learn_beads = functools.partial(calibrate.learn_beads, chunk_size=args.chunk, max_dz_nm=args.dz)
    # the slice fits localise every slice as an emitter of its own, all at once unless told otherwise
    slices = int(2 * args.window // max(args.dz, 50.0)) + 1                      # one bead's, as learning sees it
    scoring.localise = functools.partial(scoring.localise, chunk_size=slices * args.chunk)
    from smappy.calibrate.pupil import BeadSettings, OpticsSettings, RunSettings, bead_rois, camera_for, run_calibration
    # the likelihood's choice on the 676 sets in buffer (results/m5_media_pixel.jsonl)
    pixels = args.pixel or [(104.7, 106.4)]
    for acquisition in args.acquisitions:
        cond = condition(acquisition)
        files = sorted(glob.glob(os.path.join(acquisition, "*.ome.tif")))
        rois = bead_rois(files, camera_for(files[0]), dual=False)
        keep, focus, cut = select(rois)
        rois = subset(rois, keep, focus, args.window)
        decisions = {i: "include" for i in range(len(keep))}
        for pixel in pixels:
            optics = OpticsSettings(NA=args.grid_na, wavelength_nm=WAVELENGTH[cond["colour"]], n_immersion=N_IMM,
                                    n_medium=N_MED[cond["medium"]], pixel_x_nm=pixel[0], pixel_y_nm=pixel[1])
            for variant in args.variants:
                t0 = time.perf_counter()
                magnitude = use_pupil(variant)
                model = BeadSettings(zernike_order=ORDER, magnitude=magnitude, blur="free", blur_nm=30.0,
                                     bead_diameter_nm=BEAD_NM)
                run = run_calibration(files, False, optics, model, run=RunSettings(save_model=False), rois=rois,
                                      decisions=decisions)
                row = dict(acquisition=os.path.basename(os.path.normpath(acquisition)), **cond, grid_na=args.grid_na,
                           window_nm=args.window, dz_nm=args.dz, n_med=N_MED[cond["medium"]],
                           wavelength_nm=WAVELENGTH[cond["colour"]], beads=cut)
                row.update(summary(run, variant, pixel, cond, args.grid_na))
                row.update(measure(run, args.z_range))
                row["minutes"] = round((time.perf_counter() - t0) / 60, 2)
                emit(row, args.out)


if __name__ == "__main__":
    main()
