"""How much of M5's beads a single-molecule calibration explains, and what the beads need beyond it.

    python benchmarks/uipsf/real/m5_transfer.py ACQUISITION [--molecules ROW] [--variants D E S HS R] \\
        [--centre 0.5] [--window 1000] [--dz 100] [--chunk 1] [--memory 0.5] \\
        [--out benchmarks/uipsf/results/m5_transfer.jsonl]

The molecules (`m5_molecules.py`, AF647 on the coverslip in buffer, 2 November 2026) learn the objective
without a bead: γ 0.456, the phase to order 8, no blur beyond the pixel and no SAF height beyond the dyes'
own few nm.  The same day's 100 nm beads (``100nm_25beads_stack``, 676 nm, buffer, 3D lens) are fitted here
with that calibration, adding one at a time what a bead may need (the user's, 2 November 2026): the phase
learnt again, a blur σ beyond the pixel's, an effective height h of the supercritical ring or a factor s on
it.  γ stays the molecules' throughout; only the reference R learns it, the minimal model learnt on the
beads alone.  Beads of the molecules' own colour only: γ and the phase belong to one emission (each channel
learns its own, `bead_model`), so the 40 nm beads, all at 600 nm, test the colour as well; they were tried
once (E and R), the phase held in nm of path, not expecting it to hold.

One model for every variant, the product's minimal pupil (`uipsf.model.pupil.MinimalPupil`, no m ≠ 0
magnitude), the bead's sphere (40 or 100 nm) convolved along z (`BeadSettings.bead_diameter_nm`) and the pixel's blur
always under σ (the GUI's only choice):

=====  ==========  ============  ======  ======  ======
       phase       σ (beyond)    h       s       γ
=====  ==========  ============  ======  ======  ======
A      molecules'  0             0       1       held
B      free        0             0       1       held
C      free        free, round   0       1       held
D      free        0             free    1       held
P      molecules'  0             free    1       held
E      free        free, round   free    1       held
S      free        free, round   0       free    held
HS     free        free, round   free    free    held
R      free        free, round   free    1       free
=====  ==========  ============  ======  ======  ======

h 0 is the ring of a point at the glass (``saf_damping`` off), the physics a molecule's model draws; the
sphere's own ring (its dye up to 100 nm above the glass) is not in the product's model, and h stands in for
it.  s is a free factor on the supercritical light (`RingFactor`, as `m5_molecules.RingFactorPupil`): h
dims it the more the steeper the angle, s by one factor at every angle beyond the critical one, and s > 1
gives more light there than a point at the glass, which h cannot.
Every variant on the same central beads and slices, every bead included, so the objectives compare in
nats; `bead_slices.measure` gives each fit's z bend.
"""
from __future__ import annotations

import argparse
import functools
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
from m5_consensus import central  # noqa: E402
from m5_media import N_IMM, N_MED, ORDER, WAVELENGTH, acquisition, measure_beads, subset  # noqa: E402

PIXEL = (104.7, 106.4)
NA = 1.50
MOLECULES = os.path.join(HERE, "..", "results", "m5_molecules.jsonl")
#: what each variant learns beyond the beads' own positions and photons (module docstring)
VARIANTS = {"A": dict(phase=False, sigma=False, h=False, gamma=False),
            "B": dict(phase=True, sigma=False, h=False, gamma=False),
            "C": dict(phase=True, sigma=True, h=False, gamma=False),
            "D": dict(phase=True, sigma=False, h=True, gamma=False),
            "P": dict(phase=False, sigma=False, h=True, gamma=False),
            "E": dict(phase=True, sigma=True, h=True, gamma=False),
            "S": dict(phase=True, sigma=True, h=False, gamma=False, s=True),
            "HS": dict(phase=True, sigma=True, h=True, gamma=False, s=True),
            "R": dict(phase=True, sigma=True, h=True, gamma=True)}
#: a single bead's photons, times the median, by its diameter (`m5_beads.BRIGHT`): a pair of 100 nm beads is
#: about twice as bright; a 40 nm bead's light spreads with its volume, and a pair of them images as one
SINGLE = {40.0: (0.33, 3.0), 100.0: (0.5, 2.0)}


def ring_factor():
    """`MinimalPupil` with a free factor s on the supercritical light, stepped at the critical angle over a
    pupil sample, on top of the height when ``saf_damping`` learns one (`m5_molecules.RingFactorPupil`
    without forcing the height off).  Made on use: the pupil module is torch's, imported late."""
    from uipsf.model.pupil import MinimalPupil
    from uipsf.params import Bounded, ParamSpec

    class RingFactor(MinimalPupil):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            s = (self.grid.rho.real if self.grid.rho.is_complex() else self.grid.rho) * self.grid_na
            self.beyond = torch.sigmoid((s - self.n_med) / (self.grid_na / self.grid.size))

        def specs(self):
            return super().specs() + [ParamSpec("pupil.ring_s", (), init=1.0, transform=Bounded(0.0, 3.0))]

        def symmetric(self, p, cos_imm=None, kappa=None, cutoff=None):
            out = super().symmetric(p, cos_imm, kappa, cutoff)
            if cos_imm is not None:          # a profile at given points: no step there
                return out
            return out * (1 + (p["pupil.ring_s"] - 1) * self.beyond)

    return RingFactor


def molecule_row(spec: str, path: str = MOLECULES) -> dict:
    """The molecules' calibration: the last row of `MOLECULES` matching ``spec`` ("height pb r2": the SAF
    variant, ``pb`` for the pixel's blur under σ, the round)."""
    saf, pb, r = spec.split()
    rows = [json.loads(line) for line in open(path)]
    rows = [r_ for r_ in rows if r_["saf"] == saf and bool(r_.get("pixel_blur")) == (pb == "pb")
            and r_.get("round") == int(r.lstrip("r"))]
    if not rows:
        raise SystemExit(f"no molecule row {spec!r} in {path}")
    return rows[-1]


def summary(run, mol: dict, lam: float) -> dict:
    """What a fit learnt, and how far its phase is from the molecules'."""
    cal = run.cal
    p = cal.result.params.physical()
    pupil = cal.model.channels[0].psf.pupil
    phase = {j: v * lam / (2 * math.pi) for j, v in pupil.coefficients(p, "phase").items()}
    diff = {j: phase[j] - float(mol["phase_nm"].get(str(j), 0.0)) for j in phase}
    sigma = p.get("psf.sigma")
    return dict(used=len(cal.beads.used), objective=round(float(cal.result.objective), 1),
                iterations=cal.result.iterations, stop=cal.result.stop_reason,
                gamma=round(float(pupil.gamma(p)), 4),
                ring_h_nm=round(float(pupil.ring_h(p)), 2) if pupil.ring else 0.0,
                ring_s=round(float(p["pupil.ring_s"]), 4) if "pupil.ring_s" in p else None,
                sigma_nm=(sigma.detach().cpu().reshape(-1).round(decimals=2).tolist() if sigma is not None
                          else [0.0]),
                phase_nm={j: round(v, 2) for j, v in phase.items()},
                main_modes_nm={f"Z{j}": round(phase[j], 2) for j in (5, 6, 7, 8, 11, 22) if j in phase},
                phase_change_rms_nm=round(float(np.sqrt(np.mean([d * d for d in diff.values()]))), 2),
                phase_change_main_nm={f"Z{j}": round(diff[j], 2) for j in (5, 6, 7, 8, 11, 22) if j in diff},
                atlas_mean_deviance=round(float(cal.atlas.mean_deviance), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acquisition")
    ap.add_argument("--molecules", default="height pb r2", help="the molecule row: SAF variant, pb, round")
    ap.add_argument("--molecules-file", default=MOLECULES, help="the molecules' rows (`m5_molecules.py --out`)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--centre", type=float, default=0.5, help="the central fraction of the width and height")
    ap.add_argument("--max-beads", type=int, default=None, help="at most this many, the brightest")
    ap.add_argument("--z-range", type=float, default=600.0)
    ap.add_argument("--window", type=float, default=1000.0)
    ap.add_argument("--dz", type=float, default=100.0)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--memory", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "m5_transfer.jsonl"))
    args = ap.parse_args()
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(args.memory)
    import tifffile
    from smappy.calibrate.pupil import (
        BeadSettings,
        OpticsSettings,
        PSFModelSettings,
        RunSettings,
        bead_rois,
        camera_for,
        run_calibration,
    )

    import uipsf.workflows.beads as beads
    import uipsf.workflows.calibrate as calibrate
    import uipsf.workflows.scoring as scoring
    from uipsf.model.build import Held
    beads.MAX_DZ_NM = args.dz
    calibrate.learn_beads = functools.partial(calibrate.learn_beads, chunk_size=args.chunk, max_dz_nm=args.dz)
    scoring.localise = functools.partial(scoring.localise,
                                         chunk_size=int(2 * args.window // args.dz + 1) * args.chunk)
    import uipsf.model.build as build
    bead_model, minimal_pupil, ring_pupil = calibrate.bead_model, build.MinimalPupil, ring_factor()

    mol = molecule_row(args.molecules, args.molecules_file)
    name, cond, files, bead_nm = acquisition(args.acquisition)
    lam = WAVELENGTH[cond["colour"]]
    rois = bead_rois(files, camera_for(files[0]), dual=False)
    focus, photons, near = measure_beads(rois)
    bright = photons >= 0.2 * np.median(np.sort(photons)[-20:])
    ref = np.median(photons[bright & near])
    single = (photons >= SINGLE[bead_nm][0] * ref) & (photons <= SINGLE[bead_nm][1] * ref)
    keep = [int(i) for i in np.flatnonzero(bright & near & single)]
    shape = tifffile.TiffFile(files[0]).pages[0].shape[-2:]
    inner = central(rois, keep, args.centre, shape)
    if args.max_beads and len(inner) > args.max_beads:
        inner = sorted(sorted(inner, key=lambda i: -photons[i])[:args.max_beads])
    beadset = dict(found=len(photons), in_focus_bright=int((bright & near).sum()), single=len(keep),
                   central=len(inner), photons_median=round(float(np.median(photons[inner])), 0))
    print(name, beadset, flush=True)
    rois = subset(rois, inner, focus, args.window)
    optics = OpticsSettings(NA=NA, wavelength_nm=lam, n_immersion=N_IMM, n_medium=N_MED[cond["medium"]],
                            pixel_x_nm=PIXEL[0], pixel_y_nm=PIXEL[1])

    for v in args.variants:
        learn = VARIANTS[v]
        t0 = time.perf_counter()
        held = Held(phase={} if learn["phase"] else
                    {int(j): 2 * math.pi * float(nm) / lam for j, nm in mol["phase_nm"].items()},
                    gamma=None if learn["gamma"] else float(mol["gamma"]), source=f"molecules {args.molecules}")
        calibrate.bead_model = functools.partial(bead_model, held=held)
        build.MinimalPupil = ring_pupil if learn.get("s") else minimal_pupil
        psf = PSFModelSettings(pixel_blur=True, zernike_order=ORDER, magnitude="minimal",
                               blur="round" if learn["sigma"] else "fixed",
                               blur_nm=30.0 if learn["sigma"] else 0.0, saf_damping=learn["h"])
        run = run_calibration(files, False, optics, BeadSettings(psf=psf, bead_diameter_nm=bead_nm),
                              run=RunSettings(save_model=False), rois=rois,
                              decisions={i: "include" for i in range(len(rois.records))})
        row = dict(script="m5_transfer", acquisition=name, **cond, variant=v, learns=learn,
                   molecules=args.molecules, molecule_box=mol.get("box"),
                   molecule_gamma=mol["gamma"], beads=beadset, window_nm=args.window,
                   dz_nm=args.dz, centre=args.centre)
        row.update(summary(run, mol, lam))
        row.update(measure(run, args.z_range, centre_nm=[rois.records[i]["focus_nm"] for i in run.cal.beads.used]))
        row["minutes"] = round((time.perf_counter() - t0) / 60, 2)
        emit(row, args.out)
        del run
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    calibrate.bead_model, build.MinimalPupil = bead_model, minimal_pupil


if __name__ == "__main__":
    main()
