"""What one focal plane can determine: the twin, with and without phase masks (docs/uipsf/scanned-psfs.md §7a).

    python benchmarks/uipsf/synthetic/single_plane_twin.py [--photons 10000] [--out results/single_plane_twin.jsonl]

A pupil A(k) and its twin A*(−k) give the same intensity at the mirrored height, so in the focal plane itself
they cannot be told apart: the sign of every even phase mode (m even: defocus, astigmatism, spherical) is lost
(`docs/uipsf/conventions.md`, "The twin").  A mask whose design maps onto itself under the twin keeps that
ambiguity -- and the twin takes its centre c to −c as well -- one that does not (a vortex, a double helix)
should break it.  Scanned beams are often taken in one plane only, which is why this is measured first, on
the detection PSF where everything else is in place; the excitation's focusing has the same symmetry.

**What is measured.**  For each mask, a truth -- aberrations of both parities and the mask 0.05/0.03 of the
pupil radius off -- against candidate alternatives within the model (the mask's own kind and charge fixed):

- *twin*: every even mode's sign flipped, the mask's centre at −c;
- *twin, centre kept*: the even modes flipped, the centre at c;
- *centre mirrored*: only the centre at −c (the decentration's direction).

The number is the Poisson Kullback–Leibler divergence of the alternative's expected image from the truth's,
in nats per bead at ``--photons`` per plane and 10 background photons per pixel: the expected log-likelihood
ratio by which one bead's data prefer the truth.  Near 0, the data cannot choose; a few nats per bead over
tens of beads is decisive.  The emitter's position, photons and height are not re-fitted here, so a small
number is an upper bound on what the data can tell (a re-fit can only bring the alternative closer); the
fits that settle it are in ``--fits``.  Planes: one at focus, and three (−400, 0, +400 nm) for comparison.
Optics: index-matched (oil into oil, where the twin is exact for scalar light) and beads on a coverslip in
water (the mismatch breaks it weakly).
"""
from __future__ import annotations

import argparse
import json
import os

import torch

import uipsf
from uipsf.model import MaskSpec, VectorPSF, ZernikePupil, make_mask
from uipsf.optics.zernike import noll_to_nm
from uipsf.params import Params

OIL = uipsf.Optics(NA=1.45, wavelength_nm=670, n_imm=1.518, n_med=1.518, pixel_size_nm=(100.0, 100.0))
WATER = uipsf.Optics(NA=1.45, wavelength_nm=670, n_imm=1.518, n_med=1.33, pixel_size_nm=(100.0, 100.0))
#: the system's aberrations, rad: astigmatism (both), coma (both), trefoil, spherical
ABERRATIONS = {5: 0.2, 6: 0.3, 7: 0.15, 8: -0.1, 9: 0.08, 11: 0.12}
CENTRE = (0.05, 0.03)
#: a tetrapod-like design for the Zernike mask: the astigmatism family up to n = 6, as Shechtman et al. 2015's
#: tetrapods are built from (an illustration, not the DMO plate's design)
TETRAPOD = {6: 1.2, 12: 0.8, 24: 0.4}
MASKS = {"none": None, "vortex": MaskSpec(kind="vortex"), "top-hat": MaskSpec(kind="top-hat"),
         "half-moon": MaskSpec(kind="half-moon"), "double helix": MaskSpec(kind="double helix"),
         "tetrapod (Zernike)": MaskSpec(kind="zernike", zernike=TETRAPOD)}
ROI = (25, 25)
#: the beads' spread about the region's centre and the focus, nm (x, y, z)
SPREAD = torch.tensor([160.0, 160.0, 120.0])


def model(optics, spec, n_max=4):
    grid = uipsf.PupilGrid.build(optics, 64, device="cpu", dtype=torch.float64)
    pupil = ZernikePupil(grid, n_max=n_max, magnitude_modes=[], mask=make_mask(spec, grid))
    return VectorPSF(grid, pupil, ROI, fit_sigma=False)


def values(psf, aberrations, centre):
    p = Params(psf.specs(), device="cpu", dtype=torch.float64).physical()
    phase = torch.zeros(len(psf.pupil.phase_modes), dtype=torch.float64)
    for j, c in aberrations.items():
        phase[psf.pupil.phase_modes.index(j)] = c
    p["pupil.phase"] = phase
    if "mask.centre" in p:
        p["mask.centre"] = torch.tensor(centre, dtype=torch.float64)
    return p


def twin(aberrations):
    """Every even mode's sign flipped: the twin's phase (conventions.md: c_j → −(−1)^m c_j)."""
    return {j: (-c if noll_to_nm(j)[1] % 2 == 0 else c) for j, c in aberrations.items()}


def images(psf, p, planes, photons, background=10.0):
    xyz = torch.zeros(1, len(planes), 3, dtype=torch.float64)
    xyz[0, :, 2] = torch.tensor(planes, dtype=torch.float64)
    return photons * psf(p, xyz)[0] + background


def kl(mu_true, mu_alt):
    """The Poisson KL divergence of ``mu_alt`` from ``mu_true``, summed: nats per bead."""
    return float((mu_true * torch.log(mu_true / mu_alt) - mu_true + mu_alt).sum())


def table(photons):
    rows = []
    for optics_name, optics in (("oil, matched", OIL), ("water", WATER)):
        for mask_name, spec in MASKS.items():
            psf = model(optics, spec)
            minus = tuple(-c for c in CENTRE)
            truth = values(psf, ABERRATIONS, CENTRE)
            alternatives = {"twin": values(psf, twin(ABERRATIONS), minus),
                            "twin, centre kept": values(psf, twin(ABERRATIONS), CENTRE),
                            "centre mirrored": values(psf, ABERRATIONS, minus)}
            for planes_name, planes in (("focus", [0.0]), ("three planes", [-400.0, 0.0, 400.0])):
                mu = images(psf, truth, planes, photons)
                row = dict(optics=optics_name, mask=mask_name, planes=planes_name, photons=photons)
                for alt_name, q in alternatives.items():
                    row[alt_name] = round(kl(mu, images(psf, q, planes, photons)), 3)
                rows.append(row)
                print(f"{optics_name:13s} {mask_name:19s} {planes_name:13s} "
                      + "  ".join(f"{k} {row[k]:8.2f}" for k in alternatives))
    return rows


def fits(photons, n_beads=20, seed=0):
    """Each mask fitted to one simulated plane twice, from the truth and from the twin (even modes flipped,
    centre at −c, each bead's height mirrored): the phase, centre, positions, photons and backgrounds free.
    Equal objectives mean the plane cannot choose; the twin start returning to the truth means it can."""
    from uipsf.model import ForwardModel
    from uipsf.model.intensity import Bleaching
    from uipsf.simulate import parameters, simulate, template
    from uipsf.workflows.beads import fit_beads
    rows = []
    for optics_name, optics in (("oil, matched", OIL), ("water", WATER)):
        for mask_name, spec in MASKS.items():
            if spec is not None and "centre" in set(make_mask(spec, uipsf.PupilGrid.build(optics, 8)).placements):
                spec = MaskSpec(**{**spec.__dict__, "free": ("centre",)})
            psf = model(optics, spec)
            fm = ForwardModel(psf, intensity=Bleaching(order=0))
            g = torch.Generator().manual_seed(seed)
            xyz = (torch.rand(n_beads, 3, generator=g, dtype=torch.float64) - 0.5) * SPREAD
            b0 = template(n_beads, uipsf.Batch.z_from_objective(torch.tensor([0.0])), ROI, dtype=torch.float64,
                          device="cpu")

            def start(aberrations, centre, z_sign, psf=psf, b0=b0, fm=fm, xyz=xyz, spec=spec):
                phase = values(psf, aberrations, centre)["pupil.phase"]
                v = {"pupil.phase": phase, "emitter.xyz": xyz * torch.tensor([1.0, 1.0, z_sign]),
                     "emitter.photons": torch.full((n_beads,), photons),
                     "emitter.background": torch.full((n_beads,), 10.0), "psf.sigma": torch.tensor([0.0, 0.0])}
                if spec is not None and "centre" in spec.free:
                    v["mask.centre"] = torch.tensor(centre)
                return parameters(fm, b0, v, dtype=torch.float64, device="cpu")
            beads = simulate(fm, start(ABERRATIONS, CENTRE, 1.0), b0, seed=seed + 1)
            row = dict(optics=optics_name, mask=mask_name, photons=photons, beads=n_beads)
            for name, s in (("truth start", start(ABERRATIONS, CENTRE, 1.0)),
                            ("twin start", start(twin(ABERRATIONS), tuple(-c for c in CENTRE), -1.0))):
                res = fit_beads(fm, beads, start=s, first_order=None, max_dz_nm=None)
                p = res.params.physical()
                ph = dict(zip(psf.pupil.phase_modes, p["pupil.phase"].tolist(), strict=True))
                centre = [round(c, 3) for c in p["mask.centre"].tolist()] if "mask.centre" in p else None
                row[name] = dict(objective=round(res.objective, 2), phase={j: round(ph[j], 3) for j in ABERRATIONS},
                                 centre=centre)
            d = row["twin start"]["objective"] - row["truth start"]["objective"]
            row["twin minus truth (nats)"] = round(d, 2)
            print(f"{optics_name:13s} {mask_name:19s} Δ {d:9.2f} nats   twin start → "
                  f"{row['twin start']['phase']}  centre {row['twin start']['centre']}")
            rows.append(row)
    return rows


def precision(photons, n_beads=20, seed=0):
    """The CRLB of each mode and of the mask's centre, fitted from the truth, in one plane against three: the
    local determinacy beside the twin's global one.  In focus an even mode's first-order effect on the image
    vanishes at zero, so a small even aberration is determined worse than a large one; the "weak" truth has the
    even modes at a fifth."""
    from uipsf.model import ForwardModel
    from uipsf.model.intensity import Bleaching
    from uipsf.simulate import parameters, simulate, template
    from uipsf.workflows.beads import fit_beads
    weak = {j: (c / 5 if noll_to_nm(j)[1] % 2 == 0 else c) for j, c in ABERRATIONS.items()}
    rows = []
    for mask_name in ("none", "vortex", "top-hat", "double helix"):
        spec = MASKS[mask_name]
        if spec is not None:
            spec = MaskSpec(**{**spec.__dict__, "free": ("centre",)})
        psf = model(OIL, spec)
        fm = ForwardModel(psf, intensity=Bleaching(order=0))
        for truth_name, truth in (("strong", ABERRATIONS), ("weak even", weak)):
            for planes_name, planes in (("focus", [0.0]), ("three planes", [-400.0, 0.0, 400.0])):
                g = torch.Generator().manual_seed(seed)
                xyz = (torch.rand(n_beads, 3, generator=g, dtype=torch.float64) - 0.5) * SPREAD
                b0 = template(n_beads, uipsf.Batch.z_from_objective(torch.tensor(planes)), ROI, dtype=torch.float64,
                              device="cpu")
                v = {"pupil.phase": values(psf, truth, CENTRE)["pupil.phase"], "emitter.xyz": xyz,
                     "emitter.photons": torch.full((n_beads,), photons),
                     "emitter.background": torch.full((n_beads,), 10.0), "psf.sigma": torch.tensor([0.0, 0.0])}
                if spec is not None:
                    v["mask.centre"] = torch.tensor(CENTRE)
                start = parameters(fm, b0, v, dtype=torch.float64, device="cpu")
                beads = simulate(fm, start, b0, seed=seed + 1)
                res = fit_beads(fm, beads, start=start, first_order=None, max_dz_nm=None)
                sd = dict(zip(psf.pupil.phase_modes, res.std["pupil.phase"].tolist(), strict=True))
                row = dict(mask=mask_name, truth=truth_name, planes=planes_name, photons=photons, beads=n_beads,
                           crlb_rad={j: round(sd[j], 4) for j in ABERRATIONS},
                           crlb_centre=[round(c, 4) for c in res.std["mask.centre"].tolist()]
                           if "mask.centre" in res.std else None)
                print(f"{mask_name:13s} {truth_name:10s} {planes_name:13s} CRLB {row['crlb_rad']}  centre "
                      f"{row['crlb_centre']}")
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--photons", type=float, default=10_000.0)
    ap.add_argument("--fits", action="store_true", help="also fit one plane from the truth and from the twin")
    ap.add_argument("--precision", action="store_true", help="also the CRLBs in one plane against three")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results",
                                                  "single_plane_twin.jsonl"))
    args = ap.parse_args()
    rows = (table(args.photons) + (fits(args.photons) if args.fits else [])
            + (precision(args.photons) if args.precision else []))
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main()
