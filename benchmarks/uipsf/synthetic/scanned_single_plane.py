"""What one scanned plane can determine of a beam: the twin, the precision, and what fills a donut's zero
(docs/uipsf/scanned-psfs.md §7a, S3).

    python benchmarks/uipsf/synthetic/scanned_single_plane.py [--fits] [--precision] [--zero]

The widefield study (`single_plane_twin.py`) found that one focal plane loses the sign of every even phase mode
unless a mask breaks the twin.  A scanned beam is the same focusing problem with an input beam and a
polarisation in front of it, and its scan image is the beam's intensity point-reflected about the emitter
(`model.scanned`).  The twin maps the beam's spectrum s(k) to s*(−k): the even modes flip, a mask's or the beam's
centre goes to −c, a vortex's charge flips -- and so does a circular polarisation's handedness, which a model with
the λ/4 held cannot follow.  Whether that alone breaks the twin is the first question.

**The beams** (`PATTERNS`): MINFLUX's and STED's, at 640 nm through NA 1.4 oil, SimuFLUX's fill: a Gaussian
(circular), the 2-D donut (vortex, circular), the 3-D donut (top-hat, circular) and a 1-D donut (half-moon, x
polarised along its dark line: 1 % of the peak on the line, where y polarisation fills it to 46 %).
Aberrations as in the widefield study: astigmatism, coma, trefoil, spherical;
the mask 0.05/0.03 of the pupil radius off.  Dyes (open detector, `sample_weights`), scanned 33 × 33 at 25 nm,
N = 3·10⁴ (about 10⁴ counts in a Gaussian's scan), 1 background count per scan position.

- *table*: the Poisson KL divergence of each alternative's expected scan from the truth's, nats per emitter --
  "twin" (even modes, centre and handedness flipped: exact for scalar light), "twin, λ/4 held" (the handedness
  kept, as a model with the polarisation held must), "centre mirrored" -- in one plane and in three (−300, 0,
  +300 nm: the beam refocused by the SLM or a deformable mirror; on the coverslip the same as a stage).
- *--fits*: 20 emitters in one plane, the phase (order 4), the mask's centre and every emitter's position,
  photons and background free, fitted from the truth and from the twin with the λ/4 held.
- *--precision*: the CRLBs of the modes and the centre in one plane, one plane with the even modes held (from a
  widefield calibration of the objective, §7a's first remedy) and three planes.
- *--zero*: what fills a vortex's zero, the minimum against the crest: the handedness, the sample (dye, a
  fluorescent bead's fixed dyes, gold), a bead's size, astigmatism and the mask's decentration (§7, "ε has many
  sources").
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

import uipsf
from uipsf.model import (
    BeamSpec,
    Bleaching,
    ExcitationBeam,
    ForwardModel,
    MaskSpec,
    ScannedBeamPSF,
    ZernikePupil,
    make_mask,
)
from uipsf.optics.zernike import noll_to_nm
from uipsf.params import Params

SCAN_NM = 25.0
OIL = uipsf.Optics(NA=1.4, wavelength_nm=640, n_imm=1.518, n_med=1.518, pixel_size_nm=(SCAN_NM, SCAN_NM))
WATER = uipsf.Optics(NA=1.4, wavelength_nm=640, n_imm=1.518, n_med=1.33, pixel_size_nm=(SCAN_NM, SCAN_NM))
ABERRATIONS = {5: 0.2, 6: 0.3, 7: 0.15, 8: -0.1, 9: 0.08, 11: 0.12}
CENTRE = (0.05, 0.03)
#: the beam's mask and polarisation
PATTERNS = {"Gaussian": (None, "circular"), "vortex": (MaskSpec(kind="vortex"), "circular"),
            "top-hat": (MaskSpec(kind="top-hat"), "circular"), "half-moon": (MaskSpec(kind="half-moon"), "x")}
ROI = (33, 33)
PHOTONS, BACKGROUND = 3e4, 1.0
#: the emitters' spread about the scan's centre and the focus, nm (x, y, z)
SPREAD = torch.tensor([100.0, 100.0, 120.0])
PLANES = {"one plane": [0.0], "three planes": [-300.0, 0.0, 300.0]}


def even(j: int) -> bool:
    return noll_to_nm(j)[1] % 2 == 0


def model(optics, pattern, held=None, sample="dye", bead_radius_nm=0.0):
    mask, polarisation = PATTERNS[pattern]
    grid = uipsf.PupilGrid.build(optics, 64, device="cpu", dtype=torch.float64)
    if mask is not None:
        mask = MaskSpec(**{**mask.__dict__, "free": ("centre",)})
    pupil = ZernikePupil(grid, n_max=4, magnitude_modes=[], mask=make_mask(mask, grid, aperture_na=optics.NA),
                         held_phase=held)
    return ScannedBeamPSF(ExcitationBeam(grid, pupil, BeamSpec(polarisation=polarisation)), ROI, sample=sample,
                          bead_radius_nm=bead_radius_nm)


def values(psf, aberrations, centre, handedness=1):
    p = Params(psf.specs(), device="cpu", dtype=torch.float64).physical()
    p["pupil.phase"] = torch.tensor([aberrations.get(j, 0.0) for j in psf.pupil.phase_modes], dtype=torch.float64)
    if "mask.centre" in p:
        p["mask.centre"] = torch.tensor(centre, dtype=torch.float64)
    if "beam.ellipticity" in p:
        p["beam.ellipticity"] = p["beam.ellipticity"] * handedness
    return p


def twin(aberrations):
    return {j: (-c if even(j) else c) for j, c in aberrations.items()}


def scans(psf, p, planes, xyz=None):
    if xyz is None:
        xyz = torch.zeros(1, len(planes), 3, dtype=torch.float64)
        xyz[0, :, 2] = torch.tensor(planes, dtype=torch.float64)
    return PHOTONS * psf(p, xyz) + BACKGROUND


def kl(mu_true, mu_alt):
    return float((mu_true * torch.log(mu_true / mu_alt) - mu_true + mu_alt).sum())


def table():
    rows = []
    minus = tuple(-c for c in CENTRE)
    for optics_name, optics in (("oil, matched", OIL), ("water", WATER)):
        for pattern in PATTERNS:
            psf = model(optics, pattern)
            truth = values(psf, ABERRATIONS, CENTRE)
            alternatives = {"twin": values(psf, twin(ABERRATIONS), minus, -1),
                            "twin, λ/4 held": values(psf, twin(ABERRATIONS), minus),
                            "centre mirrored": values(psf, ABERRATIONS, minus)}
            for planes_name, planes in PLANES.items():
                mu = scans(psf, truth, planes)
                row = dict(study="table", optics=optics_name, pattern=pattern, planes=planes_name)
                for name, q in alternatives.items():
                    row[name] = round(kl(mu, scans(psf, q, planes)), 3)
                rows.append(row)
                print(f"{optics_name:13s} {pattern:9s} {planes_name:13s} "
                      + "  ".join(f"{k} {row[k]:9.2f}" for k in alternatives), flush=True)
    return rows


def _problem(psf, planes, n, seed):
    from uipsf.simulate import template
    fm = ForwardModel(psf, intensity=Bleaching(order=0))
    g = torch.Generator().manual_seed(seed)
    xyz = (torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5) * SPREAD
    b0 = template(n, uipsf.Batch.z_from_objective(torch.tensor(planes)), ROI, dtype=torch.float64, device="cpu")
    return fm, xyz, b0


def _start(fm, b0, psf, xyz, aberrations, centre, z_sign=1.0, handedness=1):
    from uipsf.simulate import parameters
    q = values(psf, aberrations, centre, handedness)
    n = xyz.shape[0]
    v = {"pupil.phase": q["pupil.phase"], "emitter.xyz": xyz * torch.tensor([1.0, 1.0, z_sign]),
         "emitter.photons": torch.full((n,), PHOTONS), "emitter.background": torch.full((n,), BACKGROUND)}
    if "mask.centre" in q:
        v["mask.centre"] = q["mask.centre"]
    return parameters(fm, b0, v, dtype=torch.float64, device="cpu")


def fits(n=20, seed=0):
    """One plane, fitted from the truth and from the twin a model with the λ/4 held can reach."""
    from uipsf.simulate import simulate
    from uipsf.workflows.beads import fit_beads
    rows = []
    for optics_name, optics in (("oil, matched", OIL), ("water", WATER)):
        for pattern in PATTERNS:
            psf = model(optics, pattern)
            fm, xyz, b0 = _problem(psf, [0.0], n, seed)
            data = simulate(fm, _start(fm, b0, psf, xyz, ABERRATIONS, CENTRE), b0, seed=seed + 1)
            row = dict(study="fits", optics=optics_name, pattern=pattern, emitters=n)
            for name, s in (("truth start", _start(fm, b0, psf, xyz, ABERRATIONS, CENTRE)),
                            ("twin start", _start(fm, b0, psf, xyz, twin(ABERRATIONS), tuple(-c for c in CENTRE),
                                                  -1.0))):
                res = fit_beads(fm, data, start=s, first_order=None, max_dz_nm=None)
                p = res.params.physical()
                ph = dict(zip(psf.pupil.phase_modes, p["pupil.phase"].tolist(), strict=True))
                row[name] = dict(objective=round(res.objective, 2), phase={j: round(ph[j], 3) for j in ABERRATIONS},
                                 centre=[round(c, 3) for c in p["mask.centre"].tolist()] if "mask.centre" in p
                                 else None)
            d = row["twin start"]["objective"] - row["truth start"]["objective"]
            row["twin minus truth (nats)"] = round(d, 2)
            print(f"{optics_name:13s} {pattern:9s} Δ {d:9.2f} nats   twin start → {row['twin start']['phase']}  "
                  f"centre {row['twin start']['centre']}", flush=True)
            rows.append(row)
    return rows


def precision(n=20, seed=0):
    """The CRLBs from a fit started at the truth: one plane, one plane with the even modes held at the truth,
    three planes."""
    from uipsf.simulate import simulate
    from uipsf.workflows.beads import fit_beads
    held = {j: c for j, c in ABERRATIONS.items() if even(j)}
    rows = []
    for pattern in PATTERNS:
        for case, planes, h in (("one plane", [0.0], None), ("one plane, even held", [0.0], held),
                                ("three planes", PLANES["three planes"], None)):
            psf = model(OIL, pattern, held=h)
            fm, xyz, b0 = _problem(psf, planes, n, seed)
            start = _start(fm, b0, psf, xyz, ABERRATIONS, CENTRE)
            data = simulate(fm, start, b0, seed=seed + 1)
            res = fit_beads(fm, data, start=start, first_order=None, max_dz_nm=None)
            sd = dict(zip(psf.pupil.phase_modes, res.std["pupil.phase"].tolist(), strict=True))
            row = dict(study="precision", pattern=pattern, case=case, emitters=n,
                       crlb_rad={j: round(sd[j], 4) for j in ABERRATIONS if j in sd},
                       crlb_centre=[round(c, 4) for c in res.std["mask.centre"].tolist()]
                       if "mask.centre" in res.std else None)
            if pattern == "half-moon":      # the mask is the same all along its step: no centre along it
                row["crlb_centre"][0] = None
            print(f"{pattern:9s} {case:21s} CRLB {row['crlb_rad']}  centre {row['crlb_centre']}", flush=True)
            rows.append(row)
    return rows


def zero_depth(psf, p):
    """The minimum of a scan through the focus within 100 nm of the axis against its maximum."""
    I = psf(p, torch.zeros(1, 1, 3, dtype=torch.float64))[0, 0]
    Y, X = I.shape
    y, x = torch.meshgrid(torch.arange(Y) - Y // 2, torch.arange(X) - X // 2, indexing="ij")
    near = (x ** 2 + y ** 2).double().sqrt() * SCAN_NM <= 100
    return float(I[near].min() / I.max())


def zero():
    """What fills the 2-D donut's zero, one cause at a time from a perfect vortex."""
    rows = []

    def row(cause, **kw):
        sample, radius = kw.pop("sample", "dye"), kw.pop("bead_radius_nm", 0.0)
        handedness, aberrations = kw.pop("handedness", 1), kw.pop("aberrations", {})
        centre, chi = kw.pop("centre", (0.0, 0.0)), kw.pop("chi", None)
        psf = model(OIL, "vortex", sample=sample, bead_radius_nm=radius)
        p = values(psf, aberrations, centre, handedness)
        if chi is not None:
            p["beam.ellipticity"] = torch.tensor(chi, dtype=torch.float64)
        r = dict(study="zero", cause=cause, zero=round(zero_depth(psf, p), 5))
        print(f"{cause:40s} {r['zero']:.5f}", flush=True)
        rows.append(r)

    row("none")
    for sample in ("dye", "bead", "gold"):
        row(f"wrong handedness, {sample}", handedness=-1, sample=sample)
    for radius in (10.0, 20.0, 50.0):
        row(f"bead of {2 * radius:.0f} nm", sample="bead", bead_radius_nm=radius)
    row("astigmatism 0.1 rad", aberrations={6: 0.1})
    row("coma 0.1 rad", aberrations={8: 0.1})
    row("spherical 0.1 rad", aberrations={11: 0.1})
    row("mask 0.05 of the pupil off", centre=(0.05, 0.0))
    row("λ/4 off: χ = 40° instead of 45°", chi=math.radians(40))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fits", action="store_true")
    ap.add_argument("--precision", action="store_true")
    ap.add_argument("--zero", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results",
                                                  "scanned_single_plane.jsonl"))
    args = ap.parse_args()
    rows = (table() + (fits() if args.fits else []) + (precision() if args.precision else [])
            + (zero() if args.zero else []))
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main()
