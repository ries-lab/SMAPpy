"""A radial pupil magnitude: a low-order centre and an edge of its own, against free Zernikes.

    python benchmarks/uipsf/synthetic/radial_magnitude.py [errors…] [--fits …] [--seeds 0]
                                             [--out benchmarks/uipsf/results/radial_magnitude.jsonl]

The symmetric magnitude is the part that is stable across days (`docs/uipsf/architecture.md`, "In situ
on matched M2 pairs") and that in situ holds from beads; what it has to describe sits mostly at the edge
(a soft edge, an effective NA, the apodisation), which radial Zernikes up to order 8 approximate only with
ringing that leaks into the centre -- where a bump trades with the blur σ (the user's caution, October
2026).  So here the centre keeps only low radial orders and the edge gets its own term, two ways:

- *spline*: an added piecewise-linear profile, hats at ρ = 0.80, 0.85, 0.90, 0.95, 1.00, zero inside 0.75;
- *logistic*: a multiplied edge 1 / (1 + exp((ρ − ρ_c)/w)), ρ_c in (0.85, 1.10) and w in (0.005, 0.1):
  an effective NA and a width, the physical reading (roadmap item 7).

The first runs (`results/radial_magnitude.jsonl`) kept the m ≠ 0 Zernikes as they are; since then the edge
pupils are rotationally symmetric unless named "… asym" (the user's choice, October 2026).
The truths are `magnitude.py`'s amplitude errors and a
bead-like magnitude, M2's measured symmetric modes (Noll 4, 11, 22, 37 = −0.08, −0.07, −0.05, −0.03),
which Zernikes describe exactly.  Per truth and fit: the objective, the phase bias against its CRLB, σ, and
each slice of the stack localised alone on the fitted model (`scoring.slice_fits`): the z scale (slope)
and what bends about it, the median over beads per slice within ±600 nm -- the z distortion over the
depth of field, which is what matters to the user.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace

import numpy as np
import torch
from magnitude import ROI, P, describe, simulate, with_fields
from magnitude import truth as amplitude_truth

import uipsf
from uipsf.model import ForwardModel
from uipsf.noise import Poisson
from uipsf.optics.zernike import noll_to_nm, zernike_basis, zernike_modes
from uipsf.params import Bounded, ParamSpec
from uipsf.workflows import fit_beads
from uipsf.workflows.scoring import slice_fits

OPTICS = dict(wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))
BEADLIKE = {4: -0.08, 11: -0.07, 22: -0.05, 37: -0.03}
KNOTS = (0.80, 0.85, 0.90, 0.95, 1.00)
INNER = 0.75
#: the width of a hard edge in ρ: the NA cutoff and the step onto the supercritical ring
HARD_W = 0.01
ERRORS = ("matched", "bead-like", "NA 1.40", "soft edge", "apodisation")
FITS = ("flat", "zernike", "spline", "spline c11", "logistic", "logistic c11", "mono", "mono b", "mono nu",
        "mono b nu")


#: the free monotone profile's steps: exp(−Σ d_k S((ρ − k/STEPS)·STEPS)), d_k ≥ 0, S the smoothstep
STEPS = 8


def _smooth_steps(rho: torch.Tensor) -> torch.Tensor:
    """[STEPS, ...] smoothsteps 3t² − 2t³ rising over each of `STEPS` equal intervals of ρ in [0, 1]: with
    non-negative weights their sum rises monotonically and smoothly (C¹), so exp(−sum) is any smooth,
    monotone fall from the centre, the edge's sharp parts (the ring, the cutoff) left to their own terms."""
    t = torch.clamp(rho[None] * STEPS - torch.arange(STEPS, dtype=rho.dtype, device=rho.device).reshape(
        (STEPS,) + (1,) * rho.dim()), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def hats(rho: torch.Tensor) -> torch.Tensor:
    """[K, P, P]: piecewise-linear hats at `KNOTS`, the first rising from `INNER`, the last ending at 1."""
    edges = (INNER, *KNOTS, KNOTS[-1] + (KNOTS[-1] - KNOTS[-2]))
    out = []
    for k in range(1, len(edges) - 1):
        a, b, c = edges[k - 1], edges[k], edges[k + 1]
        out.append(torch.clamp(torch.minimum((rho - a) / (b - a), (c - rho) / (c - b)), min=0.0))
    return torch.stack(out)


def _expm1_ratio(x):
    """(1 − e^(−x))/x, 1 at x = 0 (its series below 10⁻⁴)."""
    small = x < 1e-4
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1 - x / 2, -torch.expm1(-safe) / safe)


def _sphere_mean(u):
    """6(u − 2 + (u + 2)e^(−u))/u³: e^(−s h) averaged over a sphere's slices h(d − h), u = s·d; 1 at u = 0."""
    # the closed form cancels to nothing in float32 below u ~ 0.1; its series there (next term ~u⁵/1000)
    small = u < 0.5
    safe = torch.where(small, torch.ones_like(u), u)
    exact = 6 * (safe - 2 + (safe + 2) * torch.exp(-safe)) / safe ** 3
    return torch.where(small, 1 - u / 2 + 3 * u ** 2 / 20 - u ** 3 / 30 + u ** 4 / 168, exact)


class EdgePupil(uipsf.ZernikePupil):
    """A Zernike pupil whose magnitude is a low-order centre and an edge term, rotationally symmetric (the
    user's choice) unless ``asymmetric`` keeps the m ≠ 0 Zernikes too.

    ``centre``: Noll indices of m = 0 kept as Zernikes (``(4,)``, ``(4, 11)``), or ``"mono"``: no symmetric
    Zernike, the magnitude multiplied by exp(−aρ² − bρ⁴) with a, b ≥ 0 (``b`` only with ``quartic``), which
    with the edge makes the symmetric profile fall monotonically from the centre (the user's assumption).
    ``edge``: ``"spline"`` (added hats), ``"logistic"`` (multiplied 1 / (1 + exp((ρ − ρ_c)/w))) or
    ``"richards"`` (the generalised logistic (1 + ν exp((ρ − ρ_c)/w))^(−1/ν): ν ≠ 1 makes the edge asymmetric,
    a sharp shoulder and a long tail or the reverse; still monotone).  ``saf``: the supercritical light damped
    as from an emitter h above the coverslip, exp(−2πκh) with κ = √((ρ·NA)² − n_med²)/λ above the critical
    angle, one parameter h ≥ 0: a bead's dye sits up to its diameter above the glass, which the bead model
    (no depth term) puts at the glass; with ``saf_step`` also a factor s on the whole supercritical ring
    (stepped at the critical angle over `HARD_W`): the ring's transmission, whatever lowers it.  ``edge="hard"``
    is an NA cutoff at a learnt ρ_c of fixed width `HARD_W` (about a third of a pupil sample at P = 64),
    ``"none"`` the nominal aperture alone.  With ``centre="mono"`` and either, the subcritical magnitude is
    exp(−aρ²) and nothing else: the user's model (October 2026), everything that falls fast confined to the
    supercritical ring, which emitters deeper than its decay length never light.  ``centre="cos"``: the
    subcritical bend cos θ_imm^γ instead (γ ≥ 0), the shape an apodisation error has, steepening towards the
    edge where exp(−aρ²) does not.  ``centre="steps"``: any smooth, monotone fall, exp(−Σ d_k S_k(ρ)) over
    `STEPS` smoothsteps, d_k ≥ 0 (`_smooth_steps`): what cos θ^γ is checked against (the user's, October
    2026); ``steps`` puts that fall on top of cos θ^γ as a correction ("cos steps …"), which holds either
    alone (d = 0, or γ = 0) and stays monotone (both factors fall), so its gain over cos θ^γ is what the
    apodisation's shape misses; ``signed`` ("cos signed …") lets the steps go either way, a smooth correction
    that may rise.  ``saf_sd``: the spread of the dye's heights, the second cumulant,
    exp(−2πκh + 4π²κ²σ²): the intensity averaged over the heights, square-rooted, to second order.
    ``ring="layer"``: the dye uniform over h₀ … h₀ + L instead, exactly √(e^(−4πκh₀)(1 − e^(−x))/x) with
    x = 4πκL, positive and monotone for every h₀, L (the second order grows without bound at large κσ);
    ``ring="bead"``: a sphere of ``bead_nm`` (`BEAD_NM` unless told) on the glass, no parameter,
    √(6(u − 2 + (u + 2)e^(−u))/u³) with u = 4πκd.

    Held, for a model whose objective's part is fixed (`real/m5_consensus.py`): ``asym_modes`` names the m ≠ 0
    modes kept (instead of every one to ``asym_order``), ``held_magnitude`` holds some at given values
    ({Noll: value}, `ZernikePupil`), and ``hold`` holds any of this pupil's own parameters ({"pupil.centre_g":
    0.4, "pupil.edge_rho": 0.993}), which are then neither learnt nor counted."""

    def __init__(self, grid, n_max: int, centre=(4,), edge: str = "logistic", quartic: bool = False,
                 asymmetric: bool = False, saf: bool = False, saf_step: bool = False, saf_sd: bool = False,
                 ring: str = "height", asym_order: int | None = None, steps: bool = False,
                 signed: bool = False, asym_modes=None, held_magnitude: dict | None = None,
                 hold: dict | None = None, bead_nm: float | None = None):
        every = zernike_modes(n_max, exclude=("piston",))
        self.mono = centre in ("mono", "cos", "steps")
        self.cos = centre == "cos"
        self.steps = centre == "steps" or steps or signed
        self.signed = signed
        keep = () if self.mono else tuple(centre)
        top = n_max if asym_order is None else asym_order
        modes = [j for j in every if (asymmetric and noll_to_nm(j)[1] != 0 and noll_to_nm(j)[0] <= top
                                      and (asym_modes is None or j in asym_modes)) or j in keep]
        super().__init__(grid, n_max=n_max, magnitude_modes=modes, held_magnitude=held_magnitude)
        self.hold = dict(hold or {})
        self.bead_nm = BEAD_NM if bead_nm is None else float(bead_nm)
        self.edge, self.quartic, self.saf, self.saf_step, self.saf_sd = edge, quartic, saf, saf_step, saf_sd
        self.ring = ring
        o = grid.optics
        self.saf_scale = (o.NA, o.n_med, o.wavelength_nm)
        self.n_imm = o.n_imm
        # the NA cutoff's width: half a pupil sample, so that the samples it crosses give its position a gradient
        self.cut_w = max(HARD_W, 1.0 / grid.size)
        self.rho = grid.rho.real if grid.rho.is_complex() else grid.rho
        self.hat_basis = hats(self.rho).to(self.magnitude_basis.dtype)

    def truncated(self, n_max: int):
        new = super().truncated(n_max)
        out = object.__new__(EdgePupil)
        out.__dict__.update(new.__dict__)
        for k in ("edge", "quartic", "mono", "cos", "steps", "signed", "rho", "hat_basis", "saf", "saf_step", "saf_sd",
                  "saf_scale", "n_imm", "ring", "cut_w", "hold", "bead_nm"):
            setattr(out, k, getattr(self, k))
        return out

    def specs(self) -> list[ParamSpec]:
        extra = []
        if self.cos:
            extra.append(ParamSpec("pupil.centre_g", (), init=0.05, transform=Bounded(0.0, 2.0)))
        if self.steps:
            # monotone: every step falls; signed: a correction either way, smooth but no longer monotone
            low, init = (-3.0, 0.0) if self.signed else (0.0, 0.02)
            extra.append(ParamSpec("pupil.centre_d", (STEPS,), init=init, transform=Bounded(low, 3.0)))
        if self.mono and not (self.cos or self.steps):
            extra.append(ParamSpec("pupil.centre_a", (), init=0.05, transform=Bounded(0.0, 2.0)))
            if self.quartic:
                extra.append(ParamSpec("pupil.centre_b", (), init=0.02, transform=Bounded(0.0, 2.0)))
        if self.saf and self.ring in ("height", "layer"):
            extra.append(ParamSpec("pupil.saf_h", (), init=30.0, transform=Bounded(0.0, 400.0)))
        if self.saf and self.ring == "layer":
            extra.append(ParamSpec("pupil.saf_l", (), init=100.0, transform=Bounded(1.0, 3000.0)))
        if self.saf_sd:
            extra.append(ParamSpec("pupil.saf_sd", (), init=10.0, transform=Bounded(0.0, 200.0)))
        if self.saf_step:
            extra.append(ParamSpec("pupil.saf_s", (), init=0.9, transform=Bounded(0.0, 1.5)))
        if self.edge == "spline":
            extra.append(ParamSpec("pupil.edge", (len(KNOTS),)))
        elif self.edge == "hard":
            extra.append(ParamSpec("pupil.edge_rho", (), init=0.99, transform=Bounded(0.85, 1.05)))
        elif self.edge != "none":
            extra += [ParamSpec("pupil.edge_rho", (), init=1.05, transform=Bounded(0.85, 1.10)),
                      ParamSpec("pupil.edge_width", (), init=0.03, transform=Bounded(0.005, 0.1))]
            if self.edge == "richards":
                extra.append(ParamSpec("pupil.edge_nu", (), init=1.0, transform=Bounded(0.1, 10.0)))
        return [replace(s, init=self.hold[s.name], fixed=True) if s.name in self.hold else s
                for s in super().specs() + extra]

    def symmetric(self, p, rho):
        """The symmetric factor at ``rho`` (the hats added separately): centre × edge."""
        out = torch.ones_like(rho)
        NA, n_med, lam = self.saf_scale
        if self.steps:
            out = out * torch.exp(-torch.tensordot(p["pupil.centre_d"], _smooth_steps(rho), dims=([-1], [0])))
        if self.cos:
            cos_imm = torch.sqrt(torch.clamp(1 - (rho * NA / self.n_imm) ** 2, min=1e-6))
            out = out * cos_imm ** p["pupil.centre_g"][..., None, None]
        if self.mono and not (self.cos or self.steps):
            a = p["pupil.centre_a"][..., None, None]
            b = p["pupil.centre_b"][..., None, None] if self.quartic else 0.0
            out = out * torch.exp(-a * rho ** 2 - b * rho ** 4)
        if self.saf and self.ring != "height":
            kappa = torch.sqrt(torch.clamp((rho * NA) ** 2 - n_med ** 2, min=0.0)) / lam
            if self.ring == "layer":
                x = 4 * math.pi * kappa * p["pupil.saf_l"][..., None, None]
                mean = torch.exp(-4 * math.pi * kappa * p["pupil.saf_h"][..., None, None]) * _expm1_ratio(x)
            else:
                u = 4 * math.pi * kappa * self.bead_nm
                mean = _sphere_mean(u)
            out = out * torch.sqrt(mean)
        elif self.saf:
            kappa = torch.sqrt(torch.clamp((rho * NA) ** 2 - n_med ** 2, min=0.0)) / lam
            x = -2 * math.pi * kappa * p["pupil.saf_h"][..., None, None]
            if self.saf_sd:
                x = x + (2 * math.pi * kappa * p["pupil.saf_sd"][..., None, None]) ** 2
            out = out * torch.exp(x)
        if self.saf_step:
            ring = torch.sigmoid((rho - n_med / NA) / HARD_W)
            out = out * (1 + (p["pupil.saf_s"][..., None, None] - 1) * ring)
        if self.edge == "hard":
            out = out * torch.sigmoid((p["pupil.edge_rho"][..., None, None] - rho) / self.cut_w)
        if self.edge in ("logistic", "richards"):
            rc, w = p["pupil.edge_rho"][..., None, None], p["pupil.edge_width"][..., None, None]
            if self.edge == "logistic":
                out = out * torch.sigmoid((rc - rho) / w)
            else:
                nu = p["pupil.edge_nu"][..., None, None]
                # (1 + ν e^x)^(−1/ν), as exp(−log1p(ν e^x)/ν) with e^x capped, so that it stays finite
                x = torch.clamp((rho - rc) / w, max=60.0)
                out = out * torch.exp(-torch.log1p(nu * torch.exp(x)) / nu)
        return out

    def __call__(self, p):
        magnitude, phase = super().__call__(p)
        if self.edge == "spline":
            magnitude = magnitude + torch.tensordot(p["pupil.edge"], self.hat_basis, dims=([-1], [0]))
        return magnitude * self.symmetric(p, self.rho), phase

    def edge_profile(self, p, rho=None) -> list:
        """The symmetric profile at ρ = 0, 0.05 … 1 (the centre Zernikes left out), relative to ρ = 0; the cutoff
        drawn at the pupil's own width."""
        r = torch.as_tensor(np.linspace(0, 1, 21) if rho is None else rho, dtype=self.hat_basis.dtype)
        q = {k: v.detach().cpu().to(r.dtype) for k, v in p.items() if k.startswith("pupil.")}
        v = self.symmetric(q, r[None, :])[0]
        if self.edge == "spline":
            v = v + torch.tensordot(q["pupil.edge"], hats(r), dims=([-1], [0]))
        return [round(float(x / v[0]), 3) for x in v]

    def edge_values(self, p) -> dict:
        return {k.split(".", 1)[1]: round(float(v), 4) for k, v in p.items()
                if k in ("pupil.centre_a", "pupil.centre_b", "pupil.edge_rho", "pupil.edge_width", "pupil.edge_nu",
                         "pupil.saf_h", "pupil.saf_s", "pupil.saf_sd", "pupil.saf_l", "pupil.centre_g")} | (
            {"centre_d": [round(float(v), 4) for v in p["pupil.centre_d"].reshape(-1)]}
            if "pupil.centre_d" in p else {})


def edge_pupil(grid, n_max: int, fit: str) -> EdgePupil:
    """``fit`` by name: "spline", "logistic" (Noll 4 centre), "… c11" (Noll 4 and 11), "mono" (monotone
    centre, logistic edge), "mono b" (and ρ⁴), "mono nu" (a Richards edge), "mono b nu"; "… asym" keeps the
    m ≠ 0 Zernikes ("… asym4": only to order 4),
    "… saf" damps the supercritical light by a learnt height.  "flat …": the monotone centre
    with "saf" (the height), "s" (the ring's transmission) and "na" (a hard cutoff; else the nominal NA)."""
    words = fit.split()
    order = next((int(w[4:]) for w in words if w.startswith("asym") and w[4:].isdigit()), None)
    asym, saf = "asym" in words or order is not None, "saf" in words
    if words[0] in ("flat", "cos", "steps"):
        ring = "layer" if "layer" in words else "bead" if "bead" in words else "height"
        return EdgePupil(grid, n_max, centre={"flat": "mono"}.get(words[0], words[0]), steps="steps" in words,
                         signed="signed" in words,
                         edge="hard" if "na" in words else "none", asymmetric=asym, saf=saf or ring != "height",
                         saf_step="s" in words, saf_sd="sd" in words, ring=ring, asym_order=order)
    if words[0] == "mono":
        return EdgePupil(grid, n_max, centre="mono", edge="richards" if "nu" in words else "logistic",
                         quartic="b" in words, asymmetric=asym, saf=saf)
    return EdgePupil(grid, n_max, centre=(4, 11) if "c11" in words else (4,), edge=words[0], asymmetric=asym,
                     saf=saf)


def fitted_psf(fit: str, n_max: int):
    grid = uipsf.PupilGrid.build(uipsf.Optics(NA=1.43, **OPTICS), P, device="cpu")
    if fit == "flat":
        pupil = uipsf.ZernikePupil(grid, n_max=n_max, magnitude_n_max=0)
    elif fit == "zernike":
        pupil = uipsf.ZernikePupil(grid, n_max=n_max)
    else:
        pupil = edge_pupil(grid, n_max, fit)
    return uipsf.VectorPSF(grid, pupil, ROI, sigma_nm=(30.0, 30.0))


#: the bead truths' diameter, nm: dye throughout the sphere, at the heights of its slices
BEAD_NM = 100.0


def bead_saf(psf, rho):
    """The amplitude a bead's dye sends into the pupil, relative to dye at the glass: the incoherent mean over
    its heights h (weighted by the sphere's slice at h, h(d − h)) of exp(−4πκh), square-rooted."""
    o = psf.grid.optics
    kappa = torch.sqrt(torch.clamp((rho * o.NA) ** 2 - o.n_med ** 2, min=0.0)) / o.wavelength_nm
    h = torch.linspace(0.0, BEAD_NM, 41, dtype=rho.dtype)
    w = h * (BEAD_NM - h)
    mean = (w[:, None, None] * torch.exp(-4 * math.pi * kappa[None] * h[:, None, None])).sum(0) / w.sum()
    return torch.sqrt(mean)


def truth_factor(error: str, psf, rho):
    """The amplitude factor of the bead truths at ``rho``."""
    f = bead_saf(psf, rho)
    if error == "bead + apodisation":
        o = psf.grid.optics
        f = f * torch.sqrt(torch.sqrt(torch.clamp(1 - (rho * o.NA / o.n_imm) ** 2, min=0.0)))
    return f


def truth(error: str, n_max: int):
    if error.startswith("bead ") or error == "bead":
        psf, phase, field = amplitude_truth("matched", n_max)
        rho = psf.grid.rho.real if psf.grid.rho.is_complex() else psf.grid.rho
        return with_fields(psf, truth_factor(error, psf, rho).to(psf.fields.dtype)), phase, field
    if error != "bead-like":
        return amplitude_truth(error, n_max)
    psf, phase, field = amplitude_truth("matched", n_max)
    basis = zernike_basis(psf.grid, list(BEADLIKE))
    m = 1 + torch.tensordot(torch.tensor(list(BEADLIKE.values()), dtype=basis.dtype), basis, dims=([0], [0]))
    return with_fields(psf, m.to(psf.fields.dtype)), phase, field


def radial_profile(pupil, p, r) -> np.ndarray:
    """The fitted symmetric magnitude at ``r``, relative to ρ = 0: the edge pupil's factor times its centre
    Zernikes, or a Zernike pupil's m = 0 modes."""
    from uipsf.optics.zernike import zernike
    r = np.asarray(r, float)
    rt = torch.as_tensor(r, dtype=torch.float64)
    c = dict(zip(pupil.magnitude_modes, p["pupil.magnitude"].detach().cpu().double().tolist(), strict=True))
    zern = 1 + sum((v * zernike(j, r, np.zeros_like(r)) for j, v in c.items() if noll_to_nm(j)[1] == 0),
                   np.zeros_like(r))
    zern = torch.as_tensor(zern, dtype=torch.float64)
    if isinstance(pupil, EdgePupil):
        q = {k: v.detach().cpu().double() for k, v in p.items() if k.startswith("pupil.")}
        zern = zern * pupil.symmetric(q, rt[None, :])[0]
    v = zern.numpy()
    return v / v[0]


def subcritical_error(error, psf, pupil, p) -> dict | None:
    """For the bead truths: the fitted against the true symmetric magnitude below the critical angle (rms and
    largest |difference| of the profiles relative to ρ = 0), and both at ρ = 0.5, 0.8, 0.9."""
    if not error.startswith("bead"):
        return None
    o = psf.grid.optics
    r = np.linspace(0.0, o.n_med / o.NA, 31)
    t = truth_factor(error, psf, torch.as_tensor(r)[None, :]).double().numpy()[0]
    t = t / t[0]
    f = radial_profile(pupil, p, r)
    d = f - t
    at = [15, 25, 29]
    return dict(rms=round(float(np.sqrt(np.mean(d ** 2))), 4), max=round(float(np.max(np.abs(d))), 4),
                fit=[round(float(f[i]), 3) for i in at], truth=[round(float(t[i]), 3) for i in at])


def z_distortion(model, res, batch) -> dict:
    """Each slice localised alone: slope of found z against the stack's, and the median residual per slice
    about that line within ±600 nm (systematic over the beads), peak to peak and rms."""
    found, crlb, expected = slice_fits(model, res.params.physical(), batch, Poisson(), res.params)
    z, z0 = found[..., 2], expected[..., 2]
    stage = np.median(z0 - z0.mean(1, keepdims=True), 0) + 0.0      # the slices' common z
    near = np.abs(stage) <= 600.0
    ok = np.isfinite(z) & (crlb[..., 2] < 100.0)
    x, y = z0[:, near][ok[:, near]], z[:, near][ok[:, near]]
    slope, offset = np.polyfit(x, y, 1)
    r = z - (slope * z0 + offset)
    per_slice = np.array([np.median(r[ok[:, k], k]) if ok[:, k].sum() >= 6 else np.nan for k in range(z.shape[1])])
    s = per_slice[near]
    return dict(slope=round(float(slope), 4), bend_pp_nm=round(float(np.nanmax(s) - np.nanmin(s)), 1),
                bend_rms_nm=round(float(np.sqrt(np.nanmean(s ** 2))), 1),
                xy_bend_nm=[round(float(np.nanmax(v) - np.nanmin(v)), 1) for v in
                            (np.nanmedian(found[..., a] - expected[..., a], 0)[near] for a in (0, 1))])


def run(error: str, fits, seed: int, n_max: int) -> list:
    import magnitude
    magnitude.truth = truth                     # `simulate` draws from the truth of this module's errors
    batch = simulate(error, seed, n_max)
    rows = []
    for fit in fits:
        t = time.perf_counter()
        psf = fitted_psf(fit, n_max)
        model = ForwardModel(psf)
        res = fit_beads(model, batch, Poisson())
        row = dict(error=error, seed=seed, n_max=n_max, fit=fit, converged=bool(res.converged),
                   **describe(res, psf), z=z_distortion(model, res, batch))
        p = res.params.physical()
        if isinstance(psf.pupil, EdgePupil):
            row["edge"] = psf.pupil.edge_profile(p)
            row["edge_values"] = psf.pupil.edge_values(p)
        row["subcritical"] = subcritical_error(error, psf, psf.pupil, p)
        row["centre"] = {j: round(float(v), 4) for j, v in
                         zip(psf.pupil.magnitude_modes, p["pupil.magnitude"].detach().double().tolist(), strict=True)
                         if noll_to_nm(j)[1] == 0}
        row["seconds"] = round(time.perf_counter() - t, 1)
        rows.append(json.loads(json.dumps(row, default=float)))
        print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("errors", nargs="*", default=list(ERRORS))
    ap.add_argument("--fits", nargs="+", default=list(FITS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    for seed in args.seeds:
        for error in args.errors:
            for row in run(error, args.fits, seed, args.n_max):
                if args.out:
                    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                    with open(args.out, "a") as f:
                        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
