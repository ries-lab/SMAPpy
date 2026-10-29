"""The 4Pi pupils in Hao et al.'s (2017) covariant and contra-variant modes, against one set per objective.

    python benchmarks/uipsf/real/zenodo_4pi_mode_basis.py --basis modes --offset 60 0 0 [--device mps]
        [--data v1|sms] [--stage1 4,4|6,4|4,none]

Hao X, Antonello J, Allgeyer ES, Bewersdorf J, Booth MJ (2017).  Aberrations in 4Pi microscopy.  Opt
Express 25:14049, https://doi.org/10.1364/OE.25.014049.  They describe the two pupils by Q±_j: Z_j in
the upper, ±Z_j in the lower, in coordinates that mirror x and y between the objectives and flip the phase
with the direction of travel.  In `FourPiModel`'s frames, each objective's own, that is

    q+_j = (c_up − s_j c_lo)/√2,   q−_j = (c_up + s_j c_lo)/√2,   s_j = (−1)^|m_j|,

orthonormal, so only the basis turns.  For Noll 1-4 it is the model's choice already (q+ the bead's phase
and position, q− the misalignment Δ, q−_1 nothing); docs/4pi.md §19 found the same split for Noll ≥ 5 from
the Fisher matrix: q+ pinned by the interference, q− only by each arm's incoherent image.  The optimum is
the same in either basis.  What can differ is the way there: the solver damps by diag H, which is not
invariant under a rotation, and the arms are correlated at |r| = 0.6-0.95.

`ModeBasisFourPi` reads ``pupil.phase`` as q+ and ``lo.pupil.phase`` as q− (the magnitudes stay one per
objective), so the solver's linearisation runs unchanged.  ``--stage1 P,M`` learns the first stage with q+
to radial order P and q− to M (``none``: q− held at 0), then everything, as `learn_fourpi` does otherwise
(`learn`); 4,4 is `learn_fourpi`'s own.  The data (``--data``): zenodo_4pi_beads.py's v1's 18 beads, 3
mirror steps, or the 4Pi-SMS example's bead scan (`4pi_sms_nup96/Beads_stack`, 101 DCIMG slices 20 nm
apart, no mirror steps, read as the PSF calibration window reads it, `fourpi_beads.scan_beads`).  A free
photon count per observation, the vectorial model with the rotationally symmetric magnitude, n ≤ 6, as
docs/4pi.md §20.  Prints one JSON line.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from zenodo_4pi_beads import tracks, v1_batch

import uipsf
import uipsf.workflows.fourpi as wf
from uipsf.model import PerObservation
from uipsf.model.fourpi import LOWER, FourPiModel
from uipsf.model.fourpi_start import rough_start
from uipsf.noise import ShiftedPoisson
from uipsf.optics.zernike import noll_to_nm
from uipsf.params import Params, ParamSpec
from uipsf.solve import GaussNewtonLBFGS, Problem
from uipsf.workflows.beads import _like, _loosened

R = 2 ** -0.5
ROOT = Path(os.environ.get("UIPSF_DATA", Path.home() / ".cache/uipsf/zenodo"))
SMS = ROOT / "4pi_sms_nup96/Beads_stack/Cell01_642_000_000.dcimg"


def parity(modes, like) -> torch.Tensor:
    return like.new_tensor([(-1.0) ** abs(noll_to_nm(j)[1]) for j in modes])


def to_modes(modes, up, lo):
    s = parity(modes, up)
    return (up - s * lo) * R, (up + s * lo) * R


def to_arms(modes, qp, qm):
    s = parity(modes, qp)
    return (qp + qm) * R, s * (qm - qp) * R


class ModeBasisFourPi(FourPiModel):
    """`FourPiModel` with ``pupil.phase`` = q+ and ``lo.pupil.phase`` = q− (module docstring).  Both
    objectives' pupils have q+'s modes; q− may have fewer (``minus_modes``, a subset: the rest held at 0)."""

    minus_modes = None

    @property
    def plus_modes(self):
        return self.psf_up.pupil.phase_modes

    @property
    def minus(self):
        return self.plus_modes if self.minus_modes is None else self.minus_modes

    def specs(self, batch, start=None):
        out = []
        for s in super().specs(batch, start):
            if s.name == LOWER + "pupil.phase":
                s = ParamSpec(s.name, (len(self.minus),), fixed=not self.minus)
            out.append(s)
        return out

    def arms(self, p: dict) -> dict:
        if "pupil.phase" not in p:
            return p
        qp, qm = p["pupil.phase"], p[LOWER + "pupil.phase"]
        if self.minus_modes is not None:
            full = qp.new_zeros(qp.shape)
            for i, j in enumerate(self.minus):
                full[..., self.plus_modes.index(j)] = qm[..., i]
            qm = full
        out = dict(p)
        out["pupil.phase"], out[LOWER + "pupil.phase"] = to_arms(self.plus_modes, qp, qm)
        return out

    def __call__(self, p, batch):
        return super().__call__(self.arms(p), batch)

    def channel_images(self, p, xyz, phase):
        return super().channel_images(self.arms(p), xyz, phase)

    def emitter_linearisation(self, p, batch, shared_free=None):
        return super().emitter_linearisation(self.arms(p), batch, shared_free)

    def pupil_linearisation(self, p, batch):
        """One group: a mode's tangent changes both pupils, so both arms' changes are summed (the product
        rule is linear in them), where the per-objective model has one group per arm."""
        (_, up_names, f_up, apply_up), (_, lo_names, f_lo, apply_lo) = super().pupil_linearisation(
            self.arms(p), batch)

        def field(q):
            q = self.arms(q)
            return torch.stack([f_up(q), f_lo(q)])

        def apply(dA):
            return apply_up(dA[:, 0]) + apply_lo(dA[:, 1])
        return [(None, up_names + lo_names, field, apply)]


def stage_model(model: ModeBasisFourPi, plus: int, minus) -> ModeBasisFourPi:
    """``model`` with q+ to radial order ``plus`` and q− to ``minus`` (None: none)."""
    low = model.with_pupils([pu.truncated(plus) for pu in model.pupils])
    low.minus_modes = [j for j in low.plus_modes if minus is not None and noll_to_nm(j)[0] <= minus]
    return low


def moved(values: dict, a, b) -> dict:
    """``values`` of model ``a`` as model ``b``'s, by Noll mode: q+, q− and each objective's magnitude,
    zeros where ``a`` has no such mode."""
    out = dict(values)
    lists = [("pupil.phase", a.plus_modes, b.plus_modes), (LOWER + "pupil.phase", a.minus, b.minus),
             ("pupil.magnitude", a.pupils[0].magnitude_modes, b.pupils[0].magnitude_modes),
             (LOWER + "pupil.magnitude", a.pupils[1].magnitude_modes, b.pupils[1].magnitude_modes)]
    for name, src, dst in lists:
        v = values[name]
        out[name] = torch.stack([v[src.index(j)] if j in src else v.new_zeros(()) for j in dst]) if dst \
            else v.new_zeros(0)
    return out


def learn(model: ModeBasisFourPi, batch, noise, plus: int, minus):
    """`learn_fourpi` with the first stage's q+ and q− to their own orders (module docstring)."""
    solver = GaussNewtonLBFGS()
    batch = wf._device_batch(batch, model)
    dtype, device = batch.data.dtype, batch.data.device
    start = rough_start(batch, model)
    params = Params(model.specs(batch, start), batch.n_emitters, dtype=dtype, device=device)
    params = wf.best_phases(model, batch, params, noise)
    low = stage_model(model, plus, minus)
    values = moved(params.physical(), model, low)
    values["fourpi.offset"] = torch.tensor(wf.OFFSET_START_NM, dtype=dtype, device=device)
    specs = [_like(s, params.specs[s.name]) if s.shape == params.specs[s.name].shape else s
             for s in low.specs(batch, start)]
    p1 = Params(specs, batch.n_emitters, dtype=dtype, device=device, values=values).fix(*wf.SYMMETRIC)
    first = _loosened(solver).solve(Problem(low, noise, batch), p1)
    values = moved(first.params.physical(), low, model)
    params = Params(list(params.specs.values()), batch.n_emitters, dtype=dtype, device=device, values=values)
    result = solver.solve(Problem(model, noise, batch), params)
    result.history = [dict(h, stage=1) for h in first.history] + [dict(h, stage=2) for h in result.history]
    return wf.FourPiFit(model, result, start, first)


def sms_beads():
    """The 4Pi-SMS bead scan as the PSF calibration window reads it, with its camera numbers and optics as
    docs/4pi.md §10 and §17 took them.  The channels are SMAP's corners of these data (its calibration
    `Cell01_642_000_000_3dcal.mat`), typed: the window's *4Pi-SMS* preset (`channels.YALE_4PI`, x at 151,
    389, 1474, 1713) lies 40-65 px off them, and its registration kept 1 of the 5 beads."""
    from smappy.calibrate.fourpi_beads import FourPiBeadSettings, scan_beads
    from smappy.metadata import CameraMetadata
    settings = FourPiBeadSettings(layout="typed", x_corners="93, 469, 1409, 1769", y_corners="0, 0, 0, 0",
                                  width=168, height=168, mirrored="3, 4", roi_size=21)
    beads, _, _ = scan_beads([SMS], settings, CameraMetadata(conversion=0.5, offset=100.0, pixelsize_um=0.1))
    optics = uipsf.Optics(NA=1.35, wavelength_nm=680, n_imm=1.406, n_med=1.406, pixel_size_nm=(100, 100))
    return beads, optics


def phase_covariance(model, noise, batch, params):
    """The pupils' phase coefficients' covariance, beads' own parameters taken out (Schur complement), in
    the fit's own basis: [2P, 2P], the upper (or q+) block first."""
    ne = Problem(model, noise, batch).normal_equations(params)
    S = ne.H_ss - torch.einsum("eij,ejk,elk->il", ne.H_se, torch.linalg.pinv(ne.H_ee), ne.H_se)
    cov = torch.linalg.pinv(S)
    # in unconstrained units, which are the physical ones for the (untransformed) phases
    lay = {n: (lo, hi) for n, lo, hi in params.layout("shared")}
    idx = [i for n in ("pupil.phase", LOWER + "pupil.phase") for i in range(*lay[n])]
    return cov[idx][:, idx]


def both(modes, cov, basis: str):
    """The covariance in arms and in modes, from one in ``basis``."""
    P = len(modes)
    s = torch.tensor([(-1.0) ** abs(noll_to_nm(j)[1]) for j in modes], dtype=cov.dtype)
    eye = torch.eye(P, dtype=cov.dtype)
    # (q+, q−) = T (up, lo)
    T = torch.cat([torch.cat([R * eye, -R * torch.diag(s)], 1), torch.cat([R * eye, R * torch.diag(s)], 1)])
    arms = cov if basis == "arms" else T.T @ cov @ T
    return arms, T @ arms @ T.T


def per_mode(modes, cov):
    P = len(modes)
    sd = cov.diagonal().clamp_min(0).sqrt()
    r = cov[range(P), range(P, 2 * P)] / (sd[:P] * sd[P:])
    return {str(j): [float(sd[i]), float(sd[P + i]), float(r[i])] for i, j in enumerate(modes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", choices=("arms", "modes"), default="modes")
    ap.add_argument("--data", choices=("v1", "sms"), default="v1")
    ap.add_argument("--stage1", default="4,4", help="q+ and q−'s radial orders in stage 1 (modes basis)")
    ap.add_argument("--offset", type=float, nargs=3, default=list(wf.OFFSET_START_NM),
                    help="where stage 1 starts the misalignment Δ, nm")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--hold-sigma", type=float, default=None,
                    help="hold the blur σ at this, nm: on the 4Pi-SMS beads it fell to 0, where its curvature and "
                         "so its damping vanish")
    a = ap.parse_args()
    if a.data == "v1":
        batch, data = v1_batch()
        optics = uipsf.Optics(NA=1.35, wavelength_nm=600, n_imm=1.406, n_med=1.406, pixel_size_nm=(129, 129))
        steps = [2, 0, -2]
    else:
        beads, optics = sms_beads()
        batch, data, steps = beads.batch(), None, [0.0]
    grid = uipsf.PupilGrid.build(optics, 64, device=a.device)
    cls = ModeBasisFourPi if a.basis == "modes" else FourPiModel
    C, K = batch.data.shape[1], batch.data.shape[2]
    model = cls(grid, batch.data.shape[-2:], n_channels=C, n_phases=len(steps), n_max=a.n_max, steps_rad=steps,
                intensity=PerObservation())
    if a.hold_sigma is not None:
        model.psf_up.fit_sigma, model.psf_up.sigma_init = False, (a.hold_sigma, a.hold_sigma)
    wf.OFFSET_START_NM = tuple(a.offset)
    noise = ShiftedPoisson()
    plus, minus = a.stage1.split(",")
    t = time.time()
    if a.basis == "modes":
        fit = learn(model, batch, noise, int(plus), None if minus == "none" else int(minus))
    else:
        fit = wf.learn_fourpi(model, batch, noise=noise)
    seconds = time.time() - t
    shared = fit.shared()
    modes = model.psf_up.pupil.phase_modes
    phase = (shared["pupil.phase"].cpu().double(), shared[LOWER + "pupil.phase"].cpu().double())
    arms = to_arms(modes, *phase) if a.basis == "modes" else phase
    qp, qm = to_modes(modes, *arms)
    cov_arms, cov_modes = both(modes, phase_covariance(model, noise, wf._device_batch(batch, model),
                                                       fit.result.params).cpu(), a.basis)
    hist = fit.result.history
    row = {"data": a.data, "basis": a.basis, "stage1": a.stage1, "hold_sigma": a.hold_sigma,
           "offset_start": a.offset, "seconds": seconds,
           "n_beads": int(batch.data.shape[0]), "observations": int(K),
           "iterations": [sum(h.get("stage") == k for h in hist) for k in (1, 2)],
           "stop": fit.result.stop_reason, "objective": float(fit.result.objective),
           "first_objective": float(fit.first.objective) if fit.first is not None else None,
           "offset_nm": shared["fourpi.offset"].tolist(), "coherence": shared["fourpi.coherence"].tolist(),
           "sigma_nm": shared["psf.sigma"].tolist(),
           "modes": modes, "up": arms[0].tolist(), "lo": arms[1].tolist(), "q+": qp.tolist(), "q-": qm.tolist(),
           # per mode: (sd of up or q+, sd of lo or q−, their correlation)
           "sd_arms": per_mode(modes, cov_arms), "sd_modes": per_mode(modes, cov_modes)}
    if data is not None:
        row["fit"] = tracks(wf.calibration(fit, z_range_nm=600, dz_nm=25, roi_px=21), data)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    main()
