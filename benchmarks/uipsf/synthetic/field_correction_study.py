"""Field-dependent PSFs: fitting with node splines, against fitting with the local pupil, against one spline
for the field with its bias corrected afterwards.

    python benchmarks/uipsf/synthetic/field_correction_study.py [--field synthetic] [--scale 1] \\
        [--photons 5000] [--beads 40] [--out benchmarks/uipsf/results/field_correction.jsonl]

`docs/uipsf/field-dependence.md`, §11.  The truth is a field-dependent vector pupil (`FieldPolynomial`): for
``--field synthetic`` `field_interpolation.py`'s optics and its nodal field (quadratic and linear astigmatism,
linear coma) times ``--scale``, over u, v ∈ [−1, 1], u = 1 at 50 µm from the centre.  The model is known
exactly to every method: what is compared is how each represents and uses it, not how well it was learnt.

Methods:
- **pupil**: maximum likelihood with each spot's own pupil at its own position (uipsf's `localise` on the
  single-frame model): the baseline, about 10² fits/s on a CPU.
- **node splines**: what SMAPpy does (`field_splines.node_splines`, Spline 3D blending four nodes per spot).
- **central**, **average**: one spline for the field -- the model at the field's centre, or the mean of a
  9 × 9 grid's PSFs (roughly what a calibration learnt without field dependence holds).
- **central + bead correction**: the central spline, and dx, dy, dz as smooth functions of the *fitted*
  (u, v, z) subtracted afterwards.  They are regressed on simulated bead stacks over the field fitted with the
  same spline -- what an experimenter can do without a field model.  A bead's own x, y are unknown, so the
  lateral part is learnt within each bead (its mean over z removed: only the z-dependent part is
  identifiable); the beads are taken to lie on a flat coverslip, so z is absolute.  Basis: monomials of
  u, v to order 2 or 4 (``--orders``) times Legendre polynomials of z to degree 14 (`Z_DEGREE`).
- **central + model correction**: the same regression on noise-free spots of the model on a dense grid --
  the best a correction afterwards can do.

Measures.  *Bias* from noise-free images (the expected counts, fitted as data: the MLE's asymptotic bias,
with no noise floor) on a grid of field positions × z: the z error, and the lateral error with each
position's mean over z removed ("wobble"; a constant lateral offset over z is a distortion of the field,
reported apart).  *z scaling*: the slope of fitted against true z per position, and the largest deviation
from a straight line.  *Precision*: noisy repeats at four field positions, SD per z bin.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from field_interpolation import OPTICS
from smappy.calibrate.core import spline_coefficients
from smappy.field_splines import grid_nodes, node_splines
from smappy.io.calibration import SplineCalibration
from smappy.psf import SplinePSF

import uipsf
from uipsf import Batch, Poisson
from uipsf.export import sample_psf
from uipsf.workflows import BeadSettings, PSFModelSettings, bead_model
from uipsf.workflows.scoring import _single_frames, localise

BOX, ROI, DZ, Z_RANGE = 21, 13, 20.0, 800.0
Z_FIT = 500.0                      # emitters within ±Z_FIT of focus
BACKGROUND = 20.0
SCALE_UM = 50.0                    # u = 1 at this distance from the centre


# ---------------------------------------------------------------- the truth

@dataclass
class Truth:
    """A field-dependent model and where its field lies: chip = centre + u · px_per_u, u ∈ [−extent, extent]."""
    model: object
    shared: dict
    centre: tuple
    px_per_u: float
    extent: float
    pixel_nm: float


def _model(optics, centre):
    settings = BeadSettings(psf=PSFModelSettings(zernike_order=6, magnitude="fixed"), field_dependence=True,
                            bead_diameter_nm=0.0)
    return bead_model(uipsf.Optics(**optics), settings, (ROI, ROI), z_max_nm=1500, device="cpu",
                      dtype=torch.float32, field_centre_px=centre)


def _truth(optics, base: dict, field: list, sigma_nm: float, centre, extent: float) -> Truth:
    """``base`` {Noll: rad} the pupil at the field's centre; ``field`` the 14 `FieldPolynomial` coefficients
    (Noll 5, 6 over u, v, u², uv, v²; Noll 7, 8 over u, v), u in units of `SCALE_UM`."""
    model = _model(optics, centre)
    pupil = model.channels[0].psf.pupil
    phase = torch.zeros(len(pupil.phase_modes))
    for j, value in base.items():
        phase[pupil.phase_modes.index(j)] = value
    shared = {"pupil.phase": phase, "psf.sigma": torch.tensor([float(sigma_nm)]),
              "pupil.magnitude": torch.zeros(len(pupil.magnitude_modes)),
              "field.phase": torch.tensor(field, dtype=torch.float32)}
    pixel = optics["pixel_size_nm"][0]
    return Truth(model, shared, tuple(centre), SCALE_UM * 1000.0 / pixel, extent, pixel)


#: Zenodo `1ch_FD_bead`'s field as v2 learnt it (`results/zenodo_fd.json`, centre (808, 809) px), in the layout
#: above; the beads span u, v ≈ ±1.6 (177 µm)
ZENODO_FIELD = [-0.0725, 0.0097, -0.0041, 0.5079, 0.0058, -0.0302, 0.0283, 0.2686, -0.0127, -0.2226,
                -0.0013, 0.1615, 0.1826, -0.0038]
#: and its pupil at the centre, which v2's run did not keep: uiPSF v1's FD result
#: (`zenodo/results/1ch_FD_bead/psfmodel_zernike_vector_FD_single.h5`, ``res/zernike_coeff``, index = Noll − 1,
#: checked against v2's field: same terms, same signs) -- Noll 5 to 8 the constants of a quadratic in u, v fitted
#: over its 165 beads, the rest the beads' medians
ZENODO_BASE = {5: 0.021, 6: 0.703, 7: 0.209, 8: 0.006, **dict(zip(range(9, 29), (
    -0.015, 0.023, 0.165, -0.072, -0.034, -0.009, -0.006, -0.103, 0.003, 0.001, 0.001, -0.006, 0.002, 0.25,
    -0.008, -0.003, -0.017, -0.026, -0.002, 0.001), strict=True))}
ZENODO_OPTICS = dict(NA=1.5, wavelength_nm=680.0, n_imm=1.515, n_med=1.406, pixel_size_nm=(110.0, 110.0))


def truth(field: str, scale: float) -> Truth:
    """The field, its field-dependent coefficients times ``scale``.

    - *synthetic*: `field_interpolation.field_phase`'s -- Noll 6 = 0.8 + a2 (u² − v²) + a1 u, Noll 5 = 2 a2 uv
      + a1 v, Noll 7 = 0.02 + c1 u, Noll 8 = c1 v, (a2, a1, c1) = (0.15, 0.03, 0.10) rad, over u, v ∈ [−1, 1]
      (±50 µm); σ 30 nm.
    - *zenodo*: v2's learnt field and v1's centre pupil (`ZENODO_FIELD`, `ZENODO_BASE`), its optics, σ 88 nm
      (v2's 75 and 102 nm, one value here), over ±1.6 (±80 µm).
    - *m5*: the synthetic optics and centre pupil with M5's field, linear astigmatism only, 0.09 rad per
      12.5 µm half-field -- Noll 6 along u, Noll 5 along v (`docs/uipsf/field-dependence.md` §7) -- over ±0.25.
    """
    if field == "synthetic":
        a2, a1, c1 = 0.15 * scale, 0.03 * scale, 0.10 * scale
        coeff = [0, a1, 0, 2 * a2, 0, a1, 0, a2, 0, -a2, c1, 0, 0, c1]
        return _truth(OPTICS, {6: 0.8, 7: 0.02}, coeff, 30.0, (475.0, 475.0), 1.0)
    if field == "zenodo":
        return _truth(ZENODO_OPTICS, ZENODO_BASE, [c * scale for c in ZENODO_FIELD], 88.0, (808.0, 809.0), 1.6)
    if field == "m5":
        g = 0.36 * scale                                  # 0.09 rad per 0.25 u
        coeff = [0, g, 0, 0, 0, g, 0, 0, 0, 0, 0, 0, 0, 0]
        return _truth(OPTICS, {6: 0.8, 7: 0.02}, coeff, 30.0, (150.0, 150.0), 0.25)
    raise ValueError(field)


def chip(t: Truth, u, v):
    return t.centre[0] + np.asarray(u) * t.px_per_u, t.centre[1] + np.asarray(v) * t.px_per_u


def draw(model, shared, xyz_nm, field_xy, roi=ROI) -> np.ndarray:
    """[N, roi, roi] photons per photon, every spot at its own chip position (one model call per position).

    Drawn in the splines' `BOX` and cut to ``roi``: `sample_psf` drawn in a 13 px region folds the PSF's tails
    back into it -- over Zenodo's field up to 17 % of the peak against a 41 px drawing, where 21 px is within
    0.01 % -- and §11's table, drawn so, measured that mismatch as 8 nm of the node splines' z bias
    (docs/uipsf/field-dependence.md §12).  The pupil fit still draws its model in the spots' region."""
    xyz_nm, field_xy = np.atleast_2d(xyz_nm), np.atleast_2d(field_xy)
    keys, inverse = np.unique(np.round(field_xy, 4), axis=0, return_inverse=True)
    size = max(roi, BOX)
    c = (size - roi) // 2
    out = np.empty((len(xyz_nm), roi, roi))
    for k, where in enumerate(keys):
        s = inverse.reshape(-1) == k
        out[s] = sample_psf(model, shared, xyz_nm[s], size, field_xy=tuple(where)).double().numpy()[
            :, c:c + roi, c:c + roi]
    return out


# ---------------------------------------------------------------- the splines

def z_grid():
    half = int(Z_RANGE / DZ)
    return -(np.arange(2 * half + 1) - half) * DZ, float(half)          # plane 0 highest, as SMAPpy's


def stack_at(model, shared, where) -> np.ndarray:
    zg, _ = z_grid()
    return np.maximum(sample_psf(model, shared, np.c_[np.zeros((len(zg), 2)), zg], BOX,
                                 field_xy=tuple(where)).double().numpy(), 0)


def splines(t: Truth) -> dict:
    """{name: SplinePSF}: central, average, node splines; every one normalised by the centre's."""
    _, z0 = z_grid()
    model, shared, U = t.model, t.shared, t.extent
    central = stack_at(model, shared, t.centre)
    norm = float(central.sum((1, 2)).max())
    cache = {}

    def psf_at(x, y):
        key = (round(float(x), 3), round(float(y), 3))
        if key not in cache:
            cache[key] = stack_at(model, shared, key) / norm
        return cache[key]

    def fresh(x, y):                 # as the export draws them: nothing kept
        return stack_at(model, shared, (float(x), float(y))) / norm

    lo, hi = np.array(chip(t, -U, -U)), np.array(chip(t, U, U))
    t0 = time.perf_counter()
    nodes = node_splines(fresh, lo, hi)
    export_s = time.perf_counter() - t0
    n, error = (1, 0.0) if nodes is None else (nodes.shape[0], nodes.blend_error)
    g = grid_nodes(-U, U, 9)
    average = np.mean([psf_at(*chip(t, a, b)) for a in g for b in g], 0)

    def cal(psf, field=None):
        return SplinePSF(SplineCalibration(spline_coefficients(psf.astype(np.float32)), DZ, z0, x0=(BOX - 1) / 2,
                                           field_splines=field))
    return {"central": cal(central / norm), "average": cal(average),
            "node splines": cal(central / norm, nodes)}, dict(
        grid=n, blend_error=error, export_s=export_s,
        node_cells=None if nodes is None else int(nodes.coeff.shape[-1]),
        node_gb=0.0 if nodes is None else nodes.coeff.nbytes / 1e9)


def spline_fit(fitter: SplinePSF, rois, field_xy, pixel) -> np.ndarray:
    """[N, 3] nm from the ROI's centre, and z."""
    out = fitter.unpack(fitter.fit(rois.astype(np.float32), 50, 0, chip_xy=np.asarray(field_xy, float)))
    c = (ROI - 1) / 2
    return np.c_[(out["x_roi"] - c) * pixel, (out["y_roi"] - c) * pixel, out["z_nm"]]


def pupil_fit(model, shared, rois, field_xy, start, chunk: int = 500) -> np.ndarray:
    """[N, 3] nm, maximum likelihood with each spot's own pupil; started from ``start`` (a spline's fit)."""
    single = _single_frames(model)
    shapes = {s.name for s in single.specs(_frames(rois[:1], field_xy[:1])) if s.scope == "shared"}
    fixed = {k: v for k, v in shared.items() if k in shapes}
    out = []
    for i in range(0, len(rois), chunk):
        r, f, s = rois[i:i + chunk], field_xy[i:i + chunk], start[i:i + chunk]
        sums = r.reshape(len(r), -1)
        bg = np.maximum(np.percentile(sums, 20, 1), 1e-2)
        n = np.maximum(sums.sum(1) - bg * sums.shape[1], 10.0)
        fit, _ = localise(single, fixed, _frames(r, f), Poisson(), start={
            "emitter.xyz": torch.tensor(np.clip(s, [-300, -300, -Z_RANGE + 50], [300, 300, Z_RANGE - 50]),
                                        dtype=torch.float32),
            "emitter.photons": torch.tensor(n, dtype=torch.float32),
            "emitter.background": torch.tensor(bg, dtype=torch.float32)})
        out.append(fit.params.physical()["emitter.xyz"].detach().double().numpy())
    return np.concatenate(out)


def _frames(rois, field_xy) -> Batch:
    return Batch(data=torch.tensor(np.asarray(rois)[:, None, None], dtype=torch.float32),
                 z_nm=torch.zeros(1), field_xy=torch.tensor(np.asarray(field_xy, float), dtype=torch.float32))


# ---------------------------------------------------------------- the correction afterwards

#: the correction's degree in z: the bias wiggles along z -- 6 left 21 nm at worst at the corners, 10 left
#: 6.6 nm, 14 left 2.7 nm (the model's correction, field x1, order 4 in u, v)
Z_DEGREE = 14


def basis(u, v, z, order: int = 2) -> np.ndarray:
    """[N, M (Z_DEGREE + 1)]: the M monomials u^i v^j, i + j ≤ ``order``, times Legendre P0..P``Z_DEGREE`` of z
    (scaled to ±650 nm, a little beyond the emitters' range, since fitted z spreads past it)."""
    from numpy.polynomial import legendre
    u, v = np.asarray(u, float), np.asarray(v, float)
    uv = np.stack([u ** (n - j) * v ** j for n in range(order + 1) for j in range(n + 1)], 1)
    pz = legendre.legvander(np.clip(np.asarray(z) / 650.0, -1, 1), Z_DEGREE)
    return (uv[:, :, None] * pz[:, None, :]).reshape(len(u), -1)


class Correction:
    """dx, dy, dz(u, v, z_fit) by least squares.  With ``groups`` (beads) the lateral part is learnt within
    each group: a bead's own x, y are unknown, so its mean over z is removed, and with it the terms constant in
    z (P0 times each u, v monomial) -- a lateral offset constant over z is a distortion of the field, not
    learnt here.  z is always absolute (the beads on a flat coverslip).  Without ``groups`` (the model's
    correction, which knows the truth) every axis is absolute."""

    def __init__(self, uvz_fit, err, groups=None, order: int = 2, extent: float = 1.0):
        self.order, self.extent = order, extent
        B = self._basis(uvz_fit)
        self.lateral = (np.arange(B.shape[1]) % (Z_DEGREE + 1) != 0) if groups is not None \
            else np.ones(B.shape[1], bool)
        self.coef = []
        for axis in range(3):
            A, y = B.copy(), err[:, axis].copy()
            if axis < 2 and groups is not None:
                for g in np.unique(groups):
                    s = groups == g
                    A[s] -= A[s].mean(0)
                    y[s] -= y[s].mean()
            if axis < 2:
                A = A[:, self.lateral]
            self.coef.append(np.linalg.lstsq(A, y, rcond=None)[0])

    def _basis(self, uvz):
        return basis(uvz[:, 0] / self.extent, uvz[:, 1] / self.extent, uvz[:, 2], order=self.order)

    def __call__(self, uvz_fit) -> np.ndarray:
        B = self._basis(uvz_fit)
        return np.c_[B[:, self.lateral] @ self.coef[0], B[:, self.lateral] @ self.coef[1], B @ self.coef[2]]


def ok(fit, true, limit=300.0):
    return np.isfinite(fit).all(1) & (np.abs(fit[:, 2] - true[:, 2]) < limit)


# ---------------------------------------------------------------- the study

def bias_grid(extent: float, n_xy: int = 7, n_z: int = 21):
    g = np.linspace(-extent, extent, n_xy)
    zs = np.linspace(-Z_FIT, Z_FIT, n_z)
    u, v, z = (a.reshape(-1) for a in np.meshgrid(g, g, zs, indexing="ij"))
    return u, v, z


def summarise_bias(name, u, v, z, err, good, extent) -> dict:
    """z bias, lateral wobble (each position's mean over z removed), the lateral constant, z scaling."""
    out = dict(method=name, lost=float(1 - good.mean()))
    pos = np.round(u / extent * 100) * 1000 + np.round(v / extent * 100)
    wobble = err[:, :2].copy()
    const, slopes, curve = [], [], []
    for p in np.unique(pos):
        s = (pos == p) & good
        if s.sum() < 5:
            continue
        m = wobble[s].mean(0)
        wobble[s] -= m
        const.append(np.hypot(*m))
        zf = z[s] + err[s, 2]
        k = np.polyfit(z[s], zf, 1)
        slopes.append(k[0])
        curve.append(np.abs(zf - np.polyval(k, z[s])).max())
    r = np.hypot(u, v)
    for label, sel in (("all", good), ("edge", good & (r > 0.8 * extent))):
        out[f"z_rms_{label}"] = float(np.sqrt(np.mean(err[sel, 2] ** 2)))
        out[f"z_max_{label}"] = float(np.abs(err[sel, 2]).max())
        out[f"xy_wobble_rms_{label}"] = float(np.sqrt(np.mean(np.sum(wobble[sel] ** 2, 1))))
    out.update(xy_constant_max=float(np.max(const)), z_slope_min=float(np.min(slopes)),
               z_slope_max=float(np.max(slopes)), z_nonlinear_max=float(np.max(curve)))
    return out


def run(args) -> list[dict]:
    tr = truth(args.field, args.scale)
    model, shared, pixel, U = tr.model, tr.shared, tr.pixel_nm, tr.extent
    fitters, info = splines(tr)
    print(f"{args.field} field x{args.scale:g}, {args.photons:g} photons: node grid {info['grid']}x{info['grid']} "
          f"(blend error {100 * info['blend_error']:.2f} % of the peak, {info['node_cells']} cells, "
          f"{info['node_gb']:.2f} GB), export {info['export_s']:.0f} s", flush=True)
    rng = np.random.default_rng(args.seed)

    # the corrections: beads over the field, a stack each, fitted with the central spline
    # spread over the field as one picks beads for a calibration: a jittered grid reaching the corners
    # (uniform random beads left corners empty, and an order-4 correction ran off there by micrometres)
    side = int(np.ceil(np.sqrt(args.beads)))
    cells = (np.arange(side) + 0.5) / side * 2 - 1
    gu, gv = (a.reshape(-1)[:args.beads] for a in np.meshgrid(cells, cells))
    ub = np.clip(gu + rng.uniform(-1, 1, args.beads) / side, -1, 1) * U
    vb = np.clip(gv + rng.uniform(-1, 1, args.beads) / side, -1, 1) * U
    zs = np.arange(-600, 601, 20.0)
    bu, bv, bz = np.repeat(ub, len(zs)), np.repeat(vb, len(zs)), np.tile(zs, args.beads)
    bxy = np.repeat(rng.uniform(-pixel / 2, pixel / 2, (args.beads, 2)), len(zs), 0)
    btrue = np.c_[bxy, bz]
    bf = np.c_[chip(tr, bu, bv)]
    mu = draw(model, shared, btrue, bf)
    beads = rng.poisson(mu * args.bead_photons + args.bead_background).astype(float)
    fit = spline_fit(fitters["central"], beads, bf, pixel)
    s = ok(fit, btrue)
    groups = np.repeat(np.arange(args.beads), len(zs))
    corrections = {f"bead correction, order {o}": Correction(np.c_[bu, bv, fit[:, 2]][s], (fit - btrue)[s],
                                                             groups[s], o, U) for o in args.orders}
    # the model's: noise-free spots on a dense grid
    g = np.linspace(-U, U, 11)
    mu_, mv_, mz_ = (a.reshape(-1) for a in np.meshgrid(g, g, np.arange(-600, 601, 25.0), indexing="ij"))
    mtrue = np.c_[np.zeros((len(mu_), 2)), mz_]
    mf = np.c_[chip(tr, mu_, mv_)]
    mfit = spline_fit(fitters["central"], draw(model, shared, mtrue, mf) * args.photons + BACKGROUND, mf, pixel)
    s = ok(mfit, mtrue)
    corrections.update({f"model correction, order {o}": Correction(np.c_[mu_, mv_, mfit[:, 2]][s],
                                                                   (mfit - mtrue)[s], None, o, U) for o in args.orders})

    def methods(rois, field_xy, uv):
        t = {}
        res = {}
        for name in ("node splines", "central", "average"):
            t0 = time.perf_counter()
            res[name] = spline_fit(fitters[name], rois, field_xy, pixel)
            t[name] = time.perf_counter() - t0
        for name, corr in corrections.items():
            f = res["central"]
            res[f"central + {name}"] = f - corr(np.c_[uv, f[:, 2]])
            t[f"central + {name}"] = t["central"]
        t0 = time.perf_counter()
        res["pupil"] = pupil_fit(model, shared, rois, field_xy, res["node splines"])
        t["pupil"] = time.perf_counter() - t0
        return res, t

    rows = []
    # bias: noise-free images on a grid of field positions x z, emitters on the pixel centre
    u, v, z = bias_grid(U)
    true = np.c_[np.zeros((len(u), 2)), z]
    fxy = np.c_[chip(tr, u, v)]
    res, t = methods(draw(model, shared, true, fxy) * args.photons + BACKGROUND, fxy, np.c_[u, v])
    for name, f in res.items():
        r = summarise_bias(name, u, v, z, f - true, ok(f, true), U)
        r.update(kind="bias", fits_per_s=len(u) / max(t[name], 1e-9))
        rows.append(r)

    # precision: noisy repeats at four field positions, z in bins
    for pu, pv in ((0.0, 0.0), (0.5 * U, 0.0), (U, 0.0), (U, U)):
        n = args.repeats
        zt = rng.uniform(-Z_FIT, Z_FIT, n)
        true = np.c_[rng.uniform(-pixel / 2, pixel / 2, (n, 2)), zt]
        fxy = np.c_[chip(tr, np.full(n, pu), np.full(n, pv))]
        rois = rng.poisson(draw(model, shared, true, fxy) * args.photons + BACKGROUND).astype(float)
        res, _ = methods(rois, fxy, np.c_[np.full(n, pu), np.full(n, pv)])
        bins = np.digitize(zt, np.linspace(-Z_FIT, Z_FIT, 6)[1:-1])
        for name, f in res.items():
            e, good = f - true, ok(f, true)
            nan = np.full(3, np.nan)
            sd = np.array([e[good & (bins == b)].std(0) if (good & (bins == b)).sum() > 2 else nan for b in range(5)])
            med = np.array([np.median(e[good & (bins == b)], 0) if (good & (bins == b)).sum() > 2 else nan
                            for b in range(5)])
            rows.append(dict(kind="precision", method=name, u=pu / U, v=pv / U, lost=float(1 - good.mean()),
                             sd_x=float(np.nanmean(sd[:, 0])), sd_y=float(np.nanmean(sd[:, 1])),
                             sd_z=float(np.nanmean(sd[:, 2])), bias_z_rms=float(np.sqrt(np.nanmean(med[:, 2] ** 2)))))
    common = dict(field=args.field, scale=args.scale, photons=args.photons, beads=args.beads, extent=U,
                  largest_grid=args.largest_grid, **info)
    return [{**common, **r} for r in rows]


def report(rows):
    b = [r for r in rows if r["kind"] == "bias"]
    print("bias, noise-free (nm): z rms all / edge, z max, xy wobble rms all / edge, xy constant max, "
          "z slope range, z nonlinear max, fits/s")
    for r in b:
        print(f"  {r['method']:32s} {r['z_rms_all']:6.1f} {r['z_rms_edge']:6.1f} {r['z_max_all']:6.1f}  "
              f"{r['xy_wobble_rms_all']:5.1f} {r['xy_wobble_rms_edge']:5.1f} {r['xy_constant_max']:6.1f}  "
              f"{r['z_slope_min']:.3f}-{r['z_slope_max']:.3f} {r['z_nonlinear_max']:6.1f} {r['fits_per_s']:9.0f}"
              f"  lost {r['lost']:.1%}")
    print("precision (nm): sd x / z at (u, v) / extent = (0,0) (0.5,0) (1,0) (1,1)")
    for name in dict.fromkeys(r["method"] for r in rows if r["kind"] == "precision"):
        p = [r for r in rows if r["kind"] == "precision" and r["method"] == name]
        print(f"  {name:32s} " + "  ".join(f"{r['sd_x']:5.1f}/{r['sd_z']:5.1f}" for r in p))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--field", default="synthetic", choices=["synthetic", "zenodo", "m5"])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--photons", type=float, default=5000.0)
    ap.add_argument("--beads", type=int, default=40)
    ap.add_argument("--bead-photons", type=float, default=20000.0)
    ap.add_argument("--bead-background", type=float, default=50.0)
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 4], help="the correction's order in u, v")
    ap.add_argument("--repeats", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--largest-grid", type=int, default=None,
                    help="the largest node grid offered, e.g. 9 for the cap before 11 x 11 and 13 x 13")
    ap.add_argument("--out", default="benchmarks/uipsf/results/field_correction.jsonl")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.largest_grid is not None:
        import smappy.field_splines as fs
        fs.GRIDS = tuple(n for n in fs.GRIDS if n <= args.largest_grid)
    rows = run(args)
    report(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps({k: (float(x) if isinstance(x, np.floating) else x) for k, x in r.items()}) + "\n")


if __name__ == "__main__":
    main()
