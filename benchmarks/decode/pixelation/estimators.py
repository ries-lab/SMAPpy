"""Estimators of the sub-pixel equalising map, compared (2026-11-26): how much of what they move is noise.

    python benchmarks/decode/pixelation/estimators.py predict v1d_B v1e_B A_eosr_s0 G7N_s0
    python benchmarks/decode/pixelation/estimators.py known          # study 1: a known pull, no network
    python benchmarks/decode/pixelation/estimators.py netfit v1d_B ...  # study 2: netfit on test set B

`smappy.pixel_equalise` moves each place within the pixel, u, through its group's cumulative
distribution F, learnt as 40 quantile knots per cell of |z| (100 nm) x 4 uncertainty groups.  Each
cell's F is learnt from its own few thousand localizations, so part of every move is the map's
counting noise.  The candidates here:

* ``knots``  -- `pixel_equalise` as it is (|z|, 100 nm, 4 groups, 40 knots);
* ``knots_signed`` -- the same on signed z: an astigmatic PSF is sharp in x on one side of the focus,
  in y on the other;
* ``fourier_cells`` -- per cell of signed z x uncertainty, the density's Fourier coefficients
  ``a_k = 2<cos 2 pi k u>``, ``b_k = 2<sin 2 pi k u>``, each kept only where it is beyond
  ``SIGNIFICANCE`` times its counting noise ``sqrt(2 / n)``; F then has a closed form,
  ``F(u) = u + sum_k [a_k sin 2 pi k u + b_k (1 - cos 2 pi k u)] / 2 pi k``;
* ``fourier_smooth`` -- no cells: each coefficient a smooth function of (z, the uncertainty's rank),
  a tensor product of B-splines fitted by least squares to every localization's own ``2 cos 2 pi k u``
  and ``2 sin 2 pi k u`` (their means are the coefficients), with a small ridge.  Neighbouring z and
  uncertainties share their counts; there are no edges where a map jumps.

Every map is learnt on one half (even samples) and applied to the other.  Study 1 needs no network:
uniform true places bent by a known pull that changes with z and the uncertainty, so the ideal map is
known and ``rms(corrected - true)`` is the map's own error; and uniform places with no pull at all,
where every move is noise.  Study 2 is netfit's models on test set B, with the truth: held-out
`pixel_modulation`, the RMS move, the RMSE before and after, the share of moves toward the truth.

**Results (2026-11-26).**  Study 1, a pull of modulation 0.11: on flat data the knots move every
localization by their noise (0.004 - 0.019 px at 400 k - 20 k), the Fourier estimators by none
(their significance test); with 2 M localizations the smooth one leaves 13 % of the pull.  Study 2,
v1d_B (1 frame), v1e_B (3 frames), A_eosr_s0 on test set B (~58 k localizations to learn from, the
other half judged): the map's own noise (two maps from disjoint quarters) is 1.3 - 1.4 nm for the
knots, 1.6 - 1.8 nm on signed z (half the counts a cell), 0.8 - 0.9 nm for the smooth Fourier fit on
6 x 4 B-splines (z x the uncertainty's rank, K up to 10), which flattens as much (0.146 -> 0.012,
0.068 -> 0.004) and leaves A_eosr_s0's weak, near-focus pixelation (0.034) mostly alone.  Finer bases
are worse at these counts (12 x 6: 0.022, 16 x 8: 0.033 left on v1e_B): the shrinkage grows with the
parameters.  Every estimator still moves localizations 4 - 5 nm against a true pull of 0.8 - 1.2 nm
RMS (the mean error against the true place in the pixel), half of them away from the truth, and the
inlier RMSE does not improve: the correction stays cosmetic, as `pixel_equalise` says.  G7N_s0 was
left out: a recurrent model, and its modulation here (0.081) disagrees with its training's (0.008),
so this script probably does not feed it as it was trained.

**Real data (study 3, ``real RUN...``, 2026-11-26)**: the example movie's fits in the benchmark's
``fits/`` (1000 photons, p 0.5, |z| 700 nm), frames in blocks of 500, even learning (~500 - 640 k),
odd judged.  With this many localizations the two are close: held-out modulation after 0.006 - 0.016
for both, the map's noise 0.57 - 0.85 nm (smooth 6 x 4) against 0.66 - 1.20 nm (knots), the smooth fit
a little flatter far from the focus on most runs and a little noisier on the two recurrent ones.  On
Spline 3D, with nothing to correct (0.007), the smooth fit moves 0.45 nm, the knots 0.77 nm.  **The
pair check** (two localizations of one molecule, no shared frame) widens with every correction:
+0.01 - 0.06 nm on Spline 3D and the 1-frame models, up to +0.6 nm (16.5 -> 17.2) on netfit_v1, whose
pixelation is strongest (0.146; 0.45 at |z| 300 - 700 nm) -- the smooth fit by slightly more, as it
moves slightly more.  netfit_v1 far from the focus keeps 0.07 - 0.08 after either, the same with 20
harmonics or 10 z splines: what is left there depends on something neither groups by (photons, local
density), not on resolution.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np

from smappy import pixel_equalise as pe
from smappy.training.reference import pixel_modulation

MODELS = __import__("pathlib").Path.home() / ".cache" / "decode_smlm" / "netfit"
TESTSET = MODELS / "testsets" / "B.npz"
OUT = MODELS.parent / "pixelation" / "estimators"

K_MAX = 6
SIGNIFICANCE = 3.0
TWO_PI = 2 * np.pi


# ---------------------------------------------------------------------------------------------------
# the Fourier map


def fourier_map(u, a, b):
    """F(u) from the density's coefficients ``a``, ``b`` (..., K), one row per localization or one
    shared; the density clipped to stay positive, so that F is monotone (`_safe`)."""
    k = np.arange(1, a.shape[-1] + 1)
    w = TWO_PI * k * u[:, None]
    return u + ((a * np.sin(w) + b * (1 - np.cos(w))) / (TWO_PI * k)).sum(-1)


def _safe(a, b, floor=0.2):
    """Scale coefficients down where the density they give would fall below ``floor``: noise at a
    rare group can otherwise fold F back on itself."""
    grid = np.linspace(0, 1, 64, endpoint=False)
    k = np.arange(1, a.shape[-1] + 1)
    w = TWO_PI * k * grid[:, None]
    p = 1 + np.einsum("...k,gk->...g", a, np.cos(w)) + np.einsum("...k,gk->...g", b, np.sin(w))
    low = p.min(-1)
    s = np.where(low < floor, (1 - floor) / np.maximum(1 - low, 1e-9), 1.0)
    return a * s[..., None], b * s[..., None]


def _ranks(edges, v):
    """The quantile rank of each value among the learnt quantiles ``edges`` (0..1)."""
    return np.interp(v, edges, np.linspace(0, 1, len(edges)))


@dataclass
class FourierCells:
    z_edges: np.ndarray
    s_edges: np.ndarray          # quantiles of the uncertainty, per z bin (rows)
    a: np.ndarray                # (zbins, sgroups, K)
    b: np.ndarray

    @classmethod
    def fit(cls, p, z, s, z_bin=100.0, groups=4, k_max=K_MAX, significance=SIGNIFICANCE, min_n=400):
        u = pe.place(p)
        lo, hi = np.floor(z.min() / z_bin) * z_bin, np.ceil(z.max() / z_bin) * z_bin
        z_edges = np.arange(lo, hi + z_bin / 2, z_bin)
        nz = len(z_edges) - 1
        zb = np.clip(np.digitize(z, z_edges) - 1, 0, nz - 1)
        s_edges = np.zeros((nz, groups + 1))
        a = np.zeros((nz, groups, k_max))
        b = np.zeros((nz, groups, k_max))
        k = np.arange(1, k_max + 1)
        for i in range(nz):
            m = zb == i
            s_edges[i] = np.quantile(s[m], np.linspace(0, 1, groups + 1)) if m.any() else np.r_[0, [np.inf] * groups]
            g = np.clip(np.digitize(s[m], s_edges[i, 1:-1]), 0, groups - 1)
            uu = u[m]
            for j in range(groups):
                v = uu[g == j]
                if len(v) < min_n:
                    continue
                c = 2 * np.cos(TWO_PI * k * v[:, None]).mean(0)
                d = 2 * np.sin(TWO_PI * k * v[:, None]).mean(0)
                noise = np.sqrt(2.0 / len(v))
                a[i, j] = np.where(np.abs(c) > significance * noise, c, 0)
                b[i, j] = np.where(np.abs(d) > significance * noise, d, 0)
        a, b = _safe(a, b)
        return cls(z_edges, s_edges, a, b)

    def __call__(self, p, z, s):
        nz = len(self.z_edges) - 1
        zb = np.clip(np.digitize(z, self.z_edges) - 1, 0, nz - 1)
        g = np.zeros(len(p), int)
        for i in range(nz):
            m = zb == i
            g[m] = np.clip(np.digitize(s[m], self.s_edges[i, 1:-1]), 0, self.a.shape[1] - 1)
        u = pe.place(p)
        return p + fourier_map(u, self.a[zb, g], self.b[zb, g]) - u


def _bspline_basis(x, n, degree=3):
    """``n`` B-splines of ``degree`` on [0, 1], evaluated at ``x`` (clipped): (len(x), n)."""
    from scipy.interpolate import BSpline
    inner = np.linspace(0, 1, n - degree + 1)
    t = np.r_[[0.0] * degree, inner, [1.0] * degree]
    x = np.clip(x, 0, 1 - 1e-12)
    return BSpline.design_matrix(x, t, degree).toarray()


@dataclass
class FourierSmooth:
    z_range: tuple
    s_quantiles: np.ndarray
    nz: int
    ns: int
    coef_a: np.ndarray           # (nz * ns, K)
    coef_b: np.ndarray

    @classmethod
    def fit(cls, p, z, s, nz=12, ns=6, k_max=K_MAX, ridge=1.0, significance=SIGNIFICANCE):
        u = pe.place(p)
        z_range = (float(np.quantile(z, 0.001)), float(np.quantile(z, 0.999)))
        s_q = np.quantile(s, np.linspace(0, 1, 101))
        X = cls._design(z, s, z_range, s_q, nz, ns)
        k = np.arange(1, k_max + 1)
        Y = np.c_[2 * np.cos(TWO_PI * k * u[:, None]), 2 * np.sin(TWO_PI * k * u[:, None])]
        G = X.T @ X
        C = np.linalg.solve(G + ridge * np.eye(X.shape[1]), X.T @ Y)
        # each harmonic's surface against its counting noise: an observation 2 cos(...) has variance
        # about 2, so the fit's noise covariance is 2 G^-1 and c' G c / 2 is chi-squared with p degrees
        # of freedom where there is no signal.  A surface not clearly above that is dropped; the rest
        # shrunk by James-Stein, (1 - p / chi2), what the noise adds to its size taken off
        p_ = X.shape[1]
        chi2 = np.einsum("pc,pq,qc->c", C, G, C) / 2
        keep = chi2 - p_ > significance * np.sqrt(2 * p_)
        C = C * np.where(keep, np.clip(1 - p_ / np.maximum(chi2, 1e-12), 0, 1), 0.0)
        return cls(z_range, s_q, nz, ns, C[:, :k_max], C[:, k_max:])

    @staticmethod
    def _design(z, s, z_range, s_q, nz, ns):
        zx = (z - z_range[0]) / (z_range[1] - z_range[0])
        bz = _bspline_basis(zx, nz)
        bs = _bspline_basis(_ranks(s_q, s), ns)
        return (bz[:, :, None] * bs[:, None, :]).reshape(len(z), nz * ns)

    def coefficients(self, z, s):
        X = self._design(z, s, self.z_range, self.s_quantiles, self.nz, self.ns)
        return _safe(X @ self.coef_a, X @ self.coef_b)

    def __call__(self, p, z, s):
        u = pe.place(p)
        a, b = self.coefficients(z, s)
        return p + fourier_map(u, a, b) - u


def estimators(**override):
    """name -> fit(p, z, s) returning a map(p, z, s), one axis."""
    def knots(signed):
        def fit(p, z, s):
            if signed:     # pixel_equalise bins |z|: shift signed z to be non-negative, same 100 nm bins
                off = -np.floor(z.min() / 100.0) * 100.0
                ax = pe.fit_axis(p, z + off, s)
                return lambda q, zz, ss: pe.apply_axis(ax, q, zz + off, ss)
            ax = pe.fit_axis(p, z, s)
            return lambda q, zz, ss: pe.apply_axis(ax, q, zz, ss)
        return fit
    out = {"knots": knots(False), "knots_signed": knots(True),
           "fourier_cells": lambda p, z, s: FourierCells.fit(p, z, s),
           "fourier_smooth": lambda p, z, s: FourierSmooth.fit(p, z, s)}
    out.update(override)
    return out


# ---------------------------------------------------------------------------------------------------
# study 1: a known pull


def known_pull(u, z, s):
    """The true place bent toward the pixel's centre, more where the uncertainty is large and |z|
    large, with a second harmonic for a sharper peak and an odd part that changes sign with z (the
    pull an astigmatic PSF makes in x).  Monotone for these strengths."""
    r = (s - 0.05) / 0.25                           # 0 .. 1 over the uncertainties drawn
    strength = 0.03 + 0.2 * r * (0.4 + np.abs(z) / 1000)
    sharp = 0.25 * strength
    tilt = 0.03 * np.tanh(z / 300)
    return (u + strength * np.sin(TWO_PI * u) / TWO_PI + sharp * np.sin(2 * TWO_PI * u) / (2 * TWO_PI)
            + tilt * (1 - np.cos(TWO_PI * u)) / TWO_PI)


def study_known(ns=(5_000, 20_000, 100_000, 400_000), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        for pulled in (True, False):
            def draw(m):
                t = rng.uniform(0, 40, m)
                z = rng.uniform(-800, 800, m)
                s = rng.uniform(0.05, 0.3, m)
                c = np.round(t) - 0.5                  # the pixel's lower edge: centres are whole numbers
                f = c + known_pull(t - c, z, s) if pulled else t.copy()
                return t, f, z, s
            t1, f1, z1, s1 = draw(n)
            t2, f2, z2, s2 = draw(100_000)
            for name, fit in estimators().items():
                g = fit(f1, z1, s1)(f2, z2, s2)
                rows.append(dict(study="known", n=n, pulled=pulled, estimator=name,
                                 rms_to_truth_px=float(np.sqrt(np.mean((g - t2) ** 2))),
                                 rms_before_px=float(np.sqrt(np.mean((f2 - t2) ** 2))),
                                 rms_move_px=float(np.sqrt(np.mean((g - f2) ** 2))),
                                 mod_before=pixel_modulation(f2), mod_after=pixel_modulation(g)))
                r = rows[-1]
                r["left"] = r["rms_to_truth_px"] / r["rms_before_px"] if pulled else float("nan")
                print(f"n {n:7d} {'pulled' if pulled else 'flat  '} {name:15s} to truth {r['rms_to_truth_px']:.4f} "
                      f"(before {r['rms_before_px']:.4f})  move {r['rms_move_px']:.4f}  "
                      f"mod {r['mod_before']:.3f} -> {r['mod_after']:.3f}", flush=True)
    return rows


# ---------------------------------------------------------------------------------------------------
# study 2: netfit on test set B


def predict(name):
    """Run the model on test set B and keep its matched localizations with their truth."""
    import torch

    from smappy.netfit import extract, metrics, testset
    from smappy.netfit.predict import Predictor
    path = MODELS / name / "model.safetensors"
    p = Predictor(path, None)
    batch, truth, meta = testset.load(TESTSET, p.device, window=2 * p.window + 1)
    pixel_nm = tuple(meta["camera"]["pixel_nm"])
    outs = []
    with torch.no_grad():
        for a in range(0, len(batch.frames), 128):
            x = (batch.frames[a:a + 128] - p.scaling.input_offset) / p.scaling.input_scale
            outs.append(p.model(x))
    e = extract.extract(torch.cat(outs), p.scaling, p.threshold, p.extraction, p.scale)
    to = lambda v: v.detach().cpu().numpy().astype(np.float64)
    f_frame = to(e["frame"]).astype(np.int64)
    f_xyz = np.c_[to(e["x"]), to(e["y"]), to(e["z_nm"])]
    ti, fi = metrics.match(truth["frame"], truth["xyz"], f_frame, f_xyz, pixel_nm)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}.npz", frame=f_frame[fi], found=f_xyz[fi], truth=truth["xyz"][ti],
             err_px=np.c_[to(e["x_err"]), to(e["y_err"])][fi], photons=truth["photons"][ti],
             crlb=truth["crlb"][ti], pixel_nm=np.array(pixel_nm), window=p.window)
    print(f"{name}: {len(ti)} matched of {len(truth['frame'])} true, window {p.window}", flush=True)


def study_netfit(names, photons_min=100.0):
    rows = []
    for name in names:
        d = np.load(OUT / f"{name}.npz")
        keep = d["photons"] >= photons_min
        f, t, err, frame = d["found"][keep], d["truth"][keep], d["err_px"][keep], d["frame"][keep]
        pix = d["pixel_nm"]
        half = frame % 2 == 0                         # one sample a frame: even samples learn, odd are judged
        z = f[:, 2]
        for name_e, fit in estimators().items():
            moved = np.empty((int((~half).sum()), 2))
            for ax in (0, 1):
                m = fit(f[half, ax], z[half], err[half, ax])
                moved[:, ax] = m(f[~half, ax], z[~half], err[~half, ax])
            # the map's own noise: two maps learnt on disjoint quarters, applied to the same half; their
            # difference over sqrt 2 is one quarter-map's noise, times sqrt 2 again for a half's
            qa, qb = frame % 4 == 0, frame % 4 == 2
            twin = np.empty((2, int((~half).sum()), 2))
            for i, q in enumerate((qa, qb)):
                for ax in (0, 1):
                    twin[i, :, ax] = fit(f[q, ax], z[q], err[q, ax])(f[~half, ax], z[~half], err[~half, ax])
            diff = (twin[0] - twin[1]) * pix
            noise = float(np.sqrt(np.mean((diff ** 2).sum(1))) / 2)
            fo, to_ = f[~half, :2], t[~half, :2]
            d0 = (fo - to_) * pix
            d1 = (moved - to_) * pix
            mv = (moved - fo) * pix
            toward = np.abs(d1) < np.abs(d0)
            # inliers as the standard report counts them (`metrics.outliers_and_inliers`): within 5 bounds
            crlb = d["crlb"][keep][~half, :2]
            inl = (np.abs(d0) < 5 * crlb).all(1)
            # the pull itself: the mean error against the true place within the pixel, 20 bins
            ut = pe.place(to_).ravel()
            bins = np.minimum((ut * 20).astype(int), 19)
            e0 = d0.ravel()
            pull = np.array([e0[inl.repeat(2) & (bins == i)].mean() for i in range(20)])
            r = dict(study="netfit", model=name, estimator=name_e, n_learn=int(half.sum()), n_judge=int((~half).sum()),
                     mod_before=pixel_modulation(np.r_[fo[:, 0], fo[:, 1]]),
                     mod_after=pixel_modulation(np.r_[moved[:, 0], moved[:, 1]]),
                     rmse_before_nm=float(np.sqrt(np.mean((d0 ** 2).sum(1)))),
                     rmse_after_nm=float(np.sqrt(np.mean((d1 ** 2).sum(1)))),
                     rms_move_nm=float(np.sqrt(np.mean((mv ** 2).sum(1)))),
                     toward=float(toward[mv != 0].mean()) if (mv != 0).any() else float("nan"),
                     inliers=float(inl.mean()),
                     rmse_in_before_nm=float(np.sqrt(np.mean((d0[inl] ** 2).sum(1)))),
                     rmse_in_after_nm=float(np.sqrt(np.mean((d1[inl] ** 2).sum(1)))),
                     move_in_nm=float(np.sqrt(np.mean((mv[inl] ** 2).sum(1)))),
                     map_noise_nm=noise, pull_rms_nm=float(np.sqrt(np.mean(pull ** 2))), pull_nm=pull.round(2).tolist())
            # the same near the focus and far from it
            for lo, hi in ((0, 300), (300, 1000)):
                sel = (np.abs(z[~half]) >= lo) & (np.abs(z[~half]) < hi)
                r[f"mod_after_{lo}_{hi}"] = pixel_modulation(np.r_[moved[sel, 0], moved[sel, 1]])
                r[f"mod_before_{lo}_{hi}"] = pixel_modulation(np.r_[fo[sel, 0], fo[sel, 1]])
                r[f"drmse_{lo}_{hi}_nm"] = float(np.sqrt(np.mean((d1[sel] ** 2).sum(1)))
                                                 - np.sqrt(np.mean((d0[sel] ** 2).sum(1))))
            rows.append(r)
            print(f"{name:12s} {name_e:15s} mod {r['mod_before']:.3f} -> {r['mod_after']:.3f} "
                  f"(far {r['mod_before_300_1000']:.3f} -> {r['mod_after_300_1000']:.3f})  "
                  f"inlier RMSE {r['rmse_in_before_nm']:.2f} -> {r['rmse_in_after_nm']:.2f} nm ({r['inliers']:.2f})  "
                  f"move {r['move_in_nm']:.2f} nm  noise {r['map_noise_nm']:.2f} nm  pull {r['pull_rms_nm']:.2f} nm  "
                  f"toward {r['toward']:.3f}", flush=True)
    return rows


# ---------------------------------------------------------------------------------------------------
# study 3: real data, DECODE's example movie


def estimators_real():
    """The two to choose between: the knots as shipped, and the smooth Fourier fit at the basis study 2
    found best."""
    est = estimators()
    return {"knots": est["knots"],
            "smooth_6x4": lambda p, z, s: FourierSmooth.fit(p, z, s, nz=6, ns=4, k_max=10)}


def study_real(runs, block=500):
    """No truth: held-out flattening, the map's own noise, and the pair check -- a molecule on in frames
    t and t + lag (no shared frame) is localized twice, and the spread of the two positions' difference
    is the precision the data show.  The pull moves both alike and cancels in the difference; a
    correction that moves localizations at random widens it.  Frames in blocks of ``block``: even
    blocks learn, odd ones are judged (a molecule rarely spans a block's edge)."""
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1] / "suite"))
    import report as R

    from smappy.io.hdf5 import load_localizations
    rows = []
    for run in runs:
        locs = load_localizations(R.FITS / run / "example.hdf5")
        cam = R.S.camera("example")
        px = np.array([cam["pixelsize_um"] * 1000, cam.get("pixelsize_y_um", cam["pixelsize_um"]) * 1000])
        ok = R.kept(locs, {"min_prob": 0.5, "min_photons": 1000.0})
        col = {k: np.asarray(locs[k], np.float64)[ok] for k in ("frame", "x_nm", "y_nm", "z_nm", "x_err_nm", "y_err_nm")}
        lag = R.independent_lag(locs)
        frame = col["frame"].astype(np.int64)
        b = frame // block
        learn, judge = b % 2 == 0, b % 2 == 1
        qa, qb = b % 4 == 0, b % 4 == 2
        p = np.c_[col["x_nm"] / px[0], col["y_nm"] / px[1]]
        e = np.c_[col["x_err_nm"] / px[0], col["y_err_nm"] / px[1]]
        z = col["z_nm"]
        i, j = R.pairs_apart(frame[judge], col["x_nm"][judge], col["y_nm"][judge], lag)

        def spread(q):
            d = (q[i] - q[j]) * px
            return float(np.mean([1.4826 * np.median(np.abs(d[:, a] - np.median(d[:, a]))) for a in (0, 1)]))
        base = p[judge]
        r0 = dict(study="real", run=run, estimator="none", n_learn=int(learn.sum()), n_judge=int(judge.sum()),
                  lag=lag, n_pairs=len(i), mod=pixel_modulation(base.ravel()), pair_spread_nm=spread(base))
        rows.append(r0)
        print(f"{run:16s} {'none':10s} mod {r0['mod']:.3f}  pairs {r0['n_pairs']} at lag {lag}, "
              f"spread {r0['pair_spread_nm']:.2f} nm  ({r0['n_learn']} learn, {r0['n_judge']} judged)", flush=True)
        for name, fit in estimators_real().items():
            out = np.empty_like(base)
            twin = np.empty((2, *base.shape))
            for ax in (0, 1):
                out[:, ax] = fit(p[learn, ax], z[learn], e[learn, ax])(base[:, ax], z[judge], e[judge, ax])
                for k, q in enumerate((qa, qb)):
                    twin[k, :, ax] = fit(p[q, ax], z[q], e[q, ax])(base[:, ax], z[judge], e[judge, ax])
            mv = (out - base) * px
            r = dict(study="real", run=run, estimator=name, mod=pixel_modulation(out.ravel()),
                     move_nm=float(np.sqrt(np.mean((mv ** 2).sum(1)))),
                     map_noise_nm=float(np.sqrt(np.mean((((twin[0] - twin[1]) * px) ** 2).sum(1))) / 2),
                     pair_spread_nm=spread(out))
            for lo, hi in ((0, 300), (300, 700)):
                sel = (np.abs(z[judge]) >= lo) & (np.abs(z[judge]) < hi)
                r[f"mod_before_{lo}_{hi}"] = pixel_modulation(base[sel].ravel())
                r[f"mod_after_{lo}_{hi}"] = pixel_modulation(out[sel].ravel())
            rows.append(r)
            print(f"{run:16s} {name:10s} mod {r0['mod']:.3f} -> {r['mod']:.3f} "
                  f"(far {r['mod_before_300_700']:.3f} -> {r['mod_after_300_700']:.3f})  move {r['move_nm']:.2f} nm  "
                  f"noise {r['map_noise_nm']:.2f} nm  pair spread {r0['pair_spread_nm']:.2f} -> {r['pair_spread_nm']:.2f} nm",
                  flush=True)
    return rows


def _save(rows, name):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    what, names = sys.argv[1], sys.argv[2:]
    if what == "predict":
        for n in names:
            predict(n)
    elif what == "known":
        _save(study_known(), "known.jsonl")
    elif what == "real":
        _save(study_real(names), "real.jsonl")
    elif what == "netfit":
        _save(study_netfit(names), "netfit.jsonl")
