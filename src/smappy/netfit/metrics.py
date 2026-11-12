"""What a test set says of a model each epoch: the numbers the Train window's Follow draws and the
early stop reads (`smappy.training.progress`), and the benchmark's near-focus block.

Matched per frame by the Hungarian algorithm within 250 nm laterally and 500 nm in z (DECODE's
evaluation, the SMLM challenge's, the benchmark's).  Each molecule's CRLB is SMAPpy's spline fitter's
on its spot alone, noise-free, doubled in variance for an EM register (`crlbs`).
"""
from __future__ import annotations

import numpy as np

from ..training.reference import ks_uniform, robust_width, subpixel_histogram

LATERAL_NM, AXIAL_NM = 250.0, 500.0
CORE_Z_NM = 300.0
CRLB_ROI = 13


def crlbs_torch(renderer, photons, background, z_nm, pixel_nm, em: bool, roi: int = CRLB_ROI) -> np.ndarray:
    """(n, 3): each molecule's Poisson bound in x, y, z (nm), from the Fisher information of its spot
    alone in the central ``roi`` x ``roi`` pixels of the renderer's square -- Spline 3D's fit box --
    for x, y, z, photons and background; derivatives by central differences; doubled in variance
    for an EM register.  The same bound as `crlbs` (SMAPpy's C++ fitter) without needing it built:
    the training host has only Python."""
    import torch
    dev = renderer.device
    n = len(z_nm)
    out = np.full((n, 3), np.nan)
    half, size = roi // 2, renderer.size
    lo = size // 2 - half
    for a in range(0, n, 2048):
        s = slice(a, a + 2048)
        m = len(z_nm[s])
        g = torch.Generator().manual_seed(a)
        x = torch.rand(m, generator=g, dtype=torch.float64).to(dev, torch.float32) - 0.5 + 100.0
        y = torch.rand(m, generator=g, dtype=torch.float64).to(dev, torch.float32) - 0.5 + 100.0
        z = torch.as_tensor(np.asarray(z_nm[s], np.float32), device=dev)
        N = torch.as_tensor(np.asarray(photons[s], np.float32), device=dev)[:, None, None]
        bg = torch.as_tensor(np.asarray(background[s], np.float32), device=dev)[:, None, None]

        def spot(dx=0.0, dy=0.0, dz=0.0, x=x, y=y, z=z):
            sp = renderer.spots(x + dx, y + dy, z + dz)[0]
            return sp[:, lo:lo + roi, lo:lo + roi]
        h, hz = 1e-2, 1.0
        s0 = spot()
        d = [N * (spot(dx=h) - spot(dx=-h)) / (2 * h), N * (spot(dy=h) - spot(dy=-h)) / (2 * h),
             N * (spot(dz=hz) - spot(dz=-hz)) / (2 * hz), s0, torch.ones_like(s0)]
        mu = (N * s0 + bg).clamp_min(1e-6)
        J = torch.stack([v.reshape(m, -1) for v in d], -1).double()          # (m, pixels, 5)
        fisher = torch.einsum("mpi,mpj,mp->mij", J, J, 1 / mu.reshape(m, -1).double())
        cov = torch.linalg.inv(fisher)
        var = torch.diagonal(cov, dim1=1, dim2=2)[:, :3].clamp_min(0).cpu().numpy()
        out[s] = np.sqrt(var) * np.array([pixel_nm[0], pixel_nm[1], 1.0])
    return out * (np.sqrt(2.0) if em else 1.0)


def crlbs(cal, photons, background, z_nm, pixel_nm, em: bool) -> np.ndarray:
    """(n, 3): each molecule's bound in x, y, z (nm), by SMAPpy's C++ spline fitter."""
    from ..psf import SplinePSF
    from ..simulate.camera import _draw_spline
    rng = np.random.default_rng(0)
    out = np.full((len(z_nm), 3), np.nan)
    psf = SplinePSF(cal)
    r = CRLB_ROI
    for a in range(0, len(z_nm), 5000):
        s = slice(a, a + 5000)
        m = len(z_nm[s])
        image = np.repeat(np.asarray(background[s], float)[:, None, None], r, 1).repeat(r, 2)
        _draw_spline(image, np.arange(m), r // 2 + rng.uniform(-0.5, 0.5, (m, 2)), z_nm[s], photons[s], cal)
        found = psf.unpack(psf.fit(np.ascontiguousarray(image, np.float32)))
        out[s] = np.c_[found["x_err_pix"] * pixel_nm[0], found["y_err_pix"] * pixel_nm[1], found["z_err_nm"]]
    return out * (np.sqrt(2.0) if em else 1.0)


def match(t_frame, t_xyz, f_frame, f_xyz, pixel_nm):
    """Pairs (truth index, fit index), per frame; xyz in columns, rows, nm."""
    from scipy.optimize import linear_sum_assignment
    ti, fi = [], []
    for k in np.union1d(t_frame, f_frame):
        a, b = np.flatnonzero(t_frame == k), np.flatnonzero(f_frame == k)
        if not len(a) or not len(b):
            continue
        dx = (t_xyz[a, None, 0] - f_xyz[None, b, 0]) * pixel_nm[0]
        dy = (t_xyz[a, None, 1] - f_xyz[None, b, 1]) * pixel_nm[1]
        dz = t_xyz[a, None, 2] - f_xyz[None, b, 2]
        lat = np.hypot(dx, dy)
        cost = np.where((lat <= LATERAL_NM) & (np.abs(dz) <= AXIAL_NM), lat, 1e9)
        r, c = linear_sum_assignment(cost)
        good = cost[r, c] < 1e9
        ti.append(a[r[good]])
        fi.append(b[c[good]])
    if not ti:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(ti), np.concatenate(fi)


def evaluate(truth: dict, found: dict, pixel_nm) -> dict:
    """``truth``: frame, xyz (n, 3), crlb (n, 3) or None; ``found``: frame, xyz, err (m, 3) in nm."""
    ti, fi = match(truth["frame"], truth["xyz"], found["frame"], found["xyz"], pixel_nm)
    tp, fp, fn = len(ti), len(found["frame"]) - len(ti), len(truth["frame"]) - len(ti)
    jac = tp / max(tp + fp + fn, 1)
    out = {"jaccard": jac, "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1)}
    if tp < 10:
        return out
    t, f = truth["xyz"][ti], found["xyz"][fi]
    dx, dy, dz = (f[:, 0] - t[:, 0]) * pixel_nm[0], (f[:, 1] - t[:, 1]) * pixel_nm[1], f[:, 2] - t[:, 2]
    lat, ax = float(np.sqrt(np.mean(dx ** 2 + dy ** 2))), float(np.sqrt(np.mean(dz ** 2)))
    eff = lambda rmse, a: float((100 - np.sqrt((100 * (1 - jac)) ** 2 + (a * rmse) ** 2)) / 100)
    out.update(rmse_lat_nm=lat, rmse_ax_nm=ax, eff_lat=eff(lat, 1.0), eff_ax=eff(ax, 0.5),
               pixel_hist_cols=subpixel_histogram(f[:, 0]), pixel_hist_rows=subpixel_histogram(f[:, 1]),
               pixel_ks=ks_uniform(np.r_[f[:, 0], f[:, 1]]))
    err = found["err"][fi]
    out["claim_width_lat"] = robust_width(np.r_[dx / err[:, 0], dy / err[:, 1]])
    out["claim_width_ax"] = robust_width(dz / err[:, 2])
    if truth.get("crlb") is not None:
        b = truth["crlb"][ti]
        out["crlb_width_lat"] = robust_width(np.r_[dx / b[:, 0], dy / b[:, 1]])
        out["crlb_width_ax"] = robust_width(dz / b[:, 2])
    # the whole range in 200 nm bins of |z|: near the focus decides, but no bin may fail
    az_all, az = np.abs(truth["xyz"][:, 2]), np.abs(t[:, 2])
    by_z = []
    for lo in range(0, 1000, 200):
        sel_t = (az_all >= lo) & (az_all < lo + 200)
        sel = (az >= lo) & (az < lo + 200)
        if sel_t.sum() < 50 or sel.sum() < 20:
            continue
        row = {"lo": lo, "hi": lo + 200, "n": int(sel_t.sum()), "recall": float(sel.sum() / sel_t.sum()),
               "rmse_lat_nm": float(np.sqrt(np.mean(dx[sel] ** 2 + dy[sel] ** 2))),
               "rmse_ax_nm": float(np.sqrt(np.mean(dz[sel] ** 2))), "bias_z_nm": float(np.mean(dz[sel])),
               "pixel_ks": ks_uniform(np.r_[f[sel, 0], f[sel, 1]]),
               "claim_width_ax": robust_width((dz[sel] - np.median(dz[sel])) / err[sel, 2])}
        if truth.get("crlb") is not None:
            b = truth["crlb"][ti][sel]
            row["crlb_width_lat"] = robust_width(np.r_[dx[sel] / b[:, 0], dy[sel] / b[:, 1]])
            row["crlb_width_ax"] = robust_width(dz[sel] / b[:, 2])
        by_z.append(row)
    out["by_abs_z"] = by_z
    core = np.abs(t[:, 2]) < CORE_Z_NM
    if core.sum() >= 50:
        out["core_rmse_lat_nm"] = float(np.sqrt(np.mean(dx[core] ** 2 + dy[core] ** 2)))
        out["core_rmse_ax_nm"] = float(np.sqrt(np.mean(dz[core] ** 2)))
        out["core_pixel_ks"] = ks_uniform(np.r_[f[core, 0], f[core, 1]])
        if truth.get("crlb") is not None:
            b = truth["crlb"][ti][core]
            out["core_crlb_width_lat"] = robust_width(np.r_[dx[core] / b[:, 0], dy[core] / b[:, 1]])
            out["core_crlb_width_ax"] = robust_width(dz[core] / b[:, 2])
    return out
