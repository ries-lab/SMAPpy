"""A NumPy port of the TU Delft vectorial PSF (vecfit: Stallinga & Rieger), written in vecfit's own conventions.

Ported from the review's cross-check (uiPSF/docs/review_2026/scripts/crosscheck_vecfit.py), which follows
get_pupil_matrix.m, get_field_matrix_derivatives.m and get_psfs_derivatives.m: a freely rotating
dipole, stage-scan defocus, no blur.  vecfit uses the complex-conjugate convention of uiPSF and v2:
waves carry exp(−i k·r), so phases enter with a minus sign and the square root above the critical
angle takes the −i branch.  Intensities must come out the same; that is the point of the test.
Image axes here are v2's ([z, y, x], x along columns), not MATLAB's.
"""
import numpy as np


def psf(NA, lam, n_imm, n_med, n_cov, P, shape, pixel_nm, z_nm, W, x0=0.0, y0=0.0):
    """Intensity [K, Y, X] for an emitter at (x0, y0) nm and stage defocus z_nm [K]; W [P, P] phase in rad."""
    D = 2 / P
    c = np.arange(-1 + D / 2, 1, D)
    XP, YP = np.meshgrid(c, c)                      # XP along columns, as v2
    r2 = XP ** 2 + YP ** 2
    arg = 1 - r2 * NA ** 2 / n_med ** 2
    cmed = np.sqrt(np.abs(arg)) * np.where(arg < 0, -1j, 1)        # vecfit's branch
    ccov = np.sqrt(1 - r2 * NA ** 2 / n_cov ** 2 + 0j)
    cimm = np.sqrt(1 - r2 * NA ** 2 / n_imm ** 2 + 0j)
    FPmc = 2 * n_med * cmed / (n_med * ccov + n_cov * cmed)
    FSmc = 2 * n_med * cmed / (n_med * cmed + n_cov * ccov)
    FPci = 2 * n_cov * ccov / (n_cov * cimm + n_imm * ccov)
    FSci = 2 * n_cov * ccov / (n_cov * ccov + n_imm * cimm)
    FP, FS = FPmc * FPci, FSmc * FSci
    phi = np.arctan2(YP, XP)
    cp, sp = np.cos(phi), np.sin(phi)
    sth = np.sqrt(1 - cmed ** 2)
    pvec = np.stack([FP * cmed * cp, FP * cmed * sp, -FP * sth])
    svec = np.stack([-FS * sp, FS * cp, 0 * FS])
    pol = np.stack([cp * pvec - sp * svec, sp * pvec + cp * svec])          # (2, 3, P, P)
    amp = (r2 < 1) * np.sqrt(cimm) / (n_med * cmed)
    pupil = amp * np.exp(-1j * W) * pol
    kx = NA / lam * XP
    ky = NA / lam * YP
    kz = n_imm / lam * cimm
    Y, X = shape
    xs = (np.arange(X) - (X - 1) / 2) * pixel_nm[0] - x0
    ys = (np.arange(Y) - (Y - 1) / 2) * pixel_nm[1] - y0
    Ex = np.exp(-2j * np.pi * np.einsum("x,ij->xij", xs, kx))
    Ey = np.exp(-2j * np.pi * np.einsum("y,ij->yij", ys, ky))
    out = np.zeros((len(z_nm), Y, X))
    for k, z in enumerate(z_nm):
        F = np.exp(-2j * np.pi * kz * z) * pupil
        E = np.einsum("abij,yij,xij->abyx", F, Ey, Ex)
        out[k] = (np.abs(E) ** 2).sum((0, 1)) / 3
    return out
