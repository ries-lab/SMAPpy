"""Softer out-of-focus selection for a molecule stack, as a patch of smappy.calibrate.molecule_stack (a test).

Imported by m2_new.py when SOFT=1: significance 3.5; off centre within max(1.5 px, the curve's width); shape
tolerance 0.3 within ±400 nm, 0.6 beyond; brightness one window for all z from the in-focus spots (|d| < 300 nm),
from half their median to their median + 3 robust SDs (log), on photons corrected for the elliptical Gaussian's
bias on a defocused PSF (BIAS, measured on the AB model in a 21 px region).
"""
import numpy as np
import smappy.calibrate.molecule_stack as M

BIAS_D = np.arange(-800, 801, 200.0)        # measured: Gaussian / true photons on the AB model, 21 px
BIAS = np.array([0.96, 0.84, 0.76, 0.76, 0.73, 0.69, 0.64, 0.55, 0.58])


def select(s, settings, roi_px):
    h = roi_px // 2
    g = s["gauss"]
    n = len(s["frame"])
    d = s["d_nm"]
    state = ["candidate"] * n

    def out(mask, reason):
        for i in np.flatnonzero(mask):
            if state[i] == "candidate":
                state[i] = "rejected: " + reason
    fitted = np.isfinite(g["photons"]) & np.isfinite(g["sigma_x_pix"]) & np.isfinite(g["sigma_y_pix"])
    out(~fitted, "fit failed")
    import os
    width = np.hypot(s["sx_curve"], s["sy_curve"]) / np.sqrt(2)
    lim = np.maximum(M.MAX_OFF_CENTRE_PX, float(os.environ.get("OFF_SCALE", 1.0)) * width)
    tol = np.where(np.abs(d) <= 400, settings.width_tol, 2 * settings.width_tol)
    with np.errstate(invalid="ignore"):
        out(~(np.hypot(g["x_roi"] - h, g["y_roi"] - h) < lim), "off centre")
        out(~((np.abs(np.log(g["sigma_x_pix"] / s["sx_curve"])) < tol)
              & (np.abs(np.log(g["sigma_y_pix"] / s["sy_curve"])) < tol)), "shape")
    rejected, masks = M.neighbours(s, roi_px, settings.neighbour_fraction)
    out(rejected | ((~masks).mean((1, 2)) > settings.max_masked), "neighbour")
    alive = np.array([st == "candidate" for st in state])
    with np.errstate(divide="ignore", invalid="ignore"):
        logn = np.log(np.where(alive, g["photons"] / np.interp(d, BIAS_D, BIAS), np.nan))
    ref = alive & (np.abs(d) < 300) & np.isfinite(logn)
    if ref.sum() >= 20:
        med = np.median(logn[ref])
        mad = 1.4826 * np.median(np.abs(logn[ref] - med))
        bright = (logn >= med + np.log(0.5)) & (logn <= med + 3 * mad)
    else:
        bright = alive
    out(alive & ~bright, "brightness")
    return ["not drawn" if st == "candidate" else st for st in state], masks


M.select = select
