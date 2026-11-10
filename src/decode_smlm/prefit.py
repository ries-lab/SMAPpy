"""The training's ranges from a fit of the data: SMAP's ``DECODE_training_estimates``, ported.

Milestone (e) of docs/decode/poc-plan.md.  DECODE learns from frames it simulates, so it must be
told what the data look like: how bright a molecule is and for how long, the background, the z
range and the camera.  SMAP read these off the fit already loaded (``usecurrent_callback``,
``setz`` and ``finalizejson`` in ``plugins/+Analyze/+calibrate/DECODE_training_estimates.m``),
and so does this, with every rule it applied -- the constants below are SMAP's.  It needs numpy
and scipy only: the GUI calls it on every change of the form, and it must not load torch.

The rules, on SMAP's two tables: the blinks (the localizations grouped across frames, one row a
molecule's appearance) and the localizations themselves.

* **On-time** (SMAP's ``lifetime.mu``): the mean of the blinks' frame counts, ``n_in_group``,
  over ``make_statistics2``'s range, without its fit (`on_time` says why); DECODE's ``lifetime_avg`` is that less one, at
  least 1 -- a blink of mean length t in continuous time touches t + 1 frames on average.
* **Intensity** -- not SMAP's rule.  DECODE draws a molecule's photons a frame while it is on;
  a frame it is on for throughout gets that many, and the frames it switches on or off in get
  less.  So the intensity is the photons of the *fully-on* frames -- the middle frames of blinks
  seen in 3 frames or more -- of the fits whose likelihood is near the bulk (`fully_on`): their
  mean and standard deviation.  SMAP took the blinks' mean photons over ``lifetime_avg``, as
  (1, 0.2) times that, which the poor fits' photons pulled to twice the example package's own
  value (docs/decode/poc-plan.md §8, "(e) as done"); with too few fully-on frames, it falls back
  to that (`intensity`).
* **Background**: the localizations' 1st to 98th percentile, widened by 30 % of that range on
  each side, its lower end at least 0.9 times the 0.05th percentile.
* **z**: the blinks' 1st to 99th percentile, widened by 20 % on each side and clamped to the
  calibration's range; +-750 nm without z.
* **Camera**: the file's, with SMAP's read noise of 74.4 e- with EM gain and 1.5 e- without
  where the camera does not give its own, quantum efficiency 1 (photons here are detected
  photons), spurious charge 0.0015 with EM gain, 0 without (plan §6, item 8).
* **Density** is chosen, not measured: low, high or ultra-high, 15, 25 or 50 emitters in a
  40 x 40 frame (``finalizejson``).  The density in the data can change during an acquisition,
  and a model trained at a low one learns less from each frame, so the measured one
  (`measured_density`, SMAP's formula) is shown beside the choice, not used.

Where this departs from SMAP it says so at the function.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DENSITIES = {"LD": 15, "HD": 25, "UHD": 50}     # emitters in a 40 x 40 frame
DENSITY_FRAME_PX = 40
READ_NOISE_EM_E, READ_NOISE_E = 74.4, 1.5       # SMAP's, where the camera gives none
SPURIOUS_EM = 0.0015                            # DECODE's default, with EM gain
Z_WITHOUT_Z_NM = 750.0
INTENSITY_SD = 0.2                              # sd / mean of the photons per frame (SMAP's rule)
FULLY_ON_MIN_FRAMES = 3                         # a blink's middle frames are fully on from 3 frames
FULLY_ON_ENOUGH = 200                           # fewer fully-on frames: SMAP's rule instead
LOGL_MADS = 1.0                                 # a fit more MADs below the median logl_rel is left out: strict
Z_EDGE = 0.9                                    # a fit this close to the calibration's z limit is left out
SHORT_ON_TIME, BRIGHTER = 0.8, 1.3              # below this on-time, 1.3 times brighter
BG_QUANTILES, BG_WIDEN, BG_FLOOR_Q, BG_FLOOR = (0.01, 0.98), 0.3, 0.0005, 0.9
Z_QUANTILES, Z_WIDEN = (0.01, 0.99), 0.2


def _quantile(v, q):
    # MATLAB's quantile is the piecewise-linear one with knots at (k - 0.5) / n: numpy's "hazen"
    return np.quantile(np.asarray(v, np.float64), q, method="hazen")


@dataclass(frozen=True)
class Estimates:
    """What the data say, in DECODE's terms."""
    on_time_frames: float           # frames a blink is seen in, SMAP's lifetime.mu
    lifetime_frames: float          # DECODE's lifetime_avg
    intensity_mu_sig: tuple         # photons a frame while on, DECODE's intensity_mu_sig
    intensity_from: str             # "fully-on frames" or "blinks" (SMAP's rule)
    n_fully_on: int                 # the fully-on frames it was measured on
    background: tuple               # photons a pixel, DECODE's bg_uniform
    z_nm: tuple                     # DECODE's emitter extent in z
    emitters_per_frame: float       # measured, in a 40 x 40 frame: shown, not used
    n_blinks: int
    n_localizations: int


def on_time(n_in_group) -> float:
    """The mean number of frames a blink is seen in, over SMAP's range.

    SMAP (``make_statistics2``, ``meanexp``) histograms the counts from 1 up to K = max(95th
    percentile, 13), fits a sum of three exponentials to the histogram and takes the mean of the
    fitted curve over 1 to K + 1.  The cut at K is kept: it keeps the few very long blinks, often
    two molecules linked, from pulling the mean.  **The fit is not.**  A sum of decaying
    exponentials is log-convex -- the ratio of one bar to the one before can only grow along it
    -- and blinks' frame counts fall faster after the first frames than before (a blink of mean
    length t in continuous time is seen in t + 1 frames on average, few in 1, most in 1 or 2), so
    the fit runs flatter than the histogram and its mean high: on a SMAPpy simulation of 0.7
    frames' on-time, 1.95 frames for the blinks' 1.64 (true: 1.66), and on exponential blinks of
    1.5 frames 3.3 for 2.5.  The histogram's own mean over the same range is used instead.
    """
    v = np.asarray(n_in_group, np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no blinks: an on-time needs the grouped localizations")
    k = max(float(_quantile(v, 0.95)), 13.0)
    v = v[(v >= 0.5) & (v < np.floor(k) + 1.5)]             # SMAP's bins, 1 to K + 1
    return float(v.mean())


def lifetime(on_time_frames: float) -> float:
    return max(1.0, on_time_frames - 1.0)


def intensity(photons, on_time_frames: float) -> tuple:
    """Photons a frame while on: the blinks' mean photons over DECODE's lifetime."""
    per_frame = float(np.mean(photons)) / lifetime(on_time_frames)
    if on_time_frames < SHORT_ON_TIME:
        per_frame *= BRIGHTER
    return per_frame, INTENSITY_SD * per_frame


def fully_on(group_id, frame, n_in_group, logl_rel=None, z_nm=None, z_limit_nm=None,
             iterations=None, max_iterations=None, logl_mads: float = LOGL_MADS) -> np.ndarray:
    """The localizations a molecule was on for throughout, whose fits measure its brightness.

    The middle frames of the blinks seen in 3 frames or more, of the fits that are not failures:
    ``logl_rel`` at most ``logl_mads`` robust standard deviations (1.4826 MAD) below its median --
    overlapping molecules fit worse and are brighter than one; relative, because the scale of
    ``logl_rel`` depends on the camera (a median of -5 on the example package's EM frames) -- and
    neither at the iteration limit nor within 10 % of the calibration's z limit.  Losing most of the
    localizations is fine: a few hundred fully-on frames give the brightness.  On the example
    package (10 blocks of 20 frames) 1, 2, 3 and 5 MADs kept 3 286 to 4 164 fully-on frames and
    gave 12 700, 13 400, 13 800 and 14 100 photons (the package's own 14 000 at its frames' gain).
    """
    gid = np.asarray(group_id)
    f = np.asarray(frame)
    order = np.lexsort((f, gid))
    g = gid[order]
    first = np.r_[True, g[1:] != g[:-1]]
    last = np.r_[g[1:] != g[:-1], True]
    keep = np.zeros(len(gid), bool)
    keep[order] = ~first & ~last
    keep &= np.asarray(n_in_group) >= FULLY_ON_MIN_FRAMES
    if logl_rel is not None:
        ll = np.asarray(logl_rel, np.float64)
        med = np.median(ll)
        keep &= ll >= med - logl_mads * 1.4826 * np.median(np.abs(ll - med))
    if z_nm is not None and z_limit_nm is not None:
        keep &= np.abs(np.asarray(z_nm)) < Z_EDGE * z_limit_nm
    if iterations is not None and max_iterations is not None:
        keep &= np.asarray(iterations) < max_iterations
    return keep


def intensity_fully_on(photons, mask) -> tuple:
    """Photons a frame while on: the mean and standard deviation over the fully-on frames."""
    v = np.asarray(photons, np.float64)[mask]
    return float(v.mean()), float(v.std())


def background_range(background) -> tuple:
    lo, hi = _quantile(background, BG_QUANTILES)
    width = hi - lo
    low = max(lo - BG_WIDEN * width, BG_FLOOR * _quantile(background, BG_FLOOR_Q))
    return float(low), float(hi + BG_WIDEN * width)


def z_range(z_nm, limit_nm: float | None = None) -> tuple:
    """The blinks' z, widened; ``limit_nm`` the calibration's half range, about its focus."""
    if z_nm is None or len(z_nm) == 0:
        lo, hi = -Z_WITHOUT_Z_NM, Z_WITHOUT_Z_NM
    else:
        lo, hi = _quantile(z_nm, Z_QUANTILES)
        width = hi - lo
        lo, hi = lo - Z_WIDEN * width, hi + Z_WIDEN * width
    if limit_nm is not None:
        lo, hi = max(lo, -limit_nm), min(hi, limit_nm)
    return float(lo), float(hi)


def z_limit(calibration) -> float | None:
    """Half the z range of a SMAPpy `SplineCalibration`: SMAP's ``fminmax`` frames where the file
    has them, else the spline's grid on the nearer side of its focus."""
    params = getattr(calibration, "parameters", None) or {}
    fminmax = params.get("fminmax")
    dz = params.get("dz", calibration.dz)
    if fminmax is not None and np.size(fminmax) == 2:
        f = np.asarray(fminmax, np.float64).ravel()
        return float((f[1] - f[0]) * float(dz) / 2)
    nz = calibration.coeff.shape[1]
    return float(min(calibration.z0, nz - 1 - calibration.z0) * calibration.dz)


def measured_density(n_blinks: int, area_px: float, n_frames: float) -> float:
    """SMAP's measured density, emitters in a 40 x 40 frame: blinks per pixel and frame."""
    return float(n_blinks / area_px / max(n_frames, 1) * DENSITY_FRAME_PX ** 2)


def estimates(blinks: dict, localizations: dict, *, area_px: float, n_frames: int,
              z_limit_nm: float | None = None, max_iterations: int | None = None,
              logl_mads: float = LOGL_MADS) -> Estimates:
    """The estimates from the prefit's columns, as arrays.

    ``blinks``: ``photons``, ``n_in_group``, optional ``z_nm``.  ``localizations``: ``photons``,
    ``background``, and for the fully-on frames ``group_id``, ``frame``, ``n_in_group``, with
    ``logl_rel``, ``z_nm`` and ``iterations`` where the fit has them.  ``area_px`` and
    ``n_frames`` are the area and the number of frames fitted, for the measured density.
    """
    n_in_group = np.asarray(blinks["n_in_group"])
    on = on_time(n_in_group)
    z = blinks.get("z_nm")
    loc = localizations
    full = np.zeros(len(loc["background"]), bool)
    if all(k in loc for k in ("group_id", "frame", "n_in_group", "photons")):
        full = fully_on(loc["group_id"], loc["frame"], loc["n_in_group"], loc.get("logl_rel"),
                        loc.get("z_nm"), z_limit_nm, loc.get("iterations"), max_iterations, logl_mads)
    if full.sum() >= FULLY_ON_ENOUGH:
        mu_sig, source = intensity_fully_on(loc["photons"], full), "fully-on frames"
    else:
        mu_sig, source = intensity(blinks["photons"], on), "blinks"
    return Estimates(on_time_frames=on, lifetime_frames=lifetime(on), intensity_mu_sig=mu_sig,
                     intensity_from=source, n_fully_on=int(full.sum()),
                     background=background_range(loc["background"]),
                     z_nm=z_range(None if z is None else np.asarray(z), z_limit_nm),
                     emitters_per_frame=measured_density(len(n_in_group), area_px, n_frames),
                     n_blinks=int(len(n_in_group)), n_localizations=int(len(loc["background"])))


def camera(conversion: float, offset: float, pixel_size_nm, em_gain: float | None = None,
           read_noise_e: float | None = None) -> dict:
    """DECODE's ``Camera`` section.  ``pixel_size_nm`` is (x, y) in SMAPpy's order; DECODE's
    ``px_size`` is (rows, columns), so SMAP wrote it y first (``cam_pixelsize_um([2 1])``)."""
    em = em_gain is not None and em_gain > 0
    px, py = np.broadcast_to(np.asarray(pixel_size_nm, np.float64), 2)
    return {"baseline": float(offset), "e_per_adu": float(conversion), "em_gain": float(em_gain) if em else None,
            "px_size": [float(py), float(px)], "qe": 1.0,
            "read_sigma": float(read_noise_e) if read_noise_e else (READ_NOISE_EM_E if em else READ_NOISE_E),
            "spur_noise": SPURIOUS_EM if em else 0.0, "convert2photons": True}


def method_params(est: Estimates, camera_section: dict, density: str, frame_px: int = DENSITY_FRAME_PX,
                  calibration_file: str = "calibration.h5") -> dict:
    """A job's ``method.yaml``: DECODE's minimal parameter file (its ``param_friendly.yaml``), filled.

    Everything DECODE's ``reference.yaml`` has and this leaves out is filled from it when the
    file is loaded (``param_io.load_params``).  Paths are relative to the job folder, so the
    folder can move.
    """
    sim = simulation(est, density)
    extent = [-0.5, frame_px - 0.5]
    return {"Camera": dict(camera_section),
            "InOut": {"calibration_file": calibration_file, "experiment_out": "."},
            "Simulation": {"bg_uniform": sim["bg_uniform"], "emitter_av": sim["emitter_av"],
                           "emitter_extent": [extent, list(extent), sim["z_extent_nm"]],
                           "intensity_mu_sig": sim["intensity_mu_sig"], "lifetime_avg": sim["lifetime_avg"]}}


def simulation(est: Estimates, density: str) -> dict:
    """The parts of DECODE's ``Simulation`` section the estimates and the chosen density set."""
    if density not in DENSITIES:
        raise ValueError(f"density {density!r}: one of {', '.join(DENSITIES)}")
    return {"emitter_av": DENSITIES[density], "lifetime_avg": est.lifetime_frames,
            "intensity_mu_sig": [float(v) for v in est.intensity_mu_sig],
            "bg_uniform": [float(v) for v in est.background],
            "z_extent_nm": [round(est.z_nm[0]), round(est.z_nm[1])]}
