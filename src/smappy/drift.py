"""Drift correction: estimate with COMET, apply to every localization.

Sample drift is a property of the *acquisition*, not of a subset of it, so the
two halves of drift correction want different data:

* **Estimating** it works best on a clean subset -- bright, well-fitted
  localizations -- because a badly localized point contributes noise to the
  overlap cost and nothing else.  That is exactly what the display filter
  already selects, so `estimate_drift` takes the same `select` a render does.
* **Applying** it is a coordinate correction and must reach *all*
  localizations, including the ones the filter hides.  A filter is a view;
  changing it later must not require re-running the correction.

The estimator is COMET (Cost-function Optimized Maximal Overlap drift
EsTimation, https://github.com/gpufit/Comet), used as a library -- it maximises
the spatiotemporal overlap of localizations, across time windows or, by
default here, along a smooth spline in time, which needs no fiducials and no
reference structure.  Its result is a per-frame drift table
whose row index *is* the frame number, so applying it is one fancy-index per
coordinate.

**To simplify (2026-10 review, kept for now on purpose).**  Only the spline
fit is the default and the one users run; the time-window fit (`spline=False`)
is kept as legacy, for quality control and for comparison.  Its CUDA and torch
backends are gone (2026-10: nobody ran them, the torch one could not work, and
they were the only reason for the `cuda` extra).  What is left to decide is
whether to keep the time-window fit at all: without it the spline fit's two
kernels (`cpu_wrapper_chunked_approx`, `pair_indices_kdtree`) move into this
module and the rest of `smappy._comet` goes with the ~12 settings only the
time-window fit reads.  `two_stage` goes either way: the RCC prepass does that
job more cheaply.

Not wrapped: COMET's own file I/O, plotting and molecule-set output.  We hand it
an array and get an array back; reading and writing stays with `io.hdf5`, so the
corrected file is an ordinary smappy file that the viewer opens unchanged.

Units: COMET works in nm.  A table in pixels is converted for the estimate and
the drift is divided back by the pixel size when applied, so the corrected table
keeps whatever units it had.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .locs import Localizations
from .params import param

# the position columns a drift correction touches, and the drift axis each
# follows.  Deliberately not `peak_*`: those are detection positions, kept as
# the record of what was found where, not measurements of the structure.
# A fit's table has no `x_pix`; a file saved in pixels keeps its own beside
# the nm ones `io.formats.with_nm` adds, and they must move with them.
_POSITION_COLUMNS = {"x_nm": 0, "x_pix": 0,
                     "y_nm": 1, "y_pix": 1,
                     "z_nm": 2}

# The linear transformations COMET can fit per frame besides the shift: for
# each, which scales the cost divides out of a pair's difference (`_SCALES`)
# and its generators -- A(t) = I + sum_p theta_p(t) E_p, each theta_p a
# spline.  The generators are orthonormal up to a factor, so theta is read
# back from any A of the model by projection (`Drift.parameters`).  Nothing
# turns about x or y: a sample or a lens does not tilt, so the step after
# the xy models leaves z decoupled -- the xy affine map with a scale of z of
# its own -- before the full 3D one.
def _unit(r, c):
    e = np.zeros((3, 3))
    e[r, c] = 1.0
    return e


_SCALE_XY = np.diag([1.0, 1.0, 0.0])
_ROTATION_XY = _unit(1, 0) - _unit(0, 1)
_AFFINE_XY = tuple((f"a_{'xy'[r]}{'xy'[c]}", _unit(r, c))
                   for r in range(2) for c in range(2))
TRANSFORMATIONS = {
    "shift": ("", ()),
    "xy scale": ("xy", (("scale", _SCALE_XY),)),
    "xy scale + rotation": ("xy", (("scale", _SCALE_XY),
                                   ("rotation", _ROTATION_XY))),
    "xy affine": ("xy", _AFFINE_XY),
    "xy affine + z scale": ("xy+z", _AFFINE_XY + (("z_scale", _unit(2, 2)),)),
    "3D affine": ("3d", tuple((f"a_{'xyz'[r]}{'xyz'[c]}", _unit(r, c))
                              for r in range(3) for c in range(3))),
}
# How the cost measures a pair's difference (`csrc/drift.hpp`,
# `transformed_cost_and_gradient`): x and y in units of the xy part's
# scale, det(A_xy)^(1/2), and z in units of z's -- 1 where z is only
# shifted, A_zz where it is scaled on its own, and det(A)^(1/3) for both
# where the map mixes them.
_SCALES = ("xy", "xy+z", "3d")

# What a drift curve is zero at (`Drift.referenced`): left as the estimator
# made it, or the first or the last frame.
REFERENCES = ("as estimated", "first frame", "last frame")


@dataclass
class DriftSettings:
    """COMET parameters.  The defaults suit a typical SMLM movie.

    ``segmentation_mode`` sets what ``segmentation_var`` counts:
    0 = number of time windows, 1 = localizations per window,
    2 = frames per window (the default).

    A time window must hold enough localizations to define the structure --
    a few thousand is a good target -- and the movie must give at least a few
    tens of windows for the drift curve to have any time resolution.
    """

    segmentation_mode: int = param(2, label="window unit",
                                   choices=((2, "frames"),
                                            (1, "localizations"),
                                            (0, "windows")),
                                   help="what window size counts; "
                                        "time-window fit only",
                                   advanced=True)
    segmentation_var: int = param(500, label="window size", min=1,
                                  help="frames or localizations per time "
                                       "window, or the number of windows; "
                                       "time-window fit only")
    max_drift_nm: float = param(300.0, label="max drift", unit="nm", min=1,
                                help="the largest drift expected; also the "
                                     "radius within which localizations are "
                                     "paired")
    target_sigma_nm: float = param(30.0, label="target sigma", unit="nm",
                                   min=0.1,
                                   help="the finest width of the overlap "
                                        "Gaussian, where refinement stops")
    initial_sigma_nm: Optional[float] = param(None, label="initial sigma",
                                              unit="nm",
                                              help="the width the "
                                                   "refinement starts from; "
                                                   "auto: a third of max "
                                                   "drift",
                                              advanced=True)
    # The bound is for the panel as much as for sense: an unbounded spin box
    # is sized for ten digits, which took COMET's panel past the control
    # column (`tests/test_control_width.py`) once the fields were styled.
    boxcar_width: int = param(1, label="smoothing", unit="windows", min=1,
                              max=1000,
                              help="running mean over this many windows "
                                   "between refinement steps; 1 is none; "
                                   "time-window fit only",
                              advanced=True)
    interpolation: str = param("cubic", choices=("cubic", "catmull-rom"),
                               help="the curve from the window estimates to "
                                    "every frame; time-window fit only",
                               advanced=True)
    max_locs_per_segment: Optional[int] = param(None,
                                                label="max locs / window",
                                                min=1,
                                                help="a random subset of at "
                                                     "most this many per "
                                                     "window; auto: all; "
                                                     "time-window fit only",
                                                advanced=True)
    # L-BFGS-B relative tolerance.  COMET's own is 1e3*eps (~2e-13), which buys
    # nothing here: loosening it to 1e-7 costs 0.14 nm rms (0.9 nm worst case)
    # on the drift curve -- far below any localization precision -- and cuts the
    # cost evaluations from 423 to 154.  Raise it towards 1e-13 to reproduce
    # upstream COMET exactly.
    optimizer_ftol: float = param(1e-7, label="ftol",
                                  help="relative change of the cost at "
                                       "which the optimizer stops",
                                  advanced=True)
    # Skip pairs beyond 6 sigma, where each contributes less than e^-9 of a
    # coincident pair: 1.3x faster.  The Gaussian itself is exact -- the table
    # and polynomial this once used went with the numba kernel -- so the name
    # is historical.  The spline fit always uses the cutoff.  See docs/drift.md.
    approximate_kernel: bool = param(True, label="approximate kernel",
                                     help="skip pairs more than 6 sigma "
                                          "apart; the spline fit always "
                                          "does",
                                     advanced=True)
    # A looser tolerance for the coarse sigma steps.  None (the default) means
    # the same as `optimizer_ftol`: measured, loosening the coarse steps is a
    # bad trade -- 2.3x faster for tens of nm on individual windows.  See
    # docs/drift.md.
    optimizer_ftol_coarse: Optional[float] = param(None, label="ftol coarse",
                                                   help="ftol for the steps "
                                                        "above the target "
                                                        "sigma; auto: ftol; "
                                                        "time-window fit only",
                                                   advanced=True)
    # Estimate from *grouped* localizations: one entry per emitter blink rather
    # than one per frame.  20-35x faster (it removes exactly the dense, very
    # short-range pairs, which is where the pair count grows fastest) and, on
    # the clathrin dataset, recovers 96% of the improvement the full estimate
    # makes to the image.  Off by default: the full estimate is the accurate
    # one, and is now fast enough to be the thing you run.
    # Check each time window's estimate against the overlap it would have with
    # no drift correction at all, and discard the ones that do not beat it --
    # the spline then bridges them.  Costs one extra pass over the pairs.
    quality_control: bool = param(False, label="quality control",
                                  help="discard time windows whose fit does "
                                       "not beat no correction; needs fit "
                                       "spline off",
                                  advanced=True)
    # A window is discarded when its fitted drift improves its own overlap by
    # less than this, relative to no correction at all.  0.0 keeps every window
    # whose estimate is better than nothing, which is the honest bar.
    min_lift: float = param(0.0, label="min lift",
                            help="how much a window's fit must improve its "
                                 "overlap over no correction to be kept",
                            advanced=True)
    group: bool = param(True, label="merge blinks",
                        help="merge the localizations of one blink into one "
                             "before estimating")
    # None: as the session merges blinks (`merge_from_session`), so that the
    # estimate sees the blinks the Render tab shows; 50 nm and 1 frame, the
    # session's own defaults, where there is no session
    group_dx_nm: Optional[float] = param(None, label="within", unit="nm", min=0,
                                         help="how close localizations in "
                                              "consecutive frames must be to be "
                                              "merged; auto: the session's merge "
                                              "settings",
                                         advanced=True)
    group_dt: Optional[int] = param(None, label="dark frames", unit="frames", min=0,
                                    help="how many dark frames a blink may skip; "
                                         "auto: the session's merge settings",
                                    advanced=True)
    # Estimate in two passes: a grouped one to get within a few nm, then an
    # ungrouped one over a small radius to refine.  The pair count is what costs
    # time, and once the coarse pass has taken the drift out, a 30 nm radius
    # keeps a seventh of the pairs -- 18 s instead of 153 s for 99% of the
    # improvement.  The small radius also bounds the fine pass, which is what
    # keeps it from producing the runaway window the single pass has here.
    two_stage: bool = param(False, label="two stage",
                            help="a grouped pass, then an ungrouped one "
                                 "within fine radius",
                            advanced=True)
    two_stage_radius_nm: float = param(30.0, label="fine radius", unit="nm",
                                       min=0,
                                       help="the max drift of the second, "
                                            "ungrouped pass",
                                       advanced=True)
    # A coarse RCC pass first, and then COMET only has to find what RCC left
    # behind.  This is the lever when the estimate would take an afternoon:
    # COMET's cost is the pair count, which grows steeply with the search
    # radius, while RCC correlates rendered images and does not care how far it
    # looks.  So RCC takes out the hundreds of nanometres over
    # `rcc_prepass_max_drift_nm` and COMET refines the tens that are left over
    # a `max_drift_nm` of a few tens -- the same trade as `two_stage`, with a
    # first pass that is seconds instead of the expensive half.
    rcc_prepass: bool = param(False, label="RCC first",
                              help="correlate rendered time windows to take "
                                   "out the bulk of the drift, then run "
                                   "COMET over a small max drift",
                              advanced=True)
    rcc_prepass_windows: int = param(10, label="RCC windows", min=2,
                                     help="the time windows RCC correlates",
                                     advanced=True)
    # How far the prepass looks.  None is RCC's own default, which is the right
    # thing when the setting is turned on by hand; the preflight's offer puts
    # the max drift that was asked for here, since that is what the user
    # believes the stage did, and leaves `max_drift_nm` for the refinement.
    rcc_prepass_max_drift_nm: Optional[float] = param(None,
                                                      label="RCC max drift",
                                                      unit="nm",
                                                      help="auto: RCC's own "
                                                           "default",
                                                      advanced=True)
    # Fit the drift as a cubic B-spline in time instead of one free vector per
    # time window.  Drift is smooth -- slow creep, occasionally a small jump --
    # so a curve with a coefficient every `spline_knot_frames` frames has far
    # fewer parameters than a window per 500 frames, and each coefficient is
    # constrained by every localization near it in time rather than by one
    # window's worth.  There is no time window and no interpolation afterwards:
    # every localization is placed at its own frame, and the fitted curve *is*
    # the per-frame drift.  B-splines are variation-diminishing, so the curve
    # cannot overshoot the way an interpolating cubic spline through noisy
    # window estimates does.
    spline: bool = param(True, label="fit spline",
                         help="fit the drift as a smooth curve in time "
                              "rather than one value per time window")
    # 2000 frames was the knee on the clathrin dataset (26 coefficients over
    # 46 005 frames); finer knots are noisier without being sharper, and there
    # was no sign of over-smoothing even at 4000.  A dataset with faster drift
    # wants a smaller spacing, which shows up as out-of-sample sharpness getting
    # worse.  Quality control does not apply here -- there are no time windows.
    spline_knot_frames: int = param(2000, label="knot spacing", unit="frames",
                                    min=10,
                                    help="frames between the spline's "
                                         "coefficients")
    spline_penalty: float = param(0.0, label="spline penalty", min=0,
                                  help="extra penalty on the curve's "
                                       "bending; 0 is none",
                                  advanced=True)
    # A linear transformation per frame besides the shift, each of its
    # parameters a spline in time: for a sample that expands or creeps
    # differently in different places, or optics whose magnification
    # drifts (SMAP's driftcorrectionM fitted the scale alone).  The models,
    # from the least to the most free, are `TRANSFORMATIONS`; which of them
    # a dataset supports is for simulations and the data to say -- the more
    # parameters, the more an uneven density can pull on them.
    transformation: str = param("shift", label="transformation",
                                choices=tuple(TRANSFORMATIONS),
                                help="what changes over time besides the "
                                     "position; spline fit only")
    # None: the drift's own knots.  A transformation of the optics changes
    # far more slowly than the stage drifts, and a coarser spline is fewer
    # parameters for the same data.
    transformation_knot_frames: Optional[int] = param(
        None, label="transformation knots", unit="frames", min=10,
        help="frames between the transformation's spline coefficients; "
             "auto: the knot spacing",
        advanced=True)
    # A pull towards no change, for parameters the data say little about:
    # the z entries over a few hundred nm of depth, or any of them in a
    # sparse selection.  0 is none; see `_TransformationFit` for the units.
    transformation_penalty: float = param(
        0.0, label="transformation penalty", min=0,
        help="pulls the transformation towards none where the data say "
             "little; 0 is none",
        advanced=True)
    use_z: Optional[bool] = param(None, label="correct z",
                                  help="auto: if the table has z")
    # Which frame the curve is zero at (`Drift.referenced`).  "as estimated"
    # changes nothing: the overlap cost does not see a constant offset, so
    # its gradient never moves the mean of what the optimizer starts at zero
    # -- the spline's coefficients, or the time windows' drifts -- and the
    # curve comes out close to zero mean.  The first or last frame is
    # for consecutive files corrected separately (SMAP's reference choice):
    # the earlier one referenced to its last frame and the later one to its
    # first meet where they join.
    reference: str = param("as estimated", label="reference",
                           choices=REFERENCES,
                           help="the frame the drift is zero at; first or "
                                "last frame for consecutive files",
                           advanced=True)


class Drift:
    """Per-frame drift in nm, indexed by frame number -- or on a time axis.

    ``drift[f]`` is ``(dx, dy, dz)``: how far the sample had moved in frame
    ``f``, so correcting a localization *subtracts* it.

    With ``times_s`` the curve is sampled at those times instead, and a
    localization is corrected by the curve interpolated linearly at its
    ``time_s`` (held at the ends).  That is the drift of an acquisition with
    no frames: MINFLUX, whose ``frame`` is only the rank in time, so that a
    per-frame curve would be one entry per localization, valid for one
    ordering of one table; the MBM beads give a point every two seconds.
    Such a curve takes no transformation.

    With a *transformation* (`DriftSettings.transformation`) each frame also
    has a matrix: ``matrix[f]`` is the 3x3 A_f, ``centre_nm`` the point it
    acts about, and a localization at x in frame f is corrected to
    ``centre + A_f (x - centre) - drift[f]``.  Without one ``matrix`` is None
    and the correction is the plain shift.
    """

    def __init__(self, drift_nm: np.ndarray, settings: Optional[DriftSettings] = None,
                 n_used: Optional[int] = None,
                 flagged_windows: Optional[np.ndarray] = None,
                 method: str = "comet", reference: str = "as estimated",
                 matrix: Optional[np.ndarray] = None,
                 centre_nm: Optional[np.ndarray] = None,
                 transformation: str = "shift",
                 times_s: Optional[np.ndarray] = None):
        drift = np.asarray(drift_nm, dtype=np.float64)
        if drift.ndim != 2 or drift.shape[1] != 3:
            raise ValueError(f"drift must be (n_frames, 3), got {drift.shape}")
        self.drift = drift
        if times_s is not None:
            times_s = np.asarray(times_s, dtype=np.float64).reshape(-1)
            if len(times_s) != len(drift):
                raise ValueError(f"{len(times_s)} times for {len(drift)} drift values")
            if len(times_s) > 1 and np.any(np.diff(times_s) <= 0):
                raise ValueError("the drift's times must increase")
            if matrix is not None:
                raise ValueError("a drift on a time axis takes no transformation")
        self.times_s = times_s
        self.settings = settings
        self.n_used = n_used  # localizations the estimate was made from
        # time windows quality control discarded; the curve is interpolated
        # across them, so they are a caveat on the result, not a hole in it
        self.flagged_windows = flagged_windows
        # which estimator made it -- "comet", "rcc", or "rcc+comet" for the
        # prepass -- so a table's metadata and a saved curve say so
        self.method = method
        # which frame the curve is zero at (`REFERENCES`)
        self.reference = reference
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=np.float64).reshape(-1, 3, 3)
            if len(matrix) != len(drift):
                raise ValueError(f"{len(matrix)} matrices for {len(drift)} frames")
        self.matrix = matrix
        self.centre_nm = (np.zeros(3) if centre_nm is None
                          else np.asarray(centre_nm, dtype=np.float64).reshape(3))
        # the model `matrix` was fitted with (`TRANSFORMATIONS`)
        self.transformation = transformation if matrix is not None else "shift"

    def _like(self, drift, matrix, centre, reference=None) -> "Drift":
        return Drift(drift, self.settings, self.n_used, self.flagged_windows,
                     self.method, reference or self.reference, matrix, centre,
                     self.transformation, self.times_s)

    def referenced(self, reference: str) -> "Drift":
        """The curve shifted to be zero at the first or the last frame.

        Only differences of drift are measured, so a constant may be added
        freely; it moves the whole file and nothing within it.  Two
        consecutive files corrected separately, the first to its last frame
        and the second to its first, then meet where one ended and the other
        began.  ``"as estimated"`` returns the curve as it is.  A
        transformation is made the identity there too: the correction
        becomes the one of that frame undone after it, A_r^-1 (A x - d + d_r).
        """
        if reference not in REFERENCES:
            raise ValueError(f"the reference is one of {', '.join(REFERENCES)}, "
                             f"not {reference!r}")
        if reference == "as estimated" or not len(self):
            return self
        r = 0 if reference == "first frame" else -1
        if self.matrix is None:
            return self._like(self.drift - self.drift[r], None, None, reference)
        undo = np.linalg.inv(self.matrix[r])
        matrix = undo @ self.matrix
        drift = (self.drift - self.drift[r]) @ undo.T
        return self._like(drift, matrix, self.centre_nm, reference)

    def affine(self):
        """The correction as ``x -> A x + b`` per frame: (A, b), (n, 3, 3)
        and (n, 3); A is the identity without a transformation."""
        if self.matrix is None:
            return np.broadcast_to(np.eye(3), (len(self), 3, 3)), -self.drift
        c = self.centre_nm
        return self.matrix, c - self.matrix @ c - self.drift

    def then(self, other: "Drift", **kwargs) -> "Drift":
        """This correction followed by ``other``, one curve: the prepass and
        the two-stage fit, whose passes are drift of the same sample measured
        in sequence.  Without transformations the curves add; curves on a
        time axis add where they are sampled at the same times."""
        kwargs.setdefault("reference", other.reference)
        if (self.times_s is None) != (other.times_s is None) or (
                self.times_s is not None
                and not np.array_equal(self.times_s, other.times_s)):
            raise ValueError("only curves on the same axis add: both per frame, "
                             "or both at the same times")
        if self.times_s is not None:
            kwargs.setdefault("times_s", self.times_s)
        if self.matrix is None and other.matrix is None:
            return Drift(self.drift + other.drift, **kwargs)
        a1, b1 = self.affine()
        a2, b2 = other.affine()
        matrix = a2 @ a1
        b = np.einsum("fij,fj->fi", a2, b1) + b2
        c = other.centre_nm if other.matrix is not None else self.centre_nm
        drift = c - matrix @ c - b
        kwargs.setdefault("transformation", other.transformation
                          if other.matrix is not None else self.transformation)
        return Drift(drift, matrix=matrix, centre_nm=c, **kwargs)

    def parameters(self) -> Dict[str, np.ndarray]:
        """The transformation the sample underwent, per frame, by name
        (`TRANSFORMATIONS`): ``scale`` the relative change of its size (an
        expansion is positive), ``rotation`` the angle in radians,
        anticlockwise (for the small angles a drift has), ``a_xy`` an entry
        of the matrix minus the identity.  As the drift is how far the
        sample moved, this is the inverse of the correction ``matrix``.
        Empty without a transformation."""
        if self.matrix is None:
            return {}
        _, generators = TRANSFORMATIONS.get(self.transformation,
                                            TRANSFORMATIONS["3D affine"])
        delta = np.linalg.inv(self.matrix) - np.eye(3)
        return {name: np.einsum("fij,ij->f", delta, e) / float((e * e).sum())
                for name, e in generators}

    def __len__(self) -> int:
        return self.drift.shape[0]

    @property
    def frames(self) -> np.ndarray:
        return np.arange(len(self))

    def at(self, times_s) -> np.ndarray:
        """The curve at ``times_s``, (n, 3): interpolated linearly between
        its samples and held at either end.  Only for a curve on a time axis."""
        if self.times_s is None:
            raise ValueError("this drift is per frame; it has no times")
        t = np.asarray(times_s, dtype=np.float64)
        if not len(self):
            return np.zeros((len(t), 3))
        return np.column_stack([np.interp(t, self.times_s, self.drift[:, axis])
                                for axis in range(3)])

    def __str__(self) -> str:
        span = self.drift.max(axis=0) - self.drift.min(axis=0)
        over = (f"{len(self)} frames" if self.times_s is None or not len(self) else
                f"{self.times_s[-1] - self.times_s[0]:.0f} s")
        text = (f"drift over {over}, range "
                f"x {span[0]:.0f} nm, y {span[1]:.0f} nm, z {span[2]:.0f} nm")
        parameters = self.parameters()
        if parameters:
            ranges = ", ".join(f"{name} {np.ptp(values) * 1e6:.0f}"
                               for name, values in parameters.items())
            text += (f"; {self.transformation} about ({self.centre_nm[0]:.0f}, "
                     f"{self.centre_nm[1]:.0f}) nm, range {ranges} ppm")
        if self.flagged_windows is not None and len(self.flagged_windows):
            text += (f"; {len(self.flagged_windows)} time window(s) flagged and "
                     f"interpolated over: {list(self.flagged_windows)}")
        return text

    def apply(self, locs: Localizations,
              pixelsize_nm: Optional[float] = None, rows=None) -> Localizations:
        """Return a copy of the whole table with the drift subtracted.

        ``rows`` (a mask) limits it to part of the table -- one file of
        several, for *Transform*'s "drift of another file" -- and then the
        table's ``drift_correction`` record, which speaks for the whole
        table, is left to the caller.
        """
        mask = None if rows is None else np.asarray(rows, bool)
        if self.times_s is not None:
            if "time_s" not in locs:
                raise ValueError("this drift is on a time axis, and the table has no "
                                 "time_s; it has " + ", ".join(sorted(locs.columns)))
            times = np.asarray(locs["time_s"], dtype=np.float64)
            move = -self.at(times if mask is None else times[mask])
            frames = None
        else:
            frames = np.asarray(locs["frame"], dtype=np.int64)
            if mask is not None:
                frames = frames[mask]
            if len(frames) and (frames.min() < 0 or frames.max() >= len(self)):
                raise ValueError(
                    f"the table spans frames {frames.min()}..{frames.max()} but the "
                    f"drift covers 0..{len(self) - 1}; it was estimated from a "
                    f"different acquisition")
            # how far each localization moves, in nm: the drift alone, or
            # with a transformation the corrected position minus the old
            # one, which needs every axis's position at once
            move = -self.drift[frames]

        columns = dict(locs.columns)
        if self.matrix is not None:
            positions = np.zeros((len(move), 3))
            for axis, names in enumerate((("x_nm", "x_pix"), ("y_nm", "y_pix"),
                                          ("z_nm",))):
                for name in names:
                    if name in columns:
                        values = np.asarray(columns[name], dtype=np.float64)
                        if mask is not None:
                            values = values[mask]
                        if name.endswith("_pix"):
                            values = values * _pixelsize_nm(locs, pixelsize_nm)
                        positions[:, axis] = values
                        break
            relative = positions - self.centre_nm
            move = move + np.einsum("fij,fj->fi", self.matrix[frames],
                                    relative) - relative
        corrected = []
        for name, axis in _POSITION_COLUMNS.items():
            if name not in columns:
                continue
            shift = move[:, axis]
            if name.endswith("_pix"):
                shift = shift / _pixelsize_nm(locs, pixelsize_nm)
            values = np.asarray(columns[name], dtype=np.float64).copy()
            if mask is None:
                values += shift
            else:
                values[mask] += shift
            columns[name] = values.astype(np.asarray(columns[name]).dtype)
            corrected.append(name)

        metadata = dict(locs.metadata)
        if mask is not None:
            return Localizations(columns, metadata)
        metadata["drift_correction"] = {
            "method": self.method,
            "columns": corrected,
            "n_frames": len(self),
            "axis": "frame" if self.times_s is None else "time_s",
            "n_localizations_used": self.n_used,
            "flagged_windows": (None if self.flagged_windows is None
                                else [int(w) for w in self.flagged_windows]),
            "reference": self.reference,
            "transformation": self.transformation,
            "settings": asdict(self.settings) if self.settings else None,
        }
        if self.matrix is not None:
            metadata["drift_correction"]["centre_nm"] = self.centre_nm.tolist()
        return Localizations(columns, metadata)

    def to_dict(self) -> dict:
        """The curve as plain data, to be kept with the file.

        The estimate, not the run: the settings and the counts are what makes
        the curve readable a week later, and the localizations it was made
        from are in the file already.
        """
        saved = {"drift_nm": self.drift.tolist(), "method": self.method,
                 "reference": self.reference,
                 "settings": asdict(self.settings) if self.settings else None,
                 "n_used": None if self.n_used is None else int(self.n_used),
                 "flagged_windows": (None if self.flagged_windows is None
                                     else [int(w) for w in self.flagged_windows])}
        if self.matrix is not None:
            saved.update(matrix=self.matrix.reshape(-1, 9).tolist(),
                         centre_nm=self.centre_nm.tolist(),
                         transformation=self.transformation)
        if self.times_s is not None:
            saved["times_s"] = self.times_s.tolist()
        return saved

    @classmethod
    def from_dict(cls, saved: dict, settings_class=None) -> "Drift":
        """Read back what `to_dict` wrote.

        The settings come back as a ``settings_class`` (default
        `DriftSettings`, COMET's; the fiducial fit passes its own) when they
        still fit one: a curve saved by a newer version may carry a field
        this one has never heard of, and that is a reason to drop the
        settings, not the curve.
        """
        settings_class = settings_class or DriftSettings
        values = saved.get("settings")
        settings = None
        if isinstance(values, dict):
            known = {f.name for f in dataclass_fields(settings_class)}
            try:
                settings = settings_class(**{k: v for k, v in values.items()
                                             if k in known})
            except TypeError:
                settings = None
        flagged = saved.get("flagged_windows")
        matrix = saved.get("matrix")
        return cls(np.asarray(saved.get("drift_nm", []), dtype=np.float64),
                   settings=settings, n_used=saved.get("n_used"),
                   flagged_windows=None if flagged is None
                   else np.asarray(flagged, dtype=np.int64),
                   method=str(saved.get("method") or "comet"),
                   reference=str(saved.get("reference") or "as estimated"),
                   matrix=None if matrix is None
                   else np.asarray(matrix, dtype=np.float64),
                   centre_nm=saved.get("centre_nm"),
                   transformation=str(saved.get("transformation") or "shift"),
                   times_s=saved.get("times_s"))

    def plot(self, ax=None):
        """Drift vs frame, the standard sanity check."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("plotting the drift needs matplotlib: "
                              "pip install smappy-smlm[drift]") from None

        if ax is None:
            _, ax = plt.subplots()
        x = self.frames if self.times_s is None else self.times_s
        for axis, label in enumerate("xyz"):
            if np.any(self.drift[:, axis]):
                ax.plot(x, self.drift[:, axis], label=label)
        ax.set_xlabel("frame" if self.times_s is None else "time (s)")
        ax.set_ylabel("drift (nm)")
        ax.legend()
        return ax

    def plot_transformation(self, ax):
        """The transformation's parameters vs frame, in ppm: nm of movement
        per mm from the centre, which is what a scale change of 1e-6 and a
        rotation of a microradian both are."""
        for name, values in self.parameters().items():
            ax.plot(self.frames, values * 1e6, label=name)
        ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
        ax.set_xlabel("frame")
        ax.set_ylabel("transformation (ppm)")
        ax.legend()
        return ax

    def figure(self):
        """What a result shows: the drift, and under it the transformation
        when there is one (a `plugins.Plot` of two panels)."""
        if self.matrix is None:
            return self.plot
        from .plugins import Plot

        def draw(figure):
            top, bottom = figure.subplots(2, 1, sharex=True)
            self.plot(top)
            top.set_xlabel("")
            self.plot_transformation(bottom)

        return Plot(draw, panels=2, size=(6.0, 6.0))


# ------------------------------------------------------------- what it will cost
# Both the time and the memory of a drift estimate are set by one number: how
# many neighbour pairs there are within `max_drift_nm`.  Every cost evaluation
# is one `exp()` per pair, and the KD-tree search has to hold all of them at
# once.  The constants below are measured on an M1 Max (see docs/drift.md,
# "Where the time goes"); they are the right order on any CPU and are only ever used
# to decide whether to warn, never to change what is computed.
_SECONDS_PER_PAIR_EVALUATION = 2.1e-9   # the compiled kernel, across the cores
_SECONDS_PER_PAIR_SEARCH = 16e-9        # cKDTree.query_pairs
_SECONDS_PER_LOCALIZATION = 3e-7        # grouping, and applying the drift
# The sigma ladder runs a full L-BFGS-B at each level; 136 evaluations over 6
# levels measured on a 93 k-frame dataset, 154 over 6 on the clathrin one.
_EVALUATIONS_PER_SIGMA_LEVEL = 23
# scipy hands the pairs back as one int64 (N, 2) array and `pair_indices_kdtree`
# copies them into two int32 arrays, all three alive at once.  scipy builds that
# array from an internal vector, which can transiently double the 16 again, so
# this is a floor and not a worst case.
_BYTES_PER_PAIR = 24
# localizations the pair count is estimated from.  The count scales as N^2, so
# a sample answers for the whole: 60 k was within 0.1% of the true pair count
# on a 1.7 M-localization dataset, in under a tenth of a second.
_PAIR_SAMPLE = 60_000


@dataclass
class DriftCost:
    """What a drift estimate with given settings is going to cost.

    `seconds` and `memory_bytes` both follow from `n_pairs`, which is estimated
    rather than counted -- see `estimate_cost`.  They are an order-of-magnitude
    guide meant for deciding whether to start at all, not a promise.
    """

    n_localizations: int            # what the estimate runs on, after grouping
    n_pairs: float                  # neighbour pairs within max_drift_nm
    n_evaluations: int              # cost evaluations the sigma ladder will do
    seconds: float
    memory_bytes: float
    max_drift_nm: float
    machine_bytes: Optional[int] = None   # this computer's RAM, if known

    @property
    def fits_in_memory(self) -> bool:
        """Is there room for the pairs?  Unknown memory counts as room.

        Half the machine is the bar: the table, its index and the rendered
        view are already resident, and the pair estimate is a floor.
        """
        if self.machine_bytes is None:
            return True
        return self.memory_bytes < 0.5 * self.machine_bytes

    def question(self, slow_seconds: float = 300.0) -> Optional[str]:
        """What to put to the user before starting, or None to just start."""
        if not self.fits_in_memory:
            return (f"{self}\n\nThis will not fit in memory and will most "
                    f"likely die after a long time in swap.\n\n"
                    f"{_CHEAPER}\n\nRun it anyway?")
        if self.seconds > slow_seconds:
            return (f"{self}\n\n{_CHEAPER}\n\nRun it?")
        return None

    def __str__(self) -> str:
        return (f"{_count(self.n_localizations)} localizations, about "
                f"{_count(self.n_pairs)} neighbour pairs within "
                f"{self.max_drift_nm:g} nm: {_duration(self.seconds)} and at "
                f"least {_bytes(self.memory_bytes)} of memory"
                + ("" if self.machine_bytes is None
                   else f" ({_bytes(self.machine_bytes)} in this machine)"))


_CHEAPER = ("The pair count grows with the square of the localization count "
            "and steeply with the search radius, so the levers are a smaller "
            "max drift (it is the neighbour radius -- use what the stage "
            "actually drifts, not a safe-looking round number), grouping, and "
            "a tighter filter on what the estimate is made from.")


def estimate_cost(locs: Localizations, settings: Optional[DriftSettings] = None,
                  select=None, pixelsize_nm: Optional[float] = None) -> DriftCost:
    """What a drift estimate on this table will cost, without running it.

    Counting the pairs outright is the expensive part of the run itself, so
    they are counted on a sample and scaled: the count grows as N^2 for a
    fixed field, which holds to a fraction of a percent on real data.  The
    whole thing takes well under a second on millions of localizations, which
    is what lets it run before every drift correction rather than on request.
    """
    settings = settings or DriftSettings()
    index = _selection(locs, select)
    n_selected = int(len(index))
    machine = _machine_memory()
    if n_selected < 2 or "frame" not in locs:
        return DriftCost(n_selected, 0.0, 0, 0.0, 0.0,
                         settings.max_drift_nm, machine)

    x, y, z = _positions_nm(locs, pixelsize_nm, settings.use_z)
    coords = np.column_stack([x[index], y[index], z[index]])
    frames = np.asarray(locs["frame"], dtype=np.int64)[index]

    sample, ratio = _cost_sample(coords, frames, settings, pixelsize_nm)
    n_estimate = max(2, int(round(n_selected * ratio)))

    pairs = _pairs_in_radius(sample, settings.max_drift_nm, n_estimate)
    if settings.two_stage:
        # a grouped pass at the full radius, then an ungrouped one at a small
        # one; both run, so the time adds and the memory is whichever is larger
        fine = _pairs_in_radius(coords, settings.two_stage_radius_nm, n_selected)
    else:
        fine = 0.0

    levels = _sigma_levels(settings)
    evaluations = levels * _EVALUATIONS_PER_SIGMA_LEVEL
    seconds = ((pairs + fine) * (_SECONDS_PER_PAIR_SEARCH
                                 + evaluations * _SECONDS_PER_PAIR_EVALUATION)
               + len(locs) * _SECONDS_PER_LOCALIZATION)
    return DriftCost(n_localizations=n_estimate, n_pairs=pairs,
                     n_evaluations=evaluations, seconds=seconds,
                     memory_bytes=max(pairs, fine) * _BYTES_PER_PAIR,
                     max_drift_nm=settings.max_drift_nm, machine_bytes=machine)


def _pairs_in_radius(sample: np.ndarray, radius: float, n_total: int) -> float:
    """Pairs within `radius` among `n_total` points, from a sample of them."""
    from scipy.spatial import cKDTree

    if len(sample) > _PAIR_SAMPLE:
        # a fixed seed, so the same table gives the same estimate every run
        rng = np.random.default_rng(0)
        sample = sample[rng.choice(len(sample), _PAIR_SAMPLE, replace=False)]
    if len(sample) < 2:
        return 0.0
    tree = cKDTree(sample)
    pairs = (tree.count_neighbors(tree, radius) - len(sample)) / 2.0
    return float(pairs) * (n_total / len(sample)) ** 2


def _cost_sample(coords: np.ndarray, frames: np.ndarray, settings: DriftSettings,
                 pixelsize_nm: Optional[float]) -> Tuple[np.ndarray, float]:
    """Positions to estimate the pair count from, and what grouping will do.

    With `group` on, the estimate runs on grouped localizations, which are both
    fewer and differently distributed -- grouping is exactly what removes the
    densest, shortest-range pairs.  So the sample is *grouped* too: a stretch
    of frames from the middle of the acquisition, which is a fair sample of the
    field because every frame images the same one.  The ratio it groups by is
    what the whole table will group by.
    """
    if not settings.group:
        return coords, 1.0

    lo, hi = int(frames.min()), int(frames.max())
    n_frames = hi - lo + 1
    span = max(1, int(n_frames * min(1.0, 4.0 * _PAIR_SAMPLE / max(len(frames), 1))))
    start = lo + max(0, (n_frames - span) // 2)
    taken = (frames >= start) & (frames < start + span)
    if taken.sum() < 2:
        return coords, 1.0

    slice_table = Localizations({"x_nm": coords[taken, 0], "y_nm": coords[taken, 1],
                                 "z_nm": coords[taken, 2], "frame": frames[taken]})
    grouped = _grouped(slice_table, settings, pixelsize_nm)
    if len(grouped) < 2:
        return coords, 1.0
    return (np.column_stack([grouped["x_nm"], grouped["y_nm"], grouped["z_nm"]]),
            len(grouped) / float(taken.sum()))


def _report_evaluation(progress: Optional[Progress], state: dict,
                       expected: int, sigma: float) -> None:
    """Say where the optimizer is, at most once a second.

    Once a second rather than every evaluation because each one is a queued
    signal into the GUI thread, and on a small dataset they arrive hundreds of
    times a second.
    """
    if progress is None:
        return
    now = time.time()
    if now - state["reported"] < 1.0:
        return
    state["reported"] = now
    done = state["evaluations"]
    each = (now - state["started"]) / done
    left = max(0.0, (expected - done) * each)
    progress(f"sigma {sigma:.0f} nm: evaluation {done} of about {expected}, "
             f"{each:.2g} s each, ~{_duration(left)} left")


def _sigma_levels(settings: DriftSettings) -> int:
    """How many times the refinement loop runs the optimizer.

    Exactly, for the spline fit: from the initial width down to the target in
    steps of 1.5.  For the time-window fit it is a lower bound -- COMET's own
    loop keeps narrowing past the target until a step changes the drift more
    than the one before, which cannot be known before the run -- and the
    estimate says "about" for that reason as well.
    """
    sigma = (settings.initial_sigma_nm if settings.initial_sigma_nm is not None
             else settings.max_drift_nm / 3.0)
    target = settings.target_sigma_nm
    levels = 1
    while sigma > target and levels < 100:
        sigma = max(sigma / 1.5, target)
        levels += 1
    return levels


def _machine_memory() -> Optional[int]:
    if os.name == "nt":
        # no sysconf on Windows, and without this the memory warning never came there
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended", ctypes.c_ulonglong)]
        status = _Status()
        status.length = ctypes.sizeof(_Status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total)
        except (AttributeError, OSError):
            pass
        return None
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


def _count(n: float) -> str:
    for limit, suffix in ((1e9, "billion"), (1e6, "million"), (1e3, "thousand")):
        if n >= limit:
            return f"{n / limit:.1f} {suffix}"
    return f"{n:.0f}"


def _bytes(n: float) -> str:
    for limit, suffix in ((1e9, "GB"), (1e6, "MB"), (1e3, "kB")):
        if n >= limit:
            return f"{n / limit:.1f} {suffix}"
    return f"{n:.0f} B"


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} hours"


# --------------------------------------------------------------------- running
def estimate_drift(locs: Localizations, settings: Optional[DriftSettings] = None,
                   select=None, pixelsize_nm: Optional[float] = None,
                   display: bool = False,
                   progress: Optional[Progress] = None, stop=None) -> Drift:
    """Estimate drift from the selected localizations, zero where the
    settings' *reference* says (`Drift.referenced`); the rest is
    `_estimate_drift`'s."""
    settings = settings or DriftSettings()
    drift = _estimate_drift(locs, settings, select, pixelsize_nm, display,
                            progress, stop)
    return drift.referenced(settings.reference)


def _estimate_drift(locs: Localizations, settings: Optional[DriftSettings] = None,
                    select=None, pixelsize_nm: Optional[float] = None,
                    display: bool = False,
                    progress: Optional[Progress] = None, stop=None) -> Drift:
    """Estimate drift from the selected localizations.

    ``select`` is what to estimate *from*: a `LocFilter`, a boolean mask, an
    index array, or None for everything.  The result covers every frame of the
    input table, not only the selected ones.

    ``stop`` (a `threading.Event`, the plugin's `Context.stop`) ends the
    spline fit at its next cost evaluation by raising `plugins.Stopped`: a
    curve refined half way is not one to correct a table with.  The legacy
    time-window fit only looks before it starts.
    """
    _check_stop(stop, "before it started")
    from ._comet import comet_run_kd   # vendored; see _comet/
    from ._comet.core import drift_optimizer
    from ._comet.core.qc_utils import flag_flawed_segments_by_lift
    from ._comet.core.cpu_wrapper import (cpu_wrapper_chunked_approx,
                                          cpu_wrapper_chunked_fast)

    settings = settings or DriftSettings()
    if settings.quality_control and settings.spline:
        raise ValueError("quality control flags time windows, and the spline "
                         "fit has none; pass spline=False to use it")
    # The passes first: each runs this function again with its own flag off,
    # so whichever fit is asked for -- spline or time windows -- does the
    # passes' work.  With the spline checked first, the default, "RCC first"
    # and "two stage" were skipped without a word, and the slow-run dialogue's
    # "RCC first, then COMET within 50 nm" ran one fit over 50 nm and no RCC.
    if settings.rcc_prepass:
        return _rcc_prepass(locs, settings, select, pixelsize_nm, display,
                            progress, stop)
    if settings.two_stage:
        return _two_stage(locs, settings, select, pixelsize_nm, display,
                          progress, stop)
    if settings.spline:
        return _estimate_spline(locs, settings, select, pixelsize_nm, display,
                                progress, stop)
    if settings.transformation != "shift":
        raise ValueError("a transformation is fitted as a spline in time; "
                         "the time-window fit has only the shift (fit spline "
                         "on, or transformation shift)")

    # Opt in to the two changes made to the vendored COMET (docs/drift.md); its
    # own defaults are untouched for anyone importing it directly.
    drift_optimizer.CPU_WRAPPER = (cpu_wrapper_chunked_approx
                                   if settings.approximate_kernel
                                   else cpu_wrapper_chunked_fast)
    drift_optimizer.LBFGSB_OPTIONS = dict(drift_optimizer.LBFGSB_OPTIONS,
                                          ftol=float(settings.optimizer_ftol))
    drift_optimizer.LBFGSB_OPTIONS_COARSE = None if settings.optimizer_ftol_coarse is None else dict(
        drift_optimizer.LBFGSB_OPTIONS, ftol=float(settings.optimizer_ftol_coarse))

    # the drift must cover every frame of the table it will be applied to,
    # whatever subset it is estimated from
    all_frames = np.asarray(locs["frame"], dtype=np.int64)
    n_frames = int(all_frames.max()) + 1 if len(all_frames) else 0

    backend = "cpu_qc" if settings.quality_control else "cpu"
    drift_optimizer.LAST_QUALITY_CONTROL = None
    drift_optimizer.FLAG_SEGMENTS = lambda q_obs, q_null: flag_flawed_segments_by_lift(
        q_obs, q_null, settings.min_lift)

    source = locs[_selection(locs, select)]
    if settings.group:
        source = _grouped(source, settings, pixelsize_nm, display, progress)
    x, y, z = _positions_nm(source, pixelsize_nm, settings.use_z)
    dataset = np.column_stack([x, y, z,
                               np.asarray(source["frame"], dtype=np.float64)])
    if len(dataset) < 2:
        raise ValueError("need at least two localizations to estimate drift")
    # COMET reports through its own `display`; from here it is one long call,
    # so say what was handed to it before it starts
    _log(display, f"estimating drift from {len(dataset):,} localizations in "
                  f"time windows (neighbours within {settings.max_drift_nm:g} nm)",
         progress)

    # COMET's `display` is both its progress log and a blocking `plt.show()`;
    # we want the log, and to draw the drift ourselves when asked (Drift.plot)
    with _without_blocking_plots(display):
        drift = comet_run_kd(
            dataset,
            segmentation_mode=settings.segmentation_mode,
            segmentation_var=settings.segmentation_var,
            max_drift_nm=settings.max_drift_nm,
            target_sigma_nm=settings.target_sigma_nm,
            initial_sigma_nm=settings.initial_sigma_nm,
            boxcar_width=settings.boxcar_width,
            interpolation_method=settings.interpolation,
            max_locs_per_segment=settings.max_locs_per_segment,
            mode=backend,
            display=display,
            # an upfront pair-count estimate: the neighbour search runs over the
            # whole dataset at once, and is where too large a max_drift_nm turns
            # into minutes and gigabytes.  Better warned before than after.
            pair_indices_safety_check=True,
            # the estimate is made from a subset, but must cover every frame of
            # the table it will be applied to
            min_max_frames=(0, n_frames - 1),
        )
    quality = drift_optimizer.LAST_QUALITY_CONTROL
    return Drift(drift[:, :3], settings, n_used=len(dataset),
                 flagged_windows=None if quality is None else quality["flagged"])


def _spline_basis(x, n_coefficients: int, lo: float, hi: float, degree: int = 3):
    """Cubic B-spline design matrix: row per point, column per coefficient."""
    from scipy.interpolate import BSpline

    interior = np.linspace(lo, hi, n_coefficients - degree + 1)
    knots = np.concatenate([np.full(degree, lo), interior, np.full(degree, hi)])
    return BSpline.design_matrix(np.clip(np.asarray(x, dtype=float), lo, hi),
                                 knots, degree)


def _estimate_spline(locs: Localizations, settings: DriftSettings, select,
                     pixelsize_nm: Optional[float], display: bool,
                     progress: Optional[Progress] = None, stop=None) -> "Drift":
    """Fit drift as a spline in time, with no segmentation at all.

    The optimizer's variables are the spline's coefficients; the cost and its
    gradient are COMET's, over the same neighbour pairs.  Each localization
    carries its own frame, so `mu` handed to the kernel is the drift *per
    frame*, and the chain rule to the coefficients is one matrix product.
    """
    from ._comet.core.cpu_wrapper import cpu_wrapper_chunked_approx
    from ._comet.core.pair_indices import pair_indices_kdtree
    from scipy.optimize import minimize

    all_frames = np.asarray(locs["frame"], dtype=np.int64)
    n_frames = int(all_frames.max()) + 1 if len(all_frames) else 0

    source = locs[_selection(locs, select)]
    if settings.group:
        source = _grouped(source, settings, pixelsize_nm, display, progress)
    x, y, z = _positions_nm(source, pixelsize_nm, settings.use_z)
    coords = np.ascontiguousarray(np.column_stack([x, y, z]), dtype=np.float32)
    frames = np.ascontiguousarray(source["frame"], dtype=np.int32)

    # One uninterruptible scipy call that has to hold every pair at once, so
    # this is where too large a radius turns into swap and a dead process.
    # `estimate_cost` is what stands in front of it; say what is being asked
    # for, so a run that dies here says so in the log.
    _log(display, f"searching for neighbour pairs within "
                  f"{settings.max_drift_nm:g} nm among {len(coords):,} "
                  f"localizations", progress)
    idx_i, idx_j, ok = pair_indices_kdtree(coords, settings.max_drift_nm)
    if not ok:
        raise MemoryError("the neighbour search ran out of memory; reduce "
                          "max_drift_nm or use group=True")

    n_coefficients = _n_coefficients(n_frames, settings.spline_knot_frames)
    basis = _spline_basis(np.arange(n_frames), n_coefficients, 0.0, n_frames - 1.0)
    basis_t = basis.T.tocsr()
    _log(display, f"spline drift: {n_coefficients} coefficients over {n_frames} "
                  f"frames, {len(idx_i):,} pairs", progress)
    transformed = settings.transformation != "shift"
    if transformed:
        model = _TransformationFit(settings, coords, n_frames, display, progress)
        coords = model.coords

    penalty = float(settings.spline_penalty)
    if penalty:
        # second differences of the coefficients: what "not oscillating" means
        d2 = np.diff(np.eye(n_coefficients), 2, axis=0)
        curvature = d2.T @ d2

    # Every evaluation costs the same -- one pass over the pairs -- so two of
    # them are enough to say how long the rest will take.  The total is only
    # approximate (L-BFGS-B decides when it is done), which the wording says.
    state = {"evaluations": 0, "started": time.time(), "reported": 0.0}
    expected = _sigma_levels(settings) * _EVALUATIONS_PER_SIGMA_LEVEL

    def objective(flat, sigma, factor):
        # raising here is how the optimizer is left: it has no stop of its own
        _check_stop(stop, f"at sigma {sigma:.0f} nm")
        c = flat[:n_coefficients * 3].reshape(n_coefficients, 3)
        mu = np.ascontiguousarray(basis @ c)
        if transformed:
            value, gradient, extra = model.cost(
                flat[n_coefficients * 3:], mu, frames, idx_i, idx_j, sigma,
                factor, penalty)
        else:
            value, gradient = cpu_wrapper_chunked_approx(
                mu, coords, frames, idx_i, idx_j, sigma, factor)
        gradient = basis_t @ gradient.reshape(n_frames, 3)
        if penalty:
            value += penalty * float((c * (curvature @ c)).sum())
            gradient = gradient + 2.0 * penalty * (curvature @ c)
        state["evaluations"] += 1
        _report_evaluation(progress, state, expected, sigma)
        if transformed:
            return value, np.concatenate([gradient.ravel(), extra])
        return value, gradient.ravel()

    sigma = (settings.initial_sigma_nm if settings.initial_sigma_nm is not None
             else settings.max_drift_nm / 3.0)
    target = settings.target_sigma_nm
    bound = settings.max_drift_nm * 2.0
    coefficients = np.zeros(n_coefficients * 3
                            + (model.n_variables if transformed else 0))
    while True:
        result = minimize(objective, coefficients, args=(sigma, 1.0), jac=True,
                          method="L-BFGS-B", bounds=[(-bound, bound)] * coefficients.size,
                          options={"ftol": settings.optimizer_ftol, "gtol": 1e-5,
                                   "maxls": 40})
        coefficients = result.x
        _log(display, f"  sigma {sigma:6.1f} nm: {result.nfev} evaluations, "
                      f"cost {result.fun:.6g}", progress)
        state["reported"] = 0.0          # the next evaluation line starts fresh
        if sigma <= target:
            break
        sigma = max(sigma / 1.5, target)

    drift = np.asarray(basis @ coefficients[:n_coefficients * 3]
                       .reshape(n_coefficients, 3))
    if not transformed:
        return Drift(drift, settings, n_used=len(coords))
    return Drift(drift, settings, n_used=len(coords),
                 matrix=model.matrices(coefficients[n_coefficients * 3:]),
                 centre_nm=model.centre, transformation=settings.transformation)


def _n_coefficients(n_frames: int, knot_frames: int) -> int:
    return max(4, int(round(n_frames / knot_frames)) + 3)


class _TransformationFit:
    """The transformation half of the spline fit (`DriftSettings.transformation`).

    A(t) = I + sum_p theta_p(t) E_p over the model's generators, each theta_p
    a cubic B-spline with its own knots.  Three choices keep the fit well
    posed:

    * **Zero mean.**  The overlap only sees how the transformation *changes*
      -- the cost divides out a transformation common to both frames of a
      pair (`csrc/drift.hpp`, `transformed_cost_and_gradient`) -- so a
      constant theta is free, as a constant drift is.  The coefficients are
      kept to zero mean by optimising in an orthonormal basis of the
      zero-sum vectors, rather than left to wander along that direction.
    * **Units of nm, per axis.**  theta is ~1e-4 where the drift
      coefficients are tens of nm; L-BFGS-B steps both with one scale.  So
      each variable is theta times the half-width of the data along the axis
      its generator multiplies -- how far it moves the outermost
      localizations.  Along z that is a few hundred nm, not the field's ten
      um: an entry acting on z was otherwise free to swing by thousands of
      ppm for a nanometre of movement, which the overlap cannot see.
    * **A pull towards no change** (*transformation penalty*), in those
      units: lambda S / (n sigma^2) times the sum of the squared variables,
      S the overlap at the start of the width step and n the number of
      variables.  Moving every variable by sigma costs lambda times the
      overlap, so a parameter the data pin down hardly feels it, and one
      they say little about stays near none.
    """

    def __init__(self, settings: DriftSettings, coords: np.ndarray,
                 n_frames: int, display: bool, progress):
        from scipy.linalg import null_space

        scales, generators = TRANSFORMATIONS[settings.transformation]
        positions = np.asarray(coords, dtype=np.float64)
        if scales != "xy" and not np.any(positions[:, 2]):
            raise ValueError(f"{settings.transformation} needs z: the table "
                             "has no z_nm, or correct z is off")
        lo, hi = positions.min(axis=0), positions.max(axis=0)
        # the midpoint of the data, as SMAP: the shift is free, so the centre
        # only decides how the correction splits into shift and transformation
        self.centre = (lo + hi) / 2.0
        self.coords = np.ascontiguousarray(positions - self.centre, dtype=np.float32)
        self.scales = scales
        self.generators = np.stack([e for _, e in generators])
        half = np.maximum((hi - lo) / 2.0, 1.0)
        # the half-width along the axes each generator multiplies (its columns)
        self.radius = np.array([half[np.any(e != 0, axis=0)].max()
                                for e in self.generators])
        knots = settings.transformation_knot_frames or settings.spline_knot_frames
        self.n_coefficients = _n_coefficients(n_frames, knots)
        self.basis = _spline_basis(np.arange(n_frames), self.n_coefficients,
                                   0.0, n_frames - 1.0)
        self.basis_t = self.basis.T.tocsr()
        self.zero_mean = null_space(np.ones((1, self.n_coefficients)))
        self.n_variables = len(self.generators) * (self.n_coefficients - 1)
        self.curvature = None
        if settings.spline_penalty:
            d2 = np.diff(np.eye(self.n_coefficients), 2, axis=0)
            self.curvature = d2.T @ d2
        self.prior = float(settings.transformation_penalty)
        # the pull is on the relative change, theta, in units of the widest
        # axis's: (radius_max / radius_p)^2 on the nm variables, so an entry
        # acting on a thin z is held a thousand times harder than one acting
        # across the field -- a change of a few hundred nm of depth is what
        # the data say least about
        self.prior_weights = (self.radius.max() / self.radius) ** 2
        self._overlap = {}    # sigma -> the overlap at the step's first evaluation
        _log(display, f"{settings.transformation}: {len(self.generators)} "
                      f"parameter(s), {self.n_coefficients} coefficients each, "
                      f"about ({self.centre[0]:.0f}, {self.centre[1]:.0f}) nm",
             progress)

    def _variables(self, flat):
        """(n_gen, n_coefficients - 1) in the zero-sum basis, nm."""
        return flat.reshape(len(self.generators), self.n_coefficients - 1)

    def _theta_coefficients(self, flat):
        return (self.zero_mean @ self._variables(flat).T) / self.radius

    def matrices(self, flat) -> np.ndarray:
        theta = self.basis @ self._theta_coefficients(flat)
        return np.eye(3) + np.einsum("fp,pij->fij", theta, self.generators)

    def _scale_factors(self, matrix):
        """g_xy, g_z per frame (the inverse scales the kernel divides out),
        and a function of their gradients that gives the gradient with
        respect to the matrix; None where a frame's transformation is no
        physical one."""
        xy = matrix[:, :2, :2]
        if self.scales == "3d":
            det = np.linalg.det(matrix)
            if np.any(det <= 0):
                return None
            g = det ** (-1.0 / 3.0)

            def chain(d_gxy, d_gz):
                # g = det(A)^(-1/3): dg/dA = -g/3 A^-T
                d = np.zeros_like(matrix)
                d += (((d_gxy + d_gz) * g / -3.0)[:, None, None]
                      * np.linalg.inv(matrix).transpose(0, 2, 1))
                return d
            return g, g, chain
        det = np.linalg.det(xy)
        zz = matrix[:, 2, 2]
        if np.any(det <= 0) or np.any(zz <= 0):
            return None
        g_xy = det ** -0.5
        g_z = 1.0 / zz if self.scales == "xy+z" else np.ones_like(zz)

        def chain(d_gxy, d_gz):
            d = np.zeros_like(matrix)
            d[:, :2, :2] = ((d_gxy * g_xy / -2.0)[:, None, None]
                            * np.linalg.inv(xy).transpose(0, 2, 1))
            if self.scales == "xy+z":
                d[:, 2, 2] = -d_gz / zz ** 2
            return d
        return g_xy, g_z, chain

    def cost(self, flat, mu, frames, idx_i, idx_j, sigma, factor, penalty):
        """(cost, d/d mu per frame, d/d the transformation's variables), signs
        as the optimizer minimises them."""
        from . import _drift
        from ._comet.core.cpu_wrapper import CUTOFF_SIGMAS

        matrix = self.matrices(flat)
        scales = self._scale_factors(matrix)
        if scales is None:     # beyond any physical transformation
            return np.inf, np.zeros(mu.size), np.zeros(flat.size)
        g_xy, g_z, chain = scales
        params = np.ascontiguousarray(np.column_stack(
            [matrix.reshape(-1, 9), mu, g_xy, g_z]))
        total, gradient = _drift.transformed_cost_and_gradient(
            self.coords, frames, idx_i, idx_j, params, float(sigma),
            float(factor), float(CUTOFF_SIGMAS))
        gradient = -gradient
        d_matrix = (gradient[:, :9].reshape(-1, 3, 3)
                    + chain(gradient[:, 12], gradient[:, 13]))
        d_theta = np.einsum("fij,pij->fp", d_matrix, self.generators)
        # to the variables: theta coefficients = Z q / radius
        d_flat = (self.zero_mean.T @ (self.basis_t @ d_theta)).T / self.radius[:, None]
        value = -float(total)
        q = self._variables(flat)
        if penalty and self.curvature is not None:
            c = self.zero_mean @ q.T                         # (n_coef, n_gen), nm
            value += penalty * float((c * (self.curvature @ c)).sum())
            d_flat = d_flat + 2.0 * penalty * (self.zero_mean.T
                                               @ (self.curvature @ c)).T
        if self.prior:
            overlap = self._overlap.setdefault(float(sigma), abs(value))
            weight = (self.prior * overlap / (self.n_variables * sigma ** 2)
                      * self.prior_weights[:, None])
            value += float((weight * q * q).sum())
            d_flat = d_flat + 2.0 * weight * q
        return value, gradient[:, 9:12], d_flat.ravel()


# What a caller is told while this runs.  The GUI passes the plugin panel's
# `ctx.report`, the command line passes `print` through `display`.
Progress = Callable[[str], None]


def _check_stop(stop, where: str) -> None:
    """Raise `plugins.Stopped` if ``stop`` is set."""
    if stop is not None and stop.is_set():
        from .plugins import Stopped
        raise Stopped(f"COMET stopped {where}: nothing was corrected")


def _log(enabled: bool, message: str, progress: Optional[Progress] = None) -> None:
    if progress is not None:
        progress(message)
    if enabled:
        print(message)


def _two_stage(locs: Localizations, settings: DriftSettings, select,
               pixelsize_nm: Optional[float], display: bool,
               progress: Optional[Progress] = None, stop=None) -> "Drift":
    """A grouped pass to get close, then an ungrouped one over a small radius."""
    _log(display, "pass 1 of 2: grouped, over the full search radius", progress)
    coarse = estimate_drift(locs, replace(settings, two_stage=False, group=True),
                            select, pixelsize_nm, display, progress, stop)
    radius = settings.two_stage_radius_nm
    _log(display, f"pass 2 of 2: ungrouped, within {radius:g} nm", progress)
    fine = estimate_drift(
        coarse.apply(locs, pixelsize_nm),
        replace(settings, two_stage=False, group=False, max_drift_nm=radius,
                initial_sigma_nm=radius / 3.0, target_sigma_nm=radius / 5.0),
        select, pixelsize_nm, display, progress, stop)
    # the two are drift of the same sample measured in sequence, so they add
    # -- or, with a transformation, compose
    return coarse.then(fine, settings=settings, n_used=fine.n_used,
                       flagged_windows=fine.flagged_windows)


def _rcc_prepass(locs: Localizations, settings: DriftSettings, select,
                 pixelsize_nm: Optional[float], display: bool,
                 progress: Optional[Progress] = None, stop=None) -> "Drift":
    """RCC for the bulk of the drift, then COMET for what is left of it.

    The two are drift of the same sample measured one after the other, so they
    add -- as in `_two_stage`, whose shape this follows.
    """
    from .rcc import RCCSettings, estimate_drift_rcc

    rcc = RCCSettings(n_timepoints=settings.rcc_prepass_windows,
                      use_z=settings.use_z, group=settings.group,
                      group_dx_nm=settings.group_dx_nm, group_dt=settings.group_dt)
    if settings.rcc_prepass_max_drift_nm:
        rcc = replace(rcc, max_drift_nm=float(settings.rcc_prepass_max_drift_nm))
    _log(display, f"pass 1 of 2: RCC over {rcc.n_timepoints} time windows, "
                  f"within {rcc.max_drift_nm:g} nm", progress)
    coarse = estimate_drift_rcc(locs, rcc, select, pixelsize_nm, display,
                                stop=stop)
    _log(display, f"pass 2 of 2: COMET within {settings.max_drift_nm:g} nm of "
                  "what RCC left", progress)
    fine = estimate_drift(coarse.apply(locs, pixelsize_nm),
                          replace(settings, rcc_prepass=False),
                          select, pixelsize_nm, display, progress, stop)
    return coarse.then(fine, settings=settings, n_used=fine.n_used,
                       flagged_windows=fine.flagged_windows, method="rcc+comet")


def correct_drift(locs: Localizations, settings: Optional[DriftSettings] = None,
                  select=None, pixelsize_nm: Optional[float] = None,
                  display: bool = False, progress: Optional[Progress] = None,
                  stop=None):
    """Estimate from the selection, correct everything.  Returns (locs, drift)."""
    drift = estimate_drift(locs, settings, select, pixelsize_nm, display,
                           progress, stop)
    _log(display, f"applying the drift to {len(locs):,} localizations", progress)
    return drift.apply(locs, pixelsize_nm), drift


# ----------------------------------------------------------------------- saving
def drift_corrected_path(path) -> Path:
    """``foo.hdf5`` -> ``foo_driftc.hdf5``."""
    path = Path(path)
    return path.with_name(path.stem + "_driftc" + (path.suffix or ".hdf5"))


def save_drift_corrected(path, locs: Localizations, drift: Drift) -> Path:
    """Write the corrected table, with the drift curve alongside it.

    The file is an ordinary smappy localization file -- the viewer and every
    other reader take it as it is -- plus a ``/drift`` group, so the correction
    that was applied can be inspected or undone later.
    """
    import h5py

    from .io.hdf5 import save_localizations

    from . import _comet

    path = save_localizations(path, locs)
    with h5py.File(path, "a") as f:
        group = f.create_group("drift")
        for axis, name in enumerate(("x_nm", "y_nm", "z_nm")):
            group.create_dataset(name, data=drift.drift[:, axis].astype(np.float32))
        if drift.times_s is None:
            group.create_dataset("frame", data=drift.frames.astype(np.int64))
        else:
            group.create_dataset("time_s", data=drift.times_s)
        group.attrs["reference"] = drift.reference
        if drift.matrix is not None:
            # the correction's matrix per frame and the point it acts about
            group.create_dataset("matrix", data=drift.matrix)
            group.attrs["centre_nm"] = drift.centre_nm
            group.attrs["transformation"] = drift.transformation
        # the estimator is somebody else's published method: say so in the file,
        # so a result can be traced -- and cited -- back to it
        if "comet" in drift.method:
            group.attrs["method"] = _comet.UPSTREAM
            group.attrs["method_version"] = _comet.__version__
        if "rcc" in drift.method:
            group.attrs["prepass" if "comet" in drift.method else "method"] = \
                "RCC (Wang et al. 2014)"
        if "comet" not in drift.method and "rcc" not in drift.method:
            # the beads' fits (fiducials, MBM) are no published estimator;
            # the name alone says what made the curve
            group.attrs["smappy_method"] = drift.method
    return path


def load_drift(path) -> Drift:
    """Read the drift curve stored next to a corrected table."""
    import h5py

    with h5py.File(path, "r") as f:
        group = f["drift"]
        drift = np.column_stack([group[name][()]
                                 for name in ("x_nm", "y_nm", "z_nm")])
        method = str(group.attrs.get("method", ""))
        prepass = "prepass" in group.attrs
        reference = str(group.attrs.get("reference", "as estimated"))
        matrix = group["matrix"][()] if "matrix" in group else None
        centre = group.attrs.get("centre_nm")
        transformation = str(group.attrs.get("transformation", "shift"))
        times = group["time_s"][()] if "time_s" in group else None
        kept = str(group.attrs.get("smappy_method", ""))
    name = kept or ("rcc+comet" if prepass else "rcc" if method.startswith("RCC")
                    else "comet")
    return Drift(drift, method=name, reference=reference, matrix=matrix,
                 centre_nm=centre, transformation=transformation, times_s=times)


# ---------------------------------------------------------------------- helpers
MERGE_DX_NM = 50.0      # the merge settings without a session: the session's
MERGE_DT = 1            # own defaults (`session.DEFAULT_GROUP_SETTINGS`)


def merge_values(session) -> Tuple[float, int]:
    """The distance and dark frames ``session`` merges blinks with, or the
    defaults without one.

    The session may merge by precision instead of a fixed distance; the
    drift estimate links by a fixed distance only (it needs no precision
    column, and works on a table in pixels), so it takes the largest distance
    that mode allows, `dx_max`.
    """
    merge = getattr(session, "group_settings", None)
    if merge is None:
        return MERGE_DX_NM, MERGE_DT
    dx = merge.dx if merge.link == "fixed" else merge.dx_max
    return float(dx), int(merge.dt)


def merge_from_session(settings, session):
    """``settings`` (`DriftSettings` or `RCCSettings`) with the merge distance
    and dark frames left on auto taken from ``session``: what a plugin runs
    with, and records, so that the log says the numbers that were used."""
    dx, dt = merge_values(session)
    changes = {}
    if settings.group_dx_nm is None:
        changes["group_dx_nm"] = dx
    if settings.group_dt is None:
        changes["group_dt"] = dt
    return replace(settings, **changes) if changes else settings


def _grouped(locs: Localizations, settings: "DriftSettings",
             pixelsize_nm: Optional[float], display: bool = False,
             progress: Optional[Progress] = None) -> Localizations:
    """Collapse each emitter's blink into one localization before estimating."""
    from .group import GroupSettings, group

    dx = MERGE_DX_NM if settings.group_dx_nm is None else settings.group_dx_nm
    dt = MERGE_DT if settings.group_dt is None else settings.group_dt
    if "x_nm" not in locs:  # a table in pixels; the linking radius is in nm
        dx /= _pixelsize_nm(locs, pixelsize_nm)
    _log(display, f"grouping {len(locs):,} localizations", progress)
    grouped, _ = group(locs, GroupSettings(dx=dx, dt=dt))
    _log(display, f"grouped {len(locs):,} localizations into {len(grouped):,} "
                  f"blinks", progress)
    return grouped


@contextmanager
def _without_blocking_plots(active: bool):
    """Let COMET log its progress without stopping in its own `plt.show()`.

    Its diagnostic figures are closed again; a drift curve is one line here
    (`Drift.plot`) and belongs where the caller decides to draw it.
    """
    if not active:
        yield
        return
    try:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("plotting the drift needs matplotlib: "
                              "pip install smappy-smlm[drift]") from None
    except ImportError:
        yield
        return
    existing = set(plt.get_fignums())
    show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        yield
    finally:
        plt.show = show
        for number in set(plt.get_fignums()) - existing:
            plt.close(number)


def _selection(locs: Localizations, select) -> np.ndarray:
    """A LocFilter, a boolean mask or an index array -> indices."""
    n = len(locs)
    if select is None:
        return np.arange(n)
    indices = getattr(select, "indices", None)  # a LocFilter
    if indices is not None:
        return np.asarray(indices)
    select = np.asarray(select)
    if select.dtype == bool:
        if select.shape != (n,):
            raise ValueError(f"mask has {select.shape[0]} entries, table has {n}")
        return np.flatnonzero(select)
    return select.astype(np.int64)


def _pixelsize_nm(locs: Localizations, pixelsize_nm: Optional[float]) -> float:
    """The pixel size, given or from the table's own provenance."""
    if pixelsize_nm is not None:
        return float(pixelsize_nm)
    value = locs.metadata.get("pixelsize_nm")
    if value is None:
        camera = locs.metadata.get("camera") or {}
        micron = camera.get("pixelsize_um") if isinstance(camera, dict) else None
        value = None if micron is None else float(micron) * 1000.0
    if value is None:
        raise ValueError("this table is in pixels and its pixel size is not "
                         "recorded; pass pixelsize_nm=...")
    return float(value)


def _positions_nm(locs: Localizations, pixelsize_nm: Optional[float],
                  use_z: Optional[bool]):
    """x, y, z of the whole table in nm; z is zero for a 2D table."""
    if "x_nm" not in locs or "y_nm" not in locs:
        raise ValueError("no position columns x_nm, y_nm; the table has "
                         + ", ".join(sorted(locs.keys())))
    x = np.asarray(locs["x_nm"], dtype=np.float64)
    y = np.asarray(locs["y_nm"], dtype=np.float64)

    has_z = "z_nm" in locs
    if use_z is None:
        use_z = has_z
    elif use_z and not has_z:
        raise ValueError("use_z=True but the table has no z_nm column")
    z = np.asarray(locs["z_nm"], dtype=np.float64) if use_z else np.zeros_like(x)
    return x, y, z
