"""How precisely this experiment placed a molecule -- measured on the data.

Three numbers, and they are meant to disagree.

**The CRLB histogram** is what the fitter believes: from the photons, the
background and the PSF width, the Cramer-Rao bound says how well *that* spot
could have been placed.  It is a lower bound of an idealized model, blind to
everything that happens after the fit -- drift, a PSF that is not the model,
an emitter that moved.

**The pairwise displacement** is what the experiment did.  A fluorophore is
normally on for longer than one frame, so a molecule seen in frame *f* is seen
again in frame *f+1* a few nanometres away, and that displacement contains
twice the variance of one localization and nothing else -- no knowledge of the
PSF, no assumption about photons.  This is NeNA (Endesfelder et al., *Histochem
Cell Biol* 2014).

**FRC** measures the picture rather than the localizations: it splits the
acquisition in two, renders both halves and asks up to which spatial frequency
they still agree.  That folds in the labeling density, the drift and the noise
of a thin dataset as well as the precision, so it answers "what can be seen"
rather than "how well was this molecule placed" -- and it is drawn against the
blur the measured precision implies, which says which of the two is in the
way.  `smappy.frc` has the recipe and how to read it.

## The number NeNA reports is not the precision of a typical localization

The pairs NeNA can see are, by construction, the frames in which a molecule
switched on or off.  A blink that lasts about one frame is split across two,
and the two halves share one molecule's photons: if the switch happens a
fraction *u* into the exposure, the pair carries ``S^2 (1/u + 1/(1-u))`` of
variance, which is smallest when the split is even and unbounded when it is
not.  Averaged over the split, the plain NeNA sigma is therefore *worse* than
the precision of an average localization -- systematically, and by an amount
that depends on how the on-time compares with the frame time and on where the
detection threshold cut the dim half off.  Much of the folklore that "NeNA is
1.3 to 2 times the CRLB" is this effect rather than a defect of the fit.

So the plugin fits the displacements **three** times, and only the first of
the three is the number NeNA defines:

* once for a single sigma -- the NeNA number, comparable with what SMAP,
  Picasso and the paper report, and beholden to nothing but the coordinates;

* once against ``1/N_i + 1/N_j``, the pair's own **photon counts**::

      Var(displacement of pair k) = A * (1/N_i + 1/N_j)

  A precision goes as ``sqrt(A/N)``, so one free ``A`` describes every pair at
  once and the answer can then be carried to a localization of any brightness
  -- including the bright middle of a blink that NeNA never gets to see, and
  the summed photons of a grouped localization.  This is the correction for
  the effect above, and it keeps NeNA's virtue: it reads only coordinates and
  the photon column, and it is *invariant* to the calibration of that column,
  because a gain that is wrong by a factor moves ``A`` and ``N`` by opposite
  factors and leaves ``sqrt(A/N)`` where it was.  (It assumes shot noise
  dominates.  Where the background is heavy the law wants a second term in
  ``1/N^2``, and the residual at the dim end is where that shows.)

* once against each pair's **Cramer-Rao bound**, when the table carries one::

      Var(displacement of pair k) = kappa^2 * (sigma_i^2 + sigma_j^2)

  ``kappa`` is *not* a better precision -- it is a test, and it is the one
  number here that does depend on the camera calibration, because the bound
  was computed from photons that a wrong gain or offset would have got wrong.
  That is what makes it worth reporting: ``kappa = 1`` says the fitter reaches
  its own bound and the calibration is consistent with the data; ``kappa``
  away from 1 says one of the two is wrong, and does not say which.  The plain
  sigma and the photon law are unaffected either way.

The precision of a *grouped* localization follows from the photon law and not
from a measurement, because NeNA cannot measure grouped data at all: linking
has already merged everything that repeated in adjacent frames, so the pairs
that survive are re-blinks separated by a dark time, few in number and
broadened by whatever drifted in between.  Read it as ``sqrt(A / sum N)`` over
the group -- the same law, applied to the photons the group actually collected.

The plugin prints the photons of the localizations that took part in a pair
beside the photons of all of them.  When the first is much smaller, the
on-time is close to the frame time, the pairs are the split blinks, and the
distance between the plain sigma and the law's value at the median photon
count is exactly the size of that effect.

## Which pairs

Every pair within ``d_max`` whose frames differ by the gap -- not only the
nearest neighbour, despite what the name NeNA says and despite what the
original does.  All-pairs is both cleaner and what Picasso actually computes
(``_fill_dnfl`` histograms every next-frame neighbour inside the search
radius): with all pairs the contamination has a *known* shape, because pairs
of different molecules are uniform in the disc and so their distance density
is proportional to ``r``.  Taking only the nearest neighbour makes the
contamination density-dependent -- a molecule that happens to have a close
neighbour hides the true partner -- and that bias then has to be absorbed by
whatever empirical term the fit offers it.

The search volume is a **cylinder**: a disc of radius ``d_max`` laterally and,
for 3D data, ``|dz| <= dz_max``.  That keeps every background term analytic --
``2r/d_max^2`` for the lateral distance, the chord ``2*sqrt(d_max^2 - dx^2)``
normalized for one lateral axis, flat for ``dz``.

## The model

For the lateral distance, Churchman's pairwise displacement law with a true
distance of zero (two localizations of one molecule *are* at the same place),
mixed with the uniform background, and truncated at the search radius::

    p(d) = f * (d / v) exp(-d^2 / 2v) / Z  +  (1 - f) * 2d / d_max^2

with ``v = 2 s^2`` the variance of the displacement -- two localizations
contribute one ``s^2`` each -- or ``v = kappa^2 (s_i^2 + s_j^2)`` in the
CRLB-scaled fit, where it is a number per pair.  Per axis the same mixture
with a Gaussian of width ``sqrt(v)`` and the chord (or, for z, flat)
background, which is what makes the whole thing work in 3D: no closed form for
an anisotropic radial distribution is needed, ``sx``, ``sy`` and ``sz`` each
come out of their own one-dimensional fit.

Every fit is maximum likelihood on the displacements, not least squares on the
bins, so the bin count is a drawing choice and changes no number.

## Frame gap

The same fit at gaps 1, 2, ... n answers a different question.  If nothing
moves, the displacement distribution does not care about the gap and ``s`` is
flat.  If the sample drifts or vibrates, a random walk adds variance in
proportion to the gap::

    s(g)^2 = s0^2 + m * g

so the *intercept* of a line through ``s(g)^2`` is the precision with the
motion taken out, and the slope says how much motion there is between frames.
A rising curve after drift correction means the correction did not reach the
fast part -- which nothing else in the program will tell you.

## What filtering does to all this

A filter on precision (and so, through the fitter's detection threshold,
effectively on photons) does not bias these estimates -- it *conditions* them.
The displacement fit on a filtered table answers "how well is what I am
looking at placed", which is the right question for the image in front of you
and the wrong one for "how good is this sample".  Hence ``source``, which runs
both and prints them side by side: the difference between them is what
filtering bought.

Three things to know about the filtered case:

* Both partners of a pair must survive the filter, and the switch-on and
  switch-off frames of a blink are the dim ones.  A hard photon or precision
  cut therefore removes true partners preferentially, and a *very* hard one
  can remove them all -- leaving a fit with nothing but background.  The
  fitted signal fraction ``f`` is reported for exactly this reason, and a fit
  that keeps almost no signal says so.  ``kappa`` survives this better than
  the plain sigma does: it asks each surviving pair only to match its own
  bound.
* "Unfiltered" is not unconditioned.  The fitter's detection threshold is a
  filter nothing can undo; its footprint is the maximum of the photon
  histogram, below which the dim localizations were eaten
  (``Analysis/Measure/Localization Statistics`` fits from there for the same
  reason).
* The CRLB histogram *is* biased by a cut -- it is cut off -- but the estimator
  knows: its likelihood is truncated, so given the bounds it recovers the
  underlying ``sigma_c`` from the surviving part.  The bounds are read from the
  layer's filter when there is a session, and from the data otherwise.

A ROI or a z slab costs only the pairs that straddle its boundary, which for a
search radius of tens of nanometres in a field of micrometres is nothing --
except for a slab thinner than a few times the axial precision, which truncates
``dz`` itself, and which the run warns about.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..frc import (FPC, FRC, THRESHOLD as THRESHOLD_LINE, blur_envelope,
                   envelope_resolution, fpc_resolution, frc_resolution)
from ..locs import Localizations
from . import Context, Plot, Plugin, Result, param, register
from .statistics import (MIN_FOR_FIT, PRECISION_FIELDS, PRECISION_Z_FIELDS,
                         precision_density, precision_model)

# a displacement distribution is worth fitting from about here
MIN_PAIRS = 200
# the search radius, when it is not given: this many times the median
# precision.  Wide enough that the background is visible on both sides of the
# peak (which sits at sqrt(2) * sigma), narrow enough that finding the pairs
# stays cheap.
REACH = 6.0
REACH_Z = 6.0
# a slab (or z range) narrower than this many axial precisions starts to
# truncate the axial displacement rather than the sample
THIN_SLAB = 4.0
# starting fractions tried when a fit is set going
START_FRACTIONS = (0.5, 0.15, 0.85)


# ------------------------------------------------------------------ the pairs

@dataclass
class Displacements:
    """The displacement vectors of every pair found at one frame gap.

    ``variance`` and ``variance_z`` are what the Cramer-Rao bound expects of
    each pair -- ``sigma_i^2 + sigma_j^2`` from the two members' own precision
    columns.  ``inverse_photons`` is ``1/N_i + 1/N_j``, the same shape without
    the calibration: a precision goes as ``A / N``, so a pair's variance goes
    as ``A (1/N_i + 1/N_j)`` whatever the gain of the camera is said to be.
    Either lets the fit ask "how does this pair compare with what it should
    have been", one pair at a time, rather than only "how wide is this
    distribution".
    """
    gap: int
    dx: np.ndarray
    dy: np.ndarray
    dz: Optional[np.ndarray] = None
    variance: Optional[np.ndarray] = None
    variance_z: Optional[np.ndarray] = None
    photons: Optional[np.ndarray] = None      # the pair's mean photon count
    inverse_photons: Optional[np.ndarray] = None   # 1/N_i + 1/N_j
    d_max: float = 0.0
    dz_max: Optional[float] = None
    n_frames: int = 0          # frame pairs that contributed
    n_found: int = 0           # pairs found, before any subsampling

    @property
    def d(self) -> np.ndarray:
        """The lateral distance, which is what the radial fit is about."""
        return np.hypot(self.dx, self.dy)

    def __len__(self) -> int:
        return len(self.dx)

    def axes(self) -> List[Tuple[str, np.ndarray, float, str, Optional[np.ndarray]]]:
        """``(name, values, half width, background, variance)`` per axis."""
        found = [("x", self.dx, self.d_max, "chord", self.variance),
                 ("y", self.dy, self.d_max, "chord", self.variance)]
        if self.dz is not None and self.dz_max:
            found.append(("z", self.dz, self.dz_max, "flat", self.variance_z))
        return found


def displacement_pairs(x, y, frame, d_max: float, z=None,
                       dz_max: Optional[float] = None, gaps: Sequence[int] = (1,),
                       precision=None, precision_z=None, photons=None,
                       max_frame_pairs: int = 20000, max_pairs: int = 2_000_000,
                       seed: int = 0, report=None) -> Dict[int, Displacements]:
    """Every pair inside the search cylinder whose frames differ by a gap.

    One KD-tree per frame, reused as the frame walks from the later side of a
    pair to the earlier side of the next, so each frame is indexed once per gap
    and only a handful of trees are alive at a time -- a movie is a hundred
    thousand frames of a hundred localizations, and holding a tree for each
    would cost more than the search.

    Displacements are *later minus earlier*, so their mean is the drift over
    the gap rather than zero.
    """
    from scipy.spatial import cKDTree

    frame = np.asarray(frame)
    order = np.argsort(frame, kind="stable")
    xs = np.asarray(x, dtype=float)[order]
    ys = np.asarray(y, dtype=float)[order]
    zs = None if z is None else np.asarray(z, dtype=float)[order]
    ps = None if precision is None else np.asarray(precision, dtype=float)[order]
    pz = None if precision_z is None else np.asarray(precision_z, dtype=float)[order]
    nphot = None if photons is None else np.asarray(photons, dtype=float)[order]
    fs = frame[order]
    points = np.column_stack([xs, ys])

    values, starts = np.unique(fs, return_index=True)
    ends = np.append(starts[1:], len(fs))
    rng = np.random.default_rng(seed)

    found: Dict[int, Displacements] = {}
    for gap in gaps:
        wanted = values + gap
        target = np.searchsorted(values, wanted)
        inside = target < len(values)
        valid = np.zeros(len(values), dtype=bool)
        valid[inside] = values[target[inside]] == wanted[inside]
        first = np.flatnonzero(valid)
        if not len(first):
            continue
        if len(first) > max_frame_pairs:     # evenly spaced: keep the whole movie
            first = first[np.linspace(0, len(first) - 1, max_frame_pairs).astype(int)]
        second = target[first]

        trees: Dict[int, "cKDTree"] = {}

        def tree(block: int) -> "cKDTree":
            got = trees.get(block)
            if got is None:
                got = trees[block] = cKDTree(points[starts[block]:ends[block]])
            return got

        parts: List[np.ndarray] = []
        n_found = 0
        for k, (i, j) in enumerate(zip(first, second)):
            for stale in [b for b in trees if b < i]:
                del trees[stale]
            if ends[i] == starts[i] or ends[j] == starts[j]:
                continue
            pairs = tree(i).sparse_distance_matrix(tree(j), d_max,
                                                   output_type="ndarray")
            if not len(pairs):
                continue
            left = pairs["i"] + starts[i]
            right = pairs["j"] + starts[j]
            if zs is not None and dz_max:
                keep = np.abs(zs[right] - zs[left]) <= dz_max
                left, right = left[keep], right[keep]
                if not len(left):
                    continue
            parts.append(np.column_stack([left, right]))
            n_found += len(left)
            if report is not None and k and not k % 2000:
                report(f"gap {gap}: {k}/{len(first)} frames, {n_found} pairs")

        if not parts:
            continue
        both = np.concatenate(parts)
        if len(both) > max_pairs:
            both = both[rng.choice(len(both), max_pairs, replace=False)]
        left, right = both[:, 0], both[:, 1]
        found[gap] = Displacements(
            gap=gap, dx=xs[right] - xs[left], dy=ys[right] - ys[left],
            dz=None if zs is None else zs[right] - zs[left],
            variance=None if ps is None else ps[left] ** 2 + ps[right] ** 2,
            variance_z=None if pz is None else pz[left] ** 2 + pz[right] ** 2,
            photons=None if nphot is None else 0.5 * (nphot[left] + nphot[right]),
            inverse_photons=None if nphot is None else
            1.0 / np.maximum(nphot[left], 1e-9) + 1.0 / np.maximum(nphot[right], 1e-9),
            d_max=float(d_max), dz_max=None if zs is None else dz_max,
            n_frames=len(first), n_found=n_found)
    return found


# ------------------------------------------------------------------- the fits

@dataclass
class Fit:
    """One displacement distribution, described by one number.

    That number is either a precision in nanometres (``scaled`` false) or the
    factor by which the data misses each pair's own Cramer-Rao bound
    (``scaled`` true, ``kappa``).  In both cases ``sigma`` is the precision of
    one localization: for a scaled fit it is ``kappa`` times the bound these
    pairs had, which is the precision *of this population* and not of any
    other.
    """
    kind: str                  # "radial", "x", "y", "z"
    sigma: float = float("nan")
    sigma_error: float = float("nan")
    fraction: float = float("nan")      # of the pairs that are the same molecule
    n: int = 0
    half_width: float = 0.0
    offset: float = 0.0        # the mean displacement: drift over the gap
    scaled: bool = False
    scale_kind: str = ""               # "crlb" or "photons"
    kappa: float = float("nan")        # crlb: the factor the data misses it by
    kappa_error: float = float("nan")
    crlb: float = float("nan")         # crlb: the bound these pairs expected
    amplitude: float = float("nan")    # photons: A of sigma = sqrt(A / N)
    message: str = ""

    @property
    def n_signal(self) -> float:
        return self.n * self.fraction if np.isfinite(self.fraction) else float("nan")

    @property
    def ok(self) -> bool:
        return np.isfinite(self.sigma) and self.sigma > 0


def _variance(sigma: float, variance: Optional[np.ndarray]) -> np.ndarray:
    """The displacement variance the model expects: ``2 s^2``, or ``k^2`` of it.

    One expression for both fits, which is the point: the likelihood, the
    truncation and the drawing never learn which of the two they are serving.
    """
    if variance is None:
        return np.asarray(2.0 * sigma ** 2, dtype=float)
    return sigma ** 2 * np.asarray(variance, dtype=float)


def radial_density(d, sigma: float, fraction: float, d_max: float,
                   variance: Optional[np.ndarray] = None) -> np.ndarray:
    """``p(d)``: Churchman's law at zero distance, on a uniform background."""
    d = np.asarray(d, dtype=float)
    v = _variance(sigma, variance)
    norm = np.maximum(1 - np.exp(-d_max ** 2 / (2 * v)), 1e-12)
    signal = d / v * np.exp(-d ** 2 / (2 * v)) / norm
    return fraction * signal + (1 - fraction) * 2 * d / d_max ** 2


def axis_density(delta, sigma: float, fraction: float, half_width: float,
                 background: str = "chord",
                 variance: Optional[np.ndarray] = None) -> np.ndarray:
    """``p(dx)``: a Gaussian of the pair's own width on the pair background."""
    from scipy.special import erf

    delta = np.asarray(delta, dtype=float)
    width = np.sqrt(_variance(sigma, variance))
    norm = np.maximum(erf(half_width / (width * np.sqrt(2.0))), 1e-12)
    signal = (np.exp(-delta ** 2 / (2 * width ** 2))
              / (width * np.sqrt(2 * np.pi) * norm))
    return fraction * signal + (1 - fraction) * _axis_background(
        delta, half_width, background)


def _axis_background(delta, half_width: float, background: str) -> np.ndarray:
    """What one axis of a uniformly filled search volume looks like.

    Laterally the volume is a disc, so an axis of it is the chord through the
    disc at that displacement; axially it is a slab, so it is flat.
    """
    delta = np.asarray(delta, dtype=float)
    if background == "chord":
        inside = np.abs(delta) < half_width
        out = np.zeros_like(delta)
        out[inside] = (2 * np.sqrt(half_width ** 2 - delta[inside] ** 2)
                       / (np.pi * half_width ** 2))
        return out
    return np.full_like(delta, 0.5 / half_width)


def fit_radial(d, d_max: float, sigma0: Optional[float] = None,
               variance: Optional[np.ndarray] = None,
               scale_kind: str = "crlb") -> Fit:
    """``sigma`` from the lateral distances, or the scale of a given expectation."""
    d = np.asarray(d, dtype=float)
    keep = np.isfinite(d) & (d > 0) & (d <= d_max)
    if variance is not None:
        variance = np.asarray(variance, dtype=float)
        keep &= np.isfinite(variance) & (variance > 0)
        variance = variance[keep]
    d = d[keep]
    fit = Fit(kind="radial", n=len(d), half_width=float(d_max),
              scaled=variance is not None,
              scale_kind=scale_kind if variance is not None else "")
    if len(d) < MIN_PAIRS:
        fit.message = f"{len(d)} pairs: too few to fit"
        return fit
    if sigma0 is None or not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = float(np.median(d)) / np.sqrt(2.0)

    def nll(sigma: float, fraction: float) -> float:
        return _nll(radial_density(d, sigma, fraction, d_max, variance))

    return _minimize(nll, sigma0, d_max, fit, variance)


def fit_axis(delta, half_width: float, kind: str = "x",
             background: str = "chord", sigma0: Optional[float] = None,
             variance: Optional[np.ndarray] = None) -> Fit:
    """``sigma`` (or ``kappa``) from the signed displacements along one axis."""
    delta = np.asarray(delta, dtype=float)
    keep = np.isfinite(delta) & (np.abs(delta) <= half_width)
    if variance is not None:
        variance = np.asarray(variance, dtype=float)
        keep &= np.isfinite(variance) & (variance > 0)
        variance = variance[keep]
    delta = delta[keep]
    fit = Fit(kind=kind, n=len(delta), half_width=float(half_width),
              offset=float(delta.mean()) if len(delta) else float("nan"),
              scaled=variance is not None)
    if len(delta) < MIN_PAIRS:
        fit.message = f"{len(delta)} pairs: too few to fit"
        return fit
    if sigma0 is None or not np.isfinite(sigma0) or sigma0 <= 0:
        # the core is narrow and the background is wide, so a low quantile of
        # |d| is a start that survives a background of any size
        sigma0 = max(float(np.quantile(np.abs(delta), 0.25)), 1e-3)

    def nll(sigma: float, fraction: float) -> float:
        return _nll(axis_density(delta, sigma, fraction, half_width, background,
                                 variance))

    return _minimize(nll, sigma0, half_width, fit, variance)


def _nll(density: np.ndarray) -> float:
    return float(-np.sum(np.log(np.maximum(density, 1e-300))))


def _minimize(nll, sigma0: float, reach: float, fit: Fit,
              variance: Optional[np.ndarray] = None) -> Fit:
    """Maximize the mixture's likelihood in ``(log sigma, logit fraction)``.

    Unconstrained in the transformed parameters, so a plain simplex does it and
    neither sigma nor the fraction can wander out of its range.  Several
    starting fractions, because a mixture whose background dominates has a
    shallow valley and a single start can sit in it.
    """
    from scipy.optimize import minimize

    scaled = variance is not None
    # the scale is whatever turns the expectation into a precision: 1 when the
    # expectation is already in nanometres and the data meets it, and A^(1/2)
    # in nm sqrt(photons) for the photon law.  Starting it from the guessed
    # precision means neither case needs a range of its own.
    typical = float(np.sqrt(np.mean(variance) / 2)) if scaled else 1.0
    start_value = (sigma0 / typical if scaled and typical > 0 else
                   (1.0 if scaled else sigma0))
    limits = ((start_value * 1e-3, start_value * 1e3) if scaled
              else (1e-4, 10 * reach))

    def unpack(theta) -> Tuple[float, float]:
        return float(np.exp(theta[0])), float(1 / (1 + np.exp(-theta[1])))

    def objective(theta) -> float:
        sigma, fraction = unpack(theta)
        if not limits[0] < sigma < limits[1]:
            return 1e12
        value = nll(sigma, fraction)
        return value if np.isfinite(value) else 1e12

    best = None
    for fraction in START_FRACTIONS:
        start = [np.log(start_value), np.log(fraction / (1 - fraction))]
        got = minimize(objective, start, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
        if best is None or got.fun < best.fun:
            best = got
    # a simplex that has collapsed along one direction stops early on a valley
    # this shallow, and the answer then remembers where it started -- which is
    # not allowed to matter.  Restarting it at its own optimum costs a few
    # dozen evaluations and removes the dependence.
    best = minimize(objective, best.x, method="Nelder-Mead",
                    options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 4000})
    value, fraction = unpack(best.x)
    error = _parameter_error(nll, value, fraction, fit.n)
    fit.fraction = fraction
    if scaled:
        # what one localization of these pairs was expected to have: half the
        # pair's expectation, averaged.  In nanometres for the bound, in
        # 1/sqrt(photons) for the photon law -- either way, multiplying by the
        # fitted scale turns it into the precision of a typical paired
        # localization, which is what `sigma` means for every fit here.
        typical = float(np.sqrt(np.mean(variance) / 2))
        fit.sigma, fit.sigma_error = value * typical, error * typical
        if fit.scale_kind == "photons":
            fit.amplitude = float(value ** 2)
        else:
            fit.crlb = typical
            fit.kappa, fit.kappa_error = value, error
    else:
        fit.sigma, fit.sigma_error = value, error
    if fraction * fit.n < MIN_PAIRS / 2:
        fit.message = ("almost every pair is a different molecule: the filter "
                       "may have removed the repeats, or the gap is too long")
    return fit


def _parameter_error(nll, value: float, fraction: float, n: int) -> float:
    """The standard error of the fitted scale, from the observed information.

    The curvature of the log-likelihood in ``(value, fraction)`` by finite
    differences, inverted: the fraction is fitted too, so its uncertainty has
    to widen the scale's rather than be conditioned away.  Falls back to the
    Rayleigh-like ``value / sqrt(2 n_signal)`` when the curvature comes out
    singular, which happens when the fit has run to a boundary.
    """
    step_v = max(value * 1e-3, 1e-6)
    step_f = min(max(fraction, 1 - fraction) * 1e-3, 1e-3)
    try:
        def at(dv: float, df: float) -> float:
            return nll(value + dv, min(max(fraction + df, 1e-9), 1 - 1e-9))

        centre = at(0, 0)
        hvv = (at(step_v, 0) - 2 * centre + at(-step_v, 0)) / step_v ** 2
        hff = (at(0, step_f) - 2 * centre + at(0, -step_f)) / step_f ** 2
        hvf = ((at(step_v, step_f) - at(step_v, -step_f)
                - at(-step_v, step_f) + at(-step_v, -step_f))
               / (4 * step_v * step_f))
        determinant = hvv * hff - hvf ** 2
        if determinant > 0 and hvv > 0:
            return float(np.sqrt(hff / determinant))
    except (FloatingPointError, ValueError):
        pass
    signal = max(n * fraction, 1.0)
    return float(value / np.sqrt(2 * signal))


def drift_free_sigma(gaps: Sequence[int], sigmas: Sequence[float],
                     errors: Sequence[float]) -> Dict[str, float]:
    """The intercept of ``sigma(gap)^2``: the precision with the motion removed.

    A random walk between frames adds variance in proportion to the gap, so the
    line through ``sigma^2`` extrapolates to the gap the fit cannot see -- zero,
    two localizations of one molecule in the same frame.  With one gap there is
    no line and the answer is that gap's sigma.
    """
    gap = np.asarray(gaps, dtype=float)
    sigma = np.asarray(sigmas, dtype=float)
    error = np.asarray(errors, dtype=float)
    good = np.isfinite(gap) & np.isfinite(sigma) & (sigma > 0)
    gap, sigma, error = gap[good], sigma[good], error[good]
    out = {"sigma0": float("nan"), "slope": float("nan"),
           "step": float("nan"), "n": int(len(gap))}
    if not len(gap):
        return out
    if len(gap) == 1:
        out["sigma0"] = float(sigma[0])
        return out
    y = sigma ** 2
    # d(sigma^2) = 2 sigma d(sigma): the weights follow from the fitted errors
    spread = np.where(np.isfinite(error) & (error > 0), 2 * sigma * error, np.nan)
    weight = 1.0 / spread ** 2 if np.all(np.isfinite(spread)) else np.ones_like(y)
    design = np.column_stack([np.ones_like(gap), gap])
    scaled = design * weight[:, None]
    try:
        solution = np.linalg.solve(design.T @ scaled, scaled.T @ y)
    except np.linalg.LinAlgError:
        return out
    intercept, slope = float(solution[0]), float(solution[1])
    out["slope"] = slope
    out["sigma0"] = float(np.sqrt(intercept)) if intercept > 0 else float("nan")
    # a random walk of step w per frame adds w^2 g to the *displacement*
    # variance, which is 2 sigma^2 -- so the line's slope is w^2 / 2
    out["step"] = float(np.sqrt(2 * slope)) if slope > 0 else 0.0
    return out


# --------------------------------------------------------------- the analysis

@dataclass
class Measurement:
    """Everything one set of localizations had to say about its precision."""
    name: str
    n_locs: int
    radial: Optional[Fit] = None
    photon_law: Optional[Fit] = None           # the same pairs against 1/N
    scaled: Optional[Fit] = None               # the same pairs against the CRLB
    axes: Dict[str, Fit] = field(default_factory=dict)
    by_gap: Dict[int, Fit] = field(default_factory=dict)
    axes_by_gap: Dict[str, Dict[int, Fit]] = field(default_factory=dict)
    gap_line: Dict[str, float] = field(default_factory=dict)
    displacements: Dict[int, Displacements] = field(default_factory=dict)
    crlb: Dict[str, Dict[str, float]] = field(default_factory=dict)
    frc: Optional[FRC] = None
    fpc: Optional[FPC] = None                  # the resolution along each axis
    photons: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def first_gap(self) -> Optional[Displacements]:
        return (self.displacements[min(self.displacements)]
                if self.displacements else None)


def first_present(locs: Localizations, names: Sequence[str]) -> Optional[str]:
    return next((name for name in names if name in locs), None)


def median_precision(locs: Localizations, names: Sequence[str] = PRECISION_FIELDS
                     ) -> float:
    """The median of whichever precision column the table has, or NaN."""
    name = first_present(locs, names)
    if name is None:
        return float("nan")
    values = np.asarray(locs[name], dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.median(values)) if len(values) else float("nan")


def measure(locs: Localizations, name: str = "", reach: float = 0.0,
            reach_z: float = 0.0, max_gap: int = 5, pairwise: bool = True,
            crlb: bool = True, frc: bool = False, frc_pixel: float = 0.0,
            frc_blocks: int = 20, frc_repeats: int = 5, frc_axes: bool = False,
            bounds: Optional[Dict[str, Tuple]] = None,
            max_frame_pairs: int = 20000, max_pairs: int = 2_000_000,
            seed: int = 0, report=None) -> Measurement:
    """Both measurements over one table: the displacements and the CRLB.

    Takes a table and returns numbers -- a script, a test or a notebook gets
    the same answer as the plugin without a session or a window.
    """
    bounds = bounds or {}
    out = Measurement(name=name, n_locs=len(locs))
    lateral = median_precision(locs)
    axial = median_precision(locs, PRECISION_Z_FIELDS)

    if crlb:
        out.crlb = crlb_statistics(locs, bounds)
    if frc:
        out.frc = _frc_of(locs, frc_pixel, frc_blocks, frc_repeats, seed, report)
        if out.frc is not None and out.frc.message:
            out.notes.append(f"FRC: {out.frc.message}")
    if frc_axes:
        # three splits at least: the spread over them is the error bar, and a
        # 3D transform is dear enough that more than a few is not worth it
        out.fpc = _fpc_of(locs, frc_pixel, frc_blocks, max(frc_repeats // 2, 3),
                          seed, report)
        if out.fpc is None:
            out.notes.append("no z: the per-axis resolution needs 3D data")
        elif out.fpc.message:
            out.notes.append(f"FPC: {out.fpc.message}")
    if not pairwise:
        return out

    if "frame" not in locs:
        raise ValueError("the pairwise displacement needs a frame column; the "
                         f"table has {', '.join(sorted(locs.keys()))}")
    x_name = first_present(locs, ("x_nm", "x_pix"))
    y_name = first_present(locs, ("y_nm", "y_pix"))
    if x_name is None or y_name is None:
        raise ValueError("the pairwise displacement needs positions; the table "
                         f"has {', '.join(sorted(locs.keys()))}")
    z = np.asarray(locs["z_nm"], dtype=float) if "z_nm" in locs else None
    if z is not None and not np.any(np.isfinite(z) & (z != 0)):
        z = None                             # a column of zeros is not 3D data
    precision_name = first_present(locs, PRECISION_FIELDS)
    precision_z_name = first_present(locs, PRECISION_Z_FIELDS)

    if reach <= 0:
        reach = REACH * lateral if np.isfinite(lateral) and lateral > 0 else 100.0
        out.notes.append(f"search radius {reach:.0f} nm "
                         f"({REACH:g} x the median precision)")
    if z is not None and reach_z <= 0:
        reach_z = (REACH_Z * axial if np.isfinite(axial) and axial > 0
                   else 3 * reach)

    gaps = tuple(range(1, max(1, max_gap) + 1))
    if report:
        report(f"pairs within {reach:.0f} nm, frame gaps 1-{gaps[-1]}")
    out.displacements = displacement_pairs(
        np.asarray(locs[x_name], dtype=float),
        np.asarray(locs[y_name], dtype=float),
        np.asarray(locs["frame"]), reach, z=z,
        dz_max=reach_z if z is not None else None, gaps=gaps,
        precision=None if precision_name is None else locs[precision_name],
        precision_z=(None if precision_z_name is None or z is None
                     else locs[precision_z_name]),
        photons=locs["photons"] if "photons" in locs else None,
        max_frame_pairs=max_frame_pairs, max_pairs=max_pairs, seed=seed,
        report=report)
    if not out.displacements:
        out.notes.append("no pairs: no molecule was localized twice this close "
                         "together, within the frame gaps asked for")
        return out

    for gap, pairs in sorted(out.displacements.items()):
        out.by_gap[gap] = fit_radial(pairs.d, pairs.d_max, sigma0=lateral)
        for axis, values, half, background, variance in pairs.axes():
            start = axial if axis == "z" else lateral
            out.axes_by_gap.setdefault(axis, {})[gap] = fit_axis(
                values, half, kind=axis, background=background, sigma0=start)

    first = min(out.displacements)
    pairs = out.displacements[first]
    out.radial = out.by_gap.get(first)
    out.axes = {axis: fits[first] for axis, fits in out.axes_by_gap.items()
                if first in fits}
    if pairs.inverse_photons is not None:
        out.photon_law = fit_radial(pairs.d, pairs.d_max, sigma0=lateral,
                                    variance=pairs.inverse_photons,
                                    scale_kind="photons")
    if pairs.variance is not None:
        out.scaled = fit_radial(pairs.d, pairs.d_max, sigma0=lateral,
                                variance=pairs.variance, scale_kind="crlb")
    if pairs.photons is not None and "photons" in locs:
        everything = np.asarray(locs["photons"], dtype=float)
        everything = everything[np.isfinite(everything)]
        out.photons = {"paired": float(np.median(pairs.photons)),
                       "all": float(np.median(everything)) if len(everything)
                              else float("nan")}
    usable = [(gap, fit) for gap, fit in sorted(out.by_gap.items()) if fit.ok]
    if usable:
        out.gap_line = drift_free_sigma([gap for gap, _ in usable],
                                        [fit.sigma for _, fit in usable],
                                        [fit.sigma_error for _, fit in usable])
    z_fit = out.axes.get("z")
    if z is not None and z_fit is not None and z_fit.ok:
        if reach_z < THIN_SLAB * z_fit.sigma:
            out.notes.append(
                f"the axial search reaches {reach_z:.0f} nm, less than "
                f"{THIN_SLAB:g} x the axial precision: sigma_z is cut off and "
                "reads low")
    return out


def sigma_at_photons(fit: Optional[Fit], photons: float) -> float:
    """What the photon law says a localization of ``photons`` is worth.

    ``sigma = sqrt(A / N)``, and both the fit and this prediction read the same
    photon column, so a camera calibrated with the wrong gain moves ``A`` and
    ``N`` by opposite factors and leaves the answer where it was.  That is the
    point of fitting the law rather than the bound: it is as free of the
    calibration as the plain NeNA sigma is, and unlike it, it can be carried to
    localizations the pairs never contained -- the bright ones in the middle of
    a blink, or the sums a grouped table is made of.
    """
    if fit is None or not np.isfinite(fit.amplitude) or photons <= 0:
        return float("nan")
    return float(np.sqrt(fit.amplitude / photons))


def _frc_of(locs: Localizations, pixel: float, blocks: int, repeats: int,
            seed: int, report) -> Optional[FRC]:
    """The FRC of this table, or None when it has nothing to render."""
    x_name = first_present(locs, ("x_nm", "x_pix"))
    y_name = first_present(locs, ("y_nm", "y_pix"))
    if x_name is None or y_name is None or "frame" not in locs:
        return None
    return frc_resolution(np.asarray(locs[x_name], dtype=float),
                          np.asarray(locs[y_name], dtype=float),
                          np.asarray(locs["frame"]), pixelsize=pixel,
                          n_blocks=blocks, repeats=repeats, seed=seed,
                          report=report)


def _fpc_of(locs: Localizations, pixel: float, blocks: int, repeats: int,
            seed: int, report) -> Optional[FPC]:
    """The per-axis resolution of this table, or None when it is not 3D."""
    x_name = first_present(locs, ("x_nm", "x_pix"))
    y_name = first_present(locs, ("y_nm", "y_pix"))
    if x_name is None or y_name is None or "frame" not in locs or "z_nm" not in locs:
        return None
    z = np.asarray(locs["z_nm"], dtype=float)
    if not np.any(np.isfinite(z) & (z != 0)):
        return None
    return fpc_resolution(np.asarray(locs[x_name], dtype=float),
                          np.asarray(locs[y_name], dtype=float), z,
                          np.asarray(locs["frame"]), pixelsize=pixel,
                          n_blocks=blocks, repeats=repeats, seed=seed,
                          report=report)


def crlb_statistics(locs: Localizations, bounds: Optional[Dict[str, Tuple]] = None
                    ) -> Dict[str, Dict[str, float]]:
    """The fitted precision distribution, lateral and axial, with its cut.

    The bounds are the filter's, when they are known: the estimator's
    likelihood is truncated, so it recovers the whole distribution from the
    part that survived a cut instead of describing the cut.
    """
    bounds = bounds or {}
    out: Dict[str, Dict[str, float]] = {}
    for key, names in (("lateral", PRECISION_FIELDS), ("axial", PRECISION_Z_FIELDS)):
        name = first_present(locs, names)
        if name is None:
            continue
        values = np.asarray(locs[name], dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if not len(values):
            continue
        low, high = bounds.get(name, (None, None))
        stats = precision_model(values, low=low, high=high)
        stats["field"] = name
        stats["cut"] = bool(low is not None or high is not None)
        stats["values"] = values
        # sigma_c assumes the photons are exponential, which is a statement
        # about the sample and not always true -- a narrow photon distribution,
        # or a hard filter on both sides, leaves the fit describing something
        # that is not there.  Far from the median is how that shows.
        median = stats.get("median", float("nan"))
        sigma_c = stats.get("sigma_c", float("nan"))
        stats["suspect"] = bool(np.isfinite(median) and np.isfinite(sigma_c)
                                and not 0.3 * median < sigma_c < 3 * median)
        out[key] = stats
    return out


def filter_bounds(session, layer: int = 0) -> Dict[str, Tuple]:
    """The layer filter's ranges, so a truncated fit can be told about them.

    Read defensively: a script has no session, a session may have no layers,
    and a missing bound is only a bound the fit has to find for itself.
    """
    try:
        layers = session.layers
        state = layers[layer if 0 <= layer < len(layers) else 0].state
        return dict(state.sets["ungrouped"].filter.ranges)
    except (AttributeError, IndexError, KeyError, TypeError):
        return {}


# --------------------------------------------------------------------- saying

def summary(found: Sequence[Measurement]) -> str:
    """The numbers, side by side, in the order someone reads them."""
    lines: List[str] = []
    for measurement in found:
        lines.append(f"{measurement.name}: {measurement.n_locs} localizations")
        crlb = measurement.crlb.get("lateral")
        if crlb and np.isfinite(crlb.get("sigma_c", np.nan)):
            cut = ", filter bounds used" if crlb.get("cut") else ""
            lines.append(f"  CRLB      sigma_c = {crlb['sigma_c']:.2f} nm "
                         f"(median {crlb['median']:.2f} nm{cut})")
            if crlb.get("suspect"):
                lines.append("            sigma_c is far from the median: the "
                             "photons are not exponential here, so read the "
                             "median and not the fit")
        axial = measurement.crlb.get("axial")
        if axial and np.isfinite(axial.get("sigma_c", np.nan)):
            lines.append(f"  CRLB z    sigma_c = {axial['sigma_c']:.2f} nm "
                         f"(median {axial['median']:.2f} nm)")
        fit = measurement.radial
        if fit is not None and fit.ok:
            lines.append(f"  pairwise  sigma = {fit.sigma:.2f} +/- "
                         f"{fit.sigma_error:.2f} nm at gap 1, {fit.n} pairs, "
                         f"{100 * fit.fraction:.0f}% same molecule")
        for axis, axis_fit in measurement.axes.items():
            if axis_fit.ok:
                lines.append(f"  sigma_{axis}   = {axis_fit.sigma:.2f} +/- "
                             f"{axis_fit.sigma_error:.2f} nm "
                             f"(mean shift over the gap {axis_fit.offset:+.2f} nm)")
        law, photons = measurement.photon_law, measurement.photons
        if law is not None and law.ok and photons:
            lines.append(f"  law       sigma = sqrt(A/N), A^0.5 = "
                         f"{np.sqrt(law.amplitude):.0f} nm sqrt(photons), "
                         "from the same pairs and no calibration")
            for what, key in (("a paired localization", "paired"),
                              ("the median localization", "all")):
                count = photons.get(key, float("nan"))
                value = sigma_at_photons(law, count)
                if np.isfinite(value):
                    lines.append(f"            -> {value:.2f} nm for {what} "
                                 f"({count:.0f} photons)")
        scaled = measurement.scaled
        if scaled is not None and scaled.ok:
            lines.append(f"  kappa     = {scaled.kappa:.2f} +/- "
                         f"{scaled.kappa_error:.2f} x the bound these pairs "
                         f"claimed ({scaled.crlb:.2f} nm): the fit and the "
                         "photon calibration together")
        curve = measurement.frc
        if curve is not None and curve.ok:
            lines.append(f"  FRC       {curve.resolution:.1f} +/- "
                         f"{curve.error:.1f} nm at 1/7, over "
                         f"{curve.repeats} split(s) of {curve.n_blocks} blocks, "
                         f"{curve.pixelsize:.1f} nm pixels")
            sigma = (measurement.radial.sigma
                     if measurement.radial is not None and measurement.radial.ok
                     else float("nan"))
            if np.isfinite(sigma):
                lines.append(f"            the blur of {sigma:.2f} nm alone "
                             f"crosses 1/7 at {envelope_resolution(sigma):.1f} nm "
                             "-- read the curves, not the ratio")
        planes = measurement.fpc
        if planes is not None and planes.ok:
            per_axis = "  ".join(
                f"{name} {curve.resolution:.1f}"
                + (f" +/- {curve.error:.1f}" if np.isfinite(curve.error) else "")
                for name, curve in planes.axes.items() if curve.ok)
            lines.append(f"  per axis  {per_axis} nm, from planes rather than "
                         f"rings, {planes.tiles} tiles of {planes.shape[0]}x"
                         f"{planes.shape[1]}x{planes.shape[2]} voxels of "
                         f"{planes.voxel[0]:.1f}/{planes.voxel[2]:.1f} nm, "
                         f"summed over {planes.band[0]:.0f}/{planes.band[1]:.0f} nm "
                         "of frequency")
            if np.isfinite(planes.anisotropy):
                lines.append(f"            axial / lateral = "
                             f"{planes.anisotropy:.2f}")
            for name, curve in planes.axes.items():
                if curve.message:
                    lines.append(f"            {name}: {curve.message}")
        line = measurement.gap_line
        if line and np.isfinite(line.get("sigma0", np.nan)) and line.get("n", 0) > 1:
            lines.append(f"  gap -> 0  sigma = {line['sigma0']:.2f} nm, "
                         f"motion {line['step']:.2f} nm per root frame")
        for note in measurement.notes:
            lines.append(f"  {note}")
        if fit is not None and fit.message:
            lines.append(f"  {fit.message}")
    return "\n".join(lines)


# -------------------------------------------------------------------- drawing

COLORS = ("#1f77b4", "#d62728")          # one per source, in the order given


def panel_axes(target, n: int):
    """``n`` axes, from a figure or from the single axis of a one-panel plot.

    A `Plot` is handed the figure when it declares more than one panel and an
    axis when it declares one, and how many panels these draw depends on the
    data -- a table with no z has one axis fewer.  Asking what we were given is
    shorter than making every caller count first.
    """
    if hasattr(target, "subplots"):
        return list(target.subplots(n, 1, squeeze=False).ravel())
    return [target]


def averaged_density(x, fit: Fit, d_max: float, variance: np.ndarray,
                     sample: int = 2000, seed: int = 0) -> np.ndarray:
    """The scaled model's density, averaged over the pairs it was fitted to.

    A fit against each pair's own expectation has no single curve to draw: the
    density is one per pair and what the histogram shows is their mean.
    Averaging a sample of them is the honest picture, and it is the one that
    shows whether the heterogeneous model explains the tails that a single
    sigma has to leave.
    """
    scale = fit.kappa if fit.scale_kind == "crlb" else np.sqrt(fit.amplitude)
    if not np.isfinite(scale):
        return np.zeros_like(np.asarray(x, dtype=float))
    variance = np.asarray(variance, dtype=float)
    variance = variance[np.isfinite(variance) & (variance > 0)]
    if len(variance) > sample:
        variance = np.random.default_rng(seed).choice(variance, sample,
                                                      replace=False)
    grid = np.asarray(x, dtype=float)[:, None]
    stack = radial_density(grid, scale, fit.fraction, d_max, variance[None, :])
    return stack.mean(axis=1)


def draw_radial(ax, found: Sequence[Measurement], bins: int = 80) -> None:
    """The lateral displacement histogram with the mixture over it."""
    for index, measurement in enumerate(found):
        fit, pairs = measurement.radial, measurement.first_gap
        if fit is None or pairs is None or not len(pairs):
            continue
        color = COLORS[index % len(COLORS)]
        counts, edges = np.histogram(pairs.d, bins=bins, range=(0, pairs.d_max))
        centres = 0.5 * (edges[:-1] + edges[1:])
        scale = len(pairs) * (edges[1] - edges[0])
        ax.step(centres, counts, where="mid", color=color, linewidth=1.0,
                label=measurement.name)
        if fit.ok:
            x = np.linspace(1e-6, pairs.d_max, 400)
            ax.plot(x, scale * radial_density(x, fit.sigma, fit.fraction,
                                              pairs.d_max),
                    color=color, linewidth=1.6)
            ax.plot(x, scale * (1 - fit.fraction) * 2 * x / pairs.d_max ** 2,
                    color=color, linewidth=0.9, linestyle="--")
            ax.axvline(np.sqrt(2) * fit.sigma, color=color, linewidth=0.8,
                       linestyle=":")
            law = measurement.photon_law
            if law is not None and law.ok and pairs.inverse_photons is not None:
                ax.plot(x, scale * averaged_density(x, law, pairs.d_max,
                                                    pairs.inverse_photons),
                        color=color, linewidth=1.1, linestyle="-.",
                        label=f"{measurement.name}: sqrt(A/N)")
    ax.set_xlabel("lateral displacement (nm)")
    ax.set_ylabel("pairs")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(_radial_title(found), fontsize=7.5, color="0.25")


def _radial_title(found: Sequence[Measurement]) -> str:
    parts = []
    for measurement in found:
        if measurement.radial is not None and measurement.radial.ok:
            text = f"{measurement.name}: sigma = {measurement.radial.sigma:.2f} nm"
            if measurement.scaled is not None and measurement.scaled.ok:
                text += f", kappa = {measurement.scaled.kappa:.2f}"
            parts.append(text)
    return "   ".join(parts) if parts else "no fit"


def draw_axes(figure, found: Sequence[Measurement], bins: int = 80) -> None:
    """One panel per axis: the signed displacements and their Gaussian."""
    names: List[str] = []
    for measurement in found:
        for axis in measurement.axes:
            if axis not in names:
                names.append(axis)
    if not names:
        names = ["x"]
    for ax, axis in zip(panel_axes(figure, len(names)), names):
        titles = []
        for index, measurement in enumerate(found):
            fit, pairs = measurement.axes.get(axis), measurement.first_gap
            if fit is None or pairs is None:
                continue
            values = {name: value for name, value, _, _, _ in pairs.axes()}[axis]
            half = fit.half_width
            color = COLORS[index % len(COLORS)]
            counts, edges = np.histogram(values, bins=bins, range=(-half, half))
            centres = 0.5 * (edges[:-1] + edges[1:])
            ax.step(centres, counts, where="mid", color=color, linewidth=1.0,
                    label=measurement.name)
            if fit.ok:
                scale = len(values) * (edges[1] - edges[0])
                x = np.linspace(-half, half, 400)
                ax.plot(x, scale * axis_density(x, fit.sigma, fit.fraction, half,
                                                "flat" if axis == "z" else "chord"),
                        color=color, linewidth=1.6)
                titles.append(f"{measurement.name}: sigma_{axis} = "
                              f"{fit.sigma:.2f} +/- {fit.sigma_error:.2f} nm")
        ax.set_xlabel(f"{axis} displacement (nm)")
        ax.set_ylabel("pairs")
        ax.set_title("   ".join(titles) if titles else f"no fit for {axis}",
                     fontsize=7.5, color="0.25")
        if len(found) > 1:
            ax.legend(fontsize=7, frameon=False)


def draw_gaps(ax, found: Sequence[Measurement]) -> None:
    """Precision against frame gap: flat is still, rising is moving."""
    styles = {"radial": "-", "x": "--", "y": ":", "z": "-."}
    for index, measurement in enumerate(found):
        color = COLORS[index % len(COLORS)]
        series = {"radial": measurement.by_gap}
        series.update(measurement.axes_by_gap)
        for kind, fits in series.items():
            usable = [(gap, fit) for gap, fit in sorted(fits.items()) if fit.ok]
            if not usable:
                continue
            ax.errorbar([gap for gap, _ in usable],
                        [fit.sigma for _, fit in usable],
                        yerr=[fit.sigma_error for _, fit in usable], color=color,
                        linestyle=styles.get(kind, "-"), marker="o", markersize=3,
                        linewidth=1.0, capsize=2,
                        label=f"{measurement.name} {kind}")
        line = measurement.gap_line
        if line and np.isfinite(line.get("sigma0", np.nan)) and line.get("n", 0) > 1:
            gaps = np.linspace(0, max(measurement.by_gap), 50)
            ax.plot(gaps, np.sqrt(np.maximum(line["sigma0"] ** 2
                                             + line["slope"] * gaps, 0)),
                    color=color, linewidth=0.8, alpha=0.7)
            ax.plot([0], [line["sigma0"]], marker="*", markersize=9, color=color)
    ax.set_xlabel("frame gap")
    ax.set_ylabel("sigma (nm)")
    ax.set_xlim(left=0)
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title(_gap_title(found), fontsize=7.5, color="0.25")


def _gap_title(found: Sequence[Measurement]) -> str:
    parts = []
    for measurement in found:
        line = measurement.gap_line
        if line and np.isfinite(line.get("sigma0", np.nan)) and line.get("n", 0) > 1:
            parts.append(f"{measurement.name}: sigma(gap -> 0) = "
                         f"{line['sigma0']:.2f} nm, "
                         f"{line['step']:.2f} nm per root frame")
    return "   ".join(parts) if parts else "one gap: no motion to see"


def draw_crlb(figure, found: Sequence[Measurement], bins: int = 80) -> None:
    """The CRLB histograms, with the fitted law and the pairwise sigma on them."""
    keys = [key for key in ("lateral", "axial")
            if any(key in measurement.crlb for measurement in found)]
    if not keys:
        keys = ["lateral"]
    for ax, key in zip(panel_axes(figure, len(keys)), keys):
        titles = []
        for index, measurement in enumerate(found):
            stats = measurement.crlb.get(key)
            values = None if stats is None else stats.get("values")
            if values is None or not len(values):
                continue
            color = COLORS[index % len(COLORS)]
            high = float(np.percentile(values, 99.5))
            counts, edges = np.histogram(values, bins=bins, range=(0, max(high, 1e-6)))
            centres = 0.5 * (edges[:-1] + edges[1:])
            ax.step(centres, counts, where="mid", color=color, linewidth=1.0,
                    label=f"{measurement.name} ({stats['field']})")
            a = stats.get("a", float("nan"))
            if np.isfinite(a):
                ax.plot(centres,
                        len(values) * (edges[1] - edges[0])
                        * precision_density(centres, a),
                        color=color, linewidth=1.5)
                titles.append(f"{measurement.name}: sigma_c = "
                              f"{stats['sigma_c']:.2f} nm")
            fit = (measurement.radial if key == "lateral"
                   else measurement.axes.get("z"))
            if fit is not None and fit.ok:
                ax.axvline(fit.sigma, color=color, linestyle="--", linewidth=1.0,
                           label=f"{measurement.name}: pairwise")
        ax.set_xlabel(f"{key} precision (nm)")
        ax.set_ylabel("localizations")
        ax.legend(fontsize=6.5, frameon=False)
        ax.set_title("   ".join(titles) if titles else f"no {key} precision",
                     fontsize=7.5, color="0.25")


def draw_frc(ax, found: Sequence[Measurement]) -> None:
    """The FRC curves, the threshold, and the blur the precision implies."""
    titles = []
    for index, measurement in enumerate(found):
        curve = measurement.frc
        if curve is None or not len(curve.curve):
            continue
        color = COLORS[index % len(COLORS)]
        ax.plot(curve.q, curve.curve, color=color, linewidth=0.7, alpha=0.45)
        ax.plot(curve.q, curve.smoothed, color=color, linewidth=1.5,
                label=measurement.name)
        if curve.ok:
            ax.axvline(1.0 / curve.resolution, color=color, linewidth=0.9,
                       linestyle=":")
            titles.append(f"{measurement.name}: {curve.resolution:.1f} +/- "
                          f"{curve.error:.1f} nm")
        fit = measurement.radial
        if fit is not None and fit.ok:
            # not a fit to the data: what the measured precision alone leaves
            # correlated at each frequency, which the curve is read against
            ax.plot(curve.q, blur_envelope(curve.q, fit.sigma), color=color,
                    linewidth=1.0, linestyle="--",
                    label=f"{measurement.name}: blur of {fit.sigma:.1f} nm")
    ax.axhline(THRESHOLD_LINE, color="0.35", linewidth=0.9)
    ax.text(0.99, THRESHOLD_LINE, " 1/7", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color="0.35")
    ax.axhline(0.0, color="0.75", linewidth=0.6)
    ax.set_xlabel("spatial frequency (1/nm)")
    ax.set_ylabel("FRC")
    ax.set_ylim(-0.2, 1.05)
    ax.set_xlim(left=0)
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("   ".join(titles) if titles else "no FRC", fontsize=7.5,
                 color="0.25")


AXIS_COLORS = {"x": "#1f77b4", "y": "#2ca02c", "z": "#d62728"}


def draw_planes(ax, found: Sequence[Measurement]) -> None:
    """The plane correlations: one curve per axis, per source.

    With one source the axes get the colours, because that is what the panel is
    about; with two, the colours go back to the sources and the axes take the
    line styles, since comparing the same axis across sources is then the
    question.
    """
    styles = {"x": "-", "y": "--", "z": "-."}
    titles = []
    for index, measurement in enumerate(found):
        planes = measurement.fpc
        if planes is None or not planes.axes:
            continue
        color = COLORS[index % len(COLORS)]
        for name, curve in planes.axes.items():
            color = AXIS_COLORS[name] if len(found) == 1 else color
            if not len(curve.smoothed):
                continue
            ax.plot(curve.q, curve.smoothed, color=color, linewidth=1.3,
                    linestyle=styles.get(name, "-"),
                    label=name if len(found) == 1 else f"{measurement.name} {name}")
            if curve.ok:
                ax.axvline(1.0 / curve.resolution, color=color, linewidth=0.7,
                           linestyle=":")
        if planes.ok:
            titles.append(f"{measurement.name}: " + ", ".join(
                f"{name} {curve.resolution:.0f} nm"
                for name, curve in planes.axes.items() if curve.ok))
    ax.axhline(THRESHOLD_LINE, color="0.35", linewidth=0.9)
    ax.text(0.99, THRESHOLD_LINE, " 1/7", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color="0.35")
    ax.axhline(0.0, color="0.75", linewidth=0.6)
    ax.set_xlabel("spatial frequency along the axis (1/nm)")
    ax.set_ylabel("FPC")
    ax.set_ylim(-0.2, 1.05)
    ax.set_xlim(left=0)
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("   ".join(titles) if titles else "no per-axis resolution",
                 fontsize=7.5, color="0.25")


# ----------------------------------------------------------------- the plugin

@dataclass
class PrecisionSettings:
    source: str = param("both", label="localizations",
                        choices=(("selection", "as plotted (filter and ROI)"),
                                 ("all", "all, unfiltered"),
                                 ("both", "both, side by side")),
                        help="what the image is worth, what the sample is "
                             "worth, or the two compared")
    pairwise: bool = param(True, label="pairwise displacement (NeNA)",
                           help="the experimental precision, from molecules "
                                "localized in more than one frame")
    crlb: bool = param(True, label="CRLB histogram",
                       help="what the fitter expected, from the photons")
    frc: bool = param(True, label="FRC resolution",
                      help="what the picture resolves: the localization error, "
                           "the labeling density and the drift together")
    max_gap: int = param(5, label="frame gaps to", min=1, max=50,
                         help="the fit is repeated at every gap up to this: "
                              "flat means nothing moved between frames")
    reach_nm: float = param(0.0, label="search radius", unit="nm", min=0.0,
                            help=f"0: {REACH:g} x the median precision")
    reach_z_nm: float = param(0.0, label="axial search", unit="nm", min=0.0,
                              advanced=True,
                              help=f"3D data only.  0: {REACH_Z:g} x the median "
                                   "axial precision")
    frc_axes: bool = param(False, label="FRC per axis (3D)",
                           help="the resolution along x, y and z separately, "
                                "from planes of the 3D transform rather than "
                                "rings of a 2D one.  3D data, and a ROI rather "
                                "than a whole field of view")
    frc_blocks: int = param(20, label="FRC blocks", min=2, advanced=True,
                            help="the acquisition is cut into this many "
                                 "stretches of frames before the two halves "
                                 "are dealt, so that a blink stays whole")
    frc_repeats: int = param(5, label="FRC splits", min=1, max=50, advanced=True,
                             help="the curves of this many random deals are "
                                  "averaged, and their spread is the error bar")
    frc_pixel_nm: float = param(0.0, label="FRC pixel", unit="nm", min=0.0,
                                advanced=True,
                                help="0: measured roughly first, then at a "
                                     "fifth of that resolution")
    bins: int = param(80, label="bins", min=10, advanced=True,
                      help="drawing only: every fit is on the displacements")
    max_frame_pairs: int = param(20000, label="frames per gap", min=100,
                                 advanced=True,
                                 help="a longer movie is sampled evenly across "
                                      "its whole length")
    max_pairs: int = param(2_000_000, label="pairs per gap", min=1000,
                           advanced=True)
    seed: int = param(0, label="seed", advanced=True,
                      help="which pairs are kept when there are more than the "
                           "limit")


@register("Analysis/Measure/Localization Precision")
class LocalizationPrecision(Plugin):
    """The precision the data shows, beside the precision the fitter expected."""

    Settings = PrecisionSettings
    version = "2"        # 2: finds a SMAPpy 3D fit's z precision, z_err_nm

    def run(self, ctx: Context, settings: PrecisionSettings) -> Result:
        if not (settings.pairwise or settings.crlb or settings.frc
                or settings.frc_axes):
            raise ValueError("nothing to measure: tick a method")
        sources: List[Tuple[str, Localizations, bool]] = []
        everything = len(ctx.selection) == len(ctx.locs)
        if settings.source in ("selection", "both") and not everything:
            ctx.selection.require(MIN_FOR_FIT, ctx.report, "a precision")
            sources.append((ctx.selection.name or "selection",
                            ctx.selection.apply(ctx.locs), True))
        if settings.source in ("all", "both") or everything:
            sources.append(("all localizations", ctx.locs, False))

        bounds = filter_bounds(ctx.session, ctx.layer)
        found: List[Measurement] = []
        for name, locs, filtered in sources:
            if not len(locs):
                raise ValueError(f"no localizations in {name}")
            ctx.report(f"{name}: {len(locs)} localizations")
            found.append(measure(
                locs, name=name, reach=settings.reach_nm,
                reach_z=settings.reach_z_nm, max_gap=settings.max_gap,
                pairwise=settings.pairwise, crlb=settings.crlb,
                frc=settings.frc, frc_pixel=settings.frc_pixel_nm,
                frc_blocks=settings.frc_blocks, frc_repeats=settings.frc_repeats,
                frc_axes=settings.frc_axes,
                bounds=bounds if filtered else {},
                max_frame_pairs=settings.max_frame_pairs,
                max_pairs=settings.max_pairs, seed=settings.seed,
                report=ctx.report))

        plots: Dict[str, object] = {}
        if settings.pairwise and any(m.displacements for m in found):
            plots["per axis"] = Plot(
                draw=lambda figure: draw_axes(figure, found, settings.bins),
                panels=max((len(m.axes) for m in found), default=1) or 1,
                size=(5.5, 6.0))
            if settings.max_gap > 1:
                plots["frame gap"] = Plot(draw=lambda ax: draw_gaps(ax, found),
                                          size=(5.5, 3.4))
        if settings.frc and any(m.frc is not None and len(m.frc.curve)
                                for m in found):
            plots["FRC"] = Plot(draw=lambda ax: draw_frc(ax, found),
                                size=(5.5, 3.4))
        if settings.frc_axes and any(m.fpc is not None and m.fpc.axes
                                     for m in found):
            plots["per axis FRC"] = Plot(draw=lambda ax: draw_planes(ax, found),
                                         size=(5.5, 3.4))
        if settings.crlb and any(m.crlb for m in found):
            plots["CRLB"] = Plot(
                draw=lambda figure: draw_crlb(figure, found, settings.bins),
                panels=max((len(m.crlb) for m in found), default=1) or 1,
                size=(5.5, 4.4))

        main = None
        if settings.pairwise and any(m.radial is not None for m in found):
            main = Plot(draw=lambda ax: draw_radial(ax, found, settings.bins),
                        size=(5.5, 3.4))
        elif plots:
            main = plots.pop(next(iter(plots)))

        data = {"measurements": found,
                "sigma": {m.name: (m.radial.sigma if m.radial and m.radial.ok
                                   else float("nan")) for m in found},
                "kappa": {m.name: (m.scaled.kappa if m.scaled and m.scaled.ok
                                   else float("nan")) for m in found},
                "resolution": {m.name: (m.frc.resolution if m.frc and m.frc.ok
                                        else float("nan")) for m in found},
                "per_axis": {m.name: ({name: curve.resolution
                                      for name, curve in m.fpc.axes.items()}
                                      if m.fpc else {}) for m in found},
                "amplitude": {m.name: (m.photon_law.amplitude
                                       if m.photon_law and m.photon_law.ok
                                       else float("nan")) for m in found}}
        return Result(text=summary(found), data=data, plot=main, plots=plots,
                      settings=settings)

    def keep(self, result: Result):
        """The numbers, without the pairs: a precision is worth reopening.

        Plain and JSON-able, and the displacements themselves are left behind --
        they are millions of rows that the file can produce again.
        """
        kept = []
        for measurement in result.data.get("measurements") or []:
            entry = {"name": measurement.name, "n_locs": measurement.n_locs,
                     "notes": list(measurement.notes),
                     "photons": dict(measurement.photons),
                     "axes": {axis: fit.sigma
                              for axis, fit in measurement.axes.items() if fit.ok},
                     "by_gap": {str(gap): fit.sigma
                                for gap, fit in measurement.by_gap.items() if fit.ok},
                     "gap_line": dict(measurement.gap_line)}
            if measurement.radial is not None and measurement.radial.ok:
                entry.update(sigma=measurement.radial.sigma,
                             sigma_error=measurement.radial.sigma_error,
                             fraction=measurement.radial.fraction)
            if measurement.photon_law is not None and measurement.photon_law.ok:
                entry.update(amplitude=measurement.photon_law.amplitude)
            if measurement.scaled is not None and measurement.scaled.ok:
                entry.update(kappa=measurement.scaled.kappa,
                             kappa_error=measurement.scaled.kappa_error,
                             crlb_of_pairs=measurement.scaled.crlb)
            if measurement.fpc is not None and measurement.fpc.ok:
                entry["frc_per_axis"] = {
                    name: curve.resolution
                    for name, curve in measurement.fpc.axes.items() if curve.ok}
            if measurement.frc is not None and measurement.frc.ok:
                entry.update(frc_resolution=measurement.frc.resolution,
                             frc_error=measurement.frc.error,
                             frc_pixelsize=measurement.frc.pixelsize)
            entry["crlb"] = {
                key: {name: value for name, value in stats.items()
                      if isinstance(value, (int, float, str, bool))}
                for key, stats in measurement.crlb.items()}
            kept.append(entry)
        return {"measurements": kept} if kept else None
