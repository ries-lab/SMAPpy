"""Profiles along a line ROI, fitted without binning them.

Draw a line ROI over a filament, a membrane or a pair of membranes, and this
takes the localizations inside it, projects them onto the line's own
coordinates -- ``along`` the line and ``across`` it -- and fits the
distribution of one of those coordinates with a model:

* **Gaussian** -- one structure: a filament, a single membrane seen edge-on.
* **two Gaussians** -- two of them, and the number that is wanted is the
  distance between them (a nuclear envelope, the two leaflets, a doublet).
* **step (error function)** -- an edge: a region that is uniformly labelled on
  one side and empty on the other, blurred by the localization error.

SMAP's ``Analyze/measure/lineprofile`` is the original, and it fits the
*histogram*.  That is the one thing not ported.

How to fit a profile of few localizations
-----------------------------------------

A histogram of thirty localizations is mostly a statement about the bin width:
move the edges by half a bin and the fitted width changes by more than its
error bar.  Two things are wrong with fitting one.  The binning throws
information away, and least squares on Poisson counts is the wrong estimator
where the counts are small -- an empty bin carries as much information as a
full one, and a weight of ``1/sqrt(N)`` cannot express that.

So the default here is **unbinned maximum likelihood** (`method="mle"`): the
model is a probability density over the ROI window, and what is maximized is
``sum(log p(t_i))`` over the localizations themselves.  No bins exist; the
histogram in the figure is drawing only, and changing its bin width changes no
fitted number.  The window is part of the likelihood -- each density is
normalized over the ROI's own width -- so the truncation the ROI imposes costs
no bias.

Two refinements matter more than the estimator when there are few
localizations:

* **Each localization brings its own precision** (`use_precision`).  A
  localization is not a point but a Gaussian of width ``loc_precision_nm``, so
  the density is the structure convolved with *that* localization's error:
  width ``sqrt(s^2 + sigma_i^2)``.  The fitted ``s`` is then the width of the
  structure rather than of the picture of it, and for a step or a pair of
  points it is the difference between a number and a number that still has the
  microscope in it.  With few localizations this is most of the accuracy
  available: the precisions are known, and a fit that ignores them spends
  parameters re-measuring them.
* **A uniform background fraction** (`background`).  Unspecific localizations
  inside the ROI are flat, and a Gaussian asked to explain them alone comes
  back too wide.  One extra parameter, bounded to [0, 1].

What is reported is the fit's log-likelihood with AIC and BIC, so "one
Gaussian or two?" is answered by a number rather than by eye; `model="all"`
fits all three and prints the comparison.  Parameter errors come from the
curvature of the likelihood at its maximum (the observed Fisher information),
which is the usual asymptotic approximation and is optimistic for very small
samples -- with fewer than ~50 localizations read them as an order of
magnitude, and for a number that has to hold up, bootstrap the localizations.

`method="binned"` keeps the old way for comparison, as Poisson-weighted least
squares on the histogram rather than SMAP's plain least squares.  Even then
the likelihood that is reported is the unbinned one at the fitted parameters,
so the two methods -- and every model -- are compared on the same scale.

Everything here is a module-level function over arrays: `project` and
`fit_profile` need no session, no ROI and no window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..locs import Localizations
from ..regions import Region
from ..render import positions
from . import Context, Plot, Plugin, Result, param, register

PRECISION_FIELDS = ("loc_precision_nm", "loc_precision_pix")
PRECISION_Z_FIELDS = ("loc_precision_z_nm",)
MIN_FOR_FIT = 8             # below this the maximum is not where the data is
THIN = 30                   # below this the error bars are worth a warning
TINY = 1e-300
SQRT_2PI = np.sqrt(2 * np.pi)
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))
# how many distinct precisions the drawn curve averages over: the density is
# a mixture over the localizations' own widths, and 200 quantiles of them draw
# the same line as ten thousand
CURVE_SAMPLES = 200


# ------------------------------------------------------------- the geometry

def line_ends(region: Region) -> Tuple[np.ndarray, np.ndarray, float]:
    """The segment and the width of a line ROI, from its four corners.

    `Region.line` stores ``[p0+n, p1+n, p1-n, p0-n]``, so the ends are the
    midpoints of the two short sides and nothing has to be remembered.
    """
    if region.kind != "line" or len(region.points) != 4:
        raise ValueError(f"a line ROI is wanted, this is a {region.kind} ROI")
    points = np.asarray(region.points, float)
    p0 = 0.5 * (points[0] + points[3])
    p1 = 0.5 * (points[1] + points[2])
    return p0, p1, float(region.width)


def project(x, y, p0, p1) -> Tuple[np.ndarray, np.ndarray]:
    """Positions in the line's own coordinates: along it, and across it.

    ``along`` is measured from the first end, ``across`` from the line itself,
    positive to its left.  Both in the units the table came in.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    direction = p1 - p0
    length = float(np.hypot(*direction))
    if length <= 0:
        raise ValueError("the line ROI has no length")
    unit = direction / length
    dx, dy = x - p0[0], y - p0[1]
    along = dx * unit[0] + dy * unit[1]
    across = -dx * unit[1] + dy * unit[0]
    return along, across


def column(locs: Localizations, names: Sequence[str]) -> Optional[str]:
    """The first of ``names`` the table has, or None."""
    return next((n for n in names if n in locs), None)


# --------------------------------------------------------------- the models
#
# Each model is a probability *density* over the window, so a fit is a
# likelihood and two models are comparable.  A density takes the structural
# width `s` and the per-localization precisions separately, because what is
# fitted is the structure and what is known is the precision: the two add in
# quadrature and only their sum is ever seen in the picture.

def _widths(s: float, precision: Optional[np.ndarray]) -> np.ndarray:
    """The Gaussian each localization actually contributes."""
    s = max(float(s), 0.0)
    if precision is None:
        return np.atleast_1d(np.float64(max(s, 1e-6)))
    return np.maximum(np.hypot(s, np.asarray(precision, float)), 1e-6)


def _phi(z):
    return np.exp(-0.5 * z * z) / SQRT_2PI


def _Phi(z):
    from scipy.special import erf
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def gauss_density(t, mu, w, window) -> np.ndarray:
    """A Gaussian of width ``w``, normalized over the window.

    ``w`` is per localization when the precisions are folded in, and one
    number when they are not; numpy broadcasts either.
    """
    lo, hi = window
    z = (t - mu) / w
    norm = _Phi((hi - mu) / w) - _Phi((lo - mu) / w)
    return _phi(z) / w / np.maximum(norm, 1e-12)


def step_density(t, mu, w, window, side: float = 1.0) -> np.ndarray:
    """A half-plane of uniform density with an edge at ``mu``, blurred by ``w``.

    ``side = +1`` rises with t, ``-1`` falls.  The normalization is the
    integral of the error function over the window, which is closed form --
    ``(t-mu) Phi + w phi`` -- so no quadrature is needed and the density stays
    exact for a window that cuts the plateau short.
    """
    lo, hi = window
    shape = _Phi(side * (t - mu) / w)

    def integral(edge):
        z = (edge - mu) / w
        return (edge - mu) * _Phi(z) + w * _phi(z)

    rising = integral(hi) - integral(lo)
    norm = rising if side > 0 else (hi - lo) - rising
    return shape / np.maximum(norm, 1e-12)


@dataclass
class Model:
    """One shape: its parameters, where to start, and what it means."""
    key: str
    label: str
    names: Tuple[str, ...]
    start: Callable                 # start(t, precision, window) -> list
    bounds: Callable                # bounds(t, window) -> [(lo, hi), ...]
    shape: Callable                 # shape(params, t, precision, window) -> density
    describe: Callable              # describe(params, errors) -> str
    variants: Tuple[Dict, ...] = ({},)   # fits tried, best likelihood wins
    # Which parameters are widths that add to the localization precision in
    # quadrature, and so are fitted as *variances*.  A width enters the model
    # as sqrt(s^2 + sigma_i^2), which near s = 0 is sigma_i + s^2/2 sigma_i --
    # flat in s.  So s = 0 is a stationary point of the likelihood whatever
    # the data says, and a gradient method started above it walks down to the
    # bound and stops there, reporting a structure of zero width.  In s^2 the
    # same point has a slope, and the fit finds the maximum that is there.
    squares: Tuple[int, ...] = ()


def _spread(t) -> float:
    """A robust width to start from: the MAD, which a tail cannot inflate."""
    return float(max(1.4826 * np.median(np.abs(t - np.median(t))), 1e-3))


def _fwhm(s: float) -> float:
    return FWHM_PER_SIGMA * s


def _pm(value: float, error: float, digits: int = 1) -> str:
    if not np.isfinite(error):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} +- {error:.{digits}f}"


GAUSS = Model(
    key="gauss", label="Gaussian",
    names=("centre", "sigma"),
    start=lambda t, prec, window: [float(np.median(t)), _spread(t)],
    bounds=lambda t, window: [window, (0.0, window[1] - window[0])],
    shape=lambda p, t, prec, window: gauss_density(t, p[0],
                                                   _widths(p[1], prec), window),
    describe=lambda p, e: (f"centre {_pm(p[0], e[0])}, sigma {_pm(p[1], e[1])} "
                           f"(FWHM {_fwhm(p[1]):.1f})"),
    squares=(1,),
)

TWO_GAUSS = Model(
    key="two_gauss", label="two Gaussians",
    names=("centre", "distance", "sigma", "fraction"),
    start=lambda t, prec, window: [float(np.median(t)), 2 * _spread(t),
                                   0.5 * _spread(t), 0.5],
    bounds=lambda t, window: [window, (0.0, window[1] - window[0]),
                              (0.0, window[1] - window[0]), (0.0, 1.0)],
    shape=lambda p, t, prec, window: (
        p[3] * gauss_density(t, p[0] - 0.5 * p[1], _widths(p[2], prec), window)
        + (1 - p[3]) * gauss_density(t, p[0] + 0.5 * p[1],
                                     _widths(p[2], prec), window)),
    describe=lambda p, e: (f"distance {_pm(p[1], e[1])}, sigma {_pm(p[2], e[2])}, "
                           f"centre {_pm(p[0], e[0])}, "
                           f"weights {p[3]:.2f}/{1 - p[3]:.2f}"),
    squares=(2,),
)

STEP = Model(
    key="step", label="step (error function)",
    names=("edge", "sigma"),
    start=lambda t, prec, window: [float(np.median(t)), _spread(t)],
    bounds=lambda t, window: [window, (0.0, window[1] - window[0])],
    shape=lambda p, t, prec, window, side=1.0: step_density(
        t, p[0], _widths(p[1], prec), window, side),
    describe=lambda p, e: f"edge at {_pm(p[0], e[0])}, blur sigma {_pm(p[1], e[1])}",
    # which way the edge faces is not a parameter to be walked to: it is one
    # bit, and both are cheap to try
    variants=({"side": 1.0}, {"side": -1.0}),
    squares=(1,),
)

MODELS: Dict[str, Model] = {m.key: m for m in (GAUSS, TWO_GAUSS, STEP)}




# ------------------------------------------------------------------ fitting

@dataclass
class Fit:
    """A fitted model: its numbers, its quality, and the curve to draw."""
    model: str
    label: str
    names: Tuple[str, ...]
    params: np.ndarray
    errors: np.ndarray
    background: float               # the uniform fraction, 0 when not fitted
    background_error: float
    n: int
    log_likelihood: float           # always the unbinned one, whatever fitted
    method: str
    with_precision: bool
    window: Tuple[float, float]
    extra: Dict = field(default_factory=dict)
    curve: Optional[Callable] = None     # curve(t) -> density per unit length

    @property
    def k(self) -> int:
        """Free parameters, for the information criteria."""
        return len(self.params) + (1 if self.extra.get("has_background") else 0)

    @property
    def aic(self) -> float:
        return 2 * self.k - 2 * self.log_likelihood

    @property
    def bic(self) -> float:
        return self.k * np.log(max(self.n, 1)) - 2 * self.log_likelihood

    def values(self) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.params)}

    def uncertainties(self) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.errors)}

    @property
    def summary(self) -> str:
        text = MODELS[self.model].describe(self.params, self.errors)
        if self.extra.get("has_background"):
            text += f", background {100 * self.background:.0f}%"
        if self.extra.get("side", 1.0) < 0:
            text += ", falling"
        return f"{self.label}: {text}"


def _density_of(model: Model, params, window, background: float,
                has_background: bool, variant: Dict) -> Callable:
    """The density of one model at given parameters.

    Takes the precisions as an argument rather than closing over them, so the
    same function serves the likelihood -- one precision per localization --
    and the drawn curve, which averages over them by broadcasting.
    """
    lo, hi = window
    flat = 1.0 / (hi - lo)

    def density(t, precision):
        shape = model.shape(params, t, precision, window, **variant)
        if not has_background:
            return shape
        return background * flat + (1 - background) * shape

    return density


def _curve_of(density: Callable, precision) -> Callable:
    """The density as a histogram of these localizations would follow it.

    With per-localization precisions the model is a *mixture*: each
    localization is drawn from the structure smoothed by its own error, so
    what a histogram of all of them follows is the average over the sample.
    The precisions are sampled by quantile, which draws the same line as all
    of them and costs the same whatever the table's size.
    """
    if precision is None:
        return lambda t: np.asarray(density(np.asarray(t, float), None), float)
    sampled = np.asarray(precision, float)
    if len(sampled) > CURVE_SAMPLES:
        sampled = np.percentile(sampled, np.linspace(0, 100, CURVE_SAMPLES))

    def curve(t):
        t = np.atleast_1d(np.asarray(t, float))
        return np.asarray(density(t[None, :], sampled[:, None]), float).mean(axis=0)

    return curve


def _hessian(f: Callable, x: np.ndarray) -> np.ndarray:
    """Central differences, with a step scaled to each parameter."""
    n = len(x)
    h = np.maximum(np.abs(x), 1.0) * 1e-4
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            up_up, up_down, down_up, down_down = (x.copy() for _ in range(4))
            up_up[i] += h[i]; up_up[j] += h[j]
            up_down[i] += h[i]; up_down[j] -= h[j]
            down_up[i] -= h[i]; down_up[j] += h[j]
            down_down[i] -= h[i]; down_down[j] -= h[j]
            H[i, j] = H[j, i] = ((f(up_up) - f(up_down) - f(down_up)
                                  + f(down_down)) / (4 * h[i] * h[j]))
    return H


def _errors_from(hessian: np.ndarray,
                 free: Optional[np.ndarray] = None) -> np.ndarray:
    """Standard errors from the curvature, NaN where it says nothing.

    ``free`` says which parameters the fit actually moved.  One that ended on
    its bound -- a width the data puts at zero, a background fraction that
    went to nothing -- has no error bar to quote and would spoil the others'
    by making the matrix singular, so it is left out of the inversion and the
    rest are the errors *given* that it is where it is.

    A missing error bar is never a failed fit: the value is still the maximum
    of the likelihood, and printing it alone is more use than raising.
    """
    n = len(hessian)
    out = np.full(n, np.nan)
    if free is None:
        free = np.ones(n, bool)
    keep = np.flatnonzero(np.asarray(free, bool))
    block = hessian[np.ix_(keep, keep)]
    if not keep.size or not np.all(np.isfinite(block)):
        return out
    try:
        covariance = np.linalg.inv(block)
    except np.linalg.LinAlgError:
        return out
    diagonal = np.diag(covariance)
    out[keep] = np.where(diagonal > 0, np.sqrt(np.abs(diagonal)), np.nan)
    return out


def histogram(t, window, bin_size: float):
    """Counts and edges over the window: for the picture, and for `binned`."""
    lo, hi = window
    n_bins = max(int(round((hi - lo) / max(bin_size, 1e-9))), 1)
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(np.asarray(t, float), bins=edges)
    return counts.astype(float), edges


def auto_bin(t, window) -> float:
    """A bin width for the picture: Freedman-Diaconis, bounded by the window.

    Drawing only -- no fitted number depends on it -- but a histogram of two
    bins or of four hundred says nothing either.
    """
    t = np.asarray(t, float)
    lo, hi = window
    span = hi - lo
    if len(t) < 4:
        return span / 10
    q1, q3 = np.percentile(t, [25, 75])
    width = 2 * (q3 - q1) / len(t) ** (1 / 3)
    if not np.isfinite(width) or width <= 0:
        width = span / 20
    return float(np.clip(width, span / 200, span / 5))


def _clean_precision(precision, n: int):
    """The precisions, or None if there are none worth using."""
    if precision is None:
        return None
    values = np.asarray(precision, float).reshape(-1)
    if values.shape != (n,):
        raise ValueError(f"{values.shape[0]} precisions for {n} localizations")
    good = np.isfinite(values) & (values > 0)
    if not good.any():
        return None
    # a handful of missing precisions become the median rather than dropping
    # their localizations: the position is still a measurement
    return np.where(good, values, float(np.median(values[good])))


def _square(value: float, yes: bool) -> float:
    return float(value) ** 2 if yes else float(value)


def _to_fit(vector: np.ndarray, squares: Sequence[int]) -> np.ndarray:
    """Model parameters as the optimizer sees them: widths become variances."""
    out = np.asarray(vector, float).copy()
    out[squares] = out[squares] ** 2
    return out


def _from_fit(vector: np.ndarray, squares: Sequence[int]) -> np.ndarray:
    """The optimizer's parameters as the model means them."""
    out = np.asarray(vector, float).copy()
    out[squares] = np.sqrt(np.maximum(out[squares], 0.0))
    return out


def _in_widths(errors: np.ndarray, params: np.ndarray,
               squares: Sequence[int]) -> np.ndarray:
    """Errors carried from the variances back to the widths.

    ``s = sqrt(v)``, so ``ds = dv / 2s`` -- and at ``s = 0`` there is no
    linear error to quote, which is the honest answer for a width the data
    puts at the bound.
    """
    out = np.asarray(errors, float).copy()
    for i in squares:
        if i < len(out):
            s = params[i]
            out[i] = out[i] / (2 * s) if s > 0 else np.nan
    return out


def fit_profile(t, precision=None, model: str = "gauss",
                window: Optional[Tuple[float, float]] = None,
                method: str = "mle", background: bool = True,
                bin_size: float = 0.0) -> Fit:
    """Fit one coordinate of the localizations with ``model``.

    ``t`` are the positions themselves, never a histogram, and ``precision``
    is each localization's own error -- or None to fit one width for all of
    them.  ``window`` is where a localization could have been (the ROI), and
    is part of the likelihood, so the truncation costs no bias.

    ``method``: ``"mle"`` maximizes the unbinned likelihood; ``"binned"`` is
    Poisson-weighted least squares on a histogram of ``bin_size``, kept for
    comparison with the old way.  Either way the likelihood reported is the
    unbinned one at the fitted parameters, so every fit is on one scale.
    """
    from scipy.optimize import least_squares, minimize

    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    if len(t) < MIN_FOR_FIT:
        raise ValueError(f"{len(t)} localizations: too few for a profile fit "
                         f"(at least {MIN_FOR_FIT})")
    if model not in MODELS:
        raise ValueError(f"no model {model!r}: {', '.join(MODELS)}")
    spec = MODELS[model]
    if window is None:
        pad = 0.05 * (float(t.max() - t.min()) + 1e-9)
        window = (float(t.min()) - pad, float(t.max()) + pad)
    window = (float(window[0]), float(window[1]))
    precision = _clean_precision(precision, len(t))

    best: Optional[Fit] = None
    for variant in spec.variants:
        start = list(spec.start(t, precision, window))
        bounds = list(spec.bounds(t, window))
        if background:
            start.append(0.05)
            bounds.append((0.0, 0.95))
        squares = list(spec.squares)
        start = _to_fit(np.asarray(start, float), squares)
        bounds = [(_square(lo, i in squares), _square(hi, i in squares))
                  for i, (lo, hi) in enumerate(bounds)]
        lower = np.array([b[0] for b in bounds], float)
        upper = np.array([b[1] for b in bounds], float)
        start = np.clip(start, lower, upper)

        def split(vector):
            vector = _from_fit(np.asarray(vector, float), squares)
            return ((vector[:-1], float(vector[-1])) if background
                    else (vector, 0.0))

        def nll(vector) -> float:
            params, fraction = split(np.asarray(vector, float))
            density = _density_of(spec, params, window, fraction, background,
                                  variant)
            return float(-np.sum(np.log(np.maximum(density(t, precision), TINY))))

        if method == "binned":
            counts, edges = histogram(t, window, bin_size or auto_bin(t, window))
            centres = 0.5 * (edges[:-1] + edges[1:])
            widths = np.diff(edges)
            total = float(counts.sum())

            def residuals(vector):
                params, fraction = split(np.asarray(vector, float))
                density = _density_of(spec, params, window, fraction,
                                      background, variant)
                expected = total * widths * _curve_of(density, precision)(centres)
                # Poisson weights rather than SMAP's plain least squares: at
                # ten counts a bin the two differ by more than the bin width
                return (counts - expected) / np.sqrt(np.maximum(expected, 1.0))

            found = least_squares(residuals, start, bounds=(lower, upper))
            vector = np.asarray(found.x, float)
        else:
            found = minimize(nll, start, method="L-BFGS-B",
                             bounds=list(zip(lower, upper)))
            vector = np.asarray(found.x, float)
            if not np.all(np.isfinite(vector)):
                vector = start

        params, fraction = split(vector)
        span = np.maximum(upper - lower, 1e-12)
        free = (vector > lower + 1e-6 * span) & (vector < upper - 1e-6 * span)
        errors = _in_widths(_errors_from(_hessian(nll, vector), free),
                            _from_fit(vector, squares), squares)
        density = _density_of(spec, params, window, fraction, background, variant)
        fit = Fit(model=spec.key, label=spec.label, names=spec.names,
                  params=np.asarray(params, float),
                  errors=np.asarray(errors[:len(params)], float),
                  background=fraction,
                  background_error=float(errors[-1]) if background else 0.0,
                  n=len(t), log_likelihood=-nll(vector), method=method,
                  with_precision=precision is not None, window=window,
                  extra={"has_background": background, **variant},
                  curve=_curve_of(density, precision))
        if best is None or fit.log_likelihood > best.log_likelihood:
            best = fit
    return best


def fit_models(t, precision=None, models: Sequence[str] = ("gauss",),
               **kwargs) -> List[Fit]:
    """Several models on the same data, best first by AIC.

    Comparing them is the point: "one structure or two?" is a question the
    likelihood answers, and by AIC rather than by likelihood alone, because
    the second Gaussian can only ever fit better.
    """
    found = [fit_profile(t, precision, model=m, **kwargs) for m in models]
    return sorted(found, key=lambda f: f.aic)


# ------------------------------------------------------------ the profiles

AXES = {"across": "position across the line",
        "along": "position along the line",
        "z": "z"}


@dataclass
class Profile:
    """One coordinate of the localizations in a line ROI, and its window."""
    key: str
    label: str
    unit: str
    values: np.ndarray
    window: Tuple[float, float]
    precision: Optional[np.ndarray] = None
    note: str = ""              # why there is no precision, when there is none


def profiles(locs: Localizations, region: Region,
             length: float = 0.0,
             z_window: Optional[Tuple[float, float]] = None) -> Dict[str, Profile]:
    """The line's coordinates for every localization inside it.

    ``length`` overrides the ROI's own length with a window of that size
    centred on it, which is what makes two ROIs on two images comparable; the
    localizations outside it are dropped from *all* of the profiles, so the
    across profile is of the same localizations as the along profile.
    """
    p0, p1, width = line_ends(region)
    x, y = positions(locs)
    along, across = project(x, y, p0, p1)
    full = float(np.hypot(*(p1 - p0)))
    span = full if length <= 0 else float(length)
    window_along = (0.5 * (full - span), 0.5 * (full + span))
    window_across = (-0.5 * width, 0.5 * width)

    inside = ((along >= window_along[0]) & (along <= window_along[1])
              & (across >= window_across[0]) & (across <= window_across[1]))
    along, across = along[inside], across[inside]
    kept = locs[inside]

    lateral = column(kept, PRECISION_FIELDS)
    precision = np.asarray(kept[lateral], float) if lateral else None
    unit = "nm" if "x_nm" in kept else "pix"
    missing = "" if lateral else ("no localization precision in the table: one "
                                  "width is fitted for every localization")

    found = {
        "across": Profile("across", AXES["across"], unit, across, window_across,
                          precision, missing),
        "along": Profile("along", AXES["along"], unit, along, window_along,
                         precision, missing),
    }
    if "z_nm" in kept:
        z = np.asarray(kept["z_nm"], float)
        finite = z[np.isfinite(z)]
        if z_window is None and finite.size:
            pad = 0.05 * (float(finite.max() - finite.min()) + 1e-9)
            z_window = (float(finite.min()) - pad, float(finite.max()) + pad)
        z_precision_name = column(kept, PRECISION_Z_FIELDS)
        found["z"] = Profile(
            "z", AXES["z"], "nm", z, z_window or (-1.0, 1.0),
            np.asarray(kept[z_precision_name], float) if z_precision_name else None,
            "" if z_precision_name else ("no axial precision in the table: one "
                                         "width is fitted for every localization"))
    return found


# ---------------------------------------------------------------- drawing

def draw_profile(ax, profile: Profile, fits: Sequence[Fit], bin_size: float) -> None:
    """The histogram, with every fitted model over it.

    The histogram is the picture and the fits are of the localizations, so the
    curve is not a fit *to these bars*: it is the density scaled by how many
    localizations and how wide a bin, which is why it can sit above an empty
    bin without anything being wrong.
    """
    counts, edges = histogram(profile.values, profile.window, bin_size)
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
           color="0.75", edgecolor="0.45", linewidth=0.4)
    grid = np.linspace(profile.window[0], profile.window[1], 400)
    colours = ("#d62728", "#1f77b4", "#2ca02c")
    for fit, colour in zip(fits, colours):
        ax.plot(grid, len(profile.values) * bin_size * fit.curve(grid),
                color=colour, linewidth=1.5, label=fit.label)
    if len(fits) > 1:
        ax.legend(fontsize=6.5, frameon=False)
    ax.set_xlabel(f"{profile.label} ({profile.unit})")
    ax.set_ylabel("localizations")
    if fits:
        ax.set_title(fits[0].summary, fontsize=7.5, color="0.25")


def draw_scatter(figure, found: Dict[str, Profile]) -> None:
    """Where the localizations are, in the line's coordinates.

    The profile is a projection and a projection hides things -- a filament
    that leaves the ROI half way along, two structures that cross -- so the
    scatter is next to it rather than behind a menu.
    """
    keys = ["across"] + (["z"] if "z" in found else [])
    axes = figure.subplots(len(keys), 1, squeeze=False).ravel()
    along = found["along"]
    for ax, key in zip(axes, keys):
        other = found[key]
        ax.plot(along.values, other.values, ".", markersize=2, color="#1f77b4")
        ax.set_xlabel(f"{along.label} ({along.unit})")
        ax.set_ylabel(f"{other.label} ({other.unit})")
        ax.set_xlim(*along.window)
        if key == "across":
            ax.set_ylim(*other.window)
            ax.set_aspect("equal", adjustable="box")


# --------------------------------------------------------------- the plugin

@dataclass
class LineProfileSettings:
    axis: str = param("across", label="profile",
                      choices=(("across", "across the line"),
                               ("along", "along the line"),
                               ("z", "z")),
                      help="which coordinate is histogrammed and fitted")
    model: str = param("gauss", label="model",
                       choices=(("gauss", "Gaussian"),
                                ("two_gauss", "two Gaussians (a distance)"),
                                ("step", "step (error function)"),
                                ("all", "all three, compared")),
                       help="what the profile is expected to be")
    method: str = param("mle", label="fit",
                        choices=(("mle", "unbinned, maximum likelihood"),
                                 ("binned", "binned, least squares")),
                        help="the unbinned fit uses the localizations "
                             "themselves, so no fitted number depends on the "
                             "bin width; with few localizations it is the "
                             "only one worth trusting")
    use_precision: bool = param(True, label="use localization precision",
                                help="each localization is a Gaussian of its "
                                     "own precision, so the fitted width is "
                                     "the structure's rather than the "
                                     "picture's")
    background: bool = param(True, label="uniform background",
                             help="a flat fraction of unspecific "
                                  "localizations; without it a Gaussian asked "
                                  "to explain them comes back too wide")
    bin_nm: float = param(0.0, label="bin", unit="nm", min=0.0,
                          help="0: chosen from the data.  Drawing only, "
                               "unless the binned fit is chosen")
    length_nm: float = param(0.0, label="length", unit="nm", min=0.0,
                             advanced=True,
                             help="0: the ROI's own length.  Otherwise a "
                                  "window of this length centred on it, so "
                                  "two ROIs can be compared directly")


@register("Analysis/Measure/Line Profile")
class LineProfile(Plugin):
    """Profiles across a line ROI, fitted without binning them."""

    Settings = LineProfileSettings
    version = "1"

    def run(self, ctx: Context, settings: LineProfileSettings) -> Result:
        region = _line_roi(ctx)
        ctx.selection.require(MIN_FOR_FIT, ctx.report, "a profile fit")
        locs = ctx.selection.apply(ctx.locs)
        found = profiles(locs, region, length=settings.length_nm,
                         z_window=_z_window(ctx))
        if settings.axis not in found:
            raise ValueError(f"no {settings.axis} profile: the table has "
                             f"{', '.join(sorted(locs.keys()))}")
        profile = found[settings.axis]
        if len(profile.values) < MIN_FOR_FIT:
            raise ValueError(f"{len(profile.values)} localizations in the ROI: "
                             f"too few for a profile fit (at least {MIN_FOR_FIT})")
        if len(profile.values) < THIN:
            ctx.report(f"only {len(profile.values)} localizations: the fitted "
                       "values stand, their error bars are optimistic")

        precision = profile.precision if settings.use_precision else None
        bin_size = settings.bin_nm or auto_bin(profile.values, profile.window)
        models = (tuple(MODELS) if settings.model == "all" else (settings.model,))
        ctx.report(f"fitting {len(profile.values)} localizations "
                   f"({'unbinned' if settings.method == 'mle' else 'binned'})")
        fits = fit_models(profile.values, precision, models=models,
                          window=profile.window, method=settings.method,
                          background=settings.background, bin_size=bin_size)

        lines = [f"{len(profile.values)} localizations in the {region}, "
                 f"{profile.label}"]
        if profile.note and settings.use_precision:
            lines.append(profile.note)
        for fit in fits:
            lines.append(f"{fit.summary}")
            lines.append(f"    log L {fit.log_likelihood:.1f}, "
                         f"AIC {fit.aic:.1f}, BIC {fit.bic:.1f}")
        if len(fits) > 1:
            lines.append(f"best by AIC: {fits[0].label} "
                         f"(by {fits[1].aic - fits[0].aic:.1f})")

        def profile_plot(ax) -> None:
            draw_profile(ax, profile, fits, bin_size)

        def scatter_plot(figure) -> None:
            draw_scatter(figure, found)

        panels = 2 if "z" in found else 1
        return Result(
            text="\n".join(lines), settings=settings,
            data={"fits": {f.model: f for f in fits},
                  "values": {f.model: f.values() for f in fits},
                  "errors": {f.model: f.uncertainties() for f in fits},
                  "profiles": found, "bin": bin_size, "n": len(profile.values)},
            plot=profile_plot,
            plots={"scatter": Plot(draw=scatter_plot, panels=panels,
                                   size=(5.0, 2.6 * panels))})


def _line_roi(ctx: Context) -> Region:
    """The line ROI this plugin measures, refused clearly when there is none."""
    roi = getattr(ctx.session, "roi", None)
    if not isinstance(roi, Region):
        found = getattr(ctx.selection, "roi", None)
        roi = found if isinstance(found, Region) else None
    if roi is None:
        raise ValueError("no ROI: draw a line ROI in the render window")
    if roi.kind != "line":
        raise ValueError(f"the ROI is a {roi.kind}: a line ROI is what has a "
                         "direction to take a profile along")
    if roi.width <= 0:
        raise ValueError("the line ROI has no width: widen it to take in "
                         "localizations")
    return roi


def _z_window(ctx: Context) -> Optional[Tuple[float, float]]:
    """Where z could have been: the layer's own z range, when there is one.

    Taking it from the data instead would put the window's edges *at* the
    outermost localizations, and the likelihood would read that as a structure
    that stops exactly there.
    """
    session = ctx.session
    if session is None or not hasattr(session, "z_range"):
        return None
    try:
        low, high = session.z_range()
    except Exception:
        return None
    return (float(low), float(high)) if high > low else None
