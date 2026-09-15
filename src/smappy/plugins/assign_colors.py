"""Colours from the photon split between two detection channels.

A ratiometric two-colour experiment reads a molecule's species from how its
emission divides between the channels.  The dual-channel global fit leaves that
split free while sharing x, y and z (`dualfit.LINK_XYZ`), so the two photon
numbers it returns are the colour; this turns them into a ``channel`` label.

    r = (N1 - N2) / (N1 + N2)

is the coordinate throughout: bounded, symmetric between the channels, and
with a noise that has a closed form -- ``var(r) = (1 - r^2) / N`` for pure shot
noise, which is exactly the binomial variance of the split.  Two ways to cut it
up:

* **minima** -- the modes of the histogram of r are the species, the lowest
  point between two modes is the boundary, and ``dr`` around a boundary is left
  unassigned.  One decision for every localization, however bright.
* **probabilistic** -- each localization gets the posterior of each species
  given *its own* photon numbers and errors, and is assigned only if the best
  posterior reaches ``1 - crosstalk`` *and* the observed split is within
  ``tolerance`` sigma of that species' expected ratio.  The first test narrows
  the ambiguous band as 1/N, so a bright localization is assigned right up to
  the boundary and a dim one is not assigned at all.  The second is the
  question a posterior cannot ask -- whether the localization is either colour
  -- and is what refuses the valley between the modes, where a posterior alone
  picks a winner from two hypotheses that are both 20 sigma away.

`docs/dual_color_assignment.md` derives both, and in particular why the allowed
crosstalk is an upper bound on the expected fraction of misassigned
localizations among the assigned ones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..locs import Localizations
from . import Context, Plugin, Result, param, register

# candidate column pairs, in the order they are looked for: what the paired
# fit writes first, then the 1-based spelling a SMAP or hand-made table uses
CHANNEL_COLUMNS: Sequence[Tuple[str, str]] = (
    ("photons_ch0", "photons_ch1"),
    ("photons_ch1", "photons_ch2"),
    ("intensity_ch0", "intensity_ch1"),
    ("intensity_ch1", "intensity_ch2"),
)
# how far apart two maxima must be, in smoothing widths: below this they are
# one mode that the smoothing has not quite joined up
MIN_SEPARATION = 3.0
MAX_COLORS = 6
HWHM_PER_SIGMA = np.sqrt(2 * np.log(2))     # 1.1774: a Gaussian's HWHM
# a palette for the modes, in r order.  Red first because the channel the
# splitter sends the long wavelengths to is conventionally channel 1.
PALETTE = ("#d62728", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b")


def error_column(name: str) -> str:
    """The column holding ``name``'s fitted error.

    ``photons_ch0`` -> ``photons_err_ch0``, following `psf.GlobalSplinePSF`;
    anything else just gains the suffix.
    """
    head, sep, tail = name.rpartition("_ch")
    return f"{head}_err{sep}{tail}" if sep else f"{name}_err"


def channel_columns(locs: Localizations, first: Optional[str] = None,
                    second: Optional[str] = None) -> Tuple[str, str]:
    """Which two columns hold the per-channel photons.

    Named explicitly, or the first pair of `CHANNEL_COLUMNS` the table has.
    """
    if first or second:
        if not (first and second):
            raise ValueError("name both channel columns, or neither")
        for name in (first, second):
            if name not in locs:
                raise ValueError(f"no column {name!r}; the table has "
                                 f"{', '.join(sorted(locs.keys()))}")
        return first, second
    for pair in CHANNEL_COLUMNS:
        if all(name in locs for name in pair):
            return pair
    raise ValueError("no per-channel photon columns in the table: fit with "
                     "'Localize/Spline 3D 2C', or name the columns yourself")


@dataclass
class Ratios:
    """The colour coordinate and how well each localization measures it.

    ``n_eff`` is the localization's worth *in photons*: the total when the
    errors are Poisson, and less when background or EM gain has cost
    information (see the doc).  ``valid`` is where any of it means anything.
    """

    r: np.ndarray               # (n,) in [-1, 1], NaN where invalid
    total: np.ndarray           # (n,) N1 + N2
    n_eff: np.ndarray           # (n,) effective photons behind r
    valid: np.ndarray           # (n,) bool
    columns: Tuple[str, str]
    fitted_errors: bool         # were per-channel CRLBs used for n_eff?
    first: np.ndarray = None    # (n,) the two counts themselves, for the
    second: np.ndarray = None   # intensity plot

    @property
    def efficiency(self) -> float:
        """The median n_eff / total: what a photon is worth here.

        One number standing for the whole table, so that a decision can be
        drawn over a grid of intensities rather than only at the points that
        happen to have been measured.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(self.valid & (self.total > 0),
                             self.n_eff / self.total, np.nan)
        median = np.nanmedian(share) if np.any(np.isfinite(share)) else 1.0
        return float(np.clip(median, 1e-3, 1.0))


def ratios(locs: Localizations, first: Optional[str] = None,
           second: Optional[str] = None, use_errors: bool = True,
           min_photons: float = 0.0) -> Ratios:
    """r, the total, and the effective photon number behind r."""
    one, two = channel_columns(locs, first, second)
    n1 = np.asarray(locs[one], dtype=float)
    n2 = np.asarray(locs[two], dtype=float)
    total = n1 + n2
    valid = np.isfinite(n1) & np.isfinite(n2) & (total > 0)
    if min_photons > 0:
        valid &= total >= min_photons

    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(valid, (n1 - n2) / np.where(valid, total, 1.0), np.nan)
    # a fit may put a channel slightly below zero; the coordinate is still
    # meaningful at the end of its range, the variance formula is not
    r = np.clip(r, -1.0, 1.0)

    errors = [error_column(one), error_column(two)]
    fitted = use_errors and all(name in locs for name in errors)
    if fitted:
        s1 = np.asarray(locs[errors[0]], dtype=float)
        s2 = np.asarray(locs[errors[1]], dtype=float)
        good = valid & np.isfinite(s1) & np.isfinite(s2) & (s1 > 0) & (s2 > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            var_r = 4 * (n2 ** 2 * s1 ** 2 + n1 ** 2 * s2 ** 2) / total ** 4
            # N_eff is defined by matching the shot-noise form at the measured
            # point, which keeps var(r) = (1 - rho^2)/N_eff usable under any
            # hypothesis rho -- the plug-in variance would collapse at |r| = 1
            n_eff = np.where(good & (var_r > 0), (1 - r ** 2) / var_r, np.nan)
        # r exactly at an end leaves 1 - r^2 = 0 and no information at all;
        # the raw total is the honest fallback there, as it is for a localization
        # whose per-channel error did not come out of the fit
        n_eff = np.where(np.isfinite(n_eff) & (n_eff > 0), n_eff, total)
        fitted = bool(good.any())
    else:
        n_eff = total.copy()
    n_eff = np.where(valid, n_eff, np.nan)
    return Ratios(r=r, total=total, n_eff=n_eff, valid=valid,
                  columns=(one, two), fitted_errors=fitted, first=n1, second=n2)


@dataclass
class Modes:
    """What the histogram of r says the species are."""

    maxima: np.ndarray          # (k,) mode positions, ascending
    minima: np.ndarray          # (k-1,) boundaries between them
    prior: np.ndarray           # (k,) fraction of the sample in each mode
    centers: np.ndarray         # the histogram it was read from
    counts: np.ndarray
    density: np.ndarray         # smoothed counts

    def __len__(self) -> int:
        return len(self.maxima)


def _refine(index: int, y: np.ndarray, centers: np.ndarray) -> float:
    """A parabola through three bins, so a mode is not stuck on a bin centre."""
    if index <= 0 or index >= len(y) - 1:
        return float(centers[index])
    a, b, c = y[index - 1], y[index], y[index + 1]
    denominator = a - 2 * b + c
    if denominator == 0:
        return float(centers[index])
    shift = 0.5 * (a - c) / denominator
    return float(centers[index] + np.clip(shift, -1, 1) * (centers[1] - centers[0]))


def find_modes(r: np.ndarray, colors: int = 2, bins: int = 200,
               smoothing: float = 2.0,
               expected: Optional[Sequence[float]] = None) -> Modes:
    """The `colors` strongest modes of the histogram of r, and the dips between.

    The histogram is smoothed first: unsmoothed counts have a local maximum
    every few bins, and every one of them would be a species.  ``expected``
    takes the species' ratios as given -- measured on a single-label sample, or
    computed from the dyes' spectra and the splitter, as DECODE-Plex does --
    and only the boundaries and the abundances are then read off the histogram.
    """
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d

    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        raise ValueError("no localizations with two-channel photons to histogram")
    if not 1 <= colors <= MAX_COLORS:
        raise ValueError(f"colours must be between 1 and {MAX_COLORS}")
    counts, edges = np.histogram(r, bins=int(bins), range=(-1.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = (gaussian_filter1d(counts.astype(float), max(smoothing, 1e-3),
                                 mode="nearest") if smoothing > 0
               else counts.astype(float))

    if expected is not None:
        maxima = np.sort(np.asarray(expected, dtype=float))
        if not (len(maxima) and np.all(np.abs(maxima) <= 1)):
            raise ValueError("expected ratios must lie in [-1, 1]")
        chosen = [int(np.argmin(np.abs(centers - rho))) for rho in maxima]
    else:
        peaks = np.flatnonzero((density == maximum_filter1d(density, 3, mode="nearest"))
                               & (density > 0))
        separation = max(int(round(MIN_SEPARATION * smoothing)), 1)
        chosen = []
        for index in peaks[np.argsort(density[peaks])[::-1]]:
            if all(abs(index - taken) >= separation for taken in chosen):
                chosen.append(int(index))
            if len(chosen) == colors:
                break
        if len(chosen) < colors:
            raise ValueError(
                f"found {len(chosen)} mode(s) in the histogram of r, not "
                f"{colors}: lower the smoothing, raise the number of bins, ask "
                "for fewer colours, or give the expected ratios")
        chosen.sort()
        maxima = np.array([_refine(i, density, centers) for i in chosen])

    minima, cuts = [], []
    for (left, right), (rho_l, rho_r) in zip(zip(chosen, chosen[1:]),
                                             zip(maxima, maxima[1:])):
        if right - left < 2:            # two modes inside one bin: split them
            minima.append(float(rho_l + rho_r) / 2)
            cuts.append(minima[-1])
            continue
        # The middle of the lowest stretch, not its left edge: two well
        # separated species leave a wide valley of exactly zero, and the
        # boundary belongs in the middle of it rather than against a mode.
        # The lowest bins need not be contiguous -- anything in the valley,
        # a population of coincident emitters say, leaves empty stretches on
        # both sides of itself -- and the middle of *all* of them together
        # could land on that population.  So: the longest empty run, and the
        # one nearest halfway between the two modes if two runs tie.
        valley = density[left:right + 1]
        middle = (rho_l + rho_r) / 2
        def rank(run):
            centre = centers[left + (run[0] + run[1]) // 2]
            return run[1] - run[0], -abs(centre - middle)

        start, stop = max(_runs(valley <= valley.min()), key=rank)
        dip = left + (start + stop) // 2
        minima.append(_refine(dip, -density, centers))
        cuts.append(float(centers[dip]))
    minima = np.array(minima)
    # the prior is what the histogram gives away for free: how much of the
    # sample sits under each mode, between the dips that bound it
    counted = np.histogram(r, bins=np.concatenate(([-np.inf], cuts, [np.inf])))[0]
    prior = counted / max(counted.sum(), 1)
    return Modes(maxima=maxima, minima=minima, prior=prior.astype(float),
                 centers=centers, counts=counts, density=density)


def unclaimed_modes(modes: Modes, floor: float = 0.05
                    ) -> Sequence[Tuple[float, float]]:
    """Peaks between the chosen modes that were not chosen, as (r, height).

    A population sitting in a valley -- coincident emitters of both colours, a
    third dye, an artefact -- makes the model wrong rather than imprecise: the
    boundary either side of it is equally empty, so which one is taken is
    arbitrary, and whichever is taken swallows that population whole.  Worth
    saying out loud, since the histogram makes it obvious and a `channel`
    column does not.  ``floor`` is the height, as a fraction of the smaller
    neighbouring mode, below which a bump is noise rather than a population.
    """
    from scipy.ndimage import maximum_filter1d

    if len(modes) < 2:
        return []
    peaks = np.flatnonzero((modes.density == maximum_filter1d(modes.density, 3,
                                                              mode="nearest"))
                           & (modes.density > 0))
    claimed = [int(np.argmin(np.abs(modes.centers - rho))) for rho in modes.maxima]
    found = []
    for index in peaks:
        position = float(modes.centers[index])
        if not modes.maxima[0] < position < modes.maxima[-1]:
            continue
        if min(abs(index - taken) for taken in claimed) <= 1:
            continue
        side = int(np.searchsorted(modes.maxima, position))
        neighbours = modes.density[[claimed[side - 1], claimed[side]]]
        height = float(modes.density[index] / max(neighbours.min(), 1e-12))
        if height >= floor:
            found.append((position, height))
    return found


def assign_by_minima(r: np.ndarray, modes: Modes, exclusion: float = 0.0
                     ) -> np.ndarray:
    """1..k by which side of each minimum r falls, 0 within ``exclusion``."""
    r = np.asarray(r, dtype=float)
    channel = (np.searchsorted(modes.minima, r) + 1).astype(np.int32)
    channel[~np.isfinite(r)] = 0
    if exclusion > 0 and len(modes.minima):
        near = np.min(np.abs(r[:, None] - modes.minima[None, :]), axis=1)
        channel[np.isfinite(near) & (near < exclusion)] = 0
    return channel


def _concentration(rho: np.ndarray, spread: float) -> np.ndarray:
    """The Beta prior's concentration for a species whose own ratio wanders.

    ``spread`` is that wandering as a standard deviation in r.  A Beta with
    mean p = (1+rho)/2 has var(p) = p(1-p)/(kappa+1), and var(r) = 4 var(p),
    so kappa = (1 - rho^2)/spread^2 - 1 is the concentration that reproduces
    it.  A spread wider than the ratio axis allows at that mean is clamped:
    past that there is no Beta with that variance, and the species has stopped
    saying anything about the split.
    """
    return np.maximum((1 - rho ** 2) / max(spread, 1e-12) ** 2 - 1, 1e-3)


def log_likelihoods(r: np.ndarray, n_eff: np.ndarray, modes: Modes,
                    spread: float = 0.0) -> np.ndarray:
    """log P(the observed split | species), as (n, k), up to a common constant.

    The exact thing, not a Gaussian.  Photon counting is Poisson, and a Poisson
    pair factorizes into the total and the split,

        P(I1, I2 | lambda, c) = Poisson(N | lambda) Binomial(I1 | N, p_c),

    where only the second factor knows the colour and only the first knows the
    brightness.  So the brightness cancels out of the posterior without ever
    being modelled, and what is left is a binomial in the split -- whose log is
    *linear* in the two counts, ``I1 log p + I2 log(1-p)``, where a Gaussian's
    is quadratic in r.  The difference is small when both channels collect
    many photons and large when one collects few, which is exactly the
    ratiometric case worth having: a dye that sends 2% of its light to one
    channel.

    With ``spread``, p is itself drawn from a Beta of the same mean, and the
    split is beta-binomial -- the same closed form, and the same cancellation.
    """
    from scipy.special import gammaln

    r = np.clip(np.asarray(r, dtype=float), -1.0, 1.0)[:, None]
    n = np.asarray(n_eff, dtype=float)[:, None]
    rho = np.asarray(modes.maxima, dtype=float)[None, :]
    p = np.clip((1 + rho) / 2, 1e-12, 1 - 1e-12)
    # the effective counts: what n_eff photons split as r would have been
    first = n * (1 + r) / 2
    second = n - first
    with np.errstate(invalid="ignore", divide="ignore"):
        if spread <= 0:
            return first * np.log(p) + second * np.log1p(-p)
        kappa = _concentration(rho, spread)
        a, b = p * kappa, (1 - p) * kappa
        return (gammaln(first + a) + gammaln(second + b) - gammaln(n + a + b)
                - gammaln(a) - gammaln(b) + gammaln(a + b))


def variances(n_eff: np.ndarray, modes: Modes, spread: float = 0.0) -> np.ndarray:
    """var(r) under each species, as (n, k).

    The shot noise of the split each species hypothesises, ``(1 - rho_k^2) /
    n_eff``, widened by whatever intrinsic spread the species is given.  Under
    the beta-binomial the two combine as ``(1 - rho^2)(n + kappa) /
    (n (kappa + 1))``, which is the sum in quadrature to first order and right
    at both ends: pure shot noise as spread -> 0, pure spread as n -> infinity.
    """
    n = np.asarray(n_eff, dtype=float)[:, None]
    rho = np.asarray(modes.maxima, dtype=float)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        shot = (1 - rho ** 2) / n
        if spread > 0:
            kappa = _concentration(rho, spread)
            shot = shot * (n + kappa) / (kappa + 1)
    return np.maximum(shot, np.finfo(float).tiny)


def deviations(r: np.ndarray, n_eff: np.ndarray, modes: Modes,
               spread: float = 0.0) -> np.ndarray:
    """How many sigma each localization sits from each species, as (n, k).

    The *absolute* question, the one a posterior cannot ask: not which species
    is likelier, but whether the observed split is one this species could have
    produced at all.  A localization in the valley between two modes is far
    from both, and its posterior still picks a winner.
    """
    r = np.asarray(r, dtype=float)[:, None]
    rho = np.asarray(modes.maxima, dtype=float)[None, :]
    return np.abs(r - rho) / np.sqrt(variances(n_eff, modes, spread))


def posteriors(r: np.ndarray, n_eff: np.ndarray, modes: Modes,
               spread: float = 0.0, use_prior: bool = True) -> np.ndarray:
    """P(species | r, photons) per localization, as (n, k).

    A *relative* statement, and only that: it assumes the localization is one
    of the species offered, and divides the evidence between them.  See
    `deviations` for the question it cannot answer.
    """
    log_l = log_likelihoods(r, n_eff, modes, spread=spread)
    if use_prior:
        log_l = log_l + np.log(np.maximum(modes.prior, 1e-12))[None, :]
    log_l -= log_l.max(axis=1, keepdims=True)      # softmax, without overflow
    weight = np.exp(log_l)
    out = weight / weight.sum(axis=1, keepdims=True)
    return np.where(np.isfinite(out), out, np.nan)


def assign_by_probability(r: np.ndarray, n_eff: np.ndarray, modes: Modes,
                          crosstalk: float = 0.05, spread: float = 0.0,
                          use_prior: bool = True, tolerance: float = 0.0
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """The best species, if it is both the clear winner and a possible one.

    Two tests, because they refuse different localizations.

    * *Ambiguous* -- Chow's rule.  The winning posterior must reach
      ``1 - crosstalk``, which bounds the expected fraction of misassigned
      localizations among the assigned ones.
    * *Inconsistent* -- the winner must also lie within ``tolerance`` sigma of
      the observed split, so that a localization no species could have produced
      is refused rather than handed to the nearest one.  0 turns it off.

    Returns the channel (0 where either test refuses) and the posterior of the
    winner, which is what makes the achieved crosstalk reportable rather than
    assumed.
    """
    p = posteriors(r, n_eff, modes, spread=spread, use_prior=use_prior)
    best = np.nanargmax(np.where(np.isfinite(p), p, -1.0), axis=1)
    rows = np.arange(len(best))
    top = p[rows, best]
    channel = (best + 1).astype(np.int32)
    channel[~(np.isfinite(top) & (top >= 1.0 - crosstalk))] = 0
    if tolerance > 0:
        z = deviations(r, n_eff, modes, spread=spread)[rows, best]
        channel[~(np.isfinite(z) & (z <= tolerance))] = 0
    return channel, top


def _expected_ratios(settings: "AssignColorSettings") -> Optional[np.ndarray]:
    """The ratios typed into the *expected r* field, as numbers."""
    text = (settings.expected or "").strip()
    if not text:
        return None
    try:
        values = np.array([float(part) for part in text.replace(";", ",").split(",")
                           if part.strip()])
    except ValueError:
        raise ValueError(f"expected r: {text!r} is not a list of numbers")
    if len(values) != settings.colors:
        raise ValueError(f"expected r has {len(values)} ratios but {settings.colors} "
                         "colours were asked for")
    return values


@dataclass
class AssignColorSettings:
    mode: str = param("minima", label="method",
                      choices=(("minima", "split at the minima"),
                               ("probabilistic", "probabilistic")),
                      help="a cut in r, or a posterior per localization")
    colors: int = param(2, label="colours", min=1, max=MAX_COLORS,
                        help="how many species the histogram of r holds")
    exclusion: float = param(0.05, label="exclusion dr", min=0.0,
                             help="minima: leave this much of r on either side "
                                  "of a boundary unassigned")
    crosstalk: float = param(0.05, label="allowed crosstalk", min=1e-6, max=0.5,
                             help="probabilistic: the largest expected fraction "
                                  "of wrongly assigned localizations; a "
                                  "localization is assigned only if its "
                                  "posterior reaches 1 - this")
    tolerance: float = param(3.0, label="consistency", unit="sigma", min=0.0,
                             help="probabilistic: refuse a localization whose "
                                  "split is further than this from every "
                                  "species' expected ratio, in its own sigma. "
                                  "0: assign it to the nearest one anyway")
    spread: float = param(0.0, label="extra spread", min=0.0,
                          help="probabilistic: a species' own width in r, added "
                               "in quadrature to the shot noise, for a dye whose "
                               "splitting ratio varies across the field or "
                               "between molecules. 0: shot noise alone (the "
                               "summary prints what the modes suggest)")
    use_errors: bool = param(True, label="use fitted errors",
                             help="take the photon errors from the fit where "
                                  "the table has them; otherwise sqrt(photons)")
    use_prior: bool = param(True, label="use abundances", advanced=True,
                            help="probabilistic: weight each species by how much "
                                 "of the sample sits under its mode")
    min_photons: float = param(0.0, label="minimum photons", min=0.0,
                               advanced=True,
                               help="localizations with fewer photons in total "
                                    "are ignored and left unassigned")
    bins: int = param(200, label="histogram bins", min=10, max=2000,
                      advanced=True)
    smoothing: float = param(2.0, label="smoothing", unit="bins", min=0.0,
                             advanced=True,
                             help="the histogram is smoothed by this much "
                                  "before its maxima are looked for; it also "
                                  "sets how far apart two modes must be")
    expected: Optional[str] = param(None, label="expected r", advanced=True,
                                    help="the species' ratios, comma separated "
                                         "(\"-0.65, 0.96\"), measured on "
                                         "single-label samples or computed from "
                                         "the spectra. auto: the histogram's "
                                         "own maxima")
    channel1: Optional[str] = param(None, label="channel 1 column", advanced=True,
                                    help="auto: photons_ch0, or the first pair "
                                         "of per-channel columns in the table")
    channel2: Optional[str] = param(None, label="channel 2 column", advanced=True)


@register("Analysis/Dual-Color/AssignColors")
class AssignColors(Plugin):
    name = "Assign colours"
    description = ("Colour two-channel localizations by their photon split, "
                   "r = (ch1 - ch2) / (ch1 + ch2): cut the histogram of r at "
                   "its minima, or give each localization the probability of "
                   "every colour from its own photon counts.  Writes those "
                   "probabilities, and `channel`: the colour decided on, or 0 "
                   "where the probabilities are too close to choose between "
                   "and where no colour fits at all.")
    preview_help = ("draw the histogram of r with the modes, the boundaries "
                    "and what would be assigned, and the same decision over "
                    "the two channels' intensities.  Nothing is saved and the "
                    "session is not touched.")
    Settings = AssignColorSettings
    main = ("mode", "colors", "exclusion", "crosstalk", "tolerance", "spread",
            "use_errors")
    # which fields each method actually reads; the GUI greys out the rest, so
    # that a number that does nothing does not look like a number that does
    USED = {"minima": ("exclusion",),
            "probabilistic": ("crosstalk", "tolerance", "spread", "use_prior")}

    def active(self, settings: AssignColorSettings) -> Dict[str, bool]:
        """Only the chosen method's parameters are live."""
        mine = set(self.USED.get(settings.mode, ()))
        others = {name for names in self.USED.values() for name in names}
        return {name: name in mine for name in others}

    def run(self, ctx: Context, settings: AssignColorSettings) -> Result:
        return self._work(ctx, settings, apply=True)

    def preview(self, ctx: Context, settings: AssignColorSettings) -> Result:
        """The same decision, drawn instead of written."""
        return self._work(ctx, settings, apply=False)

    # ------------------------------------------------------------------ work
    def _work(self, ctx: Context, settings: AssignColorSettings,
              apply: bool) -> Result:
        ctx.selection.require(50, ctx.report, "a colour histogram")
        values = ratios(ctx.locs, settings.channel1, settings.channel2,
                        use_errors=settings.use_errors,
                        min_photons=settings.min_photons)
        ctx.report(f"r from {values.columns[0]} and {values.columns[1]}"
                   + ("" if values.fitted_errors else ", errors from sqrt(photons)"))

        # the modes are read from what the user is looking at and applied to
        # the whole table: a molecule's colour does not depend on the ROI
        seen = ctx.selection.mask & values.valid
        if not seen.any():
            raise ValueError("no selected localization has two-channel photons")
        expected = _expected_ratios(settings)
        modes = find_modes(values.r[seen], colors=settings.colors,
                           bins=settings.bins, smoothing=settings.smoothing,
                           expected=expected)
        if expected is not None:
            given = ", ".join(f"{v:+.3f}" for v in expected)
            ctx.report(f"expected ratios given: {given}")

        # The result is a probability for each colour.  `channel` is the
        # decision that follows from it -- and 0, undecided, wherever the
        # probabilities are too close together to choose between.
        per_colour = posteriors(values.r, values.n_eff, modes,
                                spread=settings.spread,
                                use_prior=settings.use_prior)
        if settings.mode == "probabilistic":
            channel, probability = assign_by_probability(
                values.r, values.n_eff, modes, crosstalk=settings.crosstalk,
                spread=settings.spread, use_prior=settings.use_prior,
                tolerance=settings.tolerance)
        elif settings.mode == "minima":
            channel = assign_by_minima(values.r, modes, settings.exclusion)
            probability = np.where(channel > 0, 1.0, 0.0)
        else:
            raise ValueError(f"unknown method {settings.mode!r}")
        channel[~values.valid] = 0
        probability = np.where(channel > 0, probability, 0.0)

        # how far the nearest species is, in its own sigma: the number the
        # posterior cannot carry.  P(colour) is a *relative* statement and sits
        # at 0 or 1 almost everywhere, so it says nothing about a localization
        # that is no colour at all -- this does, for every row, whether or not
        # it was assigned, and it is what the consistency test cuts on.
        sigma = np.nanmin(deviations(values.r, values.n_eff, modes,
                                     spread=settings.spread), axis=1)

        locs = Localizations(dict(ctx.locs.columns), dict(ctx.locs.metadata))
        locs.columns["channel"] = channel
        locs.columns["color_ratio"] = values.r.astype(np.float32)
        locs.columns["channel_p"] = probability.astype(np.float32)
        locs.columns["channel_sigma"] = sigma.astype(np.float32)
        for k in range(len(modes)):
            # P(colour k+1 | this molecule's photons), for every row: the
            # answer itself, which a single label cannot hold
            locs.columns[f"channel_p{k + 1}"] = np.where(
                values.valid, per_colour[:, k], np.nan).astype(np.float32)

        text = self._summary(values, modes, channel, probability, settings, seen)
        ctx.report(text.splitlines()[0])
        return Result(locs=locs if apply else None, text=text,
                      plot=_plotter(values, modes, channel, settings, seen),
                      plots={"intensities":
                             _intensity_plotter(values, modes, settings, seen)},
                      data={"modes": modes, "ratios": values, "channel": channel,
                            "probability": probability, "sigma": sigma,
                            "posteriors": per_colour},
                      settings=settings)

    def _summary(self, values: Ratios, modes: Modes, channel: np.ndarray,
                 probability: np.ndarray, settings: AssignColorSettings,
                 seen: np.ndarray) -> str:
        n = int(values.valid.sum())
        assigned = channel > 0
        lines = [f"{int(assigned.sum())} of {n} localizations assigned to "
                 f"{len(modes)} colours "
                 f"({100 * assigned.sum() / max(n, 1):.1f}% kept)"]
        for k, rho in enumerate(modes.maxima, 1):
            count = int((channel == k).sum())
            lines.append(f"  colour {k}: r = {rho:+.3f}, {count} localizations "
                         f"({100 * modes.prior[k - 1]:.0f}% of the sample)")
        if len(modes.minima):
            lines.append("  boundaries at r = "
                         + ", ".join(f"{m:+.3f}" for m in modes.minima))
        for position, height in unclaimed_modes(modes):
            lines.append(f"  warning: a population at r = {position:+.3f} "
                         f"({100 * height:.0f}% of the mode beside it) belongs "
                         f"to no colour: ask for more colours, or use the "
                         f"probabilistic method, which refuses it")
        if settings.mode == "probabilistic":
            achieved = (float(np.mean(1 - probability[assigned]))
                        if assigned.any() else 0.0)
            lines.append(f"  crosstalk: {100 * achieved:.2f}% expected among the "
                         f"assigned, budget {100 * settings.crosstalk:.2f}%")
            # the two tests refuse different localizations, and which one did
            # the refusing is the difference between "too close to call" and
            # "not either colour"
            refused = seen & ~assigned
            if settings.tolerance > 0 and refused.any():
                z = np.nanmin(deviations(values.r, values.n_eff, modes,
                                         spread=settings.spread), axis=1)
                far = refused & ~(np.isfinite(z) & (z <= settings.tolerance))
                lines.append(f"  refused: {int(far.sum())} further than "
                             f"{settings.tolerance:g} sigma from every colour, "
                             f"{int((refused & ~far).sum())} too close to call")
            if len(modes) == 2:
                # the exclusion zone this is equivalent to, for a median
                # localization: what mode 1 would have to be told by hand
                n_eff = float(np.nanmedian(values.n_eff[seen]))
                gap = abs(modes.maxima[1] - modes.maxima[0])
                middle = float(np.mean(modes.maxima))
                width = ((1 - middle ** 2) / n_eff + settings.spread ** 2) \
                    * np.log((1 - settings.crosstalk) / settings.crosstalk) / gap
                lines.append(f"  equivalent to dr = {width:.3f} at the median "
                             f"{n_eff:.0f} effective photons")
        else:
            lines.append(f"  exclusion zone dr = {settings.exclusion:.3f}")
        # what the modes are worth: the observed width against the shot noise,
        # which is how `spread` gets set by looking rather than by guessing
        strongest = int(np.argmin(np.abs(
            modes.maxima - modes.centers[int(np.argmax(modes.density))])))
        inside = seen & (assign_by_minima(values.r, modes) == strongest + 1)
        shot = float(np.nanmedian(np.sqrt(
            max(1 - modes.maxima[strongest] ** 2, 0.0) / values.n_eff[inside])))
        observed = _mode_width(modes, strongest, settings.smoothing)
        if np.isfinite(observed):
            lines.append(f"  colour {strongest + 1} is {observed:.3f} wide against "
                         f"{shot:.3f} from photon statistics"
                         + ("" if observed <= shot else
                            f"; spread = {np.sqrt(observed ** 2 - shot ** 2):.3f} "
                            "would account for the rest"))
        if not values.fitted_errors:
            lines.append("  no per-channel photon errors in the table: "
                         "sqrt(photons) used")
        return "\n".join(lines)


def _mode_width(modes: Modes, index: int, smoothing: float = 0.0) -> float:
    """Mode `index`'s width in r, as a standard deviation; NaN if unmeasurable.

    Read off the smoothed histogram rather than fitted: it is a number to look
    at next to the shot-noise width, not a parameter of anything.  It is taken
    from a *flank* -- the half-maximum crossing on one side, doubled -- because
    two modes that overlap never fall to half between them, and a full width
    across both of them would measure their separation instead of their width.
    A flank that runs into the neighbouring boundary before it crosses is
    discarded for the same reason; when both do, there is nothing to report.

    Two conversions make what is left comparable: a Gaussian's half width at
    half maximum is 1.177 sigma, and the smoothing applied to find the maxima
    has widened the mode by its own sigma, which comes back off in quadrature.
    """
    peak = int(np.argmin(np.abs(modes.centers - modes.maxima[index])))
    half = modes.density[peak] / 2
    if not half > 0:
        return float("nan")
    lo = (0 if index == 0
          else int(np.searchsorted(modes.centers, modes.minima[index - 1])))
    hi = (len(modes.centers) - 1 if index == len(modes) - 1
          else int(np.searchsorted(modes.centers, modes.minima[index])))
    flanks = []
    for step, limit in ((-1, lo), (1, hi)):
        i = peak
        while i != limit and modes.density[i] > half:
            i += step
        if modes.density[i] > half:
            continue                      # ran into the neighbour, not a flank
        # where the crossing actually lies, between the two bins around it
        above, below = modes.density[i - step], modes.density[i]
        fraction = ((above - half) / (above - below)) if above > below else 0.0
        crossing = modes.centers[i - step] + step * fraction * (
            modes.centers[1] - modes.centers[0])
        flanks.append(abs(crossing - modes.maxima[index]))
    if not flanks:
        return float("nan")
    sigma = float(np.mean(flanks)) / HWHM_PER_SIGMA
    step = float(modes.centers[1] - modes.centers[0])
    return float(np.sqrt(max(sigma ** 2 - (smoothing * step) ** 2, 0.0)))


def _plotter(values: Ratios, modes: Modes, channel: np.ndarray,
             settings: AssignColorSettings, seen: np.ndarray):
    """A closure over the numbers, so the figure is drawn on the GUI thread."""
    r = values.r[seen]
    assigned = channel[seen]
    n_eff = float(np.nanmedian(values.n_eff[seen]))

    def plot(ax) -> None:
        edges = np.linspace(-1, 1, settings.bins + 1)
        ax.hist(r, bins=edges, color="0.85", label="all")
        for k in range(1, len(modes) + 1):
            part = r[assigned == k]
            if len(part):
                ax.hist(part, bins=edges, color=PALETTE[(k - 1) % len(PALETTE)],
                        alpha=0.75, label=f"colour {k} ({len(part)})")
        ax.plot(modes.centers, modes.density, color="0.35", lw=1)
        for rho in modes.maxima:
            ax.axvline(rho, color="0.2", ls=":", lw=1)
        for m in modes.minima:
            ax.axvline(m, color="k", lw=1.2)
            if settings.mode == "minima" and settings.exclusion > 0:
                ax.axvspan(m - settings.exclusion, m + settings.exclusion,
                           color="k", alpha=0.10, lw=0)
        if settings.mode == "probabilistic":
            _draw_posteriors(ax, modes, n_eff, settings)
        rejected = int((assigned == 0).sum())
        ax.set_xlabel("r = (ch1 - ch2) / (ch1 + ch2)")
        ax.set_ylabel("localizations")
        ax.set_title(f"{settings.mode}: {len(r) - rejected} assigned, "
                     f"{rejected} left at 0")
        # r lives in [-1, 1] and the histogram is always binned over all of it,
        # so that two runs are comparable; the view is where the data is
        low, high = np.nanpercentile(r, (0.1, 99.9))
        margin = max(0.1 * (high - low), 2 * (edges[1] - edges[0]))
        ax.set_xlim(max(low - margin, -1), min(high + margin, 1))
        # room above the tallest bar for the legend, which would otherwise sit
        # on the posterior curves -- they run to 1 across the whole width
        ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    return plot


def _decide(first: np.ndarray, second: np.ndarray, modes: Modes,
            settings: "AssignColorSettings", efficiency: float = 1.0
            ) -> np.ndarray:
    """The same decision, for any pair of intensities.

    What makes the regions drawable: the rule depends on the two counts only
    through r and the effective photon number, so it can be evaluated over a
    grid of intensities nobody measured.  ``efficiency`` is the table's median
    n_eff / total, standing in for the fitted errors that a grid point has not
    got.
    """
    shape = np.shape(first)
    first = np.ravel(first).astype(float)   # the model works on a flat list
    second = np.ravel(second).astype(float)
    total = first + second
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(total > 0, (first - second) / total, np.nan)
    r = np.clip(r, -1.0, 1.0)
    n_eff = np.maximum(total * efficiency, 1e-9)
    if settings.mode == "minima":
        channel = assign_by_minima(r, modes, settings.exclusion)
    else:
        channel = assign_by_probability(
            r, n_eff, modes, crosstalk=settings.crosstalk, spread=settings.spread,
            use_prior=settings.use_prior, tolerance=settings.tolerance)[0]
    return channel.reshape(shape)


def _intensity_plotter(values: Ratios, modes: Modes, settings: "AssignColorSettings",
                       seen: np.ndarray):
    """The decision in the plane the photons actually live in.

    The histogram of r divides out the brightness, which is most of what a
    localization is: two channels' counts against each other show the species
    as *rays* from the origin -- one line per splitting ratio -- and show what
    the noise model claims, which is that a decision may be taken close to the
    boundary when the counts are large and not when they are small.  In log-log
    a constant ratio is a straight line of slope 1, so the regions are bands
    along the diagonal, and a band's width in this plane is what `dr` sets by
    hand and what the probabilistic method sets per localization.
    """
    first = values.first[seen]
    second = values.second[seen]
    efficiency = values.efficiency
    good = (first > 0) & (second > 0) & np.isfinite(first) & np.isfinite(second)
    first, second = first[good], second[good]

    def plot(ax) -> None:
        from matplotlib.colors import LogNorm
        if not len(first):
            ax.text(0.5, 0.5, "no localization has photons in both channels",
                    ha="center", transform=ax.transAxes)
            return
        x, y = np.log10(first), np.log10(second)
        lo = min(np.percentile(x, 0.2), np.percentile(y, 0.2)) - 0.1
        hi = max(np.percentile(x, 99.8), np.percentile(y, 99.8)) + 0.1

        # the regions, evaluated on a grid and drawn underneath
        edges = np.linspace(lo, hi, 400)
        gx, gy = np.meshgrid(edges, edges)
        region = _decide(10 ** gx, 10 ** gy, modes, settings, efficiency)
        for k in range(1, len(modes) + 1):
            ax.contourf(gx, gy, (region == k).astype(float), levels=(0.5, 1.5),
                        colors=[PALETTE[(k - 1) % len(PALETTE)]], alpha=0.22)
            ax.contour(gx, gy, (region == k).astype(float), levels=(0.5,),
                       colors=[PALETTE[(k - 1) % len(PALETTE)]], linewidths=1)
        # and the species themselves: lines of constant splitting ratio
        for k, rho in enumerate(modes.maxima, 1):
            offset = np.log10(max((1 - rho) / (1 + rho), 1e-12))
            ax.plot((lo, hi), (lo + offset, hi + offset), lw=0.8, ls=":",
                    color=PALETTE[(k - 1) % len(PALETTE)])

        counts, ex, ey = np.histogram2d(x, y, bins=160, range=((lo, hi), (lo, hi)))
        ax.pcolormesh(ex, ey, np.ma.masked_where(counts.T == 0, counts.T),
                      cmap="Greys", norm=LogNorm(), rasterized=True)
        ax.set_xlabel(f"log10 {values.columns[0]}")
        ax.set_ylabel(f"log10 {values.columns[1]}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        note = (f"dr = {settings.exclusion:g}" if settings.mode == "minima" else
                f"crosstalk {settings.crosstalk:g}, "
                + (f"{settings.tolerance:g} sigma" if settings.tolerance > 0
                   else "no consistency test"))
        ax.set_title(f"{settings.mode}: {note}"
                     + (f", n_eff = {efficiency:.2f} N" if settings.mode
                        == "probabilistic" else ""))

    return plot


def _draw_posteriors(ax, modes: Modes, n_eff: float,
                     settings: AssignColorSettings) -> None:
    """The decision as a median localization sees it, on a right-hand axis.

    The real boundaries are per-localization -- that is the whole point of the
    method -- so what is drawn is the posterior curve at the median effective
    photon number, with the shaded band where such a localization would be
    rejected.
    """
    if not np.isfinite(n_eff) or n_eff <= 0:
        return
    grid = np.linspace(-1, 1, 801)
    flat = np.full_like(grid, n_eff)
    p = posteriors(grid, flat, modes, spread=settings.spread,
                   use_prior=settings.use_prior)
    twin = ax.twinx()
    for k in range(len(modes)):
        twin.plot(grid, p[:, k], color=PALETTE[k % len(PALETTE)], lw=1, ls="--")
    twin.axhline(1 - settings.crosstalk, color="0.4", lw=0.8, ls=":")
    rejected = np.nanmax(p, axis=1) < 1 - settings.crosstalk
    if settings.tolerance > 0:
        # the consistency test rejects most of the axis when the species are
        # far apart, which is the point of it; shade it lightly enough that
        # the histogram still reads through
        z = np.nanmin(deviations(grid, flat, modes, spread=settings.spread), axis=1)
        rejected |= z > settings.tolerance
    for start, stop in _runs(rejected):
        ax.axvspan(grid[start], grid[stop], color="k", alpha=0.07, lw=0)
    twin.set_ylim(0, 1.28)
    twin.set_yticks((0, 0.5, 1))
    twin.set_ylabel(f"posterior at {n_eff:.0f} photons")


def _runs(mask: np.ndarray):
    """The (first, last) index of each run of True.  Empty when there are none."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2] - 1))
