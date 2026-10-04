"""Nuclear pore complexes: finding them, counting their corners, and the
effective labelling efficiency that follows.

Three plugins, one for each step of the ROI manager, ported from SMAP's
``segmentNPC`` with ``NPCsegmentCleanup``, ``NPCLabelingQuantify_s`` and
``NPCLabelingEfficiency`` (Thevathasan et al. 2019):

* **Segment/NPC** filters a density image with a ring, fits a circle to every
  candidate and keeps the ones that look like a pore.
* **Evaluate/NPC Corners** counts how many of a pore's eight corners hold a
  localization.
* **Analyze/NPC Labeling Efficiency** fits the histogram of those counts with
  the binomial model of the paper.

Departures from SMAP, and why:

* The ring filter is a zero-sum kernel -- a Gaussian ring minus a disc of the
  same weight -- instead of SMAP's ring followed by a difference of Gaussians
  with a cutoff in filtered-image units.  A uniform background gives zero by
  construction, and the threshold that decides becomes a number of
  localizations on the ring, which means the same thing at any pixel size.
* What SMAP split between a segmenter and a clean-up evaluator happens in one
  run here: the circle fit and the quality control are what makes a candidate
  a pore, and the user asked for them in the segmenter.  The PSF-width check
  is dropped; that is a layer filter.
* The circle fits use a soft-L1 loss, so the background inside the window
  does not pull the centre; SMAP's ``fitposring`` is plain least squares.
* The rotation of a pore is the weighted circular mean of ``8 theta`` instead
  of a least-squares fit of a sawtooth: closed form, no start value, no local
  minima.
* The histogram fit is maximum likelihood of the binomial model conditioned on
  the fit range, with a bootstrap error.  SMAP's least squares on the square
  root of the histogram is kept as a choice, to compare against.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import Context, Plot, Plugin, Result, param, register
from .roi import MAX_FINDER_PIXELS, current_file

# SMAP's NPCLabelingQuantify counts a localization towards a corner only when
# its precision is below 0.4 of the arc between two corners (15.7 nm at 50 nm)
PRECISION_OF_ARC = 0.4
# fixed-radius fits of a candidate's centre: the second is on the window
# around the first answer, which a filter peak a pixel or two off can need
FIT_PASSES = 2


# ------------------------------------------------------------------ geometry

def fit_circle(x, y, center, radius: Optional[float] = None,
               scale: float = 15.0, iterations: int = 30) -> Tuple[float, float, float]:
    """A circle through the points: ``(x0, y0, r)``.

    With ``radius`` given only the centre is fitted, which is robust even on
    a pore with two corners; without it the radius is fitted too, which is
    what says whether the structure is a ring of the expected size.  The loss
    is soft-L1 with ``scale``, so points further than about ``scale`` from the
    circle count with their distance rather than its square.

    Gauss-Newton with reweighting (the soft-L1 weight is
    ``1 / sqrt(1 + (d / scale)^2)``) rather than scipy's ``least_squares``:
    the same minimum, twenty times faster, and the segmenter fits thousands.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(center, dtype=float).copy()
    free = radius is None
    if len(x) < (3 if free else 2):
        return float(p[0]), float(p[1]), float(np.nan if free else radius)
    r = float(np.median(np.hypot(x - p[0], y - p[1]))) if free else float(radius)
    for _ in range(iterations):
        dx, dy = x - p[0], y - p[1]
        d = np.maximum(np.hypot(dx, dy), 1e-9)
        residual = d - r
        w = 1.0 / np.sqrt(1.0 + (residual / scale) ** 2)
        J = [-dx / d, -dy / d] + ([-np.ones_like(d)] if free else [])
        J = np.column_stack(J)
        step = np.linalg.lstsq(J * np.sqrt(w)[:, None], -residual * np.sqrt(w),
                               rcond=None)[0]
        p += step[:2]
        if free:
            r = abs(r + step[2])
        if np.max(np.abs(step)) < 1e-3:
            break
    return float(p[0]), float(p[1]), float(r)


def ring_kernel(radius_nm: float, width_nm: float, pixel_nm: float) -> np.ndarray:
    """A Gaussian ring minus a disc of the same weight: zero sum.

    The ring has radius ``radius_nm`` and a Gaussian profile of width
    ``width_nm``; the disc reaches to ``radius + 2 width``.  Convolved with a
    density, it gives the mean density on the ring minus the mean density
    over the disc, so a uniform background gives zero, a filled blob of the
    pore's size gives about zero or less, and a ring gives its peak at its
    centre.
    """
    extent = radius_nm + 3 * width_nm
    n = int(np.ceil(extent / pixel_nm))
    grid = np.arange(-n, n + 1) * pixel_nm
    r = np.hypot(*np.meshgrid(grid, grid))
    ring = np.exp(-(r - radius_nm) ** 2 / (2 * width_nm ** 2))
    disc = (r <= radius_nm + 2 * width_nm).astype(float)
    return ring / ring.sum() - disc / disc.sum()


# --------------------------------------------------------------- segmenting

@dataclass
class NPCSegmentSettings:
    radius_nm: float = param(55.0, label="radius", unit="nm", min=1.0,
                             help="radius of the pore: the ring the filter looks for "
                                  "and the one the bands are drawn around")
    ring_width_nm: float = param(15.0, label="ring width", unit="nm", min=1.0,
                                 help="half the width of the band counted as the ring; "
                                      "also the width of the filter's ring")
    min_locs: int = param(10, label="min localizations", min=1,
                          help="localizations a candidate needs in the ring band")
    min_radius_nm: float = param(40.0, label="min fitted radius", unit="nm", min=0.0,
                                 help="smallest fitted radius that is a pore")
    max_radius_nm: float = param(70.0, label="max fitted radius", unit="nm", min=0.0,
                                 help="largest fitted radius that is a pore")
    max_inside: float = param(0.2, label="max inside", min=0.0, max=1.0,
                              help="largest fraction of the window's localizations "
                                   "inside the ring band")
    max_outside: float = param(0.4, label="max outside", min=0.0, max=1.0,
                               help="largest fraction of the window's localizations "
                                    "outside the ring band")
    # off by default: it is what rejects an arc of a neighbouring pore, but
    # it also rejects real pores with few corners labelled -- the ones the
    # labelling efficiency is measured from -- and with the separation above
    # 2 R the simulations show no arcs left for it to catch
    min_spread_nm: float = param(0.0, label="min spread", unit="nm", min=0.0,
                                 help="smallest spread of the localizations around "
                                      "their mean, (det cov)^1/4; 0 turns it off")
    keep_rejected: bool = param(False, label="keep rejected",
                                help="add candidates that fail the checks as ROIs "
                                     "that are not used, to look at them")
    window_nm: float = param(100.0, label="window", unit="nm", min=1.0, advanced=True,
                             help="radius around a candidate that is fitted and counted")
    separation_nm: float = param(120.0, label="separation", unit="nm", min=0.0,
                                 advanced=True,
                                 help="reject a candidate this close to a stronger "
                                      "one or to an existing ROI")
    bin_nm: float = param(10.0, label="bin", unit="nm", min=0.1, advanced=True,
                          help="pixel size of the density image that is filtered")


def ring_filtered(xy: np.ndarray, settings: NPCSegmentSettings
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """The ring-filtered square root of the density, and its origin in nm.

    The square root is SMAP's: a molecule that blinks fifty times in one
    place otherwise draws a ring of response around itself that outshines
    the pores.
    """
    from scipy.signal import fftconvolve
    pixel = float(settings.bin_nm)
    pad = settings.radius_nm + 3 * settings.ring_width_nm + pixel
    origin = np.floor((xy.min(axis=0) - pad) / pixel) * pixel
    shape = np.ceil((xy.max(axis=0) + pad - origin) / pixel).astype(int) + 1
    if int(shape[0]) * int(shape[1]) > MAX_FINDER_PIXELS:
        raise ValueError("the density image exceeds 16 million pixels; "
                         "increase the bin size")
    bins = np.floor((xy - origin) / pixel).astype(int)
    density = np.zeros(tuple(shape[::-1]), dtype=float)
    np.add.at(density, (bins[:, 1], bins[:, 0]), 1)
    kernel = ring_kernel(settings.radius_nm, settings.ring_width_nm, pixel)
    return fftconvolve(np.sqrt(density), kernel, mode="same"), origin


def candidates(image: np.ndarray, origin, pixel: float, separation_nm: float
               ) -> List[Tuple[float, np.ndarray]]:
    """Positive local maxima of the filtered image, strongest first:
    ``(response, centre in nm)``."""
    from scipy.ndimage import maximum_filter
    size = max(3, 2 * int(round(separation_nm / pixel / 2)) + 1)
    maxima = (image == maximum_filter(image, size=size, mode="constant")) & (image > 0)
    yy, xx = np.nonzero(maxima)
    response = image[yy, xx]
    order = np.lexsort((xx, yy, -response))
    return [(float(response[i]),
             np.asarray(origin) + (np.array([xx[i], yy[i]]) + 0.5) * pixel)
            for i in order]


def ring_quality(x, y, center, settings: NPCSegmentSettings) -> Dict[str, Any]:
    """Fit and judge one candidate.

    The centre is fitted with the radius fixed, twice, the second time on the
    window around the first answer; that centre is the site.  The radius is
    then fitted free, and the window's localizations are split into inside,
    ring and outside by their distance from the centre:
    ``r < R - dR``, ``R - dR <= r <= R + dR`` and ``r > R + dR``.
    """
    R, dR = settings.radius_nm, settings.ring_width_nm
    c = np.asarray(center, dtype=float)
    for _ in range(FIT_PASSES):
        near = np.hypot(x - c[0], y - c[1]) <= settings.window_nm
        c = np.array(fit_circle(x[near], y[near], c, radius=R, scale=dR)[:2])
    r = np.hypot(x - c[0], y - c[1])
    near = r <= settings.window_nm
    xw, yw, rw = x[near], y[near], r[near]
    n = len(rw)
    radius = fit_circle(xw, yw, c, scale=dR)[2] if n >= 3 else np.nan
    inside = int(np.sum(rw < R - dR))
    outside = int(np.sum(rw > R + dR))
    ring = n - inside - outside
    if n >= 3:
        eig = np.clip(np.linalg.eigvalsh(np.cov(xw, yw)), 0, None)
        spread = float(np.prod(eig) ** 0.25)
    else:
        spread = 0.0
    out = {"center": c.tolist(), "radius_nm": float(radius), "n_ring": ring,
           "n_window": n, "fraction_inside": inside / n if n else np.nan,
           "fraction_outside": outside / n if n else np.nan, "spread_nm": spread}
    failed = []
    if ring < settings.min_locs:
        failed.append("localizations")
    if not settings.min_radius_nm <= radius <= settings.max_radius_nm:
        failed.append("radius")
    if not out["fraction_inside"] <= settings.max_inside:
        failed.append("inside")
    if not out["fraction_outside"] <= settings.max_outside:
        failed.append("outside")
    if spread < settings.min_spread_nm:
        failed.append("spread")
    out["failed"] = failed
    out["use"] = not failed
    return out


class KDList:
    """Centres accepted so far, asked "is anything within d of this?"."""

    def __init__(self, centres: Sequence = ()):
        self.xy = np.zeros((max(len(centres), 64), 2))
        self.n = 0
        for c in centres:
            self.add(c)

    def add(self, c) -> None:
        if self.n == len(self.xy):
            self.xy = np.concatenate([self.xy, np.zeros_like(self.xy)])
        self.xy[self.n] = c
        self.n += 1

    def near(self, c, distance: float) -> bool:
        d = self.xy[:self.n] - np.asarray(c, dtype=float)
        return bool(np.any(d[:, 0] ** 2 + d[:, 1] ** 2 < distance ** 2))


def segment_npcs(locs, settings: NPCSegmentSettings, existing: Sequence = ()
                 ) -> List[Dict[str, Any]]:
    """Every candidate with enough localizations on its ring, judged.

    One dict per candidate, strongest filter response first, from
    `ring_quality` plus the ``response``.  A candidate with fewer than
    ``min_locs`` localizations in the ring band around the filter's maximum is
    not reported at all: it is background, not a pore that failed a check.
    Candidates closer than the separation to a stronger one -- or to an
    ``existing`` centre -- are suppressed, after the fit has moved them.
    """
    from scipy.spatial import cKDTree
    for name in ("radius_nm", "ring_width_nm", "window_nm", "bin_nm"):
        if not getattr(settings, name) > 0:
            raise ValueError(f"{name} must be positive")
    xy = np.column_stack((np.asarray(locs["x_nm"], dtype=float),
                          np.asarray(locs["y_nm"], dtype=float)))
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < settings.min_locs:
        return []
    image, origin = ring_filtered(xy, settings)
    tree = cKDTree(xy)
    R, dR = settings.radius_nm, settings.ring_width_nm
    found = []
    peaks = candidates(image, origin, settings.bin_nm, settings.separation_nm)
    if not peaks:
        return found
    centres = np.array([c for _, c in peaks])
    # the ring band's count for every maximum at once: most are background
    ring = (tree.query_ball_point(centres, R + dR, return_length=True)
            - tree.query_ball_point(centres, max(R - dR, 0.0), return_length=True))
    accepted = KDList(existing)
    for (response, center), n_ring in zip(peaks, ring):
        # too few, or inside a pore already accepted: one of its own maxima
        if n_ring < settings.min_locs or accepted.near(center, settings.separation_nm):
            continue
        window = np.asarray(tree.query_ball_point(center, settings.window_nm + 2 * dR))
        site = ring_quality(xy[window, 0], xy[window, 1], center, settings)
        c = np.asarray(site["center"])
        if accepted.near(c, settings.separation_nm):
            continue
        accepted.add(c)
        site["response"] = response
        found.append(site)
    return found


def draw_quality(figure, sites: Sequence[Dict[str, Any]],
                 settings: NPCSegmentSettings) -> None:
    """The four checks as histograms, the limits drawn in, kept and rejected
    stacked: where the cut falls against the distribution."""
    checks = (("radius_nm", "fitted radius (nm)",
               (settings.min_radius_nm, settings.max_radius_nm)),
              ("fraction_inside", "fraction inside", (None, settings.max_inside)),
              ("fraction_outside", "fraction outside", (None, settings.max_outside)),
              ("spread_nm", "spread (nm)", (settings.min_spread_nm, None)))
    used = np.array([s["use"] for s in sites], dtype=bool)
    axes = figure.subplots(2, 2, gridspec_kw={"hspace": 0.55, "wspace": 0.45})
    for ax, (key, label, limits) in zip(axes.ravel(), checks):
        values = np.array([s[key] for s in sites], dtype=float)
        finite = np.isfinite(values)
        if finite.any():
            edges = np.histogram_bin_edges(values[finite], bins=20)
            ax.hist([values[finite & used], values[finite & ~used]], bins=edges,
                    stacked=True, color=["#2f7fd0", "0.75"],
                    label=["kept", "rejected"])
        for limit in limits:
            if limit is not None:
                ax.axvline(limit, color="#d62728", lw=1)
        ax.set_xlabel(label)
        ax.set_ylabel("candidates")
    figure.axes[0].legend(fontsize=7, frameon=False)


def draw_sites(ax, xy: np.ndarray, sites: Sequence[Dict[str, Any]],
               settings: NPCSegmentSettings) -> None:
    """The localizations with each candidate's fitted circle: kept in blue,
    rejected in grey."""
    from matplotlib.patches import Circle
    ax.scatter(xy[:, 0], xy[:, 1], s=0.3, c="k", linewidths=0, alpha=0.5)
    for site in sites:
        radius = site["radius_nm"] if np.isfinite(site["radius_nm"]) else settings.radius_nm
        ax.add_patch(Circle(site["center"], radius, fill=False, lw=0.8,
                            color="#2f7fd0" if site["use"] else "0.6"))
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")


@register("ROIManager/Segment/NPC")
class NPCSegment(Plugin):
    """Finds nuclear pores with a ring filter, fits a circle to each and keeps
    those that look like a pore."""

    Settings = NPCSegmentSettings
    version = "1"

    def propose(self, locs, settings: NPCSegmentSettings,
                existing: Sequence = ()) -> List[Dict[str, Any]]:
        """The sites for `ROIProject.find`: centre, whether used, and the
        checks' numbers, which go into the ROI's origin."""
        sites = segment_npcs(locs, settings, existing)
        # every candidate, for `run`'s text and figures: `find` keeps only
        # the ROIs it made, and the rejected ones are what the checks are for
        self.last = sites
        keep = [s for s in sites if s["use"] or settings.keep_rejected]
        return [{"center": s["center"], "use": s["use"],
                 "origin": {k: s[k] for k in ("radius_nm", "n_ring", "fraction_inside",
                                              "fraction_outside", "spread_nm",
                                              "failed")}}
                for s in keep]

    def run(self, ctx: Context, settings: NPCSegmentSettings) -> Result:
        project = ctx.rois
        if project is None:
            locs = ctx.locs
            sites = segment_npcs(locs, settings)
            data = {"sites": sites,
                    "centers": [s["center"] for s in sites if s["use"]]}
        else:
            file_id = current_file(project)
            if file_id is None:
                raise ValueError("no file to search: load localizations first")
            state = project.state(file_id)
            locs = state.locs[state.filter.indices]
            self.last = []
            found = project.find(file_id, plugin=self, settings=settings)
            sites = self.last
            data = {"sites": sites, "rois": found,
                    "centers": [r.center for r in found if r.use]}
        kept = sum(s["use"] for s in sites)
        text = f"{kept} pores of {len(sites)} candidates"
        if sites and kept < len(sites):
            reasons: Dict[str, int] = {}
            for s in sites:
                for f in s["failed"]:
                    reasons[f] = reasons.get(f, 0) + 1
            text += "; rejected for " + ", ".join(f"{k} ({v})" for k, v in
                                                  sorted(reasons.items()))
        xy = np.column_stack((np.asarray(locs["x_nm"], dtype=float),
                              np.asarray(locs["y_nm"], dtype=float)))
        return Result(text=text, settings=settings, data=data,
                      plot=(lambda ax: draw_sites(ax, xy, sites, settings))
                      if sites else None,
                      plots={"checks": Plot(lambda f: draw_quality(f, sites, settings),
                                            panels=4, size=(6, 5))} if sites else {})


# ---------------------------------------------------------------- corners

@dataclass
class NPCCornersSettings:
    radius_nm: float = param(50.0, label="radius", unit="nm", min=1.0,
                             help="radius of the ring the localizations are taken from")
    ring_width_nm: float = param(20.0, label="ring width", unit="nm", min=1.0,
                                 help="half the width of the ring band")
    corners: int = param(8, label="corners", min=2,
                         help="symmetry of the pore: segments around the ring")
    min_locs: int = param(1, label="min per corner", min=1,
                          help="localizations a corner needs to count as seen")


def precision_column(locs) -> Optional[str]:
    return next((n for n in ("xy_err_nm", "x_err_nm") if n in locs), None)


def count_corners(x, y, center, settings: NPCCornersSettings, precision=None
                  ) -> Dict[str, Any]:
    """How many of a pore's corners hold a localization.

    The centre is fitted with the radius fixed, and the localizations in the
    ring band whose precision is better than 0.4 of the arc between two
    corners are kept.  The pore's rotation is the circular mean of
    ``corners * theta`` weighted by ``1 / dtheta^2`` (``dtheta`` the
    precision over the distance from the centre); the ring is cut into
    segments centred on the corners, and a segment with ``min_locs``
    localizations is a corner seen.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    R, dR, n = settings.radius_nm, settings.ring_width_nm, int(settings.corners)
    step = 2 * np.pi / n
    x0, y0, _ = fit_circle(x, y, center, radius=R, scale=dR)
    near = np.hypot(x - x0, y - y0) <= R + dR
    radius = fit_circle(x[near], y[near], (x0, y0), scale=dR)[2] if near.sum() >= 3 \
        else float("nan")
    rho = np.hypot(x - x0, y - y0)
    theta = np.arctan2(y - y0, x - x0)
    ring = (rho > R - dR) & (rho < R + dR)
    if precision is not None:
        precision = np.asarray(precision, dtype=float)
        ring &= precision < PRECISION_OF_ARC * step * R
        weight = 1.0 / np.maximum(precision / np.maximum(rho, 1e-9), 1e-6) ** 2
    else:
        weight = np.ones_like(rho)
    out = {"n_corners": 0, "n_ring_locs": int(ring.sum()),
           "n_within_locs": int(np.sum(rho < R + dR)),
           "radius_nm": float(radius), "rotation_deg": float("nan"),
           "x_nm": x0, "y_nm": y0, "per_corner": [0] * n}
    if not ring.any():
        return out
    phase = np.angle(np.sum(weight[ring] * np.exp(1j * n * theta[ring]))) / n
    segment = np.floor(np.mod(theta[ring] - phase + step / 2, 2 * np.pi) / step)
    per_corner = np.bincount(segment.astype(int), minlength=n)[:n]
    out.update(n_corners=int(np.sum(per_corner >= settings.min_locs)),
               rotation_deg=float(np.degrees(phase)), per_corner=per_corner.tolist())
    return out


def draw_corners(ax, x, y, values: Dict[str, Any], settings: NPCCornersSettings,
                 precision=None) -> None:
    """The site, the ring band, and the segment borders; the localizations
    that were counted in colour."""
    from matplotlib.patches import Circle
    R, dR, n = settings.radius_nm, settings.ring_width_nm, int(settings.corners)
    x0, y0 = values["x_nm"], values["y_nm"]
    rho = np.hypot(x - x0, y - y0)
    ring = (rho > R - dR) & (rho < R + dR)
    if precision is not None:
        ring &= np.asarray(precision) < PRECISION_OF_ARC * 2 * np.pi / n * R
    ax.scatter(x[~ring] - x0, y[~ring] - y0, s=5, c="0.7", linewidths=0)
    ax.scatter(x[ring] - x0, y[ring] - y0, s=7, c="#2f7fd0", linewidths=0)
    for r, style in ((R, "-"), (R - dR, ":"), (R + dR, ":")):
        ax.add_patch(Circle((0, 0), r, fill=False, ls=style, color="k", lw=0.7))
    if np.isfinite(values["rotation_deg"]):
        phase = np.radians(values["rotation_deg"])
        seen = np.asarray(values["per_corner"]) >= settings.min_locs
        for k in range(n):
            border = phase + (k + 0.5) * 2 * np.pi / n
            ax.plot([0, 1.5 * R * np.cos(border)], [0, 1.5 * R * np.sin(border)],
                    color="0.5", lw=0.6)
            corner = phase + k * 2 * np.pi / n
            ax.plot(R * np.cos(corner), R * np.sin(corner), "o", ms=7, mew=1.2,
                    mfc="#d62728" if seen[k] else "none", mec="#d62728")
    lim = 1.6 * R
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(f"{values['n_corners']} of {n} corners, "
                 f"radius {values['radius_nm']:.1f} nm")


@register("ROIManager/Evaluate/NPC Corners")
class NPCCorners(Plugin):
    """Counts how many corners of a nuclear pore hold a localization."""

    Settings = NPCCornersSettings
    scope = "site"
    version = "1"
    # not in the default evaluation pipeline: added when the data are pores
    favorite = False

    def run(self, ctx: Context, settings: NPCCornersSettings) -> Result:
        locs = ctx.locs
        x = np.asarray(locs["x_nm"], dtype=float)
        y = np.asarray(locs["y_nm"], dtype=float)
        if not len(x):
            raise ValueError("no localizations in this ROI")
        column = precision_column(locs)
        precision = None if column is None else np.asarray(locs[column], dtype=float)
        center = (ctx.site or {}).get("center") or [float(np.mean(x)), float(np.mean(y))]
        values = count_corners(x, y, center, settings, precision)
        data = {k: v for k, v in values.items() if k != "per_corner"}
        return Result(text=f"{values['n_corners']} corners", data=data,
                      plot=lambda ax: draw_corners(ax, x, y, values, settings, precision),
                      settings=settings)


# ----------------------------------------------------- labelling efficiency

def corner_probability(efficiency, per_corner: int):
    """The chance that a corner shows: at least one of its proteins labelled."""
    return 1.0 - (1.0 - np.asarray(efficiency, dtype=float)) ** per_corner


def corner_model(efficiency: float, corners: int, per_corner: int) -> np.ndarray:
    """``P(k)``, k = 0..corners: corners seen, each independently."""
    from scipy.stats import binom
    return binom.pmf(np.arange(corners + 1), corners,
                     corner_probability(efficiency, per_corner))


def fit_likelihood(counts: np.ndarray, corners: int, per_corner: int,
                   lo: int, hi: int) -> float:
    """The efficiency that maximises the likelihood of the histogram within
    ``lo..hi``, the binomial renormalised to that range."""
    from scipy.optimize import minimize_scalar
    k = np.arange(lo, hi + 1)
    h = np.asarray(counts, dtype=float)[k]

    def negative(p):
        model = corner_model(p, corners, per_corner)[k]
        model = np.maximum(model / max(model.sum(), 1e-300), 1e-300)
        return -float(np.sum(h * np.log(model)))
    return float(minimize_scalar(negative, bounds=(1e-4, 1 - 1e-4),
                                 method="bounded", options={"xatol": 1e-6}).x)


def fit_sqrt_lsq(counts: np.ndarray, corners: int, per_corner: int,
                 lo: int, hi: int) -> float:
    """SMAP's ``fitNPClabeling``: least squares of ``sqrt(A P(k))`` against the
    square root of the histogram within ``lo..hi``, from ``A`` = the number of
    sites and an efficiency of 0.4."""
    from scipy.optimize import least_squares
    k = np.arange(lo, hi + 1)
    h = np.asarray(counts, dtype=float)

    def residual(p):
        return np.sqrt(p[0] * corner_model(p[1], corners, per_corner)[k]) - np.sqrt(h[k])
    out = least_squares(residual, [h.sum(), 0.4], bounds=([0, 0], [np.inf, 1]))
    return float(out.x[1])


FITS = {"likelihood": fit_likelihood, "sqrt least squares (SMAP)": fit_sqrt_lsq}


@dataclass
class LabelingEfficiencySettings:
    corners: int = param(8, label="corners", min=2,
                         help="corners of a pore")
    per_corner: int = param(4, label="proteins per corner", min=1,
                            help="copies of the labelled protein in one corner")
    fit_min: int = param(3, label="fit from", min=0,
                         help="fewest corners in the fit range")
    fit_max: int = param(8, label="fit to", min=0,
                         help="most corners in the fit range")
    fit: str = param("likelihood", label="fit",
                     choices=(("likelihood", "maximum likelihood"),
                              ("sqrt least squares (SMAP)", "SMAP's least squares")),
                     help="how the model is fitted to the histogram")
    bootstrap: int = param(100, label="bootstrap", min=0, advanced=True,
                           help="resamplings of the sites for the error; 0: none")


def labeling_efficiency(n_corners, settings: LabelingEfficiencySettings,
                        seed: int = 0) -> Dict[str, Any]:
    """The effective labelling efficiency from the corners seen per pore.

    ``{"efficiency", "error", "counts", "n_sites", "corner_probability"}``;
    the error is the standard deviation over ``bootstrap`` resamplings of the
    sites, with a fixed seed so a rerun gives the same number.
    """
    n = np.asarray(n_corners, dtype=float)
    n = n[np.isfinite(n)].astype(int)
    corners, per = int(settings.corners), int(settings.per_corner)
    lo, hi = int(settings.fit_min), min(int(settings.fit_max), corners)
    if lo > hi:
        raise ValueError("the fit range is empty: 'fit from' is above 'fit to'")
    if np.any((n < 0) | (n > corners)):
        raise ValueError(f"corner counts outside 0..{corners}: is 'corners' right?")
    fit = FITS[settings.fit]
    counts = np.bincount(n, minlength=corners + 1)
    in_range = int(counts[lo:hi + 1].sum())
    if in_range < 5:
        raise ValueError(f"{in_range} pores in the fit range: at least 5 are needed")
    efficiency = fit(counts, corners, per, lo, hi)
    error = float("nan")
    if settings.bootstrap > 0:
        rng = np.random.default_rng(seed)
        draws = [fit(np.bincount(rng.choice(n, len(n)), minlength=corners + 1),
                     corners, per, lo, hi) for _ in range(int(settings.bootstrap))]
        error = float(np.std(draws))
    return {"efficiency": efficiency, "error": error, "counts": counts,
            "n_sites": int(len(n)), "n_fitted": in_range,
            "corner_probability": float(corner_probability(efficiency, per))}


def true_corners(locs, corners: int = 8) -> Dict[int, int]:
    """For a simulated table: how many corners of each pore were labelled.

    The labelled molecules are drawn again from the recipe the table carries
    (as Ground Truth does), and each one is put in a corner by its angle
    around its pore's centre, with the pore's rotation found the way
    `count_corners` finds it -- on exact positions, where it cannot miss.
    ``{copy: corners labelled}``.
    """
    from ..simulate import ground_truth
    from ..simulate import SimulationSettings
    from . import settings_from
    recipe = locs.metadata.get("simulation_settings")
    copies = locs.metadata.get("copies")
    if not recipe or not copies:
        raise ValueError("this table is not a simulation")
    truth = ground_truth(settings_from(SimulationSettings, recipe))
    f = truth.fluorophores
    cx = np.asarray(copies["x_nm"], dtype=float)
    cy = np.asarray(copies["y_nm"], dtype=float)
    step = 2 * np.pi / corners
    out = {}
    for copy in range(len(cx)):
        mine = f.copy == copy
        if not mine.any():
            out[copy] = 0
            continue
        theta = np.arctan2(f.xyz[mine, 1] - cy[copy], f.xyz[mine, 0] - cx[copy])
        phase = np.angle(np.sum(np.exp(1j * corners * theta))) / corners
        segment = np.floor(np.mod(theta - phase + step / 2, 2 * np.pi) / step)
        out[copy] = int(len(np.unique(segment)))
    return out


def site_truth(project, rows, corners: int) -> Optional[np.ndarray]:
    """The labelled corners of the pore under each site, or None when the
    data are not simulated.  A site's pore is the ``copy`` most of its
    localizations belong to."""
    tables = {}
    out = []
    for row in rows:
        roi = project.rois.get(row["roi_id"])
        if roi is None:
            return None
        locs = project.extract(roi)
        if "copy" not in locs or not len(locs):
            return None
        if roi.file_id not in tables:
            try:
                tables[roi.file_id] = true_corners(project.state(roi.file_id).locs,
                                                   corners)
            except ValueError:
                return None
        copy = np.asarray(locs["copy"]).astype(int)
        copy = copy[copy >= 0]
        if not len(copy):
            out.append(np.nan)
            continue
        out.append(tables[roi.file_id].get(int(np.bincount(copy).argmax()), np.nan))
    return np.asarray(out, dtype=float)


def simulated_efficiency(project, rows) -> Optional[float]:
    """The labelling efficiency the sites' file was simulated with, if it was."""
    roi = project.rois.get(rows[0]["roi_id"]) if rows else None
    if roi is None:
        return None
    recipe = project.state(roi.file_id).locs.metadata.get("simulation_settings") or {}
    value = (recipe.get("labelling") or {}).get("efficiency")
    return None if value is None else float(value)


def corner_column(rows) -> Optional[str]:
    """``n_corners``, or the qualified name a pipeline with two corner
    counters gives it."""
    keys = list(rows[0]) if rows else []
    return next((k for k in keys if k == "n_corners"),
                next((k for k in keys if k.endswith(".n_corners")), None))


def draw_histogram(ax, fitted: Dict[str, Any], settings: LabelingEfficiencySettings,
                   truth: Optional[Dict[str, Any]] = None) -> None:
    """The corners seen and the model at the fitted efficiency, fit range
    marked; the labelled corners of a simulation beside them."""
    corners = int(settings.corners)
    k = np.arange(corners + 1)
    counts = fitted["counts"]
    width = 0.4 if truth else 0.8
    ax.bar(k - (width / 2 if truth else 0), counts, width=width, color="0.7",
           label="measured")
    lo, hi = int(settings.fit_min), min(int(settings.fit_max), corners)
    model = corner_model(fitted["efficiency"], corners, int(settings.per_corner))
    scale = fitted["n_fitted"] / max(model[lo:hi + 1].sum(), 1e-300)
    ax.plot(k, scale * model, "-", color="#d62728", lw=1)
    ax.plot(k[lo:hi + 1], scale * model[lo:hi + 1], "o", color="#d62728", ms=4,
            label=f"fit: {100 * fitted['efficiency']:.1f} %")
    if truth:
        ax.bar(k + width / 2, truth["counts"], width=width, color="#2f7fd0",
               label=f"labelled (truth): {100 * truth['efficiency']:.1f} %")
    ax.set_xlabel("corners seen")
    ax.set_ylabel("pores")
    ax.set_xticks(k)
    ax.legend(fontsize=7, frameon=False)


@register("ROIManager/Analyze/NPC Labeling Efficiency")
class NPCLabelingEfficiency(Plugin):
    """The effective labelling efficiency from the corners seen per nuclear
    pore (NPC Corners)."""

    Settings = LabelingEfficiencySettings
    version = "1"

    def run(self, ctx: Context, settings: LabelingEfficiencySettings) -> Result:
        rows = ctx.site_table
        if rows is None and ctx.rois is not None:
            rows = ctx.rois.results()
        if not rows:
            raise ValueError("evaluate some ROIs first: there is no site table")
        column = corner_column(rows)
        if column is None:
            raise ValueError("the site table has no n_corners: add NPC Corners to "
                             "the evaluation and evaluate")
        n = np.array([r.get(column, np.nan) for r in rows], dtype=float)
        fitted = labeling_efficiency(n, settings)
        text = (f"effective labelling efficiency {100 * fitted['efficiency']:.1f} "
                f"± {100 * fitted['error']:.1f} % from {fitted['n_sites']} pores "
                f"({fitted['n_fitted']} in the fit range)")
        data = {"efficiency": fitted["efficiency"], "error": fitted["error"],
                "n_sites": fitted["n_sites"], "counts": fitted["counts"].tolist(),
                "corner_probability": fitted["corner_probability"]}
        truth = None
        known = site_truth(ctx.rois, rows, int(settings.corners)) \
            if ctx.rois is not None else None
        if known is not None and np.isfinite(known).sum() >= 5:
            try:
                truth = labeling_efficiency(known, settings)
            except ValueError:
                truth = None
        simulated = simulated_efficiency(ctx.rois, rows) if ctx.rois is not None else None
        if simulated is not None:
            data["simulated_efficiency"] = simulated
            text += f"; simulated with {100 * simulated:.0f} %"
        if truth is not None:
            data["true_efficiency"] = truth["efficiency"]
            data["true_counts"] = truth["counts"].tolist()
            text += (f"; the same pores' labelled corners give "
                     f"{100 * truth['efficiency']:.1f} %")
        return Result(text=text, data=data, settings=settings,
                      plot=lambda ax: draw_histogram(ax, fitted, settings, truth))
