"""A circle through the localizations: its centre, radius and how well it fits.

SMAP's ``ROIManager/Evaluate/circleFitter``, the part of it that fits a
circle to the localizations.  ``per_region``, so it measures the view (the
drawn ROI), every ROI, or every cluster; *Centre ROIs* moves each ROI onto
the centre it finds.

**The fit is the NPC segmenter's** (`npc.fit_circle`): Gauss-Newton on the
distances from the circle with a soft-L1 loss, so a localization further
than about *outlier scale* from the circle counts with its distance rather
than its square, and the background inside the ROI does not pull the centre.
It starts at the median of the localizations, with the radius the median
distance from there.  With a *fixed radius* only the centre is fitted, which
is what holds on a ring that shows two or three of its corners; it starts
from the free circle's centre, since from the median, on the ring, the
mirror image of an arc's centre fits as well.

**Errors** are the usual ones of a least-squares fit, at the minimum: the
covariance ``s^2 (J^T W J)^-1``, with ``J`` the derivatives of the distances
by centre and radius, ``W`` the soft-L1 weights, and ``s^2`` the weighted
mean square residual (over n - p).  On a simulated ring they match the
scatter of the fitted radius over repeats (`tests/test_circle_fit.py`).

**Goodness of fit**: the root mean square distance from the circle, and,
where the table has a precision, the reduced chi-square of those distances
in units of each localization's own precision -- about 1 for a thin ring,
more for a ring wider than the precision, or a structure that is no ring.

Left out of SMAP's evaluator: the fit of a Gaussian-blurred ring to a
rendered image (a second estimate of the same circle, with a pixel size and
blur of its own), the counts in 1.5 radii and the asymmetry.  The fit there
was least squares on the algebraic distance (``implicitfit``), which the
background pulls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import Context, Plot, Plugin, Result, param, register
from .npc import fit_circle

# below this a free circle is not determined (three points fix one exactly)
MIN_FREE, MIN_FIXED = 4, 3
FIGURE_POINTS = 50_000


def circle_fit(x, y, precision=None, radius: Optional[float] = None,
               scale: float = 20.0, center=None) -> Dict[str, float]:
    """The circle through ``x``, ``y`` (nm) and its errors.

    ``radius`` fixes the radius (only the centre is fitted); ``center`` is
    where the fit starts, the median of the points when None.  Returns
    ``n_locs``, ``x_nm``, ``y_nm``, ``radius_nm`` and their errors
    (``x_err_nm`` ...; a fixed radius has error 0), ``rms_nm`` and
    ``chi2_red`` (NaN without ``precision``).  Too few points: NaN, with
    the count.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    sigma = None
    if precision is not None:
        sigma = np.asarray(precision, dtype=float)
        keep &= np.isfinite(sigma) & (sigma > 0)
        sigma = sigma[keep]
    x, y = x[keep], y[keep]
    n = len(x)
    free = radius is None or not radius > 0
    nan = float("nan")
    out = {"n_locs": n, "x_nm": nan, "y_nm": nan, "radius_nm": nan,
           "x_err_nm": nan, "y_err_nm": nan, "radius_err_nm": nan,
           "rms_nm": nan, "chi2_red": nan}
    if n < (MIN_FREE if free else MIN_FIXED):
        return out
    start = (np.median(x), np.median(y)) if center is None else center
    if not free and center is None and n >= MIN_FREE:
        # from the median, on the ring itself, a fixed radius fits an arc
        # as well with the centre on its outer side as on its inner; the
        # free circle's centre is on the right side
        free_fit = fit_circle(x, y, start, scale=scale)
        if np.all(np.isfinite(free_fit)):
            start = free_fit[:2]
    x0, y0, r = fit_circle(x, y, start, radius=None if free else float(radius),
                           scale=scale)
    p = 3 if free else 2
    dx, dy = x - x0, y - y0
    d = np.maximum(np.hypot(dx, dy), 1e-9)
    residual = d - r
    w = 1.0 / np.sqrt(1.0 + (residual / scale) ** 2)
    J = np.column_stack([-dx / d, -dy / d] + ([-np.ones(n)] if free else []))
    s2 = float(np.sum(w * residual ** 2)) / max(n - p, 1)
    try:
        cov = s2 * np.linalg.inv(J.T @ (J * w[:, None]))
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(p, np.nan)
    out.update(x_nm=float(x0), y_nm=float(y0), radius_nm=float(r),
               x_err_nm=float(err[0]), y_err_nm=float(err[1]),
               radius_err_nm=float(err[2]) if free else 0.0,
               rms_nm=float(np.sqrt(np.mean(residual ** 2))))
    if sigma is not None:
        out["chi2_red"] = float(np.sum((residual / sigma) ** 2) / max(n - p, 1))
    return out


def draw(ax, fits: List[dict]) -> None:
    """Each layer's localizations and its fitted circle, the centre marked."""
    from matplotlib.patches import Circle
    for fit in fits:
        x, y, v, colour = fit["x"], fit["y"], fit["values"], fit["colour"]
        step = max(1, len(x) // FIGURE_POINTS)
        ax.plot(x[::step], y[::step], ".", color=colour, markersize=2, alpha=0.5,
                label=fit["name"])
        if np.isfinite(v["radius_nm"]):
            ax.add_patch(Circle((v["x_nm"], v["y_nm"]), v["radius_nm"], fill=False,
                                edgecolor="k", linewidth=1.0))
            ax.plot(v["x_nm"], v["y_nm"], "+", color="k", markersize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    if len(fits) > 1:
        ax.legend(fontsize=7, frameon=False, markerscale=4)


def summary(name: str, v: Dict[str, float]) -> str:
    if not np.isfinite(v["radius_nm"]):
        return f"{name}: {v['n_locs']} localizations, too few for a circle"
    line = (f"{name}: {v['n_locs']} localizations, centre ({v['x_nm']:.1f} ± "
            f"{v['x_err_nm']:.1f}, {v['y_nm']:.1f} ± {v['y_err_nm']:.1f}) nm, "
            f"radius {v['radius_nm']:.1f} ± {v['radius_err_nm']:.1f} nm, "
            f"rms {v['rms_nm']:.1f} nm")
    if np.isfinite(v["chi2_red"]):
        line += f", reduced chi² {v['chi2_red']:.2f}"
    return line


@dataclass
class CircleFitSettings:
    radius_nm: float = param(0.0, label="fixed radius", unit="nm", min=0.0,
                             help="0: the radius is fitted.  Otherwise only the "
                                  "centre, on a circle of this radius")
    scale_nm: float = param(20.0, label="outlier scale", unit="nm", min=0.1,
                            advanced=True,
                            help="localizations further than about this from the "
                                 "circle count with their distance, not its square")


@register("Analysis/Measure/Circle fit")
class CircleFit(Plugin):
    """A circle fitted to the localizations, per layer: its centre, radius and
    their errors, and how well it fits; per ROI or cluster as well."""

    Settings = CircleFitSettings
    changes_locs = False
    per_region = True
    follow_roi = True
    version = "1"

    def run(self, ctx: Context, settings: CircleFitSettings) -> Result:
        from .histogram import _layers
        from .npc import precision_column
        fits, values, lines = [], {}, []
        for i, (name, colour, (table, selection)) in enumerate(_layers(ctx)):
            for column in ("x_nm", "y_nm"):
                if column not in table:
                    raise ValueError(f"layer {name} has no column {column!r}; it has "
                                     + ", ".join(sorted(table.keys())))
            locs = selection.apply(table)
            x = np.asarray(locs["x_nm"], dtype=float)
            y = np.asarray(locs["y_nm"], dtype=float)
            sigma_name = precision_column(locs)
            sigma = np.asarray(locs[sigma_name], float) if sigma_name else None
            v = circle_fit(x, y, sigma, radius=settings.radius_nm or None,
                           scale=settings.scale_nm)
            suffix = "" if i == 0 else f"_L{i + 1}"
            values.update({f"{key}{suffix}": value for key, value in v.items()})
            fits.append({"name": name, "colour": colour, "x": x, "y": y, "values": v})
            lines.append(summary(name, v))
        if not ctx.per_region and not any(np.isfinite(f["values"]["radius_nm"])
                                          for f in fits):
            raise ValueError(f"too few localizations for a circle (at least "
                             f"{MIN_FREE if not settings.radius_nm else MIN_FIXED})")

        def plot(ax):
            draw(ax, fits)

        return Result(text="\n".join(lines), plot=Plot(plot, size=(4.6, 4.2)),
                      values=values, data={"fits": fits}, settings=settings,
                      title="circle fit")
