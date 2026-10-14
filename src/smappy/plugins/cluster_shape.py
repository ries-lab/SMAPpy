"""The shape of a cluster: its size, extent, orientation and density.

The shape half of SMAP's ``Analyze/cluster/ClusterStatistics`` (whose pair
correlation and Ripley's K, despite its description, it never computed).
``per_region``, so it measures the view, one ROI, or -- the case it is for --
every cluster in turn, as a step of a block over ``cluster_id``; the numbers
land in the region table, where *Sort ROIs*, *Filter ROIs* and *Regions to
column* read them.

**What is measured**, of the positions as the layer shows them (its filter,
merged blinks or not), each layer on its own:

* the number of localizations and the centre (the mean position);
* the **convex hull**'s area, and in 3D its volume (`scipy.spatial.ConvexHull`,
  qhull); a hull of fewer than three points, or of points on one line, has
  no area, and gets 0;
* the **radius of gyration**, the root mean square distance from the centre,
  in x and y;
* the **covariance** of x and y: the square roots of its eigenvalues are the
  standard deviations along the long and the short axis, its leading
  eigenvector the long axis's angle; the extents along both axes are the
  positions' range after rotating onto them (SMAP's ``c_extl`` / ``c_exts``);
* the **elongation**, long / short standard deviation, and the **density**,
  localizations per um^2 of hull;
* with z, the z extent (max - min) and the hull's volume.

Where SMAP differs, and why:

* the angle is in **degrees**, 0 to 180 from the x axis, not radians;
* every value goes to the region table under a plain name; SMAP wrote the
  standard deviations, angle and extents onto the localizations
  (``c_stds`` ...) -- *Regions to column* does that here, for any value;
* no *min number of locs* or *skip first*: a filter on ``n_in_cluster``
  before the block, or *Filter ROIs* on ``n_locs`` after it, decides which
  clusters count; values that need more points than a region has are NaN;
* SMAP's ``boundary`` (an alpha shape, MATLAB's shrink factor 0.5) is left
  out: its area depends on a shrink factor with no physical meaning, while
  the hull's does not.  A concave cluster's density is underestimated by the
  hull, which the page says.

The measured standard deviations are the cluster's own spread and the
localization precision together (their squares add); nothing is subtracted,
since the precision differs from blink to blink and the subtraction is
unstable for clusters close to the precision in size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import Context, Plot, Plugin, Result, param, register

# the points drawn per layer in a figure: a scatter of a million is a grey
# square and seconds to draw
FIGURE_POINTS = 50_000
#: the ellipse drawn is this many standard deviations along each axis
ELLIPSE_SIGMAS = 2.0

NAMES_2D = ("n_locs", "centre_x_nm", "centre_y_nm", "area_nm2", "density_per_um2",
            "rg_nm", "std_long_nm", "std_short_nm", "angle_deg", "elongation",
            "extent_long_nm", "extent_short_nm")
NAMES_3D = ("centre_z_nm", "z_extent_nm", "volume_nm3")


# ------------------------------------------------------------- the algorithm

def hull_measure(points: np.ndarray) -> float:
    """The convex hull's area of (n, 2) points, or volume of (n, 3); 0 when
    the points span less than the dimension (too few, or on one line or
    plane)."""
    from scipy.spatial import ConvexHull, QhullError
    points = np.asarray(points, dtype=float)
    n, d = points.shape
    if n <= d:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except QhullError:
        return 0.0


def hull_outline(x: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    """The hull's corners, closed, as (m, 2), or None where it has no area."""
    from scipy.spatial import ConvexHull, QhullError
    points = np.column_stack((x, y)).astype(float)
    if len(points) < 3:
        return None
    try:
        corners = points[ConvexHull(points).vertices]
    except QhullError:
        return None
    return np.vstack((corners, corners[:1]))


def principal_axes(x: np.ndarray, y: np.ndarray):
    """``(std_long, std_short, angle)``: the square roots of the eigenvalues
    of the covariance (with n - 1, as MATLAB's ``cov``) and the long axis's
    angle in radians, in [0, pi)."""
    cov = np.cov(np.vstack((x, y)), ddof=1)
    values, vectors = np.linalg.eigh(cov)            # ascending
    # points on one line leave a rounding error of 1e-6 nm across it, which
    # would make an elongation of 10^8: below a millionth of the long axis
    # the short one is 0
    values = np.where(values > 1e-12 * values[1], values, 0.0)
    vx, vy = vectors[:, 1]
    return (float(np.sqrt(values[1])), float(np.sqrt(values[0])),
            float(np.mod(np.arctan2(vy, vx), np.pi)))


def cluster_shape(x, y, z=None) -> Dict[str, float]:
    """Every shape value of one set of positions in nm (see the module
    docstring), keyed by the names of the region table.

    ``z`` None measures in 2D and leaves the 3D values out.  A value that
    needs more points than there are -- a standard deviation of one point,
    an elongation or density with nothing below the line -- is NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    nan = float("nan")
    out = {name: nan for name in NAMES_2D}
    if z is not None:
        out.update({name: nan for name in NAMES_3D})
    out["n_locs"] = n
    if n == 0:
        return out
    cx, cy = float(x.mean()), float(y.mean())
    dx, dy = x - cx, y - cy
    out.update(centre_x_nm=cx, centre_y_nm=cy,
               rg_nm=float(np.sqrt(np.mean(dx * dx + dy * dy))))
    area = hull_measure(np.column_stack((x, y)))
    out["area_nm2"] = area
    if area > 0:
        out["density_per_um2"] = n / area * 1e6
    if n >= 2:
        long, short, angle = principal_axes(x, y)
        c, s = np.cos(angle), np.sin(angle)
        along, across = dx * c + dy * s, -dx * s + dy * c
        out.update(std_long_nm=long, std_short_nm=short,
                   angle_deg=float(np.degrees(angle)),
                   elongation=long / short if short > 0 else nan,
                   extent_long_nm=float(np.ptp(along)),
                   extent_short_nm=float(np.ptp(across)))
    if z is not None:
        z = np.asarray(z, dtype=float)
        out.update(centre_z_nm=float(z.mean()), z_extent_nm=float(np.ptp(z)),
                   volume_nm3=hull_measure(np.column_stack((x, y, z))))
    return out


# --------------------------------------------------------------- drawing

def draw(ax, shapes: List[dict], points: bool = True) -> None:
    """Each layer's positions with its hull and its 2-sigma ellipse, the long
    axis drawn through it.  ``shapes`` holds per layer ``name``, ``colour``,
    ``x``, ``y`` and ``values`` (`cluster_shape`'s)."""
    from matplotlib.patches import Ellipse
    for shape in shapes:
        x, y, v, colour = shape["x"], shape["y"], shape["values"], shape["colour"]
        if points:
            step = max(1, len(x) // FIGURE_POINTS)
            ax.plot(x[::step], y[::step], ".", color=colour, markersize=2,
                    alpha=0.5, label=shape["name"])
        outline = hull_outline(x, y)
        if outline is not None:
            ax.plot(outline[:, 0], outline[:, 1], "-", color=colour, linewidth=0.8)
        if np.isfinite(v["std_long_nm"]):
            angle = v["angle_deg"]
            ax.add_patch(Ellipse((v["centre_x_nm"], v["centre_y_nm"]),
                                 2 * ELLIPSE_SIGMAS * v["std_long_nm"],
                                 2 * ELLIPSE_SIGMAS * v["std_short_nm"], angle=angle,
                                 fill=False, edgecolor="k", linestyle="--",
                                 linewidth=0.9))
            r = ELLIPSE_SIGMAS * v["std_long_nm"]
            a = np.radians(angle)
            ax.plot(v["centre_x_nm"] + r * np.cos(a) * np.array([-1, 1]),
                    v["centre_y_nm"] + r * np.sin(a) * np.array([-1, 1]),
                    "k-", linewidth=0.7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    if points and len(shapes) > 1:
        ax.legend(fontsize=7, frameon=False, markerscale=4)


def summary(name: str, v: Dict[str, float]) -> str:
    """One layer's values as a line of text."""
    line = (f"{name}: {v['n_locs']} localizations, area {v['area_nm2']:.4g} nm², "
            f"{v['density_per_um2']:.4g} /µm², Rg {v['rg_nm']:.4g} nm, "
            f"std {v['std_long_nm']:.4g} × {v['std_short_nm']:.4g} nm at "
            f"{v['angle_deg']:.1f}°, elongation {v['elongation']:.3g}")
    if "z_extent_nm" in v:
        line += f", z extent {v['z_extent_nm']:.4g} nm, volume {v['volume_nm3']:.4g} nm³"
    return line


# ---------------------------------------------------------------- the plugin

@dataclass
class ClusterShapeSettings:
    use_z: bool = param(True, label="use z",
                        help="also the volume, the z extent and the z centre, "
                             "when the table has z_nm")


@register("Analysis/Measure/Cluster shape")
class ClusterShape(Plugin):
    """The size, extent, orientation and density of the localizations: the
    convex hull's area, the radius of gyration, the standard deviations
    along the long and the short axis, per cluster or per ROI as well."""

    Settings = ClusterShapeSettings
    changes_locs = False
    per_region = True
    version = "1"

    def run(self, ctx: Context, settings: ClusterShapeSettings) -> Result:
        from .histogram import _layers
        shapes, values, lines = [], {}, []
        for i, (name, colour, (table, selection)) in enumerate(_layers(ctx)):
            for column in ("x_nm", "y_nm"):
                if column not in table:
                    raise ValueError(f"layer {name} has no column {column!r}; it has "
                                     + ", ".join(sorted(table.keys())))
            locs = selection.apply(table)
            x = np.asarray(locs["x_nm"], dtype=float)
            y = np.asarray(locs["y_nm"], dtype=float)
            three_d = settings.use_z and "z_nm" in locs
            z = np.asarray(locs["z_nm"], dtype=float) if three_d else None
            keep = np.isfinite(x) & np.isfinite(y)
            if z is not None:
                keep &= np.isfinite(z)
                z = z[keep]
            x, y = x[keep], y[keep]
            v = cluster_shape(x, y, z)
            suffix = "" if i == 0 else f"_L{i + 1}"
            values.update({f"{key}{suffix}": value for key, value in v.items()})
            shapes.append({"name": name, "colour": colour, "x": x, "y": y, "values": v})
            lines.append(summary(name, v))
        if not any(len(s["x"]) for s in shapes) and not ctx.per_region:
            raise ValueError("no localizations to measure")

        def plot(ax):
            draw(ax, shapes)

        return Result(text="\n".join(lines), plot=Plot(plot, size=(4.6, 4.2)),
                      values=values, data={"shapes": shapes}, settings=settings,
                      title="cluster shape")
