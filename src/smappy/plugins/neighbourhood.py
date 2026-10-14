"""Neighbourhood: a number per localization that describes what is around it.

SMAP's ``density_calculator`` (neighbours in a circle, a cylinder or under a
Gaussian), ``SpatialFilterLocs`` (the median, mean, ... of a field around
each localization) and the density half of ``ClusterVoronoi`` (one over the
area of the localization's Voronoi cell, from SharpViSu), as one plugin with
a *quantity* choice.  `docs/smap_porting/wave2_design.md`, step 9, has the
decisions.  This segments nothing: it writes a column, which can be
filtered on and coloured by, and which a clusterer can threshold.

**Two layers**, as in SMAP: *around* -- whose localizations get the value --
and *neighbours from* -- whose localizations are counted or averaged.  Each
is read as the layer shows it (`Context.table`), with its own filter and its
own *merge blinks*, so what is counted is chosen where it always is.  The
same layer twice is the local density; two layers are a cross-density, how
many of channel B lie around each localization of channel A.

Where it departs from SMAP's:

* **Self is not counted.**  A row of the *around* table that is also among
  the neighbours (the same layer, or two layers over the same table whose
  filters both keep it) is left out of its own count, so an isolated
  localization has 0 neighbours.  SMAP's C counts it (and its Gaussian adds
  ``exp(0) = 1``).  The *statistic* keeps it, as SMAP's does: it is a
  smoothing filter, and a localization alone in its circle gets its own
  value rather than nothing.
* **In 3D the region is a cylinder**, |dz| within the *z half-height*, as the
  design asked; SMAP's ``countneighbours3Dcirc2`` is in fact an ellipsoid,
  ``(dx^2 + dy^2)/r^2 + dz^2/h^2 < 1``.  The Gaussian is SMAP's, cut at 3
  sigma laterally and 2.5 in z.
* **Every neighbour is counted.**  SMAP's sweep over x stops one short of
  the end of the sorted list, so the localization with the largest x was
  never anyone's neighbour.
* **Voronoi density** is per square micrometre, NaN for an open cell at the
  edge of the field (SMAP set those to 0, which reads as "empty" when it
  means "unknown").  Localizations at exactly the same position share one
  cell, and each gets their number over its area, as SMAP did; SMAP also
  rounded positions to whole nanometres first, which merged near-duplicates
  -- here only exact duplicates are merged.
* **Per file**: two files' positions are never mixed (SMAP had a choice).
* The search is SMAP's sweep over sorted positions made two-dimensional
  (`csrc/neighbours.cpp`): sorted by square cells a radius wide rather than
  by x alone, so a point tests the nine cells around it rather than a strip
  as tall as the field, on every core.  5 M localizations at 50 nm take
  about 5 s; a k-d tree took 17 s for the count and over a minute for the
  pairs a cylinder or a statistic needs.

**Writing the column.**  The value goes onto the localizations of the
*around* layer that pass its filter, NaN on the rest.  On a layer that shows
merged blinks, the blink's value goes onto each of its localizations through
`group_id`, as DBSCAN does.  The column is a recipe with the rule ``mean``
(`mathparse.remember`): merged again, a blink takes the mean over its
localizations -- the same value when it was computed on blinks, and the
natural blink value when it was computed on localizations -- and the NaN of
a localization outside the filter does not spoil its blink, since the mean
is over the rows that have a value (`group.group`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..locs import Localizations
from . import Context, FormState, Notice, Plot, Plugin, Result, param, register

# the scatter of the figure shows this many points at most
FIGURE_POINTS = 50_000
# where the Gaussian is cut, in sigmas: SMAP's 3 laterally, 2.5 in z
GAUSS_CUT_XY = 3.0
GAUSS_CUT_Z = 2.5
# above this many seconds a run asks first (`preflight`)
ASK_ABOVE_S = 20.0

STATISTICS = ("mean", "median", "min", "max", "std", "quantile")


# ------------------------------------------------------------- the algorithm

MODES = {"count": 0, "gauss": 1, "mean": 2, "median": 3, "min": 4, "max": 5,
         "std": 6, "quantile": 7}


def _search(around: np.ndarray, neighbours: np.ndarray, mode: str, reach: float,
            sigma: float = 0.0, z_reach: Optional[float] = None, z_sigma: float = 0.0,
            values: Optional[np.ndarray] = None, same: Optional[np.ndarray] = None,
            q: float = 0.5) -> np.ndarray:
    """The C++ search (`csrc/neighbours.cpp`): per around point, over the
    neighbours laterally within ``reach`` (and within ``z_reach`` in z)."""
    from .._neighbours import around as search
    use_z = (z_reach is not None and around.shape[1] > 2 and neighbours.shape[1] > 2)
    return search(around[:, 0], around[:, 1], around[:, 2] if use_z else None,
                  neighbours[:, 0], neighbours[:, 1],
                  neighbours[:, 2] if use_z else None,
                  values, None if same is None else np.asarray(same, dtype=np.int64),
                  MODES[mode], float(reach), float(sigma),
                  float(z_reach) if use_z else 0.0, float(z_sigma), float(q), 0)


def count_neighbours(around: np.ndarray, neighbours: np.ndarray, radius: float,
                     z_half: Optional[float] = None, gaussian: bool = False,
                     same: Optional[np.ndarray] = None) -> np.ndarray:
    """How many ``neighbours`` lie around each ``around`` point.

    Within ``radius`` laterally (a circle), and within ``z_half`` in z when
    it is given and the points are (n, 3) (a cylinder).  ``gaussian``
    weights each neighbour by ``exp(-r^2 / 2 radius^2 - dz^2 / 2 z_half^2)``
    instead, out to 3 and 2.5 sigmas -- SMAP's ``countneighbours*Gauss``.

    ``same[i]`` is the index in ``neighbours`` of around point ``i`` itself,
    or -1: a point is not its own neighbour (see the module docstring).
    """
    around = np.asarray(around, dtype=float)
    neighbours = np.asarray(neighbours, dtype=float)
    if not radius > 0:
        raise ValueError("the radius must be positive")
    if not len(around) or not len(neighbours):
        return np.zeros(len(around))
    if gaussian:
        return _search(around, neighbours, "gauss", GAUSS_CUT_XY * radius, radius,
                       None if z_half is None else GAUSS_CUT_Z * z_half,
                       z_half or 0.0, same=same)
    return _search(around, neighbours, "count", radius, z_reach=z_half, same=same)


def neighbour_statistic(around: np.ndarray, neighbours: np.ndarray,
                        values: np.ndarray, radius: float, statistic: str = "median",
                        quantile: float = 0.5, z_half: Optional[float] = None,
                        same: Optional[np.ndarray] = None) -> np.ndarray:
    """A statistic of ``values`` (one per neighbour) over the neighbours of
    each around point: ``mean``, ``median``, ``min``, ``max``, ``std`` or a
    ``quantile`` (0 to 1, linear between the order statistics, as numpy's).
    NaN where no neighbour has a finite value.

    A point that is also a neighbour is among its own neighbours, as in
    SMAP's SpatialFilterLocs: this is a smoothing filter.  ``same`` is
    accepted for symmetry with `count_neighbours` and changes nothing.
    """
    if statistic not in STATISTICS:
        raise ValueError(f"no statistic {statistic!r}: one of {', '.join(STATISTICS)}")
    around = np.asarray(around, dtype=float)
    neighbours = np.asarray(neighbours, dtype=float)
    if not radius > 0:
        raise ValueError("the radius must be positive")
    if not len(around) or not len(neighbours):
        return np.full(len(around), np.nan)
    return _search(around, neighbours, statistic, radius, z_reach=z_half,
                   values=np.asarray(values, dtype=float), q=quantile)


def _cell_areas(p: np.ndarray, box=None, bounds=None, tree=None, ids=None):
    """The Voronoi cell area of each of ``p`` from its Delaunay triangles, NaN
    where the cell is open; and, given ``box`` (x0, y0, x1, y1), whether each
    area is certain: whether every triangle around the point is one of the
    whole field's triangulation.  It is when its circumcircle holds no point
    of the field (the empty-circle property): certain at once for a circle
    inside the box, or beyond a side of the box past which there are no
    points (``bounds``, the field's own box), since the points in the box are
    all here; the rest -- a few, along the box's edge -- are asked of
    ``tree``, a k-d tree of the whole field, in which ``p`` are the rows
    ``ids``.
    """
    from scipy.spatial import Delaunay
    n = len(p)
    area = np.full(n, np.nan)
    sure = np.zeros(n, dtype=bool)
    if n < 3:
        return area, sure
    centre = p.mean(axis=0)            # circumcentres are differences of large numbers
    q = p - centre
    try:
        tri = Delaunay(q)
    except Exception:                  # noqa: BLE001 -- all on one line
        return area, sure
    a, b, c = (q[tri.simplices[:, k]] for k in range(3))
    bx, by = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    cx, cy = c[:, 0] - a[:, 0], c[:, 1] - a[:, 1]
    d = 2 * (bx * cy - by * cx)
    b2, c2 = bx * bx + by * by, cx * cx + cy * cy
    with np.errstate(divide="ignore", invalid="ignore"):
        ox = (cy * b2 - by * c2) / d
        oy = (bx * c2 - cx * b2) / d
    ux, uy = a[:, 0] + ox, a[:, 1] + oy
    # every (point, triangle), in order of angle around the point
    owner = tri.simplices.ravel()
    corner = np.repeat(np.arange(len(tri.simplices)), 3)
    vx, vy = ux[corner], uy[corner]
    order = np.lexsort((np.arctan2(vy - q[owner, 1], vx - q[owner, 0]), owner))
    owner, vx, vy, corner = owner[order], vx[order], vy[order], corner[order]
    starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    # the shoelace formula, each corner with the next of its own cell
    nxt = np.arange(len(owner)) + 1
    nxt[np.r_[starts[1:], len(owner)] - 1] = starts
    cross = vx * vy[nxt] - vx[nxt] * vy
    area[owner[starts]] = 0.5 * np.abs(np.add.reduceat(cross, starts))
    hull = np.unique(tri.convex_hull)
    area[hull] = np.nan                                 # open cells
    if box is None:
        sure[:] = True
        return area, sure
    radius = np.sqrt(ox * ox + oy * oy)
    x0, y0, x1, y1 = (v - o for v, o in zip(box, (centre[0], centre[1]) * 2))
    gx0, gy0, gx1, gy1 = (v - o for v, o in zip(bounds, (centre[0], centre[1]) * 2))
    inside = (((ux - radius >= x0) | (x0 <= gx0)) & ((ux + radius <= x1) | (x1 >= gx1))
              & ((uy - radius >= y0) | (y0 <= gy0)) & ((uy + radius <= y1) | (y1 >= gy1)))
    inside &= np.isfinite(radius)
    if tree is not None:
        ask = np.flatnonzero(~inside & np.isfinite(radius))
        if len(ask):
            # the four points nearest the centre, the triangle's own corners
            # aside: in a long thin triangle along the field's edge the circle
            # is so large that rounding puts its own corners just inside it
            distance, nearest = tree.query(
                np.column_stack([ux[ask] + centre[0], uy[ask] + centre[1]]), k=4)
            own = ids[tri.simplices[ask]]
            other = ~(nearest[:, :, None] == own[:, None, :]).any(axis=2)
            inner = other & (distance < (radius[ask] * (1 - 1e-9))[:, None])
            inside[ask[~inner.any(axis=1)]] = True
    good = np.ones(n, dtype=bool)
    np.logical_and.at(good, owner, inside[corner])
    sure = good
    # on the hull of the tile: open, but whether in the whole field too
    # cannot be told from here (`voronoi_areas` knows the field's hull)
    sure[hull] = False
    return area, sure


# points per tile of the Voronoi density: Qhull takes 9 us a point at 50 k
# and 21 us at a million, so tiles are faster even on one core
VORONOI_TILE = 50_000


def voronoi_areas(points: np.ndarray, tile: int = VORONOI_TILE, threads: int = 0
                  ) -> np.ndarray:
    """The area of each point's Voronoi cell (points distinct), NaN where it
    is open at the edge of the field.

    The field is cut into tiles of about ``tile`` points, each triangulated
    with a margin around it, on every core (Qhull lets go of the GIL); a
    point's area is kept when `_cell_areas` says it is certain, and a tile
    with points that are not is done once more with twice the margin.  What
    is left are points a nanometre or two inside the field's edge, whose
    cells reach along the edge to points far away: they are done together in
    one triangulation of a band along the field's hull, widened until every
    one is certain -- at worst the whole field.  So the answer is the one a
    single triangulation gives, in a time linear in the points: 5 M in about
    2 minutes on 4 cores where one triangulation took over 3.
    """
    from concurrent.futures import ThreadPoolExecutor
    from scipy.spatial import ConvexHull, cKDTree
    p = np.asarray(points, dtype=float)[:, :2]
    n = len(p)
    if n <= tile * 1.5:
        return _cell_areas(p)[0]
    bounds = (*p.min(axis=0), *p.max(axis=0))
    k = int(np.ceil(np.sqrt(n / tile)))
    # tiles of equal counts: strips by x quantiles, each cut by y quantiles
    xcuts = np.quantile(p[:, 0], np.linspace(0, 1, k + 1))
    xcuts[0], xcuts[-1] = -np.inf, np.inf
    strip = np.clip(np.searchsorted(xcuts, p[:, 0], "right") - 1, 0, k - 1)
    jobs = []
    for i in range(k):
        rows = np.flatnonzero(strip == i)
        if not len(rows):
            continue
        ycuts = np.quantile(p[rows, 1], np.linspace(0, 1, k + 1))
        ycuts[0], ycuts[-1] = -np.inf, np.inf
        part = np.clip(np.searchsorted(ycuts, p[rows, 1], "right") - 1, 0, k - 1)
        for j in range(k):
            mine = rows[part == j]
            if len(mine):
                jobs.append(mine)
    tree = cKDTree(p)
    xs_order = np.argsort(p[:, 0], kind="stable")
    xs = p[xs_order, 0]
    area = np.full(n, np.nan)
    hull = ConvexHull(p)
    on_hull = np.zeros(n, dtype=bool)
    on_hull[hull.vertices] = True                       # open: NaN
    left: list = []

    def one(mine):
        lo, hi = p[mine].min(axis=0), p[mine].max(axis=0)
        margin = 4 * np.sqrt(max((hi - lo).prod(), 1.0) / len(mine))
        todo = np.sort(mine[~on_hull[mine]])
        for _ in range(2):
            box = (lo[0] - margin, lo[1] - margin, hi[0] + margin, hi[1] + margin)
            a = np.searchsorted(xs, box[0], side="left")
            near = xs_order[a:np.searchsorted(xs, box[2], side="right")]
            near = np.sort(near[(p[near, 1] >= box[1]) & (p[near, 1] <= box[3])])
            got, sure = _cell_areas(p[near], box, bounds, tree, near)
            at = np.searchsorted(near, todo)
            area[todo[sure[at]]] = got[at[sure[at]]]
            todo = todo[~sure[at]]
            if not len(todo):
                return
            margin *= 2
        left.append(todo)

    workers = threads if threads > 0 else min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, jobs))
    if not left:
        return area
    todo = np.sort(np.concatenate(left))
    # each point's distance to the field's edge: inside a convex hull, the
    # nearest of the lines through its edges (Qhull's facets, normal outward)
    depth = -(p @ hull.equations[:, :2].T + hull.equations[:, 2]).max(axis=1)
    width = depth[todo].max() + 4 * np.sqrt(hull.volume / n)
    # an empty box, and points on every side of it: every triangle is asked
    # of the tree
    nowhere = (np.inf, np.inf, -np.inf, -np.inf)
    everywhere = (-np.inf, -np.inf, np.inf, np.inf)
    while len(todo):
        band = np.union1d(np.flatnonzero(depth <= width), todo)
        got, sure = _cell_areas(p[band], nowhere, everywhere, tree, band)
        at = np.searchsorted(band, todo)
        done = sure[at] | (len(band) == n)
        area[todo[done]] = got[at[done]]
        todo = todo[~done]
        width *= 2
    return area


def voronoi_density(points: np.ndarray) -> np.ndarray:
    """One over the area of each point's Voronoi cell, per square micrometre
    (points in nm).  NaN for a cell open at the edge of the field.

    Points at exactly the same position share a cell; each gets their number
    over its area (SMAP, ``VorArea``).  2D: only x and y are read.  The
    cells come from the Delaunay triangulation (`voronoi_areas`): a cell's
    corners are the centres of the circles through its point's triangles.
    """
    points = np.asarray(points, dtype=float)[:, :2]
    out = np.full(len(points), np.nan)
    if len(points) < 4:
        return out
    unique, inverse, repeats = np.unique(points, axis=0, return_inverse=True,
                                         return_counts=True)
    if len(unique) < 4:
        return out
    area = voronoi_areas(unique)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(area > 0, repeats * 1e6 / area, np.nan)
    return density[np.ravel(inverse)]


# ------------------------------------------------------------- the plugin

def _layer_names(ctx: Optional[Context]):
    session = getattr(ctx, "session", None)
    layers = getattr(session, "layers", None) or []
    return [(i + 1, f"{i + 1}: {layer.name}") for i, layer in enumerate(layers)
            if not layer.is_image]


def _default_layers():
    return [(i, f"layer {i}") for i in range(1, 7)]


@dataclass
class NeighbourhoodSettings:
    around: int = param(1, label="around", choices=_default_layers,
                        help="the layer whose localizations get the value")
    neighbours_from: int = param(1, label="neighbours from", choices=_default_layers,
                                 help="the layer whose localizations are counted; "
                                      "another layer gives a cross-density")
    quantity: str = param("count", label="quantity",
                          choices=(("count", "neighbours within r"),
                                   ("statistic", "statistic of a column"),
                                   ("voronoi", "Voronoi density")),
                          help="what is written per localization")
    radius_nm: float = param(50.0, label="radius", unit="nm", min=0.1,
                             help="r: the circle's radius, or the Gaussian's sigma")
    use_z: bool = param(True, label="in 3D",
                        help="a cylinder in x, y and z when both tables have z_nm")
    z_half_nm: float = param(100.0, label="z half-height", unit="nm", min=0.1,
                             help="how far in z a neighbour may be, or the "
                                  "Gaussian's sigma in z")
    gaussian: bool = param(False, label="Gaussian weight",
                           help="weight each neighbour by a Gaussian of its "
                                "distance rather than counting a sharp circle")
    column: str = param("", label="column", choices=lambda: [("", "-")],
                        help="the column whose statistic is taken")
    statistic: str = param("median", label="statistic",
                           choices=tuple((s, s) for s in STATISTICS),
                           help="of the column over the neighbours")
    quantile: float = param(0.9, label="quantile", min=0.0, max=1.0,
                            help="0.5 is the median")
    name: str = param("", label="new column",
                      help="empty: a name from the settings")


def default_name(settings: NeighbourhoodSettings) -> str:
    """``density_r50``, ``photons_median_r50``, ``voronoi_density`` ..."""
    r = f"r{settings.radius_nm:g}"
    other = (f"_from{settings.neighbours_from}"
             if settings.neighbours_from != settings.around else "")
    if settings.quantity == "voronoi":
        return "voronoi_density"
    if settings.quantity == "statistic":
        stat = (f"q{settings.quantile * 100:g}" if settings.statistic == "quantile"
                else settings.statistic)
        return f"{settings.column or 'column'}_{stat}_{r}{other}"
    return f"density_{'gauss_' if settings.gaussian else ''}{r}{other}"


def _points(table: Localizations, use_z: bool) -> np.ndarray:
    if "x_nm" not in table or "y_nm" not in table:
        raise ValueError("Neighbourhood needs x_nm and y_nm: the table has "
                         f"{', '.join(sorted(table.keys()))}")
    columns = ["x_nm", "y_nm"] + (["z_nm"] if use_z and "z_nm" in table else [])
    return np.column_stack([np.asarray(table[c], dtype=float) for c in columns])


def _files(table: Localizations) -> np.ndarray:
    return (np.asarray(table["filenumber"]) if "filenumber" in table
            else np.zeros(len(table), dtype=np.int64))


def neighbourhood(around: Tuple[Localizations, np.ndarray],
                  neighbours: Tuple[Localizations, np.ndarray],
                  settings: NeighbourhoodSettings, report=None) -> np.ndarray:
    """The value per row of the around table, NaN outside its mask.

    ``around`` and ``neighbours`` are each ``(table, mask)``: the table a
    layer shows and its filter over it.  File by file.
    """
    table_a, use_a = around
    table_b, use_b = neighbours
    use_a = np.asarray(use_a, dtype=bool).copy()
    use_b = np.asarray(use_b, dtype=bool).copy()
    s = settings
    if s.quantity == "voronoi" and (table_a is not table_b or s.around != s.neighbours_from):
        raise ValueError("Voronoi density is of one layer's own localizations: "
                         "set *neighbours from* to the *around* layer, or count "
                         "neighbours within r for a cross-density")
    use_z = s.use_z and s.quantity != "voronoi" and "z_nm" in table_a and "z_nm" in table_b
    pa = _points(table_a, use_z)
    pb = _points(table_b, use_z)
    use_a &= np.isfinite(pa).all(axis=1)
    use_b &= np.isfinite(pb).all(axis=1)
    values = None
    if s.quantity == "statistic":
        if not s.column:
            raise ValueError("choose a column")
        if s.column not in table_b:
            raise ValueError(f"no column {s.column!r} in the neighbours' table: it has "
                             + ", ".join(sorted(table_b.keys())))
        values = np.asarray(table_b[s.column], dtype=float)
    files_a, files_b = _files(table_a), _files(table_b)
    out = np.full(len(table_a), np.nan)
    z_half = s.z_half_nm if use_z else None
    files = np.unique(files_a[use_a])
    for k, f in enumerate(files):
        if report:
            report(f"file {k + 1} of {len(files)}" if len(files) > 1 else "neighbours")
        rows_a = np.flatnonzero(use_a & (files_a == f))
        rows_b = np.flatnonzero(use_b & (files_b == f))
        same = None
        if table_a is table_b:
            # a row of both is one point: never its own neighbour
            where = np.full(len(table_b), -1, dtype=np.int64)
            where[rows_b] = np.arange(len(rows_b))
            same = where[rows_a]
        if s.quantity == "voronoi":
            out[rows_a] = voronoi_density(pa[rows_a])
        elif s.quantity == "statistic":
            out[rows_a] = neighbour_statistic(pa[rows_a], pb[rows_b], values[rows_b],
                                              s.radius_nm, s.statistic, s.quantile,
                                              z_half, same)
        else:
            out[rows_a] = count_neighbours(pa[rows_a], pb[rows_b], s.radius_nm,
                                           z_half, s.gaussian, same)
    return out


def per_localization(locs: Localizations, grouped: Localizations,
                     values: np.ndarray) -> np.ndarray:
    """A blink's value on each of its localizations, through ``group_id``;
    NaN for a localization whose blink is not in ``grouped``."""
    if "group_id" not in locs or "group_id" not in grouped:
        raise ValueError("the merged blinks have no group_id to map the values "
                         "back to the localizations: run it with blinks not merged")
    gid = np.asarray(grouped["group_id"])
    order = np.argsort(gid, kind="stable")
    wanted = np.asarray(locs["group_id"])
    out = np.full(len(locs), np.nan)
    if not len(gid):
        return out
    at = np.clip(np.searchsorted(gid[order], wanted), 0, len(gid) - 1)
    hit = gid[order][at] == wanted
    out[hit] = np.asarray(values, dtype=float)[order[at[hit]]]
    return out


def expected_seconds(n_around: int, n_neighbours: int, area_nm2: float,
                     settings: NeighbourhoodSettings) -> float:
    """A rough run time, from 5 M localizations on 4 cores: sorting and
    reading cost about 0.5 us a point, testing a candidate 2 ns (a point
    tests the nine cells a search radius wide around it), a statistic 5 ns
    more a neighbour, and the Voronoi cells 12 us a point."""
    n_a, n_b = max(n_around, 0), max(n_neighbours, 0)
    if settings.quantity == "voronoi":
        return 12e-6 * n_a
    reach = settings.radius_nm * (GAUSS_CUT_XY if settings.gaussian
                                  and settings.quantity == "count" else 1.0)
    density = n_b / max(area_nm2, 1.0)
    candidates = 9 * reach ** 2 * density
    seconds = 0.5e-6 * (n_a + n_b) + 2e-9 * n_a * candidates
    if settings.quantity == "statistic":
        seconds += 5e-9 * n_a * np.pi * reach ** 2 * density
    return seconds


@register("Analysis/Process/Neighbourhood")
class Neighbourhood(Plugin):
    """Per localization, what is around it: the neighbours within a radius,
    a statistic of a column over them, or the Voronoi density."""

    Settings = NeighbourhoodSettings
    changes_locs = True
    version = "1"

    def describe(self, ctx: Optional[Context], settings: NeighbourhoodSettings
                 ) -> FormState:
        q = settings.quantity
        lateral = q in ("count", "statistic")
        state = FormState(shown={
            "radius_nm": lateral, "use_z": lateral,
            "z_half_nm": lateral, "gaussian": q == "count",
            "column": q == "statistic", "statistic": q == "statistic",
            "quantile": q == "statistic" and settings.statistic == "quantile"})
        state.active["z_half_nm"] = settings.use_z
        state.hints["name"] = default_name(settings)
        layers = _layer_names(ctx)
        if layers:
            state.choices["around"] = layers
            state.choices["neighbours_from"] = layers
        if ctx is not None and ctx.session is not None and len(ctx.locs):
            state.choices["column"] = [
                (n, n) for n in ctx.locs.keys()
                if np.asarray(ctx.locs[n]).ndim == 1
                and np.asarray(ctx.locs[n]).dtype.kind in "iuf"]
        if q == "voronoi" and settings.around != settings.neighbours_from:
            state.notice = Notice("Voronoi density is of one layer: the "
                                  "neighbours are the around layer's own")
        return state

    def _tables(self, ctx: Context, settings: NeighbourhoodSettings):
        """((table, mask), (table, mask)) of the two layers, as they show."""
        def layer(n: int):
            if ctx.session is None:
                return ctx.locs, ctx.selection
            index = n - 1
            if not 0 <= index < len(ctx.session.layers):
                raise ValueError(f"there is no layer {n}")
            return ctx.table(index)
        a = layer(settings.around)
        b = a if settings.neighbours_from == settings.around else \
            layer(settings.neighbours_from)
        return a, b

    def preflight(self, ctx: Context, settings: NeighbourhoodSettings):
        try:
            (ta, sa), (tb, sb) = self._tables(ctx, settings)
            ma, mb = np.asarray(sa.mask, bool), np.asarray(sb.mask, bool)
            n_a, n_b = int(ma.sum()), int(mb.sum())
            xy = np.column_stack([np.asarray(ta["x_nm"])[ma], np.asarray(ta["y_nm"])[ma]])
            area = float(np.prod(np.maximum(np.ptp(xy, axis=0), 1.0))) if len(xy) else 1.0
        except Exception:                                  # noqa: BLE001
            return None             # the run says what is wrong
        seconds = expected_seconds(n_a, n_b, area, settings)
        if seconds < ASK_ABOVE_S:
            return None
        took = f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} minutes"
        return (f"{n_a:,} localizations around, {n_b:,} neighbours: this will take "
                f"about {took}" + ("" if settings.quantity == "voronoi"
                                   else ".  A smaller radius is faster") + ".  Run anyway?")

    def run(self, ctx: Context, settings: NeighbourhoodSettings) -> Result:
        from ..mathparse import recipes, remember
        (ta, sa), (tb, sb) = self._tables(ctx, settings)
        whole, _ = ctx.ungrouped()
        # a blink's value goes onto its localizations through group_id, so
        # handing back the ungrouped table is right (as DBSCAN)
        ctx.served_grouped = False
        name = settings.name.strip() or default_name(settings)
        own = {r["field"] for r in recipes(whole)}
        if name in whole and name not in own:
            raise ValueError(f"{name} is a column of the localizations' own: "
                             "choose another name for the new column")
        values = neighbourhood((ta, sa.mask), (tb, sb.mask), settings, ctx.report)
        grouped = ta is not whole
        column = per_localization(whole, ta, values) if grouped else values
        out = Localizations({**whole.columns, name: column.astype(np.float32)},
                            dict(whole.metadata))
        remember(out, name, "", grouped="mean")
        finite = np.isfinite(values)
        data: Dict[str, Any] = {"column": name, "n": int(finite.sum())}
        if finite.any():
            data.update(median=float(np.median(values[finite])),
                        mean=float(np.mean(values[finite])))
        what = {"count": "neighbours", "statistic": settings.statistic,
                "voronoi": "Voronoi density (/um^2)"}[settings.quantity]
        text = (f"{name}: {what} for {data['n']} "
                + ("blinks" if grouped else "localizations")
                + (f", median {data['median']:.4g}" if finite.any() else "")
                + "; NaN elsewhere")
        xy = np.column_stack([np.asarray(ta["x_nm"], float), np.asarray(ta["y_nm"], float)])
        shown = values
        # the settings as given: an empty name stays the default it is, so a
        # run again at another radius names its column after that radius
        return Result(locs=out, text=text, settings=settings,
                      data=data,
                      plot=Plot(lambda figure: draw_values(figure, xy, shown, name),
                                panels=2, size=(8, 3.6)))


def draw_values(figure, xy: np.ndarray, values: np.ndarray, name: str) -> None:
    """The localizations coloured by the value, and its histogram."""
    left, right = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.3, 1]})
    have = np.flatnonzero(np.isfinite(values))
    pick = (np.random.default_rng(0).choice(have, FIGURE_POINTS, replace=False)
            if len(have) > FIGURE_POINTS else have)
    if len(pick):
        lo, hi = np.percentile(values[pick], [1, 99])
        dots = left.scatter(xy[pick, 0], xy[pick, 1], s=1, c=values[pick],
                            cmap="viridis", vmin=lo, vmax=max(hi, lo + 1e-12),
                            linewidths=0)
        figure.colorbar(dots, ax=left, shrink=0.8)
        right.hist(values[have], bins=60, range=(lo, max(hi, lo + 1e-12)), color="0.6")
    left.set_aspect("equal")
    left.set_xlabel("x (nm)")
    left.set_ylabel("y (nm)")
    left.set_title(name, fontsize=9)
    right.set_xlabel(name)
    right.set_ylabel("localizations")
