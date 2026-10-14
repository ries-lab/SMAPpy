"""Clusters as regions: *Find clusters*, by DBSCAN, Voronoi or HDBSCAN.

`docs/regions.md` ("Clusters") has the decisions.  A clusterer writes
``cluster_id`` -- 1, 2, ... by size, 0 for a localization in no cluster -- and
``n_in_cluster``, the localizations of its cluster, onto every row; the table
then names its regions, and a block repeats over them (`blocks.py`), a filter
on ``n_in_cluster`` keeps the large ones, and *Column to ROIs*
(`region_column.py`) makes an ROI of each to review and annotate -- of any
clusterer's id, not only this one's.  Both columns are recipes with the rule
``max`` for the grouped table (`mathparse.remember`), so a blink keeps its
cluster.

**DBSCAN** (Ester et al. 1996), as SMAP's ``Cluster DBSCAN`` (the MIiSR
version, Caetano et al. 2015) defines it: a localization is a *core* point
when at least *k* others lie within *eps* of it; core points within *eps* of
each other are one cluster; a localization within *eps* of a core point
joins it (a *border* point) and the rest are noise.  Where it departs from
SMAP's:

* a border point joins the cluster of its **nearest** core point, not of
  whichever core point's neighbourhood was expanded first, so the answer
  does not depend on the order of the table;
* clusters are numbered **by size**, the largest 1;
* the radius defaults to **50 nm** rather than SMAP's automatic one, which
  is the radius of a ball holding *k* points at the mean density of the
  whole field: in a cell, mostly empty, that is hundreds of nanometres and
  merges everything (0 still asks for it);
* the search is a k-d tree (`scipy.spatial.cKDTree`) and the clusters the
  connected components of the core points' graph, rather than SMAP's sweep
  over x -- the same clusters, a million localizations in seconds;
* each **file** of the session is clustered on its own, so two files'
  positions are never mixed;
* on a **grouped** layer the blinks are clustered, and every localization
  takes its blink's cluster.

**Voronoi** (SR-Tesseler's, Levet et al. 2015): a localization is dense when
its Voronoi density (`neighbourhood.voronoi_cells`) is above a multiple of
the field's mean -- its localizations over their convex hull's area -- and
dense cells that touch, Delaunay neighbours of the whole field, are one
cluster; a cluster below a minimum size is none.  The design named SharpViSu
for it, whose ClusterViSu (Andronov et al. 2016) instead puts the threshold
where the data's distribution of cell areas crosses a simulated random one;
the fixed multiple is SR-Tesseler's, simpler and with no simulation.  SMAP's
``ClusterVoronoi`` wrote only the density.

**HDBSCAN** (Campello et al. 2013) is scikit-learn's (1.3 and later), an
optional dependency: ``pip install "smappy-smlm[clusters]"``.  It is quadratic
in practice -- 48 s at 100 k localizations, 7.6 min at 300 k -- and one call
that Stop cannot interrupt, so `FindClusters.preflight` asks before a long
run.  SMAP had none.

All three write the same columns through `with_clusters`, numbered by size
(`by_size`), one file at a time, blinks or localizations as the layer shows.
What was measured per cluster goes back onto its localizations with
*Regions to column* (`region_column.py`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ..locs import Localizations
from . import (Context, FormState, Notice, Plot, Plugin, Result, Stopped, param,
               register)

# the rows clustered at once are this many at most in a figure: a scatter of a
# million points is a grey square and seconds to draw
FIGURE_POINTS = 50_000


# ------------------------------------------------------------- the algorithm

def auto_eps(points: np.ndarray, k: int) -> float:
    """SMAP's neighbourhood radius when none is given: the radius of a ball
    that would hold *k* points at the mean density of the points' bounding
    box, ``(V k Gamma(d/2 + 1) / (m pi^(d/2)))^(1/d)``."""
    from math import gamma, pi
    m, d = points.shape
    extent = np.ptp(points, axis=0)
    volume = float(np.prod(np.where(extent > 0, extent, 1.0)))
    return (volume * k * gamma(0.5 * d + 1) / (m * pi ** (0.5 * d))) ** (1.0 / d)


# the points whose neighbours are counted at once: the step at which the
# count reports progress and a stop is honoured
COUNT_CHUNK = 200_000


def dbscan(points: np.ndarray, eps: float, k: int,
           progress: Optional[Callable[[float], None]] = None) -> np.ndarray:
    """A cluster label per point, 1, 2, ... largest first, 0 for noise.

    ``points`` is (n, 2) or (n, 3); a point is core when at least ``k``
    others are within ``eps``; a border point joins its nearest core point's
    cluster (see the module docstring).

    ``progress(fraction)`` is called as it goes -- which may raise, to stop
    (the plugin raises `Stopped`).  The count goes in chunks; the pairs of
    core points, their connected components and the border points are one
    scipy call each, so a stop waits for the one running: on 2 M
    localizations those take 10, 13 and 2 s, the count 9 s.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    points = np.asarray(points, dtype=float)
    n = len(points)
    labels = np.zeros(n, dtype=np.int64)
    if n == 0:
        return labels
    if not eps > 0:
        raise ValueError("the radius must be positive")
    def step(fraction):
        if progress is not None:
            progress(fraction)

    tree = cKDTree(points)
    counts = np.empty(n, dtype=np.int64)
    for start in range(0, n, COUNT_CHUNK):
        rows = slice(start, min(start + COUNT_CHUNK, n))
        counts[rows] = tree.query_ball_point(points[rows], eps, return_length=True)
        step(0.3 * rows.stop / n)
    core = np.flatnonzero(counts >= k + 1)          # itself and k others
    if not len(core):
        return labels
    core_tree = cKDTree(points[core])
    pairs = core_tree.query_pairs(eps, output_type="ndarray")
    step(0.6)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(core), len(core)))
    _, component = connected_components(graph, directed=False)
    step(0.9)
    # border points: the nearest core point within eps
    rest = np.setdiff1d(np.arange(n), core)
    found = np.full(n, -1, dtype=np.int64)
    found[core] = component
    if len(rest):
        distance, nearest = core_tree.query(points[rest], distance_upper_bound=eps)
        near = np.isfinite(distance)
        found[rest[near]] = component[nearest[near]]
    step(1.0)
    return by_size(found)


def by_size(found: np.ndarray) -> np.ndarray:
    """Labels 1, 2, ... numbered by size, the largest 1, from any cluster
    numbers (0 and up; below 0 for none), which become 0.  Ties keep the
    order of the numbers given (a stable sort)."""
    found = np.asarray(found, dtype=np.int64)
    labels = np.zeros(len(found), dtype=np.int64)
    member = found >= 0
    if not member.any():
        return labels
    sizes = np.bincount(found[member])
    order = np.argsort(-sizes, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)
    labels[member] = rank[found[member]]
    return labels


def sizes_of(labels: np.ndarray) -> np.ndarray:
    """Each row's cluster size, 0 for noise."""
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels)
    out = counts[labels]
    out[labels == 0] = 0
    return out


def with_clusters(locs: Localizations, labels: np.ndarray, name: str = "cluster_id"
                  ) -> Localizations:
    """The table with ``name`` and ``n_in_cluster``, both recipes that keep
    the largest value on the grouped table: a blink keeps its cluster."""
    from ..mathparse import remember
    labels = np.asarray(labels, dtype=np.int64)
    out = Localizations({**locs.columns, name: labels, "n_in_cluster": sizes_of(labels)},
                        dict(locs.metadata))
    remember(out, name, "", grouped="max")
    remember(out, "n_in_cluster", "", grouped="max")
    return out


def voronoi_clusters(points: np.ndarray, factor: float = 2.0, min_size: int = 10,
                     progress: Optional[Callable[[float], None]] = None
                     ) -> Tuple[np.ndarray, float]:
    """Voronoi clustering (SR-Tesseler's, Levet et al. 2015): a cluster label
    per point, 1, 2, ... largest first, 0 for none; and the field's mean
    density, per square micrometre.

    A point is dense when the density of its Voronoi cell (one over its area,
    `neighbourhood.voronoi_cells`) is above ``factor`` times the field's mean
    -- its points over the area of their convex hull.  Dense cells that touch
    are one cluster; a cluster of fewer than ``min_size`` points is none.
    Points at one position share a cell and so a cluster.  2D: x and y.
    ``progress(fraction)`` as for `dbscan`: the cells are most of the time.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import ConvexHull
    from .neighbourhood import voronoi_cells
    points = np.asarray(points, dtype=float)[:, :2]
    n = len(points)
    labels = np.zeros(n, dtype=np.int64)
    if n < 4:
        return labels, float("nan")
    try:
        hull_um2 = ConvexHull(points).volume / 1e6         # in 2D, volume is area
    except Exception:                                      # noqa: BLE001 -- on a line
        return labels, float("nan")
    mean = n / hull_um2
    cell, density, edges = voronoi_cells(
        points, None if progress is None else (lambda f: progress(0.9 * f)))
    dense = np.nan_to_num(density, nan=0.0) > factor * mean
    edges = edges[dense[edges[:, 0]] & dense[edges[:, 1]]]
    m = len(density)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(m, m))
    _, component = connected_components(graph, directed=False)
    component[~dense] = -1
    found = component[cell]
    sizes = np.bincount(found[found >= 0], minlength=m)
    found[(found >= 0) & (sizes[np.maximum(found, 0)] < min_size)] = -1
    if progress is not None:
        progress(1.0)
    return by_size(found), mean


# scikit-learn added HDBSCAN in 1.3; it is an optional dependency
HDBSCAN_NEEDS = ("HDBSCAN needs scikit-learn 1.3 or newer: "
                 "pip install \"smappy-smlm[clusters]\" (or scikit-learn)")


def hdbscan_available() -> bool:
    """Whether scikit-learn's HDBSCAN can be imported, without importing it."""
    import importlib.util
    if importlib.util.find_spec("sklearn") is None:
        return False
    try:
        from importlib.metadata import version
        major, minor = (int(v) for v in version("scikit-learn").split(".")[:2])
    except Exception:                                      # noqa: BLE001 -- on a line
        return False
    return (major, minor) >= (1, 3)


def hdbscan_clusters(points: np.ndarray, min_cluster_size: int = 10,
                     min_samples: Optional[int] = None) -> np.ndarray:
    """HDBSCAN (Campello et al. 2013, as scikit-learn implements it): a cluster
    label per point, 1, 2, ... largest first, 0 for noise.  ``min_samples``
    None is ``min_cluster_size``, scikit-learn's default."""
    if not hdbscan_available():
        raise ImportError(HDBSCAN_NEEDS)
    from sklearn.cluster import HDBSCAN
    points = np.asarray(points, dtype=float)
    if len(points) < max(2, min_cluster_size):
        return np.zeros(len(points), dtype=np.int64)
    found = HDBSCAN(min_cluster_size=int(min_cluster_size),
                    min_samples=None if min_samples is None else int(min_samples),
                    copy=True).fit_predict(points)
    return by_size(found)


# above this many seconds a run asks first (`FindClusters.preflight`)
ASK_ABOVE_S = 20.0


def expected_seconds(algorithm: str, counts) -> float:
    """A rough run time for files of these many localizations (see
    `FindClusters.preflight` for the measurements)."""
    counts = np.asarray(counts, dtype=float)
    if algorithm == "hdbscan":
        return float((48.0 * (counts / 1e5) ** 2).sum())
    if algorithm == "voronoi":
        return float(20e-6 * counts.sum())
    return float(1e-5 * counts.sum())


# --------------------------------------------------------------- the plugin

@dataclass
class DBSCANSettings:
    radius_nm: float = param(50.0, label="radius", unit="nm", min=0.0,
                             help="eps: how close a neighbour is; 0: chosen from "
                                  "the density, as SMAP does")
    neighbours: int = param(10, label="neighbours", min=1,
                            help="k: the neighbours within the radius that make "
                                 "a core point")
    use_z: bool = param(True, label="in 3D", help="cluster in x, y and z when the "
                                                  "table has z_nm")


@dataclass
class VoronoiSettings:
    factor: float = param(2.0, label="density factor", min=0.0,
                          help="dense: a Voronoi density above this many times "
                               "the field's mean")
    min_size: int = param(10, label="minimum size", min=1,
                          help="localizations in a cluster; smaller ones are none")


@dataclass
class HDBSCANSettings:
    min_cluster_size: int = param(10, label="minimum size", min=2,
                                  help="the fewest localizations a cluster has")
    min_samples: Optional[int] = param(None, label="neighbours", min=1,
                                       help="auto: the minimum size; more is more "
                                            "conservative, with more noise")
    use_z: bool = param(True, label="in 3D", help="cluster in x, y and z when the "
                                                  "table has z_nm")


ALGORITHMS = (("dbscan", "DBSCAN"), ("voronoi", "Voronoi"), ("hdbscan", "HDBSCAN"))


@dataclass
class FindClustersSettings:
    algorithm: str = param("dbscan", label="algorithm", choices=ALGORITHMS,
                           help="DBSCAN: a radius and a count; Voronoi: dense "
                                "cells that touch; HDBSCAN: no radius")
    dbscan: DBSCANSettings = param(default_factory=DBSCANSettings, label="DBSCAN")
    voronoi: VoronoiSettings = param(default_factory=VoronoiSettings, label="Voronoi")
    hdbscan: HDBSCANSettings = param(default_factory=HDBSCANSettings, label="HDBSCAN")


@register("Analysis/Regions/Find clusters")
class FindClusters(Plugin):
    """Finds clusters -- by DBSCAN, Voronoi density or HDBSCAN -- and names
    them: cluster_id and n_in_cluster."""

    Settings = FindClustersSettings
    changes_locs = True
    version = "1"

    def describe(self, ctx: Optional[Context], settings: FindClustersSettings
                 ) -> FormState:
        """Only the chosen algorithm's part; how to install HDBSCAN when it
        is chosen and missing."""
        algorithm = settings.algorithm
        state = FormState(shown={key: algorithm == key for key, _ in ALGORITHMS})
        if algorithm == "hdbscan" and not hdbscan_available():
            state.notice = Notice(HDBSCAN_NEEDS)
        return state

    def preflight(self, ctx: Context, settings: FindClustersSettings):
        """A question before a Voronoi or HDBSCAN run that will be long.

        Measured on 4 cores, clusters in a background: Voronoi about 20 us a
        localization; scikit-learn's HDBSCAN 48 s at 100 k and 7.6 min at
        300 k -- growing as the square, so a million is hours -- and one
        call that Stop cannot interrupt.  DBSCAN, a second per 100 k, asks
        nothing.
        """
        if settings.algorithm == "dbscan":
            return None
        try:
            mask = np.asarray(ctx.selection.mask, dtype=bool)
            files = (np.asarray(ctx.locs["filenumber"])[mask] if "filenumber" in ctx.locs
                     else np.zeros(int(mask.sum()), dtype=np.int64))
            counts = np.unique(files, return_counts=True)[1].astype(float)
        except Exception:                                  # noqa: BLE001
            return None             # the run says what is wrong
        seconds = expected_seconds(settings.algorithm, counts)
        if seconds < ASK_ABOVE_S:
            return None
        took = (f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} minutes"
                if seconds < 5400 else f"{seconds / 3600:.0f} hours")
        advice = ("HDBSCAN grows as the square of the localizations: merge blinks, "
                  "or run it in an ROI, or use DBSCAN.  Stop cannot end it early.  "
                  if settings.algorithm == "hdbscan" else "")
        return (f"{int(counts.sum()):,} localizations: this will take about {took}.  "
                + advice + "Run anyway?")

    def run(self, ctx: Context, settings: FindClustersSettings) -> Result:
        algorithm = settings.algorithm
        if algorithm not in dict(ALGORITHMS):
            raise ValueError(f"no algorithm {algorithm!r}: one of "
                             + ", ".join(k for k, _ in ALGORITHMS))
        if algorithm == "hdbscan" and not hdbscan_available():
            raise ValueError(HDBSCAN_NEEDS)
        name = dict(ALGORITHMS)[algorithm]
        table, selection = ctx.locs, ctx.selection    # what the layer shows
        whole, _ = ctx.ungrouped()
        grouped = table is not whole
        # the localizations get their blink's cluster through group_id, so
        # handing back the ungrouped table is right (see `Context.ungrouped`)
        ctx.served_grouped = False
        use = np.asarray(selection.mask, dtype=bool)
        columns = [c for c in ("x_nm", "y_nm") if c in table]
        if len(columns) < 2:
            raise ValueError(f"{name} needs x_nm and y_nm: the table has "
                             f"{', '.join(sorted(table.keys()))}")
        use_z = {"dbscan": settings.dbscan.use_z, "hdbscan": settings.hdbscan.use_z,
                 "voronoi": False}[algorithm]
        if use_z and "z_nm" in table:
            columns.append("z_nm")
        points = np.column_stack([np.asarray(table[c], dtype=float) for c in columns])
        use &= np.isfinite(points).all(axis=1)
        files = (np.asarray(table["filenumber"]) if "filenumber" in table
                 else np.zeros(len(table), dtype=np.int64))
        labels = np.zeros(len(table), dtype=np.int64)
        used = []                     # per file: DBSCAN's radius, Voronoi's mean
        offset = 0
        files_used = np.unique(files[use])
        for i, f in enumerate(files_used):
            def progress(fraction, i=i):
                done = (i + fraction) / len(files_used)
                # a half-clustered table is worth nothing: a stop keeps nothing
                if ctx.stopped:
                    raise Stopped(f"{name} stopped at {100 * done:.0f}%: nothing written")
                ctx.report(f"{name}: {100 * done:.0f}%")

            rows = np.flatnonzero(use & (files == f))
            if algorithm == "dbscan":
                s = settings.dbscan
                eps = s.radius_nm or auto_eps(points[rows], s.neighbours)
                used.append(eps)
                found = dbscan(points[rows], eps, s.neighbours, progress)
            elif algorithm == "voronoi":
                s = settings.voronoi
                found, mean = voronoi_clusters(points[rows], s.factor, s.min_size,
                                               progress)
                used.append(mean)
            else:
                s = settings.hdbscan
                progress(0.0)
                found = hdbscan_clusters(points[rows], s.min_cluster_size, s.min_samples)
            progress(1.0)
            found[found > 0] += offset
            offset = max(offset, int(found.max(initial=0)))
            labels[rows] = found
        if grouped:
            labels = _per_localization(whole, table, labels)
        n_clusters = int(labels.max(initial=0))
        clustered = int((labels > 0).sum())
        text = (f"{n_clusters} clusters of {clustered} localizations, "
                f"{int(len(labels) - clustered)} in none; " + _how(settings, used)
                + (" (blinks)" if grouped else ""))
        locs = with_clusters(whole, labels)
        data: Dict[str, Any] = {"clusters": n_clusters, "algorithm": algorithm}
        if algorithm == "dbscan":
            data["radius_nm"] = used
        elif algorithm == "voronoi":
            data["mean_density_um2"] = used
        sizes = np.bincount(labels)[1:] if n_clusters else np.zeros(0, int)
        xy = np.column_stack([np.asarray(whole["x_nm"], float),
                              np.asarray(whole["y_nm"], float)])
        return Result(locs=locs, text=text, settings=settings, data=data,
                      values={"clusters": n_clusters, "clustered": clustered},
                      plot=Plot(lambda figure: draw_clusters(figure, xy, labels, sizes),
                                panels=2, size=(8, 3.6)))


def _how(settings: FindClustersSettings, used) -> str:
    """The settings the result line names, with what *auto* resolved to."""
    if settings.algorithm == "dbscan":
        return ("DBSCAN, radius " + ", ".join(f"{e:.1f}" for e in used)
                + f" nm, {settings.dbscan.neighbours} neighbours")
    if settings.algorithm == "voronoi":
        v = settings.voronoi
        return (f"Voronoi, dense above {v.factor:g} x the mean density ("
                + ", ".join(f"{m:.0f}" for m in used) + f" /um^2), at least "
                f"{v.min_size} localizations")
    h = settings.hdbscan
    return (f"HDBSCAN, minimum size {h.min_cluster_size}, "
            f"{h.min_samples or h.min_cluster_size} neighbours")


def _per_localization(locs: Localizations, grouped: Localizations,
                      labels: np.ndarray) -> np.ndarray:
    """A blink's cluster on each of its localizations, through ``group_id``."""
    if "group_id" not in locs or "group_id" not in grouped:
        raise ValueError("the merged blinks have no group_id to map the clusters "
                         "back to the localizations: cluster with blinks not merged")
    gid = np.asarray(grouped["group_id"])
    order = np.argsort(gid, kind="stable")
    wanted = np.asarray(locs["group_id"])
    at = np.clip(np.searchsorted(gid[order], wanted), 0, len(gid) - 1)
    hit = gid[order][at] == wanted
    out = np.zeros(len(locs), dtype=np.int64)
    out[hit] = labels[order[at[hit]]]
    return out


def draw_clusters(figure, xy: np.ndarray, labels: np.ndarray, sizes: np.ndarray) -> None:
    """The clusters coloured, the rest grey; and how large they are."""
    left, right = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.3, 1]})
    n = len(xy)
    pick = (np.random.default_rng(0).choice(n, FIGURE_POINTS, replace=False)
            if n > FIGURE_POINTS else np.arange(n))
    noise = pick[labels[pick] == 0]
    member = pick[labels[pick] > 0]
    left.scatter(xy[noise, 0], xy[noise, 1], s=1, c="0.8", linewidths=0)
    left.scatter(xy[member, 0], xy[member, 1], s=2, c=labels[member] % 20,
                 cmap="tab20", linewidths=0)
    left.set_aspect("equal")
    left.set_xlabel("x (nm)")
    left.set_ylabel("y (nm)")
    left.set_title(f"{len(sizes)} clusters", fontsize=9)
    if len(sizes):
        right.hist(sizes, bins=min(50, max(5, len(sizes) // 5)), color="0.6")
    right.set_xlabel("localizations per cluster")
    right.set_ylabel("clusters")
