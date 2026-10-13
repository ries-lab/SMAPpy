"""Clusters as regions: DBSCAN.

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

What was measured per cluster goes back onto its localizations with
*Regions to column* (`region_column.py`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..locs import Localizations
from . import Context, Plot, Plugin, Result, param, register

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


def dbscan(points: np.ndarray, eps: float, k: int) -> np.ndarray:
    """A cluster label per point, 1, 2, ... largest first, 0 for noise.

    ``points`` is (n, 2) or (n, 3); a point is core when at least ``k``
    others are within ``eps``; a border point joins its nearest core point's
    cluster (see the module docstring).
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
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, eps, return_length=True)
    core = np.flatnonzero(counts >= k + 1)          # itself and k others
    if not len(core):
        return labels
    core_tree = cKDTree(points[core])
    pairs = core_tree.query_pairs(eps, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(core), len(core)))
    _, component = connected_components(graph, directed=False)
    # border points: the nearest core point within eps
    rest = np.setdiff1d(np.arange(n), core)
    found = np.full(n, -1, dtype=np.int64)
    found[core] = component
    if len(rest):
        distance, nearest = core_tree.query(points[rest], distance_upper_bound=eps)
        near = np.isfinite(distance)
        found[rest[near]] = component[nearest[near]]
    # numbered by size, the largest 1
    member = found >= 0
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


# --------------------------------------------------------------- DBSCAN

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


@register("Analysis/Regions/Find/DBSCAN")
class DBSCAN(Plugin):
    """Finds clusters by density and names them: cluster_id and n_in_cluster."""

    Settings = DBSCANSettings
    changes_locs = True
    version = "1"

    def run(self, ctx: Context, settings: DBSCANSettings) -> Result:
        table, selection = ctx.locs, ctx.selection    # what the layer shows
        whole, _ = ctx.ungrouped()
        grouped = table is not whole
        # the localizations get their blink's cluster through group_id, so
        # handing back the ungrouped table is right (see `Context.ungrouped`)
        ctx.served_grouped = False
        use = np.asarray(selection.mask, dtype=bool)
        columns = [c for c in ("x_nm", "y_nm") if c in table]
        if len(columns) < 2:
            raise ValueError("DBSCAN needs x_nm and y_nm: the table has "
                             f"{', '.join(sorted(table.keys()))}")
        if settings.use_z and "z_nm" in table:
            columns.append("z_nm")
        points = np.column_stack([np.asarray(table[c], dtype=float) for c in columns])
        use &= np.isfinite(points).all(axis=1)
        files = (np.asarray(table["filenumber"]) if "filenumber" in table
                 else np.zeros(len(table), dtype=np.int64))
        labels = np.zeros(len(table), dtype=np.int64)
        eps_used = []
        offset = 0
        for f in np.unique(files[use]):
            rows = np.flatnonzero(use & (files == f))
            eps = settings.radius_nm or auto_eps(points[rows], settings.neighbours)
            eps_used.append(eps)
            found = dbscan(points[rows], eps, settings.neighbours)
            found[found > 0] += offset
            offset = max(offset, int(found.max(initial=0)))
            labels[rows] = found
        if grouped:
            labels = _per_localization(whole, table, labels)
        n_clusters = int(labels.max(initial=0))
        clustered = int((labels > 0).sum())
        text = (f"{n_clusters} clusters of {clustered} localizations, "
                f"{int(len(labels) - clustered)} in none; radius "
                + ", ".join(f"{e:.1f}" for e in eps_used) + f" nm, {settings.neighbours} "
                "neighbours" + (" (blinks)" if grouped else ""))
        locs = with_clusters(whole, labels)
        data: Dict[str, Any] = {"clusters": n_clusters, "radius_nm": eps_used}
        sizes = np.bincount(labels)[1:] if n_clusters else np.zeros(0, int)
        xy = np.column_stack([np.asarray(whole["x_nm"], float),
                              np.asarray(whole["y_nm"], float)])
        return Result(locs=locs, text=text, settings=settings, data=data,
                      values={"clusters": n_clusters, "clustered": clustered},
                      plot=Plot(lambda figure: draw_clusters(figure, xy, labels, sizes),
                                panels=2, size=(8, 3.6)))


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
