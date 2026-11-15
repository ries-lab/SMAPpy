"""Clusters as regions: *Find clusters*, by DBSCAN, Voronoi, HDBSCAN or
local maxima.

`docs/regions.md` ("Clusters") has the decisions.  A clusterer writes
``cluster_id`` -- 1, 2, ... by size, -1 for a localization in no cluster -- and
``n_in_cluster``, the localizations of its cluster, onto every row; the table
then names its regions, and a block repeats over them (`blocks.py`), a filter
on ``n_in_cluster`` keeps the large ones, and *Segment regions* (`segment.py`,
*from an id column*) makes an ROI of each to review and annotate -- of any
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
* clusters are numbered **by size**, the largest 1, within a run;
* the radius defaults to **50 nm** rather than SMAP's automatic one, which
  is the radius of a ball holding *k* points at the mean density of the
  whole field: in a cell, mostly empty, that is hundreds of nanometres and
  merges everything (0 still asks for it);
* the search is SMAP's sweep over sorted positions made two-dimensional
  (`csrc/neighbours.cpp`, as Neighbourhood's) and the clusters are joined by
  union-find during it -- the same clusters, 2 M localizations in 2 s;
* each **file** of the session is clustered on its own, so two files'
  positions are never mixed;
* a run writes afresh only the files and channels its layer shows; the
  others keep their clusters, and the new ones are numbered above them
  (`region_runs.replaced_rows`, `merge_ids`), so two files or two colours
  on two layers are clustered one after the other;
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

**HDBSCAN** (Campello et al. 2013) is the ``hdbscan`` package's (McInnes et
al. 2017), an optional dependency: ``pip install "smappy-smlm[clusters]"``.
Not scikit-learn's, which builds the minimum spanning tree only by Prim's
algorithm, every point against every other: 48 s at 100 k localizations and
7.6 min at 300 k, a million in hours.  The package builds it by dual-tree
Boruvka on a k-d tree, looking only at nearby points: 2.5 s at 100 k and 53 s
at a million.  ``approx_min_span_tree=False`` asks for the exact tree, no
slower here; the package's default approximate one agreed with scikit-learn
at an adjusted Rand index of 0.92, the exact one 0.99 -- the rest is ties,
the mutual reachability being a core distance for many edges at once, so
several trees are equally minimal.  It is one call that Stop cannot
interrupt, so `FindClusters.preflight` asks before a long run.  SMAP had
none.

**Local maxima** is Picasso's SMLM clusterer (jungmannlab/picasso, MIT;
`clusterer._cluster`; Schlichthaerle et al. 2021, Reinhardt et al. 2023),
made for DNA-PAINT: a point with more neighbours within the radius than any
of its neighbours is a site's centre, and its neighbours its cluster.  Two
sites closer than DBSCAN's chain of core points stay apart, where DBSCAN
joins them; the radius is about twice the precision (Picasso recommends twice
the NeNA value) and so a few nanometres.  Ported in `local_maxima`; it departs
only in taking the maxima most neighbours first, not in the table's order.
In 3D z is scaled by the radius over a *z radius*, Picasso's anisotropic
neighbourhood.  Picasso's default radius and minimum size, 10 nm and 10.

**Frame analysis** (Picasso's `clusterer._frame_analysis`, on by default in
its SMLM clusterer, a tick here for every algorithm) makes noise of a cluster
whose localizations bunch in time: the mean of their place in the
acquisition outside 20-80 %, or more than 80 % of them in one twentieth of
it.  A DNA-PAINT site binds at random through the whole acquisition; a
sticky imager or dirt binds once, for long.  The thresholds and the place in
the acquisition are *Binding kinetics*' (`binding_kinetics.frame_spread`),
so a cluster that fails here has the values that fail there.

All four write the same columns through `with_clusters`, numbered by size
(`by_size`), one file at a time, blinks or localizations as the layer shows.
What was measured per cluster goes back onto its localizations with
*Regions to attribute* (`region_column.py`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ..locs import Localizations
from ..region_runs import merge_ids, replaced_rows
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
    """A cluster label per point, 1, 2, ... largest first, -1 for noise.

    ``points`` is (n, 2) or (n, 3); a point is core when at least ``k``
    others are within ``eps``; a border point joins its nearest core point's
    cluster (see the module docstring).

    The search is the Neighbourhood one (`csrc/neighbours.cpp`): the points
    sorted into square cells ``eps`` wide, each testing the nine cells around
    it, on every core.  The core points are then joined by union-find while
    their cells are scanned again (`Index.dbscan_join`), so the pairs within
    ``eps`` are never stored.  On 2 M localizations, 28 s became 2 s: scipy's
    k-d tree took 9 s to count, but 23 s went on writing out the 26 M core
    pairs, making a sparse graph of them and walking it.  The clusters are
    the same, numbered alike (tests/test_clusters.py keeps the scipy version
    to check).

    ``progress(fraction)`` is called as it goes -- which may raise, to stop
    (the plugin raises `Stopped`): after each `COUNT_CHUNK` points counted,
    and after the join, which is one call of about a second on 2 M.
    """
    from .._neighbours import Index, cell_order
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n == 0:
        return np.full(0, -1, dtype=np.int64)
    if not eps > 0:
        raise ValueError("the radius must be positive")
    use_z = points.shape[1] > 2
    x, y = points[:, 0], points[:, 1]
    z = points[:, 2] if use_z else None
    x0, y0, x1 = float(x.min()), float(y.min()), float(x.max())
    index = Index(x, y, z, None, x0, y0, x1, float(eps))
    order = cell_order(x, y, x0, y0, x1, float(eps))
    counts = np.empty(n)
    for start in range(0, n, COUNT_CHUNK):
        rows = order[start:start + COUNT_CHUNK]
        # mode 8: within eps in x, y (and z), a sphere; itself counted
        counts[rows] = index.query(x[rows], y[rows], None if z is None else z[rows],
                                   None, 8, 0.0, float(eps) if use_z else 0.0)
        if progress is not None:
            progress(0.7 * min(start + COUNT_CHUNK, n) / n)
    core = counts >= k + 1                        # itself and k others
    found = np.asarray(index.dbscan_join(core, use_z))
    if progress is not None:
        progress(1.0)
    return by_size(found)


def by_size(found: np.ndarray) -> np.ndarray:
    """Labels 1, 2, ... numbered by size, the largest 1, from any cluster
    numbers (0 and up; below 0 for none, which become -1, `ids.NONE`).  Ties
    keep the order of the numbers given (a stable sort)."""
    found = np.asarray(found, dtype=np.int64)
    labels = np.full(len(found), -1, dtype=np.int64)
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
    """Each row's cluster size, 0 for noise (-1)."""
    labels = np.asarray(labels, dtype=np.int64)
    member = labels >= 0
    out = np.zeros(len(labels), dtype=np.int64)
    if member.any():
        out[member] = np.bincount(labels[member])[labels[member]]
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
    per point, 1, 2, ... largest first, -1 for none; and the field's mean
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
    labels = np.full(n, -1, dtype=np.int64)
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


# the hdbscan package (see the module docstring); an optional dependency
HDBSCAN_NEEDS = ("HDBSCAN needs the hdbscan package: "
                 "pip install \"smappy-smlm[clusters]\" (or hdbscan)")


def hdbscan_available() -> bool:
    """Whether the hdbscan package can be imported, without importing it."""
    import importlib.util
    return importlib.util.find_spec("hdbscan") is not None


def hdbscan_clusters(points: np.ndarray, min_cluster_size: int = 10,
                     min_samples: Optional[int] = None) -> np.ndarray:
    """HDBSCAN (Campello et al. 2013, as the hdbscan package implements it): a
    cluster label per point, 1, 2, ... largest first, -1 for noise.
    ``min_samples`` None is ``min_cluster_size``, the package's default; it
    counts the neighbours without the point itself (scikit-learn's counts it,
    one more for the same clusters)."""
    if not hdbscan_available():
        raise ImportError(HDBSCAN_NEEDS)
    from hdbscan import HDBSCAN
    points = np.asarray(points, dtype=float)
    if len(points) < max(2, min_cluster_size):
        return np.full(len(points), -1, dtype=np.int64)
    found = HDBSCAN(min_cluster_size=int(min_cluster_size),
                    min_samples=None if min_samples is None else int(min_samples),
                    approx_min_span_tree=False).fit_predict(points)
    return by_size(found)


def local_maxima(points: np.ndarray, radius: float, min_points: int,
                 progress: Optional[Callable[[float], None]] = None) -> np.ndarray:
    """Picasso's SMLM clusterer (`clusterer._cluster`): a cluster label per
    point, 1, 2, ... largest first, -1 for none.

    Each point's neighbours within ``radius`` are counted, itself included.
    A **local maximum** has more than ``min_points`` and no neighbour with
    more; each maximum takes its neighbours that no earlier maximum took, and
    a maximum already taken hands its free neighbours to the cluster that
    took it -- two maxima closer than the radius are one cluster.  Clusters
    of fewer than ``min_points`` are none.  ``points`` (n, 2) or (n, 3); in
    3D the caller scales z so the neighbourhood is a sphere of ``radius``
    (Picasso's anisotropic radius, `cluster_3D`).

    Where it departs from Picasso: the maxima are taken **most neighbours
    first**, ties in the order of the cell search, where Picasso takes them
    in the order of the table -- the same clusters, but not depending on how
    the table is sorted.  The counts and the neighbour lists come from the
    cell search of `csrc/neighbours.cpp` (`Index.query`, `Index.pairs`), in
    chunks, rather than from a k-d tree's lists.
    """
    from .._neighbours import Index, cell_order
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n == 0:
        return np.full(0, -1, dtype=np.int64)
    if not radius > 0:
        raise ValueError("the radius must be positive")
    use_z = points.shape[1] > 2
    x, y = points[:, 0], points[:, 1]
    z = points[:, 2] if use_z else None
    x0, y0, x1 = float(x.min()), float(y.min()), float(x.max())
    r = float(radius)
    index = Index(x, y, z, None, x0, y0, x1, r)
    order = cell_order(x, y, x0, y0, x1, r)

    def zs(rows):
        return None if z is None else z[rows]

    counts = np.empty(n, dtype=np.int64)
    for start in range(0, n, COUNT_CHUNK):
        rows = order[start:start + COUNT_CHUNK]
        counts[rows] = index.query(x[rows], y[rows], zs(rows), None, 8, 0.0,
                                   r if use_z else 0.0)
        if progress is not None:
            progress(0.4 * min(start + COUNT_CHUNK, n) / n)
    # the most neighbours any neighbour has; every point is its own neighbour,
    # so each point asked about has a run of pairs, in the order asked
    most = np.empty(n, dtype=np.int64)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    # asked in cell order, as the counts were: the search reuses its cells
    candidates = np.flatnonzero(counts > min_points)
    candidates = candidates[np.argsort(rank[candidates])]
    for start in range(0, len(candidates), COUNT_CHUNK):
        rows = candidates[start:start + COUNT_CHUNK]
        asked, found, _ = index.pairs(x[rows], y[rows], zs(rows), use_z)
        asked, found = np.asarray(asked), np.asarray(found)
        starts = np.flatnonzero(np.r_[True, asked[1:] != asked[:-1]])
        most[rows[asked[starts]]] = np.maximum.reduceat(counts[found], starts)
        if progress is not None:
            progress(0.4 + 0.3 * min(start + COUNT_CHUNK, len(candidates))
                     / max(len(candidates), 1))
    peaks = candidates[counts[candidates] == most[candidates]]
    # most neighbours first; `candidates` is in cell order, kept for ties
    peaks = peaks[np.argsort(-counts[peaks], kind="stable")]
    found = np.full(n, -1, dtype=np.int64)
    if len(peaks):
        asked, near, _ = index.pairs(x[peaks], y[peaks], zs(peaks), use_z)
        asked, near = np.asarray(asked), np.asarray(near)
        bounds = np.searchsorted(asked, np.arange(len(peaks) + 1))
        for k, i in enumerate(peaks):
            neighbours = near[bounds[k]:bounds[k + 1]]
            label = found[i] if found[i] >= 0 else k
            found[neighbours[found[neighbours] < 0]] = label
    if progress is not None:
        progress(1.0)
    member = found >= 0
    sizes = np.bincount(found[member], minlength=len(peaks))
    found[member & (sizes[np.maximum(found, 0)] < min_points)] = -1
    return by_size(found)


def frame_analysis(labels: np.ndarray, frame: np.ndarray, file: Optional[np.ndarray],
                   spans: Dict[int, Tuple[int, int]]) -> Tuple[np.ndarray, int]:
    """Picasso's frame analysis (`clusterer._frame_analysis`) as a filter:
    ``(labels, rejected)``, the labels with every cluster that fails set to
    0 and the rest numbered again 1, 2, ... in their order, and how many
    failed.

    A cluster fails when the mean of its localizations' place in the
    acquisition -- ``(frame - first + 1/2) / frames`` of its file, as
    *Binding kinetics*' `frame_spread` -- is outside `FRAME_MEAN_BOUNDS`, or
    when one twentieth of the acquisition holds more than `FRAME_PEAK_MAX` of
    them: a sticky imager, dirt that binds for a while, a site that appeared
    late.  ``spans`` is each file's first and last frame.  All clusters at
    once, by counting into bins, not a loop over clusters.
    """
    from ..binding_kinetics import FRAME_BINS, FRAME_MEAN_BOUNDS, FRAME_PEAK_MAX
    labels = np.asarray(labels, dtype=np.int64).copy()
    frame = np.asarray(frame, dtype=float)
    file = (np.zeros(len(frame), np.int64) if file is None
            else np.asarray(file, np.int64))
    member = np.flatnonzero(labels >= 0)          # -1: in no cluster
    if not len(member):
        return labels, 0
    rel = np.full(len(member), np.nan)
    for f, (lo, hi) in spans.items():
        here = file[member] == f
        rel[here] = (frame[member][here] - lo + 0.5) / (hi - lo + 1)
    known = np.isfinite(rel)
    if not known.all():
        raise ValueError("frame analysis needs each file's frames: a file has none")
    rel = np.clip(rel, 0.0, 1.0)
    lab = labels[member]
    m = int(lab.max()) + 1
    n = np.bincount(lab, minlength=m).astype(float)
    mean = np.bincount(lab, weights=rel, minlength=m) / np.maximum(n, 1)
    bins = np.minimum((rel * FRAME_BINS).astype(np.int64), FRAME_BINS - 1)
    peak = np.bincount(lab * FRAME_BINS + bins,
                       minlength=m * FRAME_BINS).reshape(m, FRAME_BINS).max(axis=1)
    lo, hi = FRAME_MEAN_BOUNDS
    fails = (mean < lo) | (mean > hi) | (peak > FRAME_PEAK_MAX * n)
    rejected = int(fails[n > 0].sum())
    labels[member[fails[lab]]] = -1
    kept = np.unique(labels[labels >= 0])
    renumber = np.full(m, -1, dtype=np.int64)
    renumber[kept] = np.arange(1, len(kept) + 1)
    return np.where(labels >= 0, renumber[np.maximum(labels, 0)], -1), rejected


# above this many seconds a run asks first (`FindClusters.preflight`)
ASK_ABOVE_S = 20.0


def expected_seconds(algorithm: str, counts) -> float:
    """A rough run time for files of these many localizations (see
    `FindClusters.preflight` for the measurements)."""
    counts = np.asarray(counts, dtype=float)
    if algorithm == "hdbscan":
        return float(50e-6 * counts.sum())
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


@dataclass
class LocalMaximaSettings:
    radius_nm: float = param(10.0, label="radius", unit="nm", min=0.1,
                             help="a site's localizations lie within it; about "
                                  "twice the precision (NeNA)")
    min_points: int = param(10, label="minimum size", min=1,
                            help="localizations a cluster has at least")
    use_z: bool = param(True, label="in 3D", help="cluster in x, y and z when the "
                                                  "table has z_nm")
    z_radius_nm: float = param(20.0, label="z radius", unit="nm", min=0.1, beside=True,
                               help="the neighbourhood's half height in z")


ALGORITHMS = (("dbscan", "DBSCAN"), ("voronoi", "Voronoi"), ("hdbscan", "HDBSCAN"),
              ("local_maxima", "local maxima"))


@dataclass
class FindClustersSettings:
    algorithm: str = param("dbscan", label="algorithm", choices=ALGORITHMS,
                           help="DBSCAN: a radius and a count; Voronoi: dense "
                                "cells that touch; HDBSCAN: no radius; local "
                                "maxima: one cluster per DNA-PAINT site")
    dbscan: DBSCANSettings = param(default_factory=DBSCANSettings, label="DBSCAN")
    voronoi: VoronoiSettings = param(default_factory=VoronoiSettings, label="Voronoi")
    hdbscan: HDBSCANSettings = param(default_factory=HDBSCANSettings, label="HDBSCAN")
    local_maxima: LocalMaximaSettings = param(default_factory=LocalMaximaSettings,
                                              label="local maxima")
    frame_analysis: bool = param(False, label="frame analysis",
                                 help="no cluster where its localizations bunch "
                                      "in time: a sticky imager")


@register("Analysis/Regions/Find clusters")
class FindClusters(Plugin):
    """Finds clusters -- by DBSCAN, Voronoi density, HDBSCAN or local maxima
    -- and names them: cluster_id and n_in_cluster."""

    Settings = FindClustersSettings
    changes_locs = True
    version = "3"
    # tried on one ROI before the field: Picasso's "test clustering"
    follow_roi = True

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
        localization; HDBSCAN about 50 us (2.5 s at 100 k, 53 s at a
        million), one call that Stop cannot interrupt.  DBSCAN, a second per
        100 k, asks nothing.
        """
        if settings.algorithm in ("dbscan", "local_maxima"):
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
        advice = ("Stop cannot end HDBSCAN early: merge blinks, or run it in an "
                  "ROI, for a shorter run.  "
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
                 "voronoi": False, "local_maxima": settings.local_maxima.use_z}[algorithm]
        if use_z and "z_nm" in table:
            columns.append("z_nm")
        points = np.column_stack([np.asarray(table[c], dtype=float) for c in columns])
        if algorithm == "local_maxima" and points.shape[1] > 2:
            # Picasso's anisotropic radius: z scaled so the sphere is the radius
            lm = settings.local_maxima
            points[:, 2] *= lm.radius_nm / lm.z_radius_nm
        use &= np.isfinite(points).all(axis=1)
        files = (np.asarray(table["filenumber"]) if "filenumber" in table
                 else np.zeros(len(table), dtype=np.int64))
        labels = np.full(len(table), -1, dtype=np.int64)
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
            elif algorithm == "local_maxima":
                s = settings.local_maxima
                found = local_maxima(points[rows], s.radius_nm, s.min_points, progress)
            else:
                s = settings.hdbscan
                progress(0.0)
                found = hdbscan_clusters(points[rows], s.min_cluster_size, s.min_samples)
            progress(1.0)
            found[found >= 0] += offset         # labels from 1: above the last file's
            offset = max(offset, int(found.max(initial=0)))
            labels[rows] = found
        if grouped:
            labels = _per_localization(whole, table, labels)
        rejected = None
        if settings.frame_analysis:
            # on the localizations, as Picasso's: a blink's frames all count
            labels, rejected = frame_analysis(labels, whole["frame"], (
                whole["filenumber"] if "filenumber" in whole else None),
                _spans(ctx, whole))
        n_clusters = int(labels.max(initial=0))
        # the files and channels the layer shows are clustered afresh; the
        # others keep their clusters, numbered above them (`replaced_rows`)
        replaced = replaced_rows(whole, table, use)
        clustered = int((labels >= 0).sum())
        labels, n_kept = merge_ids(whole["cluster_id"] if "cluster_id" in whole
                                   else None, labels, replaced)
        text = (f"{n_clusters} clusters of {clustered} localizations, "
                f"{int(replaced.sum()) - clustered} in none; " + _how(settings, used)
                + (" (blinks)" if grouped else "")
                + (f"; {rejected} failed the frame analysis" if rejected is not None
                   else "")
                + (f"; {n_kept} clusters of other files or channels kept"
                   if n_kept else ""))
        locs = with_clusters(whole, labels)
        data: Dict[str, Any] = {"clusters": n_clusters, "algorithm": algorithm,
                                "kept": n_kept}
        if rejected is not None:
            data["frame_analysis_rejected"] = rejected
        if algorithm == "dbscan":
            data["radius_nm"] = used
        elif algorithm == "voronoi":
            data["mean_density_um2"] = used
        sizes = np.bincount(labels[labels >= 0])
        sizes = sizes[sizes > 0]
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
    if settings.algorithm == "local_maxima":
        m = settings.local_maxima
        return (f"local maxima, radius {m.radius_nm:g} nm, at least "
                f"{m.min_points} localizations")
    h = settings.hdbscan
    return (f"HDBSCAN, minimum size {h.min_cluster_size}, "
            f"{h.min_samples or h.min_cluster_size} neighbours")


def _spans(ctx: Context, whole: Localizations) -> Dict[int, Tuple[int, int]]:
    """Each file's first and last frame of the acquisition, as *Binding
    kinetics* finds them: the file's ``n_frames`` where it is recorded, else
    its first and last localization."""
    from ..binding_kinetics import spans_of
    from .binding_kinetics import acquisition
    if "frame" not in whole:
        raise ValueError("frame analysis needs the frame column")
    file = whole["filenumber"] if "filenumber" in whole else None
    seen = spans_of(whole["frame"], file)
    known = acquisition(ctx, whole, list(seen), None) or {}
    return {**seen, **known}


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
    out = np.full(len(locs), -1, dtype=np.int64)
    out[hit] = labels[order[at[hit]]]
    return out


def draw_clusters(figure, xy: np.ndarray, labels: np.ndarray, sizes: np.ndarray) -> None:
    """The clusters coloured, the rest grey; and how large they are."""
    left, right = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.3, 1]})
    n = len(xy)
    pick = (np.random.default_rng(0).choice(n, FIGURE_POINTS, replace=False)
            if n > FIGURE_POINTS else np.arange(n))
    noise = pick[labels[pick] < 0]
    member = pick[labels[pick] >= 0]
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
