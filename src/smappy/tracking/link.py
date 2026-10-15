"""Linking localizations into tracks: two linkers, the same candidates.

Single-particle tracking here is fitted from single molecules, so it is
**sparse**: a jump comparable to the distance between molecules would mean
motion blur or overlapping spots that cannot be fitted.  Deciding which
localization of the next frame continues a track is therefore rarely in
doubt; what a linker has to get right is the frames where the fluorophore
was dark, dim or badly fitted -- a track must survive them.  That is why
u-track (Jaqaman et al. 2008), with its motion models and a second, global
gap-closing pass, was never used in the end, and why the default here is
the simple linker (the benchmark in NOTES.md, "Tracking", and on the page).

**Candidates** (`candidates`): every pair of localizations at most
``memory + 1`` frames apart (``memory`` dark frames bridged) and at most
the maximum jump apart, from the C++ cell index of Neighbourhood
(`csrc/neighbours.cpp`, `Index.pairs`).  One index serves every frame: x is
shifted by the frame times a stride wider than the field plus twice the
jump, so a point asked about at ``x + (frame + lag) * stride`` can only meet
points of frame ``frame + lag`` -- one query per lag, no loop over frames
(the trick `simulate.localizations.close_groups` plays on a k-d tree).  The
same maximum jump holds for a gap as for one frame, as in Crocker and Grier
(1996) and trackpy: Brownian motion over k frames goes sqrt(k) further, so
a fast particle's gaps are bridged less often than a slow one's.

**simple** (`link_simple`): frame by frame, each localization continues the
nearest open track end within the jump, ends kept open for ``memory``
frames; where two want one, the nearest pair first (at equal distance the
shorter gap).  Visiting all candidates in the order (frame, distance, gap)
and taking a link when both its ends are free is exactly that, so it is one
loop in C++ (`greedy_links`) with no Python per frame.

**global assignment** (`link_global`): per frame, the links that minimise
the sum of squared jumps with every track end or localization left unlinked
costing the squared maximum jump -- Crocker and Grier's criterion, which
their code (SMAP's ``mytrack.m``) solves by enumerating each subnetwork and
this by `scipy.optimize.linear_sum_assignment` on each connected group of
candidates (maximising ``R^2 - d^2`` over the links is the same).  Links
nobody else wants are taken without a solve.  A Python loop over frames:
slower, and on sparse data the same links (the benchmark).  The design's
option of a cost scaled by the precisions was left out: at the jumps and
densities where the choice matters at all the two linkers already agree.

Both write the same ids through `track_ids`: tracks 1, 2, ... in the order
they start, 0 for a localization in a track shorter than the minimum
(a lone one included), and the localizations in each track.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

ALGORITHMS = (("simple", "simple"), ("global", "global assignment"))


@dataclass
class Candidates:
    """Candidate links, earlier point to later: indices into the points
    given, the squared distance (nm^2) and the frames between them."""
    from_: np.ndarray
    to: np.ndarray
    d2: np.ndarray
    lag: np.ndarray

    def __len__(self) -> int:
        return len(self.from_)


def candidates(x, y, frame, z=None, max_jump_nm: float = 500.0, memory: int = 2,
               threads: int = 0) -> Candidates:
    """Every pair of points 1 to ``memory + 1`` frames apart within
    ``max_jump_nm`` (in 3D, with ``z``, a sphere)."""
    from .._neighbours import Index
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    frame = np.asarray(frame, np.int64)
    n = len(x)
    if max_jump_nm <= 0:
        raise ValueError("the maximum jump must be positive")
    if memory < 0:
        raise ValueError("memory is 0 or more frames")
    empty = Candidates(*(np.zeros(0, np.int64),) * 2, np.zeros(0), np.zeros(0, np.int64))
    if n < 2:
        return empty
    f = frame - frame.min()
    lags = int(memory) + 1
    x_min, x_max = float(x.min()), float(x.max())
    stride = (x_max - x_min) + 2.0 * max_jump_nm + 1.0
    shifted = x + f * stride
    zz = None if z is None else np.asarray(z, float)
    index = Index(shifted, y, zz, None, x_min, float(y.min()),
                  x_max + (float(f.max()) + lags) * stride, float(max_jump_nm), threads)
    parts = []
    for lag in range(1, lags + 1):
        a, b, d2 = index.pairs(shifted + lag * stride, y, zz, zz is not None, threads)
        parts.append((a, b, d2, np.full(len(a), lag, np.int64)))
    if not parts:
        return empty
    return Candidates(*(np.concatenate(p) for p in zip(*parts)))


def _in_frame_order(c: Candidates, frame: np.ndarray) -> np.ndarray:
    """The candidates by the frame they lead to, then nearest, then the
    shorter gap: the order the simple linker visits them in."""
    return np.lexsort((c.lag, c.d2, frame[c.to]))


def link_simple(c: Candidates, frame, n: int) -> np.ndarray:
    """Each point's successor in its track (-1 for none): frame by frame,
    the nearest free pair first."""
    from .._neighbours import greedy_links
    if not len(c):
        return np.full(n, -1, np.int64)
    order = _in_frame_order(c, np.asarray(frame, np.int64))
    return np.asarray(greedy_links(c.from_[order], c.to[order], n))


def link_global(c: Candidates, frame, n: int, max_jump_nm: float,
                progress: Optional[Callable[[float], None]] = None) -> np.ndarray:
    """Each point's successor (-1 for none): per frame the links with the
    least squared jump, a point left unlinked costing ``max_jump_nm^2``."""
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    frame = np.asarray(frame, np.int64)
    nxt = np.full(n, -1, np.int64)
    if not len(c):
        return nxt
    order = _in_frame_order(c, frame)
    a, b, d2 = c.from_[order], c.to[order], c.d2[order]
    # a tiny margin keeps a link at exactly the maximum jump worth taking
    gain = max_jump_nm ** 2 * (1 + 1e-9) - d2
    to_frame = frame[b]
    bounds = np.flatnonzero(np.diff(to_frame)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(a)]])
    for k, (s, e) in enumerate(zip(starts, ends)):
        if progress is not None and k % 2000 == 0:
            progress(k / len(starts))
        ia, ib, g = a[s:e], b[s:e], gain[s:e]
        free = nxt[ia] < 0               # an end already continued is not one
        ia, ib, g = ia[free], ib[free], g[free]
        if not len(ia):
            continue
        ua, ra = np.unique(ia, return_inverse=True)
        ub, rb = np.unique(ib, return_inverse=True)
        if len(ua) == len(ia) and len(ub) == len(ib):
            nxt[ia] = ib                 # nobody competes: every link stands
            continue
        na, nb = len(ua), len(ub)
        graph = coo_matrix((np.ones(len(ia)), (ra, na + rb)), shape=(na + nb,) * 2)
        _, part = connected_components(graph, directed=False)
        edge_part = part[ra]
        lone = np.bincount(edge_part)[edge_part] == 1
        nxt[ia[lone]] = ib[lone]
        rest = np.flatnonzero(~lone)
        if not len(rest):
            continue
        rest = rest[np.argsort(edge_part[rest], kind="stable")]
        cuts = np.flatnonzero(np.diff(edge_part[rest])) + 1
        for group in np.split(rest, cuts):
            ga, ra_g = np.unique(ra[group], return_inverse=True)
            gb, rb_g = np.unique(rb[group], return_inverse=True)
            matrix = np.zeros((len(ga), len(gb)))
            matrix[ra_g, rb_g] = g[group]
            rows, cols = linear_sum_assignment(matrix, maximize=True)
            take = matrix[rows, cols] > 0
            nxt[ua[ga[rows[take]]]] = ub[gb[cols[take]]]
    return nxt


def track_ids(nxt: np.ndarray, frame, min_length: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """``track_id`` (1, 2, ... in the order the tracks start; 0 for a track
    of fewer than ``min_length`` localizations) and ``track_length`` (the
    localizations of the track, 0 where the id is 0), from each point's
    successor."""
    nxt = np.asarray(nxt, np.int64)
    frame = np.asarray(frame, np.int64)
    n = len(nxt)
    pred = np.full(n, -1, np.int64)
    linked = nxt >= 0
    pred[nxt[linked]] = np.flatnonzero(linked)
    # pointer jumping: every point's head in log2(longest track) steps
    head = np.where(pred >= 0, pred, np.arange(n))
    while True:
        up = head[head]
        if np.array_equal(up, head):
            break
        head = up
    sizes = np.bincount(head, minlength=n)
    heads = np.flatnonzero((pred < 0) & (sizes >= max(int(min_length), 1)))
    heads = heads[np.lexsort((heads, frame[heads]))]
    rank = np.zeros(n, np.int64)
    rank[heads] = np.arange(1, len(heads) + 1)
    track = rank[head]
    length = np.where(track > 0, sizes[head], 0)
    return track, length


def link_tracks(x, y, frame, z=None, *, max_jump_nm: float = 500.0, memory: int = 2,
                min_length: int = 2, algorithm: str = "simple", groups=None,
                threads: int = 0, progress: Optional[Callable[[float], None]] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """``track_id`` and ``track_length`` per point (see `track_ids`), linked
    with ``algorithm`` (``simple`` or ``global``).  ``groups``, an integer
    per point, links each group on its own -- a file, a channel -- and a
    point whose group is below 0, or whose position is not finite, is in no
    track."""
    if algorithm not in dict(ALGORITHMS):
        raise ValueError(f"no linker {algorithm!r}: one of "
                         + ", ".join(k for k, _ in ALGORITHMS))
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    frame = np.asarray(frame, np.int64)
    n = len(x)
    zz = None if z is None else np.asarray(z, float)
    groups = np.zeros(n, np.int64) if groups is None else np.asarray(groups, np.int64)
    ok = np.isfinite(x) & np.isfinite(y) & (groups >= 0)
    if zz is not None:
        ok &= np.isfinite(zz)
    track = np.zeros(n, np.int64)
    length = np.zeros(n, np.int64)
    found = np.unique(groups[ok])
    offset = 0
    for i, g in enumerate(found):
        rows = np.flatnonzero(ok & (groups == g))

        def part(fraction, i=i):
            if progress is not None:
                progress((i + fraction) / len(found))

        part(0.0)
        f = frame[rows]
        sub_z = None if zz is None else zz[rows]
        c = candidates(x[rows], y[rows], f, sub_z, max_jump_nm, memory, threads)
        if algorithm == "simple":
            nxt = link_simple(c, f, len(rows))
        else:
            nxt = link_global(c, f, len(rows), max_jump_nm, part)
        t, ln = track_ids(nxt, f, min_length)
        t[t > 0] += offset
        offset = max(offset, int(t.max(initial=0)))
        track[rows], length[rows] = t, ln
    if progress is not None:
        progress(1.0)
    return track, length


def true_links(truth, frame) -> Tuple[np.ndarray, np.ndarray]:
    """The links of the true tracks among the points: each point to the next
    one of its track (``truth`` > 0), however many frames apart."""
    truth = np.asarray(truth, np.int64)
    frame = np.asarray(frame, np.int64)
    order = np.lexsort((frame, truth))
    order = order[truth[order] > 0]
    same = truth[order][1:] == truth[order][:-1]
    return order[:-1][same], order[1:][same]


def score(track_id, truth, frame) -> Dict[str, float]:
    """How a linking compares with the truth (``track_id_true``).

    ``links_found``: of the true links -- each localization of a track to
    the next one seen, however long the gap -- the fraction the tracks hold;
    ``links_wrong``: of the links the tracks hold, the fraction that are
    not true ones; ``tracks_whole``: of the true tracks of two or more
    localizations, the fraction found exactly, one track holding all of its
    localizations and nothing else.
    """
    track_id = np.asarray(track_id, np.int64)
    truth = np.asarray(truth, np.int64)
    n = len(track_id)
    ta, tb = true_links(truth, frame)
    fa, fb = true_links(track_id, frame)
    key = lambda a, b: a.astype(np.int64) * n + b           # noqa: E731
    true_keys, found_keys = key(ta, tb), key(fa, fb)
    hit = np.isin(found_keys, true_keys)
    # whole: every member of a true track in one found track of the same size
    real = truth > 0
    t_size = np.bincount(truth[real])
    f_size = np.bincount(track_id[track_id > 0]) if (track_id > 0).any() else np.zeros(1, int)
    rows = np.flatnonzero(real)
    t, f = truth[rows], track_id[rows]
    order = np.lexsort((f, t))
    t, f = t[order], f[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(t)) + 1]) if len(t) else np.zeros(0, int)
    counts = np.diff(np.concatenate([starts, [len(t)]]))
    first_f = f[starts]
    last_f = f[starts + counts - 1]
    tracks = t[starts]
    whole = ((first_f == last_f) & (first_f > 0)
             & (f_size[np.minimum(first_f, len(f_size) - 1)] == counts))
    eligible = t_size[tracks] >= 2
    return {"links_true": float(len(ta)), "links_made": float(len(fa)),
            "links_found": float(hit.sum() / len(ta)) if len(ta) else float("nan"),
            "links_wrong": float((~hit).sum() / len(fa)) if len(fa) else 0.0,
            "tracks_whole": (float((whole & eligible).sum() / eligible.sum())
                             if eligible.any() else float("nan"))}
