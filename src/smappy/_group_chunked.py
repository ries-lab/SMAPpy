"""Linking in parallel: chunk the frame axis, link the chunks, stitch the seams.

`connect` is 59% of the time it takes to open a large file and runs on one core
(`csrc/group.hpp` is a single sequential walk; the binding releases the GIL but
starts no threads).  Every *link* it makes is at most ``dt + 1`` frames long, so
the frame axis can be cut and the pieces linked at the same time -- what a cut
costs is the traces that cross it, and only those have to be repaired.

The algorithm, in the order the code does it:

1.  **Cut** the sorted table into ``n_chunks`` pieces at frame boundaries,
    balanced by number of localizations.  A chunk is a contiguous slice of the
    (frame, x) sorted order, so each one is already in the form
    `connect_single` wants.

2.  **Link** the chunks at the same time, one thread each.  The extension
    releases the GIL, so these run properly in parallel.

3.  **Offset** each chunk's 1-based ids by the groups before it, so ids are
    globally unique.

4.  **Stitch** each seam.  A trace that crossed it was cut in two: a *tail*
    ending in the last frames of chunk k and a *head* starting in the first
    frames of chunk k+1.  For every tail we rebuild the running position the
    sequential walk would have carried across the seam -- ``xh = (xh + x)/2``
    over the group's members -- and then run that walk's own inner loop into
    chunk k+1's first frames.  Where it lands on the *seed* of a head group,
    the two are merged.

5.  **Renumber** through the merges, contiguously: `combine` counts to
    ``gi.max()``, so a gap in the ids would become an empty row with a zero
    ``n_in_group`` and NaNs across it.

Where this differs from the sequential walk, and it does differ:

* The sequential walk seeds particles in (frame, x) order over the whole table,
  and a particle claims localizations as it goes, taking them out of reach of
  particles seeded later.  Across a seam that order is lost: chunk k+1 seeds its
  own particles with no knowledge of the tails reaching into it.  The stitch
  restores the first link across the seam but not the consequences of the
  claims that link would have prevented.
* A tail is offered the heads in order of where it ends, which is the
  sequential order only to the extent that a trace's seed order follows its
  end order.
* Rebuilding the running position folds only the members the chunk holds.  The
  fold halves the weight of every earlier member, so a member more than ~24
  links back is below float precision anyway; a trace that started in an
  *earlier chunk* is the real case, and there the tail starts from its own
  chunk-local seed rather than the true one.

None of that is worth fixing blind -- it is worth measuring, which is what
``scripts/check_chunked_grouping.py`` does against the sequential result.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .group import sorted_order

try:
    from . import _group
except ImportError:                                   # pragma: no cover
    _group = None


def chunk_edges(frame: np.ndarray, n_chunks: int, lo: int = 0,
                hi: Optional[int] = None) -> List[int]:
    """Cut ``frame[lo:hi]`` into ``n_chunks`` slices of about equal size.

    ``frame`` is the sorted frame column, so the cuts are snapped forward to
    the next frame change: one frame never straddles two chunks, which would
    split a trace in a way the seam repair is not written for.
    """
    hi = len(frame) if hi is None else hi
    n = hi - lo
    if n_chunks <= 1 or n < 2 * n_chunks:
        return [lo, hi]
    edges = [lo]
    for k in range(1, n_chunks):
        cut = lo + (n * k) // n_chunks
        if cut <= edges[-1]:
            continue
        # forward to where the frame changes, so a frame stays whole
        cut = lo + int(np.searchsorted(frame[lo:hi], frame[cut], side="right"))
        if edges[-1] < cut < hi:
            edges.append(cut)
    edges.append(hi)
    return edges


def _link_chunks(ids, x, y, frame, z, dx, dt, dz, edges, workers):
    """Every chunk linked at once, then the ids made unique across them.

    `ids` is indexed like the sorted table, so everything after this -- the
    seam repair especially -- can use one set of indices.
    """
    def one(k):
        a, b = edges[k], edges[k + 1]
        zb = None if z is None else z[a:b]
        return _group.connect(x[a:b], y[a:b], frame[a:b], dx, dt, zb, dz)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, range(len(edges) - 1)))

    offset = 0
    for k, (part, count) in enumerate(results):
        ids[edges[k]:edges[k + 1]] = np.asarray(part) + offset
        offset += int(count)
    return offset


class _Merges:
    """Union-find over group ids, and the contiguous renumbering at the end."""

    def __init__(self, n: int):
        self.parent = np.arange(n + 1, dtype=np.int64)   # ids are 1-based

    def find(self, a: int) -> int:
        p = self.parent
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return int(a)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)       # keep the earlier id

    def renumber(self, ids: np.ndarray) -> Tuple[np.ndarray, int]:
        """Resolve every id through the merges, then close the gaps.

        Gaps are not cosmetic: `combine` bincounts to ``max(ids)`` and slices
        off index 0, so a missing id is an all-NaN row with ``n_in_group`` 0.
        """
        # pointer doubling rather than a Python loop over every id: the
        # chains are seam-length, so this settles in two or three passes
        root = self.parent
        while True:
            nxt = root[root]
            if np.array_equal(nxt, root):
                break
            root = nxt
        used = np.zeros(len(self.parent), bool)
        used[root[1:]] = True
        used[0] = False
        dense = np.zeros(len(self.parent), np.int64)
        dense[used] = np.arange(1, int(used.sum()) + 1, dtype=np.int64)
        return dense[root[ids]], int(used.sum())


def _running_state(xs, ys, zs):
    """The walk's own recursive mean: ``xh = (xh + x_new) / 2``, in order."""
    xh, yh, zh = float(xs[0]), float(ys[0]), (float(zs[0]) if zs is not None else 0.0)
    for i in range(1, len(xs)):
        xh = (float(xs[i]) + xh) / 2
        yh = (float(ys[i]) + yh) / 2
        if zs is not None:
            zh = (float(zs[i]) + zh) / 2
    return xh, yh, zh


def _stitch(x, y, frame, z, dx, dt, dz, ids, a, b, c, merges) -> Tuple[int, int]:
    """Repair one seam: chunk [a, b) against chunk [b, c).

    Returns (traces rejoined, tails that reached the seam).  Both sides are
    small -- only the last and first ``dt + 2`` frames take part -- so this is
    Python over a few thousand rows however large the table is.
    """
    if b <= a or c <= b:
        return 0, 0
    reach = dt + 1
    first = int(frame[b])

    # --- the tails: groups of chunk k still alive within reach of the seam ---
    tail_from = a + int(np.searchsorted(frame[a:b], first - reach, side="left"))
    if tail_from >= b:
        return 0, 0
    tail_ids = np.unique(ids[tail_from:b])
    if not len(tail_ids):
        return 0, 0

    # their members, over the whole chunk, so the running mean is the real one
    member = np.isin(ids[a:b], tail_ids)
    m_ids, m_x, m_y = ids[a:b][member], x[a:b][member], y[a:b][member]
    m_f = frame[a:b][member]
    m_z = z[a:b][member] if z is not None else None
    order = np.argsort(m_ids, kind="stable")             # frame order kept inside
    m_ids, m_x, m_y, m_f = m_ids[order], m_x[order], m_y[order], m_f[order]
    if m_z is not None:
        m_z = m_z[order]
    starts = np.concatenate(([0], np.flatnonzero(m_ids[1:] != m_ids[:-1]) + 1,
                             [len(m_ids)]))

    tails = []
    for s, e in zip(starts[:-1], starts[1:]):
        if int(m_f[e - 1]) < first - reach:               # died before the seam
            continue
        xh, yh, zh = _running_state(m_x[s:e], m_y[s:e], None if m_z is None else m_z[s:e])
        tails.append((int(m_f[e - 1]), xh, yh, zh, int(m_ids[s])))
    if not tails:
        return 0, 0
    tails.sort()                    # by where they end, then by position

    # --- the heads: the first frames of chunk k+1, and which of them seed ----
    head_to = b + int(np.searchsorted(frame[b:c], first + reach, side="right"))
    hx, hy, hf = x[b:head_to], y[b:head_to], frame[b:head_to]
    hz = z[b:head_to] if z is not None else None
    h_ids = ids[b:head_to]
    # a head may be joined only at its seed: taking it mid-trace would drag a
    # group that the sequential walk had already given to someone else
    seed = np.zeros(len(h_ids), bool)
    _, first_of = np.unique(h_ids, return_index=True)
    seed[first_of] = True
    taken = np.zeros(len(h_ids), bool)

    joined = 0
    for fh, xh, yh, zh, gid in tails:
        dark = 0
        while dark <= dt:
            nxt = fh + 1
            lo = int(np.searchsorted(hf, nxt, side="left"))
            hi = int(np.searchsorted(hf, nxt, side="right"))
            found = False
            if hi > lo:
                # sorted by x inside a frame, so walk only the window
                j0 = lo + int(np.searchsorted(hx[lo:hi], xh - dx, side="left"))
                for j in range(j0, hi):
                    if hx[j] >= xh + dx:
                        break
                    if taken[j] or not seed[j]:
                        continue
                    if not (yh - dx < hy[j] < yh + dx):
                        continue
                    if hz is not None and not (zh - dz < hz[j] < zh + dz):
                        continue
                    merges.union(gid, int(h_ids[j]))
                    taken[j] = True
                    joined += 1
                    found = True
                    break
            if found:
                break
            fh = nxt
            dark += 1
    return joined, len(tails)


def connect_chunked(x, y, frame, dx: float = 50.0, dt: int = 1,
                    blocks: Optional[np.ndarray] = None, z=None,
                    dz: Optional[float] = None, n_chunks: int = 8,
                    workers: Optional[int] = None, progress=None,
                    stats: Optional[dict] = None) -> np.ndarray:
    """`smappy.group.connect`, with the linking spread over threads.

    Same arguments and same return -- 1-based group ids in the input order --
    plus ``n_chunks``, how many pieces the frame axis is cut into.  One chunk
    is the sequential algorithm exactly.  ``stats``, if given, is filled with
    ``rejoined`` (traces put back together across a cut) and ``at_seams`` (how
    many groups reached a cut at all, the population an error can come from).
    """
    if stats is not None:
        stats.setdefault("rejoined", 0)
        stats.setdefault("at_seams", 0)
    if _group is None:                                    # pragma: no cover
        raise RuntimeError("the _group extension is not built")
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    frame = np.asarray(np.rint(np.asarray(frame, np.float64)), np.int64)
    z = None if (z is None or dz is None) else np.asarray(z, np.float64)
    dz = 0.0 if dz is None else float(dz)

    keys: Tuple[np.ndarray, ...] = ()
    if blocks is not None:
        bl = np.asarray(blocks)
        keys = tuple(bl.T) if bl.ndim == 2 else (bl,)
    if progress is not None:
        progress("connect (sorting)", 0.0)
    order = sorted_order(x, frame, keys, workers)

    xs, ys, fs = x[order], y[order], frame[order]
    zs = None if z is None else z[order]

    if keys:
        stacked = np.stack([np.asarray(k)[order] for k in keys], axis=1)
        block_edges = np.concatenate(
            ([0], np.flatnonzero(np.any(stacked[1:] != stacked[:-1], axis=1)) + 1,
             [x.size]))
    else:
        block_edges = np.array([0, x.size])

    out = np.zeros(x.size, np.int64)
    ids = np.zeros(x.size, np.int64)           # in sorted order, per block
    total = 0
    n_blocks = len(block_edges) - 1
    for bi, (lo, hi) in enumerate(zip(block_edges[:-1], block_edges[1:])):
        if progress is not None:
            label = "connect" if n_blocks == 1 else f"connect (block {bi + 1}/{n_blocks})"
            progress(label, lo / max(x.size, 1))
        edges = chunk_edges(fs, n_chunks, int(lo), int(hi))
        n_local = _link_chunks(ids, xs, ys, fs, zs, dx, dt, dz, edges, workers)
        merges = _Merges(n_local)
        for k in range(len(edges) - 2):
            joined, seen = _stitch(xs, ys, fs, zs, dx, dt, dz, ids,
                                   edges[k], edges[k + 1], edges[k + 2], merges)
            if stats is not None:
                stats["rejoined"] += joined
                stats["at_seams"] += seen
        block, n_groups = merges.renumber(ids[lo:hi])
        out[order[lo:hi]] = block + total
        total += n_groups
    return out
