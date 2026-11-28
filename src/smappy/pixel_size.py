"""The camera's pixel in the sample, from beads imaged at several stage positions.

A slide of beads is moved by the stage between frames -- a grid, a line, a
few random places -- and each frame is one image near focus.  The beads move
on the camera by the stage's step divided by the pixel, turned by the
camera's angle to the stage and mirrored if the light path mirrors; so the
camera's offset between frames, against the stage's, is the pixel size, the
rotation and the mirror at once:

    o_f = M s_f + c                (o in pixels, s in micrometres)

with ``M`` a 2x2 matrix in pixels per micrometre.  Its inverse's columns are
the stage's move for one pixel along camera x and along camera y: their
lengths are the pixel's size, their angle the camera's rotation, the sign of
its determinant the mirror (`decompose`).

**The beads first, the pixel after.**  A frame may share only a few beads
with the one before -- the step is most of a field -- so matching each frame
to the next alone is fragile: three beads, one of them missed, and the chain
breaks.  Nothing about the pixel is needed to find the offsets, though, so it
is not assumed:

1. **Offsets between frames, by a vote** (`frame_shift`): every pair of a
   spot in one frame and a spot in another votes for the shift between them;
   the beads the frames share agree, the rest scatter.  The shift with the
   most votes counts only where that many votes could not have come by
   chance among so many spots (`chance_votes`, a Poisson bound) -- a dense
   field's pairs agree by accident more often.  The same vote as
   `smappy.mbm_registration.match`, binned so that a frame of a few hundred
   beads takes milliseconds.  Every pair of frames is compared (or each with
   its `NEIGHBOURS` nearest by stage position, for many frames): two frames
   far apart in the scan with nothing between them in common still link
   through the ones that saw them both.
2. **All the offsets at once** (`solve_offsets`): the frames are the nodes
   of a graph and each agreeing pair an edge that says how far one is from
   the other; a least squares solution of the whole graph puts every frame
   where its edges together say, and an edge that disagrees with the rest
   by more than the tolerance -- a wrong vote -- is dropped and the graph
   solved again.  Frames not connected to the largest group are left out.
3. **Each bead's identity** (`link_beads`): every spot put back by its
   frame's offset lands where its bead is; spots within the tolerance of
   each other are one bead.  A bead is then seen in every frame that saw
   it, however far apart in the scan, and a group with two spots of one
   frame -- two beads too close to tell apart -- is left out.
4. **Offsets and beads together** (`refine`): the offsets and the beads'
   positions that fit every spot best, by alternating means, so that a
   frame's offset rests on every bead it saw, each bead measured in every
   frame.  Spots far from their bead's place are dropped.

**Then the stage** (`fit_stage`): the offsets against the stage positions,
a linear least squares fit of ``M`` and ``c``, and, when asked, the
**backlash**: a stage's position lags behind the command by a fixed amount
in the direction it last moved, so a raster that comes back along a row
lands a constant step off.  The model adds a term per stage axis that is +1
or -1 by the direction that axis last moved (`approach_directions`) -- the
users' scripts this replaced left out the steps after a reversal instead,
which on a serpentine raster still leaves the rows offset against each other
when positions, not steps, are fitted.  Frames whose residual is far beyond
the others' (a missed step) are left out, a few rounds.  The errors are a
delete-one jackknife over the frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: how far two spots' shifts may differ and still agree, pixels: a bead is fitted to a few
#: hundredths of a pixel, so this leaves room for the field's distortion and a bead out of focus,
#: and no more -- each halving of it makes an accidental agreement of k spots 2^(2k - 2) times rarer
TOLERANCE_PX = 0.5
#: the vote bins differences on a dense grid up to this many bins, and while there are no more
#: than 16 bins to a difference; sparsely else
DENSE_BINS = 4_000_000
#: the fewest beads two frames must share to be linked by them
MIN_VOTES = 3
#: frames compared with each frame when there are too many to compare every pair
NEIGHBOURS = 20
#: every pair of frames is compared up to this many pairs
ALL_PAIRS = 5000
#: and up to this many differences of spots in all (a few seconds' vote): beyond, each frame with
#: its `NEIGHBOURS` nearest only
MAX_DIFFERENCES = 2e7
#: the chance that any pair of frames of a scan is linked by accident, over every shift each
#: could have: divided among the pairs compared.  A wrong link is often caught after -- it
#: disagrees with its frames' other links (`solve_offsets`), or puts a frame off the stage's line
#: (`fit_stage`) -- but not where it is a frame's only one
FALSE_LINK = 0.05
#: how far from where the stage's rough map puts a group of frames its shift is searched,
#: pixels, plus five times the map's uncertainty there (`StageFit.uncertainty`)
GUIDED_RADIUS_PX = 1.0
#: the least a map's rms residual is taken to be, pixels: a map from as many frames as it has
#: unknowns fits them exactly, and is not exact
RMS_FLOOR_PX = 0.2
#: how far a guessed axis may be off, as a share of the step: pixels this far from square, or
#: axes this far from perpendicular (0.15 rad, 9 degrees)
ASPECT = 0.15
#: frames a stage fit needs beyond its unknowns, so that a jackknife and the outliers mean something
SPARE_FRAMES = 3
#: a frame (or a spot) is an outlier this many robust standard deviations from the fit
OUTLIER_SIGMAS = 5.0
#: what a residual below this, pixels, is never called an outlier for: a stage that steps to a
#: few nanometres gives a robust sigma far below a bead's precision
OUTLIER_FLOOR_PX = 0.05


# ------------------------------------------------------------------ the vote

def chance_votes(n_a: int, n_b: int, area_px: float, tolerance_px: float,
                 p: float = FALSE_LINK, searched_px: Optional[float] = None) -> int:
    """The fewest votes a shift needs between frames of ``n_a`` and ``n_b`` spots on ``area_px``
    square pixels for an accident to be rarer than ``p``.

    The differences of unrelated spots lie at most ``n_a n_b / area`` per square pixel (at a
    small shift, less where the frames overlap less), so a disc of the tolerance holds a Poisson
    number of them, at most ``lam``.  Over every shift two frames can have, the expected number of
    discs with k or more is the Poisson tail times the discs there are, the overlap's falling
    density counted -- a k-th power of a triangle on each axis integrates to (2 / (k + 1))^2 of a
    frame; over ``searched_px`` square pixels around a predicted shift it is the full density
    there."""
    disc = np.pi * tolerance_px ** 2
    lam = n_a * n_b * disc / max(area_px, 1.0)
    k = np.arange(1, 10 * int(lam) + 50)
    discs = (np.full(len(k), float(searched_px)) if searched_px is not None
             else area_px * (2.0 / (k + 1)) ** 2) / disc
    return _first_beyond(k, lam, discs, p)


def _first_beyond(k: np.ndarray, lam: float, discs: np.ndarray, p: float) -> int:
    """The first ``k`` whose Poisson tail P(X >= k) at ``lam``, times ``discs``, is at most ``p``."""
    from scipy.special import pdtrc            # P(X > k - 1): a ufunc, where scipy.stats is not
    ok = pdtrc(k - 1, lam) * np.maximum(discs, 1.0) <= p
    return int(k[np.argmax(ok)]) if ok.any() else int(k[-1])


def frame_shift(a: np.ndarray, b: np.ndarray, tolerance_px: float = TOLERANCE_PX,
                expected: Optional[Tuple[float, float]] = None,
                around: Optional[Tuple[np.ndarray, float]] = None
                ) -> Tuple[np.ndarray, int]:
    """The shift ``d`` with ``b ≈ a + d`` most pairs of spots agree on, and how many do (`vote`).

    ``a``, ``b`` are (n, 2) positions in pixels; every difference ``b_j - a_i`` votes.
    ``expected`` (lo, hi) keeps only shifts of that length, pixels -- where the pixel is roughly
    known, from the stage's step; ``around`` (shift, radius) only shifts within the radius of that
    one, where the stage's map predicts it.
    """
    a = np.asarray(a, float)[:, :2]
    b = np.asarray(b, float)[:, :2]
    if not len(a) or not len(b):
        return np.full(2, np.nan), 0
    d = (b[None, :, :] - a[:, None, :]).reshape(-1, 2)
    if expected is not None:
        length = np.hypot(d[:, 0], d[:, 1])
        d = d[(length >= expected[0]) & (length <= expected[1])]
    if around is not None:
        d = d[np.hypot(*(d - np.asarray(around[0], float)).T) <= around[1]]
    return vote(d, tolerance_px)


def vote(d: np.ndarray, tolerance_px: float = TOLERANCE_PX) -> Tuple[np.ndarray, int]:
    """The shift most of the differences ``d`` (n, 2) agree on within the tolerance, and how many
    do.  Binned at the tolerance, each bin's count summed with its eight neighbours', so a cluster
    split by a bin edge counts whole; the winner refined to the median of the differences near it
    and counted within the tolerance.  Milliseconds for the 10^5 differences of two frames of a
    few hundred beads, where comparing every difference with every other would not finish."""
    d = np.asarray(d, float).reshape(-1, 2)
    if not len(d):
        return np.full(2, np.nan), 0
    keys = np.floor(d / tolerance_px).astype(np.int64)
    low = keys.min(axis=0) - 1
    keys = keys - low
    height, width = (int(v) for v in keys.max(axis=0) + 2)
    code = keys[:, 0] * width + keys[:, 1]
    if height * width <= min(DENSE_BINS, 16 * len(d)):
        # a histogram on the grid and a 3 x 3 box sum: tens of milliseconds for 10^5 differences
        counts = np.bincount(code, minlength=height * width).reshape(height, width)
        padded = np.pad(counts, 1)
        total = sum(padded[1 + dx:1 + dx + height, 1 + dy:1 + dy + width]
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        best = int(np.argmax(total))
    else:
        bins, counts = np.unique(code, return_counts=True)
        total = np.zeros(len(bins), np.int64)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbour = bins + dx * width + dy
                at = np.minimum(np.searchsorted(bins, neighbour), len(bins) - 1)
                hit = bins[at] == neighbour
                total[hit] += counts[at[hit]]
        best = int(bins[np.argmax(total)])
    centre = (np.array([best // width, best % width]) + low + 0.5) * tolerance_px
    # the refinement looks no further than this: the shift moves by less than a tolerance
    d = d[np.max(np.abs(d - centre), axis=1) <= 3.0 * tolerance_px]
    shift = np.median(d[np.max(np.abs(d - centre), axis=1) <= 1.5 * tolerance_px], axis=0)
    for _ in range(3):
        close = np.hypot(*(d - shift).T) <= tolerance_px
        if not close.any():
            break
        shift = np.median(d[close], axis=0)
    close = np.hypot(*(d - shift).T) <= tolerance_px
    return shift, int(close.sum())


@dataclass
class Edge:
    """Two frames that share beads: ``offset[second] - offset[first] = shift``."""
    first: int
    second: int
    shift: np.ndarray
    votes: int


def frame_pairs(stage_um: np.ndarray, neighbours: int = NEIGHBOURS,
                all_pairs: int = ALL_PAIRS) -> List[Tuple[int, int]]:
    """The pairs of frames to compare: every pair while there are at most ``all_pairs``, else each
    frame with its ``neighbours`` nearest by stage position -- the ones that can overlap."""
    n = len(stage_um)
    if n * (n - 1) // 2 <= all_pairs:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    from scipy.spatial import cKDTree
    stage = np.nan_to_num(np.asarray(stage_um, float), nan=1e12)
    # frames at one stage position are all each other's nearest: a little noise breaks the ties
    tree = cKDTree(stage + 1e-9 * np.arange(n)[:, None])
    _, near = tree.query(stage, k=min(neighbours + 1, n))
    pairs = {(min(i, j), max(i, j)) for i, row in enumerate(near) for j in row if i != j}
    return sorted(pairs)


def pair_shifts(positions: Sequence[np.ndarray], pairs: Sequence[Tuple[int, int]],
                frame_area_px: float, tolerance_px: float = TOLERANCE_PX,
                min_votes: int = MIN_VOTES, stage_um: Optional[np.ndarray] = None,
                pixel_hint_nm: Optional[float] = None, report=None) -> List[Edge]:
    """The `Edge` of every pair of frames whose vote beats chance (`chance_votes`) and
    ``min_votes``.  ``positions`` holds each frame's spots, (n, 2) pixels.  With ``pixel_hint_nm``
    and ``stage_um`` a pair is only searched for shifts of about the length the stage's step makes
    (from half to twice it: the pixel is not known, only roughly), and a pair the step puts more
    than a frame apart is not compared at all."""
    edges = []
    side = np.sqrt(frame_area_px)
    for k, (i, j) in enumerate(pairs):
        a, b = positions[i], positions[j]
        if len(a) < min_votes or len(b) < min_votes:
            continue
        expected = None
        if pixel_hint_nm and stage_um is not None:
            step = 1000.0 * float(np.hypot(*(stage_um[j] - stage_um[i]))) / pixel_hint_nm
            if not np.isfinite(step) or step > 1.5 * side * np.sqrt(2.0):
                continue
            expected = (0.5 * step - 3 * tolerance_px, 2.0 * step + 3 * tolerance_px)
        shift, votes = frame_shift(a, b, tolerance_px, expected)
        need = chance_votes(len(a), len(b), frame_area_px, tolerance_px,
                            FALSE_LINK / max(len(pairs), 1))
        if votes >= max(min_votes, need):
            edges.append(Edge(i, j, shift, votes))
        if report is not None and k % 500 == 499:
            report(f"compared {k + 1} of {len(pairs)} pairs of frames")
    return edges


def rough_maps(offsets: np.ndarray, stage_um: np.ndarray, backlash: bool = True
               ) -> Tuple[List["StageFit"], float]:
    """The stage's maps onto the camera the frames placed so far allow, and how far to trust them,
    pixels (their rms residual; for a guess, a share of the step).

    With frames that span two directions it is one, `fit_stage` of them.  With frames along one
    line only, the camera's move per micrometre along that line is known and the other is
    guessed: the same length turned a quarter either way -- pixels about square, the camera
    mirrored or not -- two maps, searched both, each a share `ASPECT` of the step wide.  None
    while fewer than three frames are placed."""
    try:
        fit = fit_stage(offsets, stage_um, backlash, spare=1)
    except ValueError:
        fit = None
    if fit is not None:
        r = fit.residual_px[fit.used]
        return [fit], float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))
    rows = np.flatnonzero(np.all(np.isfinite(offsets), axis=1)
                          & np.all(np.isfinite(stage_um), axis=1))
    if len(rows) < 3:
        return [], 0.0
    centred = stage_um[rows] - stage_um[rows].mean(axis=0)
    u = np.linalg.svd(centred, full_matrices=False)[2][0]           # the line's direction
    along = centred @ u
    if np.ptp(along) <= 0:
        return [], 0.0
    a = np.column_stack([along, np.ones(len(rows))])
    coef, *_ = np.linalg.lstsq(a, offsets[rows], rcond=None)
    m = coef[0]                                                      # pixels per um along u
    v = np.array([-u[1], u[0]])
    directions = approach_directions(stage_um)
    maps = []
    for turn in (1.0, -1.0):
        m_v = turn * np.array([-m[1], m[0]])
        matrix = np.outer(m, u) + np.outer(m_v, v)
        constant = coef[1] - matrix @ stage_um[rows].mean(axis=0)
        maps.append(StageFit(matrix, constant, np.zeros((2, 2)), np.zeros(len(offsets), bool),
                             np.full(offsets.shape, np.nan), directions, stage_um))
    return maps, -ASPECT


def join_groups(positions: Sequence[np.ndarray], offsets: np.ndarray, group: np.ndarray,
                main: int, maps: Sequence["StageFit"], trust: float, stage_um: np.ndarray,
                frame_area_px: float, tolerance_px: float = TOLERANCE_PX,
                min_votes: int = MIN_VOTES) -> int:
    """Join the groups of frames the blind vote left apart to the ``main`` group, in place:
    ``offsets`` of a joined group's frames made the main group's, its ``group`` set to ``main``.
    Returns how many were joined.

    A group's frames are placed among themselves; only its offset against the main group is
    missing, one shift for all of them.  So every pair of a frame of it and a frame of the main
    group that can overlap votes for that one shift, and the votes are pooled: three frames each
    sharing two beads with their neighbours are six agreeing votes.  They are searched only near
    the shift the stage's ``maps`` (`rough_maps`) predict -- within the map's uncertainty there
    (``trust``, its rms residual in pixels, through `StageFit.uncertainty`; negative: a guessed
    map, off by that share of the distance) -- so that a few votes are beyond chance there, as they are not among every shift."""
    side = np.sqrt(frame_area_px)
    disc = np.pi * tolerance_px ** 2
    others = [g for g in np.unique(group) if g != main]
    others.sort(key=lambda g: -int(np.sum(group == g)))
    p = FALSE_LINK / max(len(others) * len(maps), 1)
    joined = 0
    for g in others:
        mine, theirs = np.flatnonzero(group == main), np.flatnonzero(group == g)
        best = (0, None)
        for rough in maps:
            predicted = rough.predict(stage_um, rough.directions)
            # the group's own offsets are relative: the shift that puts them where the map does
            guess = np.median(predicted[theirs] - offsets[theirs], axis=0)
            if trust < 0:
                far = np.hypot(*(np.mean(predicted[theirs], axis=0)
                                 - np.mean(predicted[mine], axis=0)))
                radius = GUIDED_RADIUS_PX - trust * far
            else:
                radius = GUIDED_RADIUS_PX + 5.0 * rough.uncertainty(
                    theirs, max(trust, RMS_FLOOR_PX))
            radius = min(radius, side / 2)
            pooled, lam = [], 0.0
            for j in theirs:
                b = positions[j] - offsets[j]
                for i in mine:
                    step = offsets[j] + guess - offsets[i]
                    if np.max(np.abs(step)) > side + radius or not len(positions[i]) \
                            or not len(b):
                        continue
                    a = positions[i] - offsets[i]
                    d = (b[None, :, :] - a[:, None, :]).reshape(-1, 2)
                    pooled.append(d[np.hypot(*(d - guess).T) <= radius])
                    lam += len(a) * len(b) * disc / frame_area_px
            if not pooled:
                continue
            shift, votes = vote(np.concatenate(pooled), tolerance_px)
            need = chance_votes_pooled(lam, np.pi * radius ** 2 / disc, p)
            if votes >= max(min_votes, need) and votes > best[0]:
                best = (votes, shift)
        if best[1] is not None:
            offsets[theirs] += best[1]
            group[theirs] = main
            joined += 1
    return joined


def chance_votes_pooled(lam: float, discs: float, p: float) -> int:
    """The fewest votes for a shift to beat chance at ``p``: a Poisson count of mean ``lam`` per
    disc of the tolerance (the sum over the pooled pairs of frames), over ``discs`` of them."""
    k = np.arange(1, 10 * int(lam) + 50)
    return _first_beyond(k, lam, np.full(len(k), float(discs)), p)


# ------------------------------------------------------------------ the graph

def solve_offsets(n_frames: int, edges: Sequence[Edge], tolerance_px: float = TOLERANCE_PX
                  ) -> Tuple[np.ndarray, np.ndarray, List[Edge]]:
    """Each frame's offset from the edges, by least squares weighted by the votes, in each
    connected group of frames (its first frame at 0; a frame in no edge a group of its own).  An
    edge off the solution by more than the tolerance is dropped, the worst first, and the graph
    solved again.  Returns (offsets (n, 2), the group of each frame, the edges kept)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import lsqr
    kept = list(edges)
    while True:
        i = np.array([e.first for e in kept], np.int64)
        j = np.array([e.second for e in kept], np.int64)
        graph = coo_matrix((np.ones(len(kept)), (i, j)), shape=(n_frames, n_frames))
        _, label = connected_components(graph, directed=False)
        first = np.zeros(n_frames, bool)
        first[np.unique(label, return_index=True)[1]] = True       # held at 0
        index = np.full(n_frames, -1)
        index[~first] = np.arange(int((~first).sum()))
        offsets = np.zeros((n_frames, 2))
        if not kept:
            return offsets, label, []
        rows, cols, vals = [], [], []
        for r, e in enumerate(kept):
            w = np.sqrt(e.votes)
            if index[e.second] >= 0:
                rows.append(r), cols.append(index[e.second]), vals.append(w)
            if index[e.first] >= 0:
                rows.append(r), cols.append(index[e.first]), vals.append(-w)
        a = coo_matrix((vals, (rows, cols)),
                       shape=(len(kept), max(int((~first).sum()), 1))).tocsr()
        weights = np.sqrt([e.votes for e in kept])
        if (~first).any():
            for axis in range(2):
                rhs = weights * np.array([e.shift[axis] for e in kept])
                offsets[~first, axis] = lsqr(a, rhs, atol=1e-12, btol=1e-12)[0]
        off = np.array([np.hypot(*(offsets[e.second] - offsets[e.first] - e.shift))
                        for e in kept])
        worst = int(np.argmax(off))
        if off[worst] <= tolerance_px:
            return offsets, label, kept
        kept.pop(worst)


def link_beads(frame: np.ndarray, xy_px: np.ndarray, offsets: np.ndarray,
               tolerance_px: float = TOLERANCE_PX) -> np.ndarray:
    """A bead id per spot (-1 for none): spots that land within the tolerance of each other once
    their frames' offsets are taken off, in groups seen in two frames or more and never twice in
    one frame."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    frame = np.asarray(frame, np.int64)
    out = np.full(len(frame), -1, np.int64)
    ok = np.all(np.isfinite(offsets[frame]), axis=1)
    rows = np.flatnonzero(ok)
    if len(rows) < 2:
        return out
    placed = xy_px[rows] - offsets[frame[rows]]
    pairs = cKDTree(placed).query_pairs(tolerance_px, output_type="ndarray")
    n = len(rows)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    size = np.bincount(label)
    # how many distinct frames each group has: one spot per frame, else it is two beads
    distinct = np.unique(np.column_stack([label, frame[rows]]), axis=0)
    frames_seen = np.bincount(distinct[:, 0], minlength=len(size))
    good = (size >= 2) & (frames_seen == size)
    ids = np.full(len(size), -1, np.int64)
    ids[good] = np.arange(int(good.sum()))
    out[rows] = ids[label]
    return out


def refine(frame: np.ndarray, xy_px: np.ndarray, bead: np.ndarray, offsets: np.ndarray,
           iterations: int = 500, outlier_sigmas: float = OUTLIER_SIGMAS
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The offsets and the beads' positions that fit the spots best (``spot = bead + offset``):
    alternating means, the first frame with beads held where it is.  A spot further from its
    bead than ``outlier_sigmas`` robust standard deviations is taken out (its id set to -1) and
    the fit made again.  Returns (offsets, bead positions (n_beads, 2), the bead ids used)."""
    frame = np.asarray(frame, np.int64)
    bead = np.array(bead, np.int64)
    xy_px = np.asarray(xy_px, float)
    offsets = np.array(offsets, float)
    n_frames = len(offsets)
    p = np.zeros((0, 2))
    for _ in range(4):
        use = bead >= 0
        f, b, q = frame[use], bead[use], xy_px[use]
        n_beads = int(b.max()) + 1 if len(b) else 0
        if not n_beads:
            return offsets, np.zeros((0, 2)), bead
        held = int(f.min())
        seen = np.bincount(f, minlength=n_frames) > 0
        o = np.where(np.isfinite(offsets), offsets, 0.0)
        count_b = np.bincount(b, minlength=n_beads)[:, None]
        count_f = np.maximum(np.bincount(f, minlength=n_frames), 1)[:, None]
        for _ in range(iterations):
            p = np.column_stack([np.bincount(b, q[:, k] - o[f, k], n_beads)
                                 for k in range(2)]) / count_b
            new = np.column_stack([np.bincount(f, q[:, k] - p[b, k], n_frames)
                                   for k in range(2)]) / count_f
            new -= new[held]
            step = np.max(np.abs(new[seen] - o[seen]))
            o = new
            if step < 1e-8:
                break
        p = np.column_stack([np.bincount(b, q[:, k] - o[f, k], n_beads)
                             for k in range(2)]) / count_b
        residual = np.hypot(*(q - p[b] - o[f]).T)
        sigma = 1.4826 * np.median(residual)
        bad = residual > max(outlier_sigmas * sigma, OUTLIER_FLOOR_PX)
        offsets = np.where(seen[:, None], o, np.nan)
        if not bad.any():
            return offsets, p, bead
        bead[np.flatnonzero(use)[bad]] = -1
        # a bead left with one spot says nothing about the offsets
        alone = np.bincount(bead[bead >= 0], minlength=n_beads) < 2
        bead[(bead >= 0) & alone[np.maximum(bead, 0)]] = -1
        _, bead[bead >= 0] = np.unique(bead[bead >= 0], return_inverse=True)
    return offsets, p, bead


# ------------------------------------------------------------------ the stage

def approach_directions(stage_um: np.ndarray, still_um: Optional[float] = None) -> np.ndarray:
    """Per frame and stage axis, the direction (+1 or -1) the axis last moved in to get there; the
    frames before an axis first moves take that first move's.  A step below ``still_um`` (auto:
    1 % of the median step) is no move.  0 for an axis that never moved."""
    stage = np.asarray(stage_um, float)
    out = np.zeros(stage.shape)
    for axis in range(stage.shape[1]):
        step = np.diff(stage[:, axis])
        moved = np.abs(step[np.isfinite(step) & (np.abs(step) > 0)])
        if not len(moved):
            continue
        small = 0.01 * float(np.median(moved)) if still_um is None else still_um
        direction, current = np.zeros(len(stage)), 0.0
        for k, s in enumerate(step, start=1):
            if np.isfinite(s) and abs(s) > small:
                current = np.sign(s)
            direction[k] = current
        if np.any(direction):
            direction[direction == 0] = direction[np.flatnonzero(direction)[0]]
        out[:, axis] = direction
    return out


@dataclass
class StageFit:
    """``offset = matrix @ stage + constant + backlash @ directions``: the matrix in pixels per
    micrometre, the backlash's columns in pixels per stage axis (zero where not fitted)."""
    matrix: np.ndarray
    constant: np.ndarray
    backlash: np.ndarray
    used: np.ndarray                    # frames in the fit
    residual_px: np.ndarray             # (n, 2), NaN where the offset is unknown
    directions: np.ndarray              # (n, 2), `approach_directions`
    stage_um: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    axes: List[int] = field(default_factory=list)       # stage axes with a backlash term
    #: (A^T A)^-1 of the fit's design matrix, for how well it predicts elsewhere; None: a guess
    inverse: Optional[np.ndarray] = None

    def design(self, stage_um: np.ndarray, directions: np.ndarray) -> np.ndarray:
        columns = [stage_um[:, 0], stage_um[:, 1], np.ones(len(stage_um))]
        return np.column_stack(columns + [directions[:, axis] for axis in self.axes])

    def uncertainty(self, rows: np.ndarray, rms_px: float) -> float:
        """How far off the map may put the frames ``rows``, pixels, at most: the residual's rms
        times the square root of one plus each frame's leverage -- far from the frames it was
        fitted to, a map of few frames extrapolates."""
        if self.inverse is None:
            return float("inf")
        x = self.design(np.atleast_2d(self.stage_um[rows]), self.directions[rows])
        leverage = np.einsum("ij,jk,ik->i", x, self.inverse, x)
        return float(rms_px * np.sqrt(1.0 + np.max(leverage)))

    def predict(self, stage_um: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """The offsets, pixels, at these stage positions approached from these directions."""
        return stage_um @ self.matrix.T + self.constant + directions @ self.backlash.T


def _lstsq(stage_um, offsets, directions, axes, rows):
    columns = [stage_um[rows, 0], stage_um[rows, 1], np.ones(len(rows))]
    columns += [directions[rows, axis] for axis in axes]
    coef, *_ = np.linalg.lstsq(np.column_stack(columns), offsets[rows], rcond=None)
    backlash = np.zeros((2, 2))
    for k, axis in enumerate(axes):
        backlash[:, axis] = coef[3 + k]
    return coef[:2].T, coef[2], backlash


def backlash_axes(directions: np.ndarray, rows: np.ndarray,
                  stage_um: Optional[np.ndarray] = None) -> List[int]:
    """The stage axes whose backlash the frames ``rows`` can tell apart: whose last-move direction
    changes over them, and -- with ``stage_um`` -- not in step with the position.  Two rows of a
    serpentine raster are the case: x is approached forward in the one and back in the other, so
    the direction is the row, and a backlash in x cannot be told from the step in y."""
    axes = [axis for axis in range(directions.shape[1])
            if len(np.unique(directions[rows, axis])) == 2]
    if stage_um is None:
        return axes
    kept: List[int] = []
    base = [stage_um[rows, 0], stage_um[rows, 1], np.ones(len(rows))]
    for axis in axes:
        x = np.column_stack(base + [directions[rows, a] for a in kept + [axis]])
        x = x / np.maximum(np.abs(x).max(axis=0), 1e-12)
        if np.linalg.matrix_rank(x, tol=1e-6) == x.shape[1]:
            kept.append(axis)
    return kept


def fit_stage(offsets: np.ndarray, stage_um: np.ndarray, backlash: bool = True,
              outlier_sigmas: float = OUTLIER_SIGMAS, rounds: int = 3,
              spare: int = SPARE_FRAMES) -> StageFit:
    """`StageFit` of the frames whose offset is known; frames off by more than
    ``outlier_sigmas`` robust standard deviations left out, ``rounds`` times.  ``spare`` frames
    beyond the unknowns are needed (fewer raise), and kept when leaving outliers out."""
    offsets = np.asarray(offsets, float)
    stage = np.asarray(stage_um, float)
    directions = approach_directions(stage)
    used = np.all(np.isfinite(offsets), axis=1) & np.all(np.isfinite(stage), axis=1)
    unknowns = 3 + (len(backlash_axes(directions, np.flatnonzero(used), stage)) if backlash
                    else 0)
    if used.sum() < unknowns + spare:
        raise ValueError(f"{int(used.sum())} frames could be placed: {unknowns + spare} at "
                         f"least are needed -- the steps are too large for the field, or too few "
                         f"beads were found")
    centred = stage[used] - stage[used].mean(axis=0)
    if np.linalg.matrix_rank(centred, tol=1e-6 * max(np.abs(centred).max(), 1e-12)) < 2:
        raise ValueError("the stage moved along one line only: a pixel size and an angle need "
                         "steps in two directions")
    for _ in range(rounds + 1):
        rows = np.flatnonzero(used)
        axes = backlash_axes(directions, rows, stage) if backlash else []
        matrix, constant, lag = _lstsq(stage, offsets, directions, axes, rows)
        residual = offsets - (stage @ matrix.T + constant + directions @ lag.T)
        length = np.hypot(*residual.T)
        sigma = 1.4826 * np.median(length[used])
        bad = used & (length > max(outlier_sigmas * sigma, OUTLIER_FLOOR_PX))
        if not bad.any() or used.sum() - bad.sum() < unknowns + spare:
            break
        used = used & ~bad
    rows = np.flatnonzero(used)
    fit = StageFit(matrix, constant, lag, used, residual, directions, stage, axes)
    x = fit.design(stage[rows], directions[rows])
    fit.inverse = np.linalg.pinv(x.T @ x)
    return fit


# ------------------------------------------------------------------ the numbers

def decompose(matrix: np.ndarray) -> Dict[str, float]:
    """What ``matrix`` (pixels per micrometre) says about the camera: ``pixel_x_nm`` and
    ``pixel_y_nm``, the stage's move for one pixel along camera x and y; ``angle_deg``, the
    direction of camera x in the stage's coordinates (0: along the stage's +x, 180: against it);
    ``mirrored``, 1 when camera x to camera y turns the other way than stage x to stage y;
    ``skew_deg``, how far the camera's axes are from square in the stage's coordinates."""
    p = 1000.0 * np.linalg.inv(np.asarray(matrix, float))         # nm per pixel
    a, b = p[:, 0], p[:, 1]
    px, py = float(np.hypot(*a)), float(np.hypot(*b))
    angle = float(np.degrees(np.arctan2(a[1], a[0])))
    skew = float(np.degrees(np.arcsin(np.clip(a @ b / (px * py), -1.0, 1.0))))
    return {"pixel_x_nm": px, "pixel_y_nm": py, "angle_deg": angle,
            "mirrored": float(np.linalg.det(p) < 0), "skew_deg": skew}


def camera_matrix(pixel_nm: Tuple[float, float], angle_deg: float = 0.0,
                  mirrored: bool = False) -> np.ndarray:
    """The matrix (pixels per micrometre) of a camera with this pixel, angle and mirror and square
    axes: `decompose`'s inverse."""
    t = np.radians(angle_deg)
    a = np.array([np.cos(t), np.sin(t)]) * pixel_nm[0]
    turn = -1.0 if mirrored else 1.0
    b = np.array([-np.sin(t), np.cos(t)]) * turn * pixel_nm[1]
    return np.linalg.inv(np.column_stack([a, b]) / 1000.0)


def _wrapped(values: np.ndarray, centre: float) -> np.ndarray:
    return (values - centre + 180.0) % 360.0 - 180.0 + centre


def backlash_nm(matrix: np.ndarray, lag: np.ndarray) -> np.ndarray:
    """The hysteresis along each stage axis, nm: how much further the stage is when it arrives
    moving forward than moving back -- twice each term, in the stage's coordinates; negative is
    the usual lag."""
    p = 1000.0 * np.linalg.inv(np.asarray(matrix, float))
    moved = p @ np.asarray(lag, float)                    # column per stage axis, nm
    return np.array([2.0 * moved[0, 0], 2.0 * moved[1, 1]])


def jackknife(offsets: np.ndarray, stage_um: np.ndarray, fit: StageFit,
              backlash: bool = True) -> Dict[str, float]:
    """Delete-one jackknife standard errors over the frames of ``fit``: ``pixel_x_nm_err`` and the
    like, the stage fit made again without each frame in turn (the offsets kept)."""
    stage = np.asarray(stage_um, float)
    directions = fit.directions
    rows = np.flatnonzero(fit.used)
    full = decompose(fit.matrix)
    estimates: Dict[str, List[float]] = {k: [] for k in full if k != "mirrored"}
    lags: List[np.ndarray] = []
    for leave in rows:
        keep = rows[rows != leave]
        axes = [a for a in fit.axes if len(np.unique(directions[keep, a])) == 2]
        matrix, _, lag = _lstsq(stage, offsets, directions, axes, keep)
        for k, v in decompose(matrix).items():
            if k in estimates:
                estimates[k].append(v)
        lags.append(backlash_nm(matrix, lag))
    n = len(rows)
    out = {}
    for k, values in estimates.items():
        v = np.asarray(values)
        if k == "angle_deg":
            v = _wrapped(v, full[k])
        out[f"{k}_err"] = float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))
    lag = np.asarray(lags)
    spread = np.sqrt((n - 1) / n * np.sum((lag - lag.mean(axis=0)) ** 2, axis=0))
    out["backlash_x_nm_err"], out["backlash_y_nm_err"] = float(spread[0]), float(spread[1])
    return out


# ------------------------------------------------------------------ all of it

@dataclass
class PixelSizeResult:
    """Everything `measure` found: the numbers, the stage fit and what it rested on."""
    values: Dict[str, float]
    fit: StageFit
    offsets_px: np.ndarray              # (n_frames, 2), NaN for frames not placed
    stage_um: np.ndarray                # (n_frames, 2)
    bead: np.ndarray                    # per spot, -1 none
    bead_xy_px: np.ndarray              # (n_beads, 2) in the first placed frame's camera
    edges: List[Edge] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def pixel_nm(self) -> Tuple[float, float]:
        return self.values["pixel_x_nm"], self.values["pixel_y_nm"]

    def text(self) -> str:
        v = self.values
        n_frames = len(self.offsets_px)
        lines = [
            f"pixel: {v['pixel_x_nm']:.2f} ± {v['pixel_x_nm_err']:.2f} nm along camera x, "
            f"{v['pixel_y_nm']:.2f} ± {v['pixel_y_nm_err']:.2f} nm along y",
            f"camera x at {v['angle_deg']:.2f} ± {v['angle_deg_err']:.2f}° to the stage's x, "
            + ("mirrored" if v["mirrored"] else "not mirrored")
            + f"; axes {v['skew_deg']:+.2f} ± {v['skew_deg_err']:.2f}° from square",
            f"{int(v['n_frames'])} of {n_frames} frames, {int(v['n_beads'])} beads; residual "
            f"{v['residual_nm']:.1f} nm rms",
        ]
        lags = [f"{v[f'backlash_{a}_nm']:.0f} ± {v[f'backlash_{a}_nm_err']:.0f} nm in {a}"
                for k, a in enumerate("xy") if self.fit.backlash[:, k].any()]
        if lags:
            lines.append("backlash " + ", ".join(lags))
        return "\n".join(lines + self.notes)


def measure(frame: np.ndarray, xy_px: np.ndarray, stage_um: np.ndarray,
            frame_shape: Tuple[int, int], tolerance_px: float = TOLERANCE_PX,
            min_votes: int = MIN_VOTES, backlash: bool = True,
            pixel_hint_nm: Optional[float] = None, report=None) -> PixelSizeResult:
    """The pixel size, angle and mirror from bead spots (``frame``, ``xy_px``: one row per spot,
    positions in camera pixels) and each frame's stage position (``stage_um``, (n_frames, 2)).
    ``frame_shape`` is (height, width) in pixels.  The steps are the module docstring's.  With
    ``pixel_hint_nm`` the vote looks only for shifts the stage's steps make at about that pixel;
    when that leaves frames unplaced it is tried again without."""
    frame = np.asarray(frame, np.int64)
    xy = np.asarray(xy_px, float)[:, :2]
    stage = np.asarray(stage_um, float)[:, :2]
    n_frames = len(stage)
    say = report or (lambda text: None)
    positions = [xy[frame == f] for f in range(n_frames)]
    area = float(frame_shape[0] * frame_shape[1])
    pairs = frame_pairs(stage)
    counts = np.array([len(p) for p in positions], float)
    if np.mean(counts) ** 2 * len(pairs) > MAX_DIFFERENCES:
        # many beads in many frames: each frame with its nearest by the stage, which are the
        # ones that overlap; the rest are linked through them
        pairs = frame_pairs(stage, all_pairs=0)
    say(f"comparing {len(pairs)} pairs of {n_frames} frames")
    notes: List[str] = []
    edges = pair_shifts(positions, pairs, area, tolerance_px, min_votes, stage, pixel_hint_nm,
                        report)
    offsets, group, kept = solve_offsets(n_frames, edges, tolerance_px)
    if pixel_hint_nm and _largest(group)[1] < n_frames:
        blind = pair_shifts(positions, pairs, area, tolerance_px, min_votes, report=report)
        again = solve_offsets(n_frames, blind, tolerance_px)
        if _largest(again[1])[1] > _largest(group)[1]:
            offsets, group, kept = again
            notes.append(f"the pixel of {pixel_hint_nm:g} nm left frames apart: searched "
                         f"without it")
    # the frames placed so far map the stage onto the camera, roughly: the groups still apart
    # are searched where that map puts them, until no more join
    main = _largest(group)[0]
    while True:
        placed = group == main
        if placed.all():
            break
        maps, trust = rough_maps(np.where(placed[:, None], offsets, np.nan), stage, backlash)
        if not maps:
            break
        say(f"{int(placed.sum())} frames placed together; looking for the "
            f"{len(np.unique(group[~placed]))} groups apart where the stage puts them"
            + (", the camera's other axis guessed" if trust < 0 else ""))
        if not join_groups(positions, offsets, group, main, maps, trust, stage, area,
                           tolerance_px, min_votes):
            break
    offsets = np.where((group == main)[:, None], offsets, np.nan)
    placed = np.isfinite(offsets[:, 0])
    if placed.sum() < 4:
        raise ValueError(
            f"only {int(placed.sum())} frames could be placed together by the beads they share: "
            f"the steps are too large for the field, or too few beads were found")
    if placed.sum() < n_frames:
        missing = np.flatnonzero(~placed)
        notes.append(f"{len(missing)} frames share no beads with the rest and are left out: "
                     + ", ".join(str(int(k)) for k in missing[:12])
                     + (" ..." if len(missing) > 12 else ""))
    say(f"{int(placed.sum())} frames placed; linking the beads")
    bead = link_beads(frame, xy, offsets, tolerance_px)
    offsets, bead_xy, bead = refine(frame, xy, bead, offsets)
    fit = fit_stage(offsets, stage, backlash)
    values = decompose(fit.matrix)
    values.update(jackknife(offsets, stage, fit, backlash))
    lag = backlash_nm(fit.matrix, fit.backlash)
    values["backlash_x_nm"], values["backlash_y_nm"] = float(lag[0]), float(lag[1])
    for k, axis in enumerate("xy"):
        if not fit.backlash[:, k].any():          # never reversed: not measured
            values[f"backlash_{axis}_nm"] = values[f"backlash_{axis}_nm_err"] = float("nan")
    mean_pixel = 0.5 * (values["pixel_x_nm"] + values["pixel_y_nm"])
    r = fit.residual_px[fit.used]
    values["residual_nm"] = float(np.sqrt(np.mean(np.sum(r ** 2, axis=1)))) * mean_pixel
    values["n_frames"] = float(fit.used.sum())
    values["n_beads"] = float(len(bead_xy))
    left = np.flatnonzero(np.isfinite(offsets[:, 0]) & ~fit.used)
    if len(left):
        notes.append(f"{len(left)} frames off the stage's line and left out: "
                     + ", ".join(str(int(k)) for k in left[:12]))
    return PixelSizeResult(values, fit, offsets, stage, bead, bead_xy, kept, notes)


def _largest(group: np.ndarray) -> Tuple[int, int]:
    """(the largest group, its number of frames)."""
    labels, counts = np.unique(group, return_counts=True)
    k = int(np.argmax(counts))
    return int(labels[k]), int(counts[k])


# ------------------------------------------------------------------ the stage's tags

#: tags Micro-Manager writes the XY stage's position under, per frame, in micrometres
STAGE_TAGS = (("XPositionUm", "YPositionUm"),)


def stage_columns(tags: Dict[str, np.ndarray], x_tag: str = "", y_tag: str = ""
                  ) -> Tuple[str, str]:
    """The names of the stage's x and y among a file's frame tags (`smappy.frametags`): the given
    ones, else Micro-Manager's ``XPositionUm`` / ``YPositionUm``, else a pair of tags that differ
    by an x and a y alone (``Stage-X`` and ``Stage-Y``).  Raises naming the tags there are."""
    names = [k for k in tags if k != "frame"]
    if x_tag or y_tag:
        missing = [t for t in (x_tag, y_tag) if t and t not in tags]
        if missing or not (x_tag and y_tag):
            raise ValueError(f"no frame tag {', '.join(missing) or 'for both axes'}; the tags "
                             f"that change are: {', '.join(names) or 'none'}")
        return x_tag, y_tag
    for x, y in STAGE_TAGS:
        if x in tags and y in tags:
            return x, y
    for x in names:
        for at in [k for k, c in enumerate(x) if c in "xX"]:
            y = x[:at] + ("Y" if x[at] == "X" else "y") + x[at + 1:]
            if y in tags and y != x:
                return x, y
    raise ValueError("the stage's position is not among the frames' tags; the tags that change "
                     f"are: {', '.join(names) or 'none'} -- name the stage's x and y tags")


def stage_positions(tags: Dict[str, np.ndarray], n_frames: int, x_tag: str = "",
                    y_tag: str = "", unit_um: float = 1.0) -> np.ndarray:
    """(n_frames, 2) stage positions in micrometres from the frame tags (``unit_um``: one of the
    tag's units in micrometres, 0.001 for nm); NaN for a frame no tag was read for."""
    x, y = stage_columns(tags, x_tag, y_tag)
    frames = np.asarray(tags["frame"], np.int64)
    out = np.full((n_frames, 2), np.nan)
    inside = (frames >= 0) & (frames < n_frames)
    out[frames[inside], 0] = np.asarray(tags[x], float)[inside] * unit_um
    out[frames[inside], 1] = np.asarray(tags[y], float)[inside] * unit_um
    return out
