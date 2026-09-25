"""Grouping: merging localizations of one emitter across consecutive frames.

A single emitter is localized in many consecutive frames, so a raw table counts
one blink many times.  Bright, long-lived emitters become hot spots in a
histogram render, and every statistic downstream sees a cluster of "independent"
points where there is one molecule.  Grouping collapses each such run into one
localization -- with a better precision, since it averages several measurements.

Two parts, following SMAP's ``Grouper.m`` and ``connectsingle2c.c``:

``connect``   greedy frame-to-frame linking (in C++, see ``csrc/group.hpp``).
``combine``   reduce each group to one row.

The combine rules are SMAP's, which are per column and not arbitrary:

===============================  ==========================================
positions, PSF size, colour      weighted mean, ``w = 1 / precision^2``
photons, background              sum
precisions and ``*_err``         ``1 / sqrt(sum(1 / e^2))``
log-likelihood                   max (the best frame's)
frame                            min (where the group starts)
===============================  ==========================================

Two departures, both because the blanket rule is wrong for the column:

* the error of a **summed** quantity adds in quadrature, ``sqrt(sum(e^2))``, not
  by the precision rule -- so ``photons_err`` and ``background_err`` use that.
  This is the rule that keeps shot noise consistent: with ``e_i = sqrt(N_i)`` it
  gives ``sqrt(sum(N_i))`` exactly, which is the shot noise of the summed
  photons.  SMAP applies its ``*_err`` rule to them instead, understating the
  error of a four-member group by a factor of six.
* each coordinate is weighted by **its own** error -- ``x_nm`` by ``x_err_nm``,
  ``y_nm`` by ``y_err_nm``, ``z_nm`` by ``z_err_nm`` -- where the table has
  them.  SMAP weights all three with the pooled lateral precision because that
  is the one weight it computes, and flags it as a shortcut; under astigmatism
  ``x_err`` and ``y_err`` diverge with z (that divergence is what encodes z), so
  the pooled weight is the right one only when they happen to be equal.
* the lateral precision ``xy_err_nm`` is not combined at all where the table
  has ``x_err_nm`` and ``y_err_nm``: it is derived from the combined pair, the
  RMS of the two, as it is everywhere else (`columns.add_xy_err`), so the
  grouped table cannot disagree with itself.  Combining it by the precision
  rule would give a slightly different number -- the rule is not linear in
  the square.
* ``logl_rel`` is a likelihood *per pixel* and stays comparable under max, but
  the raw ``logl`` of a group is the sum over its members' fits; taking the max
  of it is meaningless across groups of different size, so it is dropped rather
  than given a wrong value.  ``logl_rel`` -- which is what the filter uses --
  is kept.

Grouping is expensive (the linking is sequential and cannot be vectorised), so
it is done once and the grouped table kept alongside the original rather than
recomputed when the display switches between them.

Most of what grouping cost was not the linking.  On a 57 M localization file
`connect` took 71.7 s, and 68 s of that was the `np.lexsort` in front of the
walk; the walk itself is 5.5 s.  `sorted_order` below returns the same order --
element for element -- in 2.1 s.

Opening that file went 134 s -> 56 s, and its peak memory 20 GB -> 9.7: the
sort, the linking cut into chunks that run at once (`link_chunks`, and
`smappy._group_chunked` for what it costs in exactness), and the reduction
compiled (`csrc/combine.hpp`).  What is left of the 56 s: 32 s reading the
file, 9 s `combine`, 6 s the linking, 2 s the sort, 6 s the indices and
filters.

**To revisit: read fewer columns**, which is a memory problem before it is a
speed one.  30 columns are read from that file and 13 are read by anything
downstream; the four `bg*` ones are float64.  Opening it peaks at 20 GB, so it
does not open at all on the 8-16 GB a microscope PC usually has.  Keeping only
the columns that are used would take that to about 8 GB, the 31 s read to about
13, and `combine` -- one pass per column -- with it.  See NOTES.md: which
columns those are is per format and belongs in a setting, not in a constant.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .columns import xy_err
from .locs import Localizations
from .mathparse import (COMBINE_RULES, RECOMPUTE, apply_recipes,
                        derived_names, recipes)

# ``progress(text, fraction)``: what stage this is in, and how far through the
# table it is.  Both are advisory -- the fraction is coarse and the text is for
# a human to read, not to branch on.
Progress = Callable[[str, float], None]

try:
    from . import _group
except ImportError:  # pure-Python fallback if the extension is not built
    _group = None


# How each column is combined.  Taken from SMAP's `Grouper.m`; names translated
# to this package's, and the two corrections described in the module docstring.
COMBINE_MODES: Dict[str, str] = {
    "x_nm": "mean", "y_nm": "mean", "z_nm": "mean",
    "x_pix": "mean", "y_pix": "mean",
    "peak_x_nm": "mean", "peak_y_nm": "mean",
    "peak_x_pix": "mean", "peak_y_pix": "mean",
    "sigma_nm": "mean", "sigma_pix": "mean",
    "sigma_x_nm": "mean", "sigma_y_nm": "mean",
    "sigma_x_pix": "mean", "sigma_y_pix": "mean",
    "photons": "sum", "background": "sum",
    "photons_err": "quad", "background_err": "quad",
    "xy_err_nm": "precision", "xy_err_pix": "precision",
    "x_err_nm": "precision", "y_err_nm": "precision", "z_err_nm": "precision",
    "x_err_pix": "precision", "y_err_pix": "precision",
    "logl_rel": "max",
    "frame": "min",
    "iterations": "max",
}

# Columns that have no meaningful group value; see the module docstring.
DROP_ON_GROUPING = ("logl",)

# Columns grouping itself writes, on both tables.  `n_in_group` is the on-time
# in frames and `group_id` says which group a localization went into -- 1-based,
# as `connect` numbers them, and on the grouped table simply its own row.
# Keeping the id on the *ungrouped* table is what makes a column computed later
# combinable without linking again: the expensive part of grouping is the walk,
# and it has already been done.  It costs 4 bytes a localization.
GROUP_COLUMNS = ("group_id", "n_in_group")

# The precision columns the general weight is taken from, best first (SMAP's
# order).  It is a lateral precision, so it is the right weight for x and y.
WEIGHT_FIELDS = ("xy_err_nm", "xy_err_pix", "x_err_nm", "x_err_pix")

# Columns that carry their own uncertainty and are weighted by it rather than by
# the pooled lateral precision.  SMAP weights everything with `locprecnm`
# because that is the one weight it computes -- a comment in `Grouper.m` flags
# it as a shortcut -- but the fitter returns a separate error per coordinate and
# under astigmatism they genuinely differ: `x_err` and `y_err` diverge with z
# (that is what encodes z), and `z_err` varies by a factor of several over the
# range.  Weighting each coordinate by its own error is the correct inverse-
# variance estimate; the pooled one is only right when the errors are equal.
WEIGHT_FOR: Dict[str, Tuple[str, ...]] = {
    "x_nm": ("x_err_nm",), "y_nm": ("y_err_nm",), "z_nm": ("z_err_nm",),
    "x_pix": ("x_err_pix",), "y_pix": ("y_err_pix",),
}


def _mode(name: str) -> str:
    if name in COMBINE_MODES:
        return COMBINE_MODES[name]
    return "precision" if name.endswith(("_err", "err")) else "mean"


# One 16-bit digit per radix pass: numpy's stable sort is a real radix sort for
# uint8 and uint16 and a comparison sort for everything wider.  On 20 M values
# that is 0.18 s against 9.9 s for the same numbers as int64, which is the whole
# reason this module sorts the way it does.
DIGIT = 16
MASK = (1 << DIGIT) - 1



def _monotonic(x: np.ndarray) -> Tuple[np.ndarray, int]:
    """``x`` as an unsigned integer that sorts the way the float does.

    IEEE 754 floats are already ordered by their bit pattern within a sign, so
    flipping the sign bit for positives and every bit for negatives gives a
    plain unsigned key.  float32-valued data -- which is what smappy stores --
    needs half the digits, so it is worth the one pass to notice.
    """
    narrow = x.astype(np.float32)
    if np.array_equal(narrow.astype(np.float64), x):
        u = narrow.view(np.uint32).copy()
        sign = (u >> np.uint32(31)).astype(bool)
        u[sign] = ~u[sign]
        u[~sign] = u[~sign] | np.uint32(0x80000000)
        return u.astype(np.uint64), 32
    u = np.ascontiguousarray(x, np.float64).view(np.uint64).copy()
    sign = (u >> np.uint64(63)).astype(bool)
    u[sign] = ~u[sign]
    u[~sign] = u[~sign] | (np.uint64(1) << np.uint64(63))
    return u, 64


def _digits(u: np.ndarray, bits: int) -> List[np.ndarray]:
    """``u`` cut into 16-bit digits, least significant first."""
    return [((u >> np.uint64(s)) & np.uint64(MASK)).astype(np.uint16)
            for s in range(0, bits, DIGIT)]


def radix_argsort(values: np.ndarray) -> np.ndarray:
    """Stable argsort of non-negative integers, by 16-bit digits.

    The same trick `sorted_order` runs on: numpy's stable sort is a radix sort
    for uint16 and a comparison sort for int64.  Grouping 57 M localizations
    into 40 M groups, this is 1.5 s where ``np.argsort(kind="stable")`` is 4.4.
    Identical output -- both are stable.
    """
    u = np.asarray(values).astype(np.uint64)
    bits = max(DIGIT, int(int(u.max()).bit_length() if u.size else 1))
    bits = ((bits + DIGIT - 1) // DIGIT) * DIGIT
    order = None
    for d in _digits(u, bits):
        order = (np.argsort(d, kind="stable") if order is None
                 else order[np.argsort(d[order], kind="stable")])
    return order


def _codes(keys) -> Optional[Tuple[np.ndarray, int]]:
    """The block keys as one dense integer, or None if they are not integers.

    Block keys are file and channel numbers: a handful of small integers, so
    subtracting the minimum is all the packing they need and no sort is
    involved in working out the codes.
    """
    code = np.zeros(1, np.uint64)
    span = 1
    for k in keys:                      # keys[0] is the most significant
        a = np.asarray(k)
        if a.dtype.kind not in "iub":
            return None
        lo, hi = int(a.min()), int(a.max())
        width = hi - lo + 1
        if width <= 0 or span * width > MASK + 1:
            return None
        code = code * np.uint64(width) + (a.astype(np.int64) - lo).astype(np.uint64)
        span *= width
    return code, span


def _radix_order(x, frame, keys, workers) -> Optional[np.ndarray]:
    """The (keys, frame, x) order by radix, or None if this data does not suit.

    The sort key is cut where it can be: everything above the low 16 bits of
    the frame -- the block keys and the frame's high bits -- is a *partition*,
    computed straight from the values with no sort at all, and a partition is
    then sorted on its own by 16-bit digits.

    Nothing here assumes the input is in any order.  A partition is a *key*
    every localization carries, and the pass that gathers them is itself a
    stable counting sort, so the grouping is produced rather than found: an
    arbitrarily shuffled table gives the same permutation as a tidy one.  Partitions are independent, so
    they go across threads; within one, the digits are ``x`` low to high and
    then the frame's low half, least significant first, as a radix sort wants.
    """
    if np.isnan(x).any():               # the bit trick has no answer for NaN
        return None
    block = _codes(keys)
    if block is None:
        return None
    code, span = block
    fu = (frame - int(frame.min())).astype(np.uint64)
    chunks = int(fu.max() >> np.uint64(DIGIT)) + 1
    if span * chunks > MASK + 1:        # the partition must fit one radix pass
        return None
    part = (code * np.uint64(chunks) + (fu >> np.uint64(DIGIT))).astype(np.uint16)

    u, bits = _monotonic(np.asarray(x, np.float64))
    digits = _digits(u, bits) + [(fu & np.uint64(MASK)).astype(np.uint16)]

    base = np.argsort(part, kind="stable")           # one radix pass, O(n)
    edges = np.flatnonzero(part[base][1:] != part[base][:-1]) + 1
    edges = np.concatenate(([0], edges, [len(base)]))

    def one(k):
        piece = base[edges[k]:edges[k + 1]]
        for d in digits:
            piece = piece[np.argsort(d[piece], kind="stable")]
        return piece

    if len(edges) <= 2:
        return one(0)
    with ThreadPoolExecutor(max_workers=workers or min(16, (os.cpu_count() or 4))) as pool:
        return np.concatenate(list(pool.map(one, range(len(edges) - 1))))


def sorted_order(x, frame, keys=(), workers: Optional[int] = None) -> np.ndarray:
    """The (block, frame, x) order `connect_single` wants, without lexsort.

    This sort, not the linking, is what grouping a large table cost: on a 57 M
    localization file `np.lexsort` takes **66 s** and the C++ walk after it
    5.5 s.  lexsort is a stable argsort per key through an index, and at this
    size every pass is a cache miss per element.

    A radix sort is the alternative, and numpy has one hiding in it: its stable
    sort is radix for uint8 and uint16 and a comparison sort above that, 55x
    apart on the same numbers.  So the key is cut into 16-bit digits and passed
    least significant first -- x, then the frame -- which is the textbook
    least-significant-digit order.

    The part above the frame's low 16 bits does not need sorting at all: the
    block keys and the frame's high bits are a partition, worked out from the
    values, and each partition is then sorted by itself on a thread.  **2.1 s**
    on that file, and element for element what ``np.lexsort((x, frame) +
    keys)`` returns -- ties included, since every pass is stable.

    Data the digit trick has no answer for (NaN in x, block keys that are not
    small integers) falls back to lexsort rather than guessing.

    Three things that were tried and are not here.  Sorting x *globally* first
    and radix-passing the frame over it with numpy's own stable sorts is exact
    and 72 s -- those sorts are comparison sorts at these widths, which is the
    whole point of cutting the key into 16-bit digits instead.  Sorting the
    block keys and the frame first and then x inside each (block, frame) bucket
    with threaded lexsorts is exact and 5.1 s.  Discretizing x into a single
    16-bit digit saves 0.4 s and stops being exact -- 0.19% of localizations
    change places with a neighbour -- a bad trade when the control is the
    sequential walk.
    """
    x = np.asarray(x)
    frame = np.asarray(frame)
    if len(frame) < 2:
        return np.arange(len(frame), dtype=np.intp)
    order = _radix_order(x, frame, keys, workers)
    if order is None:
        return np.lexsort((x, frame) + tuple(reversed(keys)))
    return order


def connect(x, y, frame, dx: float = 50.0, dt: int = 1,
            blocks: Optional[np.ndarray] = None,
            progress: Optional[Progress] = None) -> np.ndarray:
    """Assign every localization a 1-based group id, in the input order.

    ``dx`` is the half-width of the search box in the units of x and y, ``dt``
    the number of dark frames a particle may skip.  ``blocks`` labels groups of
    localizations that linking may not cross (a file or channel number); linking
    is run once per block, rather than SMAP's trick of zeroing the frame at each
    boundary, which leaves the array no longer sorted by frame.

    Linking is lateral only.  A z window was offered and is gone: an emitter's
    fitted z wanders by more than the axial precision between frames, so a
    window tight enough to separate two emitters above each other also broke
    the blinks of one, and nobody could set it to anything that helped.
    """
    if _group is None:
        raise RuntimeError("the _group extension is not built; "
                           "run setup.py build_ext --build-lib src")
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    frame = np.asarray(np.rint(np.asarray(frame, np.float64)), np.int64)
    if not (x.shape == y.shape == frame.shape) or x.ndim != 1:
        raise ValueError("x, y and frame must be 1-D and the same length")

    keys: Tuple[np.ndarray, ...] = ()
    if blocks is not None:
        b = np.asarray(blocks)
        keys = tuple(b.T) if b.ndim == 2 else (b,)

    # the linker needs (frame, x) ascending, within a block
    order = sorted_order(x, frame, keys)
    out = np.zeros(x.size, np.int64)

    if keys:
        stacked = np.stack([np.asarray(k)[order] for k in keys], axis=1)
        edges = np.concatenate(
            ([0], np.flatnonzero(np.any(stacked[1:] != stacked[:-1], axis=1)) + 1,
             [x.size]))
    else:
        edges = np.array([0, x.size])

    offset = 0
    n_blocks = len(edges) - 1
    for i, (begin, end) in enumerate(zip(edges[:-1], edges[1:])):
        if progress is not None:
            label = "connect" if n_blocks == 1 else f"connect (block {i + 1}/{n_blocks})"
            progress(label, begin / max(x.size, 1))
        block = order[begin:end]
        ids, n_groups = _group.connect(x[block], y[block], frame[block], dx, dt)
        out[block] = ids + offset
        offset += n_groups
    return out


def _inverse_variance(values) -> np.ndarray:
    w = 1.0 / np.asarray(values, np.float64) ** 2
    return np.where(np.isfinite(w), w, 1.0)           # SMAP: infinite weight -> 1


def _weights(locs: Localizations) -> np.ndarray:
    """Per-localization weight for the means: ``1 / precision^2``, SMAP's order."""
    for name in WEIGHT_FIELDS:
        if name in locs:
            return _inverse_variance(locs[name])
    if "photons" in locs:
        return np.asarray(locs["photons"], np.float64)
    return np.ones(len(locs))


def combine(locs: Localizations, group_index: np.ndarray,
            fields: Optional[Sequence[str]] = None,
            progress: Optional[Progress] = None) -> Localizations:
    """Reduce each group to one row, one column at a time by its own rule.

    ``progress(text, fraction)`` is called per column, which is the natural
    unit here: every column is one pass of the same shape over the table.
    """
    gi = np.asarray(group_index, np.int64)
    if gi.size and gi.min() < 1:
        raise ValueError("group ids must be 1-based")
    n_groups = int(gi.max()) if gi.size else 0
    size = n_groups + 1

    default_w = _weights(locs)
    weights: Dict[str, np.ndarray] = {"": default_w}
    sums: Dict[str, np.ndarray] = {}
    n_in_group = np.bincount(gi, minlength=size)[1:]

    # A derived column does not follow the per-column rules: its recipe says
    # what it means on a grouped table -- recomputed from the expression, or
    # reduced by the rule the user chose.  Both are done below, after the
    # measured columns.
    derived = recipes(locs) if fields is None else []
    if fields is not None:
        names = list(fields)
    else:
        skip = set(DROP_ON_GROUPING) | set(GROUP_COLUMNS) | derived_names(locs)
        names = [n for n in locs.keys() if n not in skip]
    # derived from the combined x and y errors below, not combined itself
    lateral = [n for n in ("xy_err_nm", "xy_err_pix") if n in names
               and n.replace("xy_", "x_") in names
               and n.replace("xy_", "y_") in names]
    names = [n for n in names if n not in lateral]

    # Group order, built once and read by every column.  `connect` numbers
    # groups densely, so `starts` has one entry per group plus the end.
    order = radix_argsort(gi)
    starts = np.concatenate(
        ([0], np.flatnonzero(gi[order][1:] != gi[order][:-1]) + 1, [len(gi)]))
    # How each combine mode accumulates.  The C++ side sums (or takes the
    # extreme); what turns a sum into a mean, a precision or an error in
    # quadrature is the cheap per-group arithmetic below, which stays here
    # where the rules are written down.
    ACCUMULATE = {"sum": 0, "mean": 1, "precision": 2, "quad": 3, "min": 4, "max": 5}

    def accumulate(values, kind: int, w=None) -> np.ndarray:
        """One pass over the table for one column, in C++ if it is built.

        The numpy fallback is the reference the tests compare against, and is
        what runs if the extension is missing.  It needs the column as float64
        and the weighted product as an array -- two 457 MB temporaries on a
        57 M localization table -- which is most of why the other one exists.
        """
        if _group is not None and hasattr(_group, "combine"):
            return np.asarray(_group.combine(np.ascontiguousarray(values),
                                             w if kind == 1 else None,
                                             order, starts, kind))
        v = np.asarray(values, np.float64)
        if kind == 0:
            return np.bincount(gi, weights=v, minlength=size)[1:]
        if kind == 1:
            return np.bincount(gi, weights=v * w, minlength=size)[1:]
        if kind == 2:
            return np.bincount(gi, weights=1.0 / v ** 2, minlength=size)[1:]
        if kind == 3:
            return np.bincount(gi, weights=v ** 2, minlength=size)[1:]
        reduce = np.minimum if kind == 4 else np.maximum
        return reduce.reduceat(v[order], starts[:-1])


    def weight_column(name: str) -> str:
        """Which weight this column is combined with: its own, or the pooled."""
        for column in WEIGHT_FOR.get(name, ()):
            if column in locs:
                return column
        return ""

    # every weight the columns will ask for, and its per-group sum
    for column in {weight_column(n) for n in names}:
        if column:
            weights[column] = _inverse_variance(locs[column])
        sums[column] = accumulate(weights[column], 0)

    def weighting(name: str) -> Tuple[np.ndarray, np.ndarray]:
        column = weight_column(name)
        return weights[column], sums[column]

    def reduce_column(name: str) -> np.ndarray:
        values = locs[name]
        mode = _mode(name)
        if mode == "mean":
            w, sum_w = weighting(name)
            return accumulate(values, 1, w) / sum_w
        if mode == "precision":
            return 1.0 / np.sqrt(accumulate(values, 2))
        if mode == "quad":     # the error of a sum, not of a mean
            return np.sqrt(accumulate(values, 3))
        if mode in ACCUMULATE:
            return accumulate(values, ACCUMULATE[mode])
        raise ValueError(f"unknown combine mode {mode!r} for {name!r}")

    # One column at a time, on purpose.  The columns are independent and numpy
    # drops the GIL inside bincount, so threading them looks free and is not:
    # a column in flight holds a float64 output over the groups and a float64
    # temporary over the table -- 0.78 GB on a 57 M localization file, where
    # the table itself is already 8 GB.  Measured there, threads against
    # seconds: 1: 12.5, 2: 10.5, 3: 12.9, 4: 14.8, 6: 23.1, 8: 31.8.  Two
    # threads win 2 s and every further one loses more, and that is on a
    # machine with 34 GB; on the 8-16 GB a microscope PC usually has, the
    # second thread is what pushes it into swap.  The way to make this faster
    # is to give it fewer columns, not more threads -- see the note at the top
    # of the module.
    columns: Dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        if progress is not None:
            progress(f"combine ({i + 1}/{len(names)})", i / len(names))
        columns[name] = reduce_column(name).astype(np.float32)

    for name in ("frame", "iterations"):
        if name in columns:
            columns[name] = columns[name].astype(np.int64 if name == "frame"
                                                 else np.int32)
    for name in lateral:
        columns[name] = xy_err(columns[name.replace("xy_", "x_")],
                               columns[name.replace("xy_", "y_")])
    columns["n_in_group"] = n_in_group.astype(np.int32)
    # the same ids as the ungrouped table's `group_id`, so a row can be found
    # from a localization and the other way round
    columns["group_id"] = np.arange(1, size, dtype=np.int32)

    # the derived fields that are *reduced*: one more pass of the same shape,
    # using the accumulators above rather than a second implementation
    for recipe in derived:
        rule = recipe["grouped"]
        if rule not in COMBINE_RULES or recipe["field"] not in locs:
            continue                       # recomputed, or left off, below
        values = locs[recipe["field"]]
        if rule == "mean":
            reduced = accumulate(values, ACCUMULATE["sum"]) / n_in_group
        elif rule in ("any", "all"):
            extreme = accumulate(values, ACCUMULATE["max" if rule == "any" else "min"])
            reduced = (extreme != 0).astype(np.float32)
        else:
            reduced = accumulate(values, ACCUMULATE[rule])
        columns[recipe["field"]] = np.asarray(reduced, np.float32)

    metadata = dict(locs.metadata)
    metadata["grouped"] = True
    grouped = Localizations(columns, metadata)
    # recomputed here: the ones whose recipe says so, and the ones there was
    # nothing to reduce -- a field defined in `n_in_group` has no ungrouped
    # column until the first grouping, which is this one
    apply_recipes(grouped, only=[r["field"] for r in derived
                                 if r["grouped"] == RECOMPUTE
                                 or r["field"] not in locs])
    return grouped


@dataclass(frozen=True)
class GroupSettings:
    """How to link.  ``dx`` is in the units of the table (nm, normally)."""

    dx: float = 50.0
    dt: int = 1
    block_fields: Sequence[str] = ("filenumber", "channel")
    # How many pieces the frame axis is cut into for the linking, which runs
    # one thread per piece.  Not a free choice: a cut trace is repaired at the
    # seam but the repair is not exact (see `smappy._group_chunked`), so the
    # grouping depends on this number.  It is a setting rather than a thread
    # count for that reason -- the same file groups the same way on any
    # machine, whatever it has to run on.  1 is the sequential walk.
    link_chunks: int = 8


def attach(locs: Localizations, grouped: Localizations,
           group_index: np.ndarray) -> None:
    """Write the group id and the on-time back onto the ungrouped table.

    Grouping is the only thing that knows which localizations belong together,
    and that answer is as much a property of a localization as its photon
    count: SMAP's ``numberInGroup`` is a field of the ungrouped table and its
    plugins filter on it.  Writing it back here is also what lets a field
    computed afterwards be combined without linking again -- `group_id` is the
    index a `np.bincount` needs -- and what makes an expression in
    ``n_in_group`` mean something on both tables.

    Columns are added in place: nothing downstream is derived from *which*
    columns a table has (the filter keeps the bounds it was given, the index is
    built from x and y), so a new column costs nothing to add and a new table
    would cost every layer its state.
    """
    gi = np.asarray(group_index, np.int64)
    locs.columns["group_id"] = gi.astype(np.int32)
    locs.columns["n_in_group"] = np.asarray(grouped["n_in_group"])[gi - 1]
    # a recipe in `n_in_group` can only now be computed -- see `mathparse`
    apply_recipes(locs)


def group(locs: Localizations, settings: Optional[GroupSettings] = None,
          progress: Optional[Progress] = None, attach_columns: bool = True
          ) -> Tuple[Localizations, np.ndarray]:
    """Group a table.  Returns the grouped table and the per-input group id.

    The ids are **1-based**, as `connect` produces them, so the row of the
    grouped table a localization ended up in is ``group_index - 1``.

    ``attach_columns`` writes `group_id` and `n_in_group` onto ``locs`` itself;
    see `attach` for why that is the default rather than a courtesy.

    ``progress(text, fraction)`` reports the two stages; it is what the GUI
    puts in its status bar while this runs off the main thread.
    """
    settings = settings or GroupSettings()
    from .render import positions

    x, y = positions(locs)
    if "frame" not in locs:
        raise KeyError("grouping needs a 'frame' column")

    present = [locs[name] for name in settings.block_fields if name in locs]
    blocks = np.stack(present, axis=1) if present else None
    if settings.link_chunks > 1:
        from ._group_chunked import connect_chunked
        group_index = connect_chunked(x, y, locs["frame"], settings.dx, settings.dt,
                                      blocks, n_chunks=settings.link_chunks,
                                      progress=progress)
    else:
        group_index = connect(x, y, locs["frame"], settings.dx, settings.dt, blocks,
                              progress=progress)
    grouped = combine(locs, group_index, progress=progress)
    if attach_columns:
        attach(locs, grouped, group_index)
    return grouped, group_index
