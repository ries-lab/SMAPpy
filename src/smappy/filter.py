"""Selecting which localizations to display.

Filtering is a range per field -- localization precision below 30 nm, relative
log-likelihood above -2, z within a slab -- and the displayed set is the
intersection of all of them.  Recomputing that from scratch on every render is
what makes interactive filtering slow, so, as in SMAP, **one boolean array is
cached per field** and only the field that changed is recomputed; the
intersection is then an AND over a handful of arrays, which is memory-bandwidth
work and takes a few milliseconds even for millions of localizations.

The result is exposed two ways.  ``mask`` is the boolean array, for combining
with other selections; ``indices`` is the compacted version, which is what the
renderer wants -- it indexes three or four columns instead of copying the whole
table.

NaN is excluded by any range, because a comparison against NaN is false.  That
is deliberate: a localization with no z should not appear in a z slab.

**Spans.**  A filter can be restricted to row spans (`restrict`): every
range is then tested on those rows only and the rest are False, without
reading them.  It is what a layer showing one file of several does
(`layer_state.ViewState`): a file is one contiguous block of rows, and on a
table of 10 files of 10 M a slider costs what it costs on the file alone --
0.02 s instead of 0.28 s (`docs/performance.md`, "Large tables").  The masks
stay as long as the table, so everything that combines them with other
selections is unchanged; the rows outside the spans are zeros that are never
written (``np.zeros`` takes them from the system untouched).

Not ported from SMAP: ROI/polygon masks and the grouped-localization filters.
``set_mask`` takes an arbitrary boolean array under a name, which is where such
a selection -- or the viewer's "inside the field of view" -- would plug in.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .locs import Localizations


class LocFilter:
    """Cached range filters over the fields of a :class:`Localizations` table."""

    def __init__(self, locs: Localizations,
                 spans: Optional[Sequence[Tuple[int, int]]] = None,
                 **ranges: Tuple[float, float]):
        self._locs = locs
        self._n = len(locs)
        self._ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        self._masks: Dict[str, np.ndarray] = {}
        self._combined: Optional[np.ndarray] = None
        self._indices: Optional[np.ndarray] = None
        self._spans: Optional[List[Tuple[int, int]]] = self._clean(spans)
        for field, (lo, hi) in ranges.items():
            self.set(field, lo, hi)

    # ------------------------------------------------------------- setting up
    def set(self, field: str, lo: Optional[float] = None,
            hi: Optional[float] = None) -> "LocFilter":
        """Keep localizations with ``lo <= field <= hi``; ``None`` is unbounded.

        Only this field's array is recomputed.
        """
        self._ranges[field] = (lo, hi)
        self._masks[field] = self._range_mask(field, lo, hi)
        return self._invalidate()

    def _range_mask(self, field: str, lo: Optional[float], hi: Optional[float],
                    start: int = 0) -> np.ndarray:
        """The mask for one range, over the rows from ``start`` on -- inside
        the spans only, when the filter is restricted to some."""
        column = self._locs[field]          # raises if the column does not exist
        if self._spans is not None:
            mask = np.zeros(len(column) - start, dtype=bool)
            for a, b in self._spans:
                a, b = max(a, start), b
                if a < b:
                    mask[a - start:b - start] = _in_range(column[a:b], lo, hi)
            return mask
        return _in_range(column[start:], lo, hi)

    @staticmethod
    def _clean(spans) -> Optional[List[Tuple[int, int]]]:
        if spans is None:
            return None
        return [(int(a), int(b)) for a, b in sorted(spans) if int(b) > int(a)]

    def restrict(self, spans: Optional[Sequence[Tuple[int, int]]]) -> "LocFilter":
        """Keep only rows inside these ``(start, stop)`` spans, and test the
        ranges on those rows alone; None lifts it.  Every range is tested
        again; a hand-set mask (`set_mask`) is kept as it is."""
        spans = self._clean(spans)
        if spans == self._spans:
            return self
        self._spans = spans
        for name, (lo, hi) in self._ranges.items():
            self._masks[name] = self._range_mask(name, lo, hi)
        return self._invalidate()

    @property
    def spans(self) -> Optional[List[Tuple[int, int]]]:
        """The row spans the filter is restricted to; None for every row."""
        return None if self._spans is None else list(self._spans)

    def append(self, n_new: int) -> "LocFilter":
        """Take in ``n_new`` rows appended to the table since the last call.

        Only the new rows are tested, against the bounds already set: appending
        never re-derives a limit, so what the user typed keeps meaning exactly
        what they typed while data arrives.  A mask set by hand (`set_mask`)
        has no rule to extend it, so its new rows are excluded until it is set
        again -- visible and recoverable, unlike silently including them.
        """
        start, self._n = self._n, self._n + int(n_new)
        if len(self._locs) != self._n:
            raise ValueError(f"the table has {len(self._locs)} rows, "
                             f"expected {self._n} after appending {n_new}")
        for name, mask in self._masks.items():
            # rows past the spans are outside them, as rows a hand-set mask
            # has no rule for are
            tail = (self._range_mask(name, *self._ranges[name], start=start)
                    if name in self._ranges and self._spans is None
                    else np.zeros(int(n_new), dtype=bool))
            self._masks[name] = np.concatenate([mask, tail])
        return self._invalidate()

    def set_mask(self, name: str, mask: np.ndarray) -> "LocFilter":
        """Add an arbitrary selection under a name (a ROI, a manual pick)."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self._n,):
            raise ValueError(f"a mask must have {self._n} entries, got {mask.shape}")
        self._ranges.pop(name, None)
        self._masks[name] = mask
        return self._invalidate()

    def remove(self, name: str) -> "LocFilter":
        self._masks.pop(name, None)
        self._ranges.pop(name, None)
        return self._invalidate()

    def clear(self) -> "LocFilter":
        self._masks.clear()
        self._ranges.clear()
        return self._invalidate()

    def _invalidate(self) -> "LocFilter":
        self._combined = None
        self._indices = None
        return self

    # --------------------------------------------------------------- the result
    @property
    def mask(self) -> np.ndarray:
        """Boolean array over the table: the intersection of every filter."""
        if self._combined is None:
            self._combined = self._and(list(self._masks.values()))
            if self._spans is not None:
                # compacted per span now, while the spans are known: the
                # renderer would otherwise find them in a pass over every row
                self._indices = self._span_indices(self._combined)
        return self._combined

    def mask_without(self, *names: str) -> np.ndarray:
        """The mask with the filters on ``names`` left out.

        For a plugin that must see what a filter hides: a layer set to one
        colour still has to hand both colours to the colour assignment.
        """
        return self._and([m for name, m in self._masks.items() if name not in names])

    def _and(self, masks: List[np.ndarray]) -> np.ndarray:
        """The AND of ``masks`` over every row, or inside the spans only."""
        if self._spans is None:
            if not masks:
                return np.ones(self._n, dtype=bool)
            return np.logical_and.reduce(masks)
        out = np.zeros(self._n, dtype=bool)
        for a, b in self._spans:
            out[a:b] = (np.logical_and.reduce([m[a:b] for m in masks]) if masks
                        else True)
        return out

    def _span_indices(self, mask: np.ndarray) -> np.ndarray:
        parts = [a + np.flatnonzero(mask[a:b]) for a, b in self._spans]
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)

    @property
    def indices(self) -> np.ndarray:
        """The positions of the kept localizations; what the renderer indexes with."""
        if self._indices is None:
            mask = self.mask
            if self._indices is None:
                self._indices = np.flatnonzero(mask)
        return self._indices

    @property
    def cached_indices(self) -> Optional[np.ndarray]:
        """`indices` if they have been computed for the current mask, else None."""
        return self._indices

    def keep_indices(self, mask: np.ndarray, indices: np.ndarray) -> None:
        """Take ``indices`` computed elsewhere -- by a render thread, from a
        snapshot's mask -- if ``mask`` is still this filter's.

        Compacting five million rows is 10-20 ms; a render thread that had to
        do it hands the answer back, so the next render does not.  Both arrays
        are only ever replaced, never written, so identity says they agree.
        """
        if self._indices is None and mask is self._combined:
            self._indices = indices

    @property
    def ranges(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        return dict(self._ranges)

    def __len__(self) -> int:
        if self._spans is not None:
            return int(len(self.indices))
        return int(self.mask.sum())

    @property
    def n_rows(self) -> int:
        """The rows of the table this filter covers -- kept or not."""
        return self._n

    def __iter__(self) -> Iterator[str]:
        return iter(self._masks)

    def __contains__(self, name: str) -> bool:
        return name in self._masks

    def apply(self) -> Localizations:
        """The filtered table.  Copies every column; prefer ``indices``."""
        return self._locs[self.mask]

    def counts(self) -> Dict[str, int]:
        """How many localizations each filter keeps on its own.

        What a GUI shows next to each slider, and the quickest way to see which
        filter is doing the cutting.
        """
        if self._spans is not None:
            return {name: sum(int(mask[a:b].sum()) for a, b in self._spans)
                    for name, mask in self._masks.items()}
        return {name: int(mask.sum()) for name, mask in self._masks.items()}

    def __str__(self) -> str:
        if not self._masks:
            return f"no filter, {self._n} localizations"
        parts = []
        for name in self._masks:
            lo, hi = self._ranges.get(name, (None, None))
            if name in self._ranges:
                parts.append(f"{name} "
                             f"[{'' if lo is None else f'{lo:g}'}"
                             f"..{'' if hi is None else f'{hi:g}'}]")
            else:
                parts.append(f"{name} (mask)")
        return (f"{len(self)} of {self._n} localizations: " + ", ".join(parts))


def _in_range(values: np.ndarray, lo: Optional[float], hi: Optional[float]) -> np.ndarray:
    mask = np.ones(values.shape[0], dtype=bool)
    if lo is not None:
        mask &= values >= lo
    if hi is not None:
        mask &= values <= hi
    return mask


def histogram_range(values: np.ndarray) -> Tuple[float, float]:
    """The span a field's histogram is drawn over: the bulk, not the tail.

    The 1st to 99th percentile, cut further to three interquartile ranges
    beyond the quartiles (Tukey's far-out fences).  The percentiles alone are
    not enough for a column with a long tail: a simulation keeps spots down to
    10 photons, whose precision runs to hundreds of nanometres, so the 99th
    percentile of ``xy_err_nm`` is 120 nm over a median of 5 and the histogram
    is one bar at the left.  The fences put its edge at 24 nm, and leave a
    symmetric or exponential column (positions, frames, photons) at its
    percentiles.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    p1, q1, q3, p99 = np.quantile(values, (0.01, 0.25, 0.75, 0.99))
    iqr = q3 - q1
    lo, hi = float(max(p1, q1 - 3 * iqr)), float(min(p99, q3 + 3 * iqr))
    if hi <= lo:                            # a column of (nearly) one value
        lo, hi = float(p1), float(p99)
    if hi <= lo:
        hi = lo + 1.0
    return (lo, hi)


def quantile_range(locs: Localizations, field: str, lower: float = 0.001,
                   upper: float = 0.999) -> Tuple[float, float]:
    """A sensible starting range for a field, ignoring its tails.

    Used to initialise filter limits and colour ranges without a stray outlier
    setting the scale.
    """
    values = np.asarray(locs[field], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 0.0)
    return (float(np.quantile(values, lower)), float(np.quantile(values, upper)))
