"""Filter thresholds from the data itself, for MINFLUX's quality attributes.

A fixed cut on ``cfr`` or ``efo`` is right for one sample and buffer and
wrong for the next; these find it in the measurement, as pyMINFLUX does
(`Quality filter`, `plugins/quality_filter.py`):

* **cfr** (`robust_upper`): the centre-frequency ratio of single molecules
  scatters about a sample's typical value; two molecules on at once, or
  background, push it up.  The bound is the median plus ``k`` robust
  standard deviations (1.4826 x the median absolute deviation), k = 2 by
  default -- pyMINFLUX's automatic cfr threshold.
* **efo** (`efo_valley`): the photon rate of one molecule forms a peak; two
  molecules emitting at once form a second near twice its rate.  The bound
  is the lowest point of the smoothed histogram between the first peak
  and twice it -- the valley pyMINFLUX cuts at.  Where there is no second
  peak the histogram falls monotonically there, and the bound is where it
  has fallen to ``tail_fraction`` of the peak, not the end of the range.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

#: the median absolute deviation of a normal distribution, in sigmas
MAD = 1.4826


def robust_upper(values, k: float = 2.0) -> float:
    """The median plus ``k`` robust standard deviations of ``values``'
    finite entries; NaN with none."""
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan")
    med = float(np.median(v))
    return med + k * MAD * float(np.median(np.abs(v - med)))


@dataclass
class Valley:
    """The efo bound and how it was found: the single-molecule ``peak``,
    the ``bound``, whether a second peak made it a valley (``valley``),
    and the smoothed histogram (``centres``, ``counts``)."""
    peak: float
    bound: float
    valley: bool
    centres: np.ndarray
    counts: np.ndarray


def efo_valley(values, bins: int = 120, smooth_bins: float = 2.0,
               tail_fraction: float = 0.05) -> Valley:
    """The upper efo bound (see the module docstring), from ``values``'
    positive finite entries up to their 99.5th percentile."""
    from scipy.ndimage import gaussian_filter1d
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 20:
        return Valley(float("nan"), float("nan"), False, np.zeros(0), np.zeros(0))
    hi = float(np.quantile(v, 0.995))
    counts, edges = np.histogram(v[v <= hi], bins=bins, range=(0, hi))
    centres = 0.5 * (edges[1:] + edges[:-1])
    s = gaussian_filter1d(counts.astype(float), smooth_bins)
    j = int(np.argmax(s))
    peak = float(centres[j])
    window = (centres > peak) & (centres <= 2 * peak)
    if not window.any():
        return Valley(peak, hi, False, centres, s)
    idx = np.flatnonzero(window)
    k = idx[int(np.argmin(s[idx]))]
    after = s[k:min(len(s), 2 * j + 2)] if k + 1 < len(s) else s[k:k + 1]
    # a valley: the histogram rises again after the minimum, toward 2 x peak
    rises = len(after) > 2 and after.max() > s[k] * 1.2 + 1.0
    if rises:
        return Valley(peak, float(centres[k]), True, centres, s)
    below = np.flatnonzero((centres > peak) & (s <= tail_fraction * s[j]))
    bound = float(centres[below[0]]) if len(below) else hi
    return Valley(peak, bound, False, centres, s)
