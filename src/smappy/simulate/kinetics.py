"""Labels -> fluorophores -> blinks -> photons per frame.

The part both simulators share, and the part a user adjusts to match a dye.

**Labelling.**  A label is labelled with probability ``efficiency`` and then
carries exactly ``fluorophores`` (rounded) or a Poisson number with that mean;
each fluorophore sits off its label by a Gaussian linkage error.

**Blinking.**  One model for STORM, PALM and PAINT, in continuous time:
a fluorophore is dark for an exponential ``off_time``, on for an exponential
``on_time``, and after each blink bleaches with probability ``bleaching`` --
so it blinks a geometric number of times, ``1/bleaching`` on average if the
measurement is long enough.  PAINT bleaches too (the dye in the imager is
replenished, the docking strand is not), so it is the same model with a long
off-time.  Switching happens at any moment within a frame, and a frame gets
the photons of the time the fluorophore was on in it: the first and last
frames of a blink are dimmer, as in real data, and a blink of 1.5 frames on
average touches 2.5 frames.

The off-time is usually not something one knows; what one knows is how often
a molecule came back, so it is set from ``blinks``, the mean number of blinks
a fluorophore shows *within the measurement*.  With ``K`` activations of an
unbleached fluorophore in the measurement, Poisson with mean
``T / (off + on)``, and ``G`` blinks before bleaching, the number seen is
``min(G, K)``, whose mean is

    E = sum_k (1 - p)^(k - 1) P(K >= k)

and `off_time_for` solves that for the off-time.  More than ``1/p`` blinks
cannot be had, and asking for them is refused rather than silently capped.

**Activation.**  At a fixed rate most blinks come early, while the pool is
full, and fall away as it bleaches (``decay``).  In a dSTORM experiment one
raises the activation as the pool bleaches to keep the density constant, and
that is the default (``constant``): the same fluorophores, the same number of
blinks each, their start times mapped monotonically so that together they are
spread evenly over the measurement.  The mapping keeps each fluorophore's own
order, so its dark times are longer early and shorter late -- what a rising
activation does.  The third choice, ``all``, is SMAP's "Dye" model: every
fluorophore shows *all* its blinks, geometric in number with mean ``blinks``
(a bleaching probability of ``1 / blinks``), whatever the number of frames,
spread evenly the same way -- the number of blinks is then the dye's alone,
not the dye's and the measurement's.  (SMAP's version maps the times by the
inverse of the ranks rather than the ranks, which keeps them uniform but
scrambles each fluorophore's order; here the order is kept.)

**Linkage error** comes in two kinds, as in SMAP: a fixed offset per
fluorophore (`label`), shared by its blinks, and a free one drawn for every
blink (``linkage_free_nm``, `blink_offsets`), for a dye that turns on a
flexible linker.

**Photons.**  Each blink draws a total from a gamma distribution with the
given mean and standard deviation (a standard deviation of 0 is exactly the
mean), which it emits at the constant rate ``total / on_time`` while it is on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .settings import BlinkingSettings, LabellingSettings
from .structure import Labels


@dataclass
class Fluorophores:
    xyz: np.ndarray          # (n, 3) nm
    dye: np.ndarray
    copy: np.ndarray
    label: np.ndarray        # index into the structure's labels

    def __len__(self) -> int:
        return len(self.xyz)


@dataclass
class Blinks:
    """One row per blink, sorted by fluorophore then time; times in frames."""
    owner: np.ndarray        # which fluorophore
    start: np.ndarray
    end: np.ndarray          # cut at the end of the measurement
    rate: np.ndarray         # photons per frame while on
    off_time: float          # the off-time that was used (the mean, when spread)
    offset: np.ndarray = None  # (n, 3) nm, the free linkage error of each blink

    def __len__(self) -> int:
        return len(self.owner)


@dataclass
class Emission:
    """One row per fluorophore and frame it was on in, sorted by frame."""
    owner: np.ndarray
    frame: np.ndarray        # int64
    photons: np.ndarray      # expected photons in that frame
    blink: np.ndarray = None  # which blink the row is (the first, if two share it)

    def __len__(self) -> int:
        return len(self.owner)


def draw(mean: float, std: float, n: int, rng) -> np.ndarray:
    """``n`` values with this mean and standard deviation, never negative: a
    gamma distribution, or exactly the mean when ``std`` is 0."""
    if mean <= 0:
        return np.zeros(n)
    if std <= 0:
        return np.full(n, float(mean))
    shape = (mean / std) ** 2
    return rng.gamma(shape, mean / shape, n)


def label(labels: Labels, settings: LabellingSettings, rng) -> Fluorophores:
    n = len(labels)
    kept = rng.random(n) < settings.efficiency
    if settings.poisson:
        count = rng.poisson(settings.fluorophores, n)
    else:
        count = np.full(n, int(round(settings.fluorophores)))
    count = np.where(kept, count, 0)
    index = np.repeat(np.arange(n), count)
    xyz = labels.xyz[index]
    if settings.linkage_nm > 0:
        xyz = xyz + rng.normal(0, settings.linkage_nm, xyz.shape)
    return Fluorophores(xyz, labels.dye[index], labels.copy[index], index)


# ------------------------------------------------------------------ blinking
def expected_blinks(off_time: float, on_time: float, bleaching: float,
                    n_frames: float) -> float:
    """The mean number of blinks a fluorophore shows within ``n_frames``."""
    from scipy.stats import poisson
    mu = n_frames / (off_time + on_time)
    q = 1.0 - bleaching
    # enough terms for both the bleaching and the Poisson tail to vanish
    k_max = int(mu + 10 * np.sqrt(mu) + 20)
    if 0 < q < 1:
        k_max = min(k_max, int(np.log(1e-12) / np.log(q)) + 2)
    k = np.arange(1, k_max + 1)
    return float(np.sum(q ** (k - 1) * poisson.sf(k - 1, mu)))


def off_time_for(blinks: float, on_time: float, bleaching: float,
                 n_frames: float) -> float:
    """The off-time that gives ``blinks`` blinks per fluorophore on average."""
    ceiling = np.inf if bleaching <= 0 else 1.0 / bleaching
    if blinks >= ceiling * (1 - 1e-6):
        raise ValueError(
            f"{blinks:g} blinks cannot be had with a bleaching probability of "
            f"{bleaching:g}: a fluorophore blinks 1/p = {ceiling:.3g} times on "
            f"average before it bleaches, however short its off-time")
    lo, hi = np.log(1e-6 * n_frames), np.log(1e6 * n_frames)
    for _ in range(100):                     # blinks fall as the off-time grows
        mid = 0.5 * (lo + hi)
        if expected_blinks(np.exp(mid), on_time, bleaching, n_frames) > blinks:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def blink(n: int, n_frames: int, settings: BlinkingSettings, rng) -> Blinks:
    """Every blink of ``n`` fluorophores within ``n_frames``."""
    if settings.activation == "all":
        return _every_blink(n, n_frames, settings, rng)
    on, p = settings.on_time, settings.bleaching
    off = settings.off_time or off_time_for(settings.blinks, on, p, n_frames)
    # blinks before bleaching, capped at more than the measurement could hold
    cap = int(3 * n_frames / (off + on) + 10 * np.sqrt(n_frames / (off + on)) + 10)
    g = rng.geometric(p, n) if p > 0 else np.full(n, cap)
    g = np.minimum(g, cap)
    owner = np.repeat(np.arange(n), g)
    m = len(owner)
    first = np.zeros(m, bool)
    first[np.concatenate([[0], np.cumsum(g)[:-1]])[g > 0]] = True
    on_for = rng.exponential(on, m)
    previous_on = np.concatenate([[0.0], on_for[:-1]])
    previous_on[first] = 0.0
    step = rng.exponential(off, m) + previous_on
    # a cumulative sum restarted at each fluorophore's first blink
    total = np.cumsum(step)
    base = np.maximum.accumulate(np.where(first, total - step, 0.0))
    start = total - base
    keep = start < n_frames
    owner, start, on_for = owner[keep], start[keep], on_for[keep]
    if settings.activation == "constant" and len(start):
        start = _spread(owner, start, on_for, n_frames, rng)
        keep = start < n_frames
        owner, start, on_for = owner[keep], start[keep], on_for[keep]
    elif settings.activation not in ("constant", "decay"):
        raise ValueError(f"activation {settings.activation!r}: constant, decay or all")
    end = np.minimum(start + on_for, n_frames)
    rate = draw(settings.photons, settings.photons_std, len(owner), rng) / on
    return Blinks(owner, start, end, rate, off)


def _every_blink(n, n_frames, settings: BlinkingSettings, rng) -> Blinks:
    """All the blinks each fluorophore has before it bleaches, spread evenly
    over the measurement in each one's order (SMAP's "Dye")."""
    if settings.blinks < 1:
        raise ValueError(f"every blink until bleached: at least one blink each, "
                         f"not {settings.blinks:g}")
    on = settings.on_time
    g = rng.geometric(1.0 / settings.blinks, n)
    owner = np.repeat(np.arange(n), g)
    first = np.zeros(len(owner), bool)
    first[np.concatenate([[0], np.cumsum(g)[:-1]])[g > 0]] = True
    # any time scale will do: only the order survives the spreading
    step = rng.exponential(1.0, len(owner))
    total = np.cumsum(step)
    start = total - np.maximum.accumulate(np.where(first, total - step, 0.0))
    on_for = rng.exponential(on, len(owner))
    start = _spread(owner, start, on_for, n_frames, rng)
    keep = start < n_frames
    owner, start, on_for = owner[keep], start[keep], on_for[keep]
    end = np.minimum(start + on_for, n_frames)
    same = owner[1:] == owner[:-1]
    dark = (start[1:] - end[:-1])[same]
    off = float(dark.mean()) if dark.size else float("nan")
    rate = draw(settings.photons, settings.photons_std, len(owner), rng) / on
    return Blinks(owner, start, end, rate, off)


def blink_offsets(n: int, width_nm: float, rng) -> np.ndarray:
    """The free linkage error of ``n`` blinks, (n, 3) nm."""
    if width_nm <= 0:
        return np.zeros((n, 3))
    return rng.normal(0, width_nm, (n, 3))


def _spread(owner, start, on_for, n_frames, rng) -> np.ndarray:
    """The start times mapped, monotonically, onto as many uniform ones: the
    blinks fall evenly over the measurement and each fluorophore keeps its
    order.  A dark time squeezed shorter than the blink before it would put
    the fluorophore on twice at once, so such a blink waits for the last."""
    order = np.argsort(start, kind="stable")
    spread = np.empty_like(start)
    spread[order] = np.sort(rng.uniform(0, n_frames, len(start)))
    same = np.concatenate([[False], owner[1:] == owner[:-1]])
    for _ in range(1000):
        end_before = np.concatenate([[-np.inf], (spread + on_for)[:-1]])
        clash = same & (spread < end_before)
        if not clash.any():
            break
        spread[clash] = end_before[clash]
    return spread


def emission(blinks: Blinks, n_frames: int) -> Emission:
    """The blinks cut into frames: photons by the time on in each frame, and
    one row for a fluorophore that blinked twice within one frame."""
    f0 = np.floor(blinks.start).astype(np.int64)
    f1 = np.maximum(np.ceil(blinks.end).astype(np.int64) - 1, f0)
    count = f1 - f0 + 1
    idx = np.repeat(np.arange(len(f0)), count)
    within = np.arange(len(idx)) - np.repeat(np.cumsum(count) - count, count)
    frame = f0[idx] + within
    overlap = (np.minimum(blinks.end[idx], frame + 1)
               - np.maximum(blinks.start[idx], frame))
    ok = (overlap > 0) & (frame < n_frames)
    idx, frame, photons = idx[ok], frame[ok], blinks.rate[idx[ok]] * overlap[ok]
    owner = blinks.owner[idx]
    key = frame * (int(owner.max(initial=0)) + 1) + owner
    unique, inverse = np.unique(key, return_inverse=True)
    summed = np.bincount(inverse, weights=photons, minlength=len(unique))
    first = np.zeros(len(unique), np.int64)
    first[inverse[::-1]] = np.arange(len(inverse))[::-1]
    return Emission(owner[first], frame[first], summed, idx[first])
