"""The channel move: how far a split camera's second channel lies from where the bead registration
puts it, on the chip, measured from the movie (docs/decode/gui-plan.md §4b; Jonas, 2026-11-23).

A splitter moves between the bead calibration and a measurement, by whole pixels and more (DECODE-Plex's
Fig. 2a: 5.8 rows), and may drift during one.  The move is a translation on the chip -- the second
channel's image moved by (columns, rows) -- added to where the beads' registration puts each partner.
It is measured, never assumed, with the model that made the registration: a peak's maximum is not the
spline's centre, the difference depends on z and differs between two planes or two dyes, so peak
positions measure the move plus a smeared bias.  Hence two steps:

* **A vote** (`vote`), coarse, to a pixel: each channel's peaks, channel 0's mapped into channel 1 by the
  registration, every difference within ``reach`` pixels in one histogram; partners pile up at the
  move, chance pairs spread flat.  Its only job is to put each partner in the right box.
* **The fit** (`measure`): the two-channel fitter with x and y free per channel (*link x, y* off), the
  vote's move applied, on blocks spread over the whole movie; each well-fitted pair says where its
  second channel lies against the link (``dx_nm_ch1``, ``dy_nm_ch1``).  The move is their
  error-weighted robust mean (`constant`), or on request a robust weighted smoothing spline over the
  frames (`over_time`, `beads.fit_offsets_and_spline` as *Fiducials* fits beads).

`ChannelMove` holds the result in chip pixels and hands it to each fitter in its own terms: the channel
layer's drift for Spline 3D 2C and Gaussian 2D 2C (`layer_drift`, `multichannel.MultiChannelEngine`), a
move per block for the DECODE fitter (`at`).  It is kept beside the movie (`cache_path`), keyed by what
it was measured with, so a preview's measurement serves the run after it and every run after that.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

VOTE_REACH_PX = 10.0          # differences the vote looks at, either way
VOTE_BIN_PX = 0.5             # its histogram's bins
VOTE_MIN_STRENGTH = 5.0       # a peak this many times the histogram's typical bin: a move; else none
ROBUST_SIGMAS = 3.0           # pairs this many robust sds off the move are left out, iterated
ROBUST_ROUNDS = 5
MIN_PAIRS = 20                # fewer well-fitted pairs: no measurement
CURVE_SAMPLES = 200           # a move over time is kept as its curve at this many frames


@dataclass
class ChannelMove:
    """The second channel's move on the chip, (columns, rows) pixels: constant (one row of ``offsets``) or
    over ``frames`` (interpolated between, held beyond).  ``record`` says how it was found."""
    offsets: np.ndarray                               # (m, 2) columns, rows
    frames: Optional[np.ndarray] = None               # (m,) for a move over time
    record: dict = field(default_factory=dict)

    @classmethod
    def typed(cls, columns: float = 0.0, rows: float = 0.0) -> "ChannelMove":
        return cls(np.array([[float(columns), float(rows)]]), None, {"how": "typed"})

    @property
    def over_time(self) -> bool:
        return self.frames is not None and len(self.frames) > 1

    @property
    def mean(self) -> np.ndarray:
        return np.asarray(self.offsets, float).mean(0)

    def at(self, frame) -> np.ndarray:
        """(n, 2) columns, rows at each of ``frame``."""
        frame = np.atleast_1d(np.asarray(frame, float))
        if not self.over_time:
            return np.repeat(np.asarray(self.offsets, float)[:1], len(frame), 0)
        return np.c_[np.interp(frame, self.frames, self.offsets[:, 0]),
                     np.interp(frame, self.frames, self.offsets[:, 1])]

    def is_zero(self) -> bool:
        return not np.any(np.abs(np.asarray(self.offsets, float)) > 1e-9)

    def layer_drift(self, layout):
        """The move as the channel layer's drift (`multichannel.ChannelDrift`): channel 1's own pixels,
        which a mirror in the cut flips along its axis."""
        from .multichannel import ChannelDrift
        flips = (layout.flips or [(False, False)] * layout.n_channels)[1]
        sign = np.array([-1.0 if flips[0] else 1.0, -1.0 if flips[1] else 1.0])
        frames = np.asarray(self.frames, float) if self.over_time else np.array([0.0])
        offsets = np.zeros((len(frames), layout.n_channels, 2))
        offsets[:, 1] = np.asarray(self.offsets, float)[:len(frames)] * sign
        return ChannelDrift(frames, offsets)

    def line(self) -> str:
        """The move in a line, for the form and the run's text."""
        c, r = self.mean
        how = self.record.get("how", "")
        text = f"channel 2 moved {r:+.2f} rows, {c:+.2f} columns"
        if self.over_time:
            span = np.ptp(np.asarray(self.offsets, float), 0)
            text += f", over time by up to {max(span):.2f} px"
        err = self.record.get("error_px")
        if err is not None:
            text += f" (+-{err:.2f})"
        if self.record.get("pairs"):
            text += f", from {self.record['pairs']} pairs"
        return text + (f"; {how}" if how and how != "typed" else "")

    def to_dict(self) -> dict:
        return {"offsets_px": np.asarray(self.offsets, float).tolist(),
                "frames": None if self.frames is None else np.asarray(self.frames, float).tolist(),
                "record": self.record}

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelMove":
        frames = d.get("frames")
        return cls(np.asarray(d["offsets_px"], float), None if frames is None else np.asarray(frames, float),
                   dict(d.get("record") or {}))


# ------------------------------------------------------------------------------------- the vote
def vote(found0, found1, layout, reach: float = VOTE_REACH_PX, bin_px: float = VOTE_BIN_PX):
    """The coarse move, ((columns, rows) on the chip, strength), from each channel's peaks
    (`detect.Candidates`, each in its own channel's pixels): channel 0's mapped into channel 1, every
    difference within ``reach`` in one histogram, its peak refined by the centroid of the bins about
    it.  ``strength``: the peak over the histogram's typical bin; below `VOTE_MIN_STRENGTH` it found
    nothing (a move of 0 is returned)."""
    diffs = []
    for frame in np.intersect1d(np.unique(found0.frame), np.unique(found1.frame)):
        a = found0[found0.frame == frame]
        b = found1[found1.frame == frame]
        mapped = layout.to_channel(1, np.c_[a.x, a.y].astype(float))
        d = np.c_[b.x, b.y].astype(float)[None, :, :] - mapped[:, None, :]
        d = d.reshape(-1, 2)
        diffs.append(d[np.all(np.abs(d) <= reach, 1)])
    d = np.concatenate(diffs) if diffs else np.zeros((0, 2))
    if len(d) < MIN_PAIRS:
        return np.zeros(2), 0.0
    edges = np.arange(-reach, reach + bin_px / 2, bin_px)
    h, _, _ = np.histogram2d(d[:, 0], d[:, 1], bins=(edges, edges))
    i, j = np.unravel_index(np.argmax(h), h.shape)
    typical = max(float(np.median(h)), float(np.mean(h)), 1e-9)
    strength = float(h[i, j] / typical)
    if strength < VOTE_MIN_STRENGTH:
        return np.zeros(2), strength
    centres = (edges[:-1] + edges[1:]) / 2
    sl = (slice(max(i - 1, 0), i + 2), slice(max(j - 1, 0), j + 2))
    w = h[sl]
    cx, cy = np.meshgrid(centres[sl[0]], centres[sl[1]], indexing="ij")
    layer = np.array([float((w * cx).sum() / w.sum()), float((w * cy).sum() / w.sum())])
    return to_chip(layer, layout), strength


def to_chip(layer_shift, layout) -> np.ndarray:
    """A shift in channel 1's own pixels as (columns, rows) on the chip: the cut's mirror undone."""
    flips = (layout.flips or [(False, False)] * layout.n_channels)[1]
    s = np.asarray(layer_shift, float).copy()
    if flips[0]:
        s[..., 0] *= -1
    if flips[1]:
        s[..., 1] *= -1
    return s


# ------------------------------------------------------------------------------- from the fit
def pair_shifts(locs, layout, pixel_nm, base: Optional[ChannelMove] = None):
    """Each unlinked pair's second channel against the link, as (columns, rows) on the chip, with
    ``(frame, shift (n, 2), error px (n,))``: ``dx_nm_ch1`` / ``dy_nm_ch1`` are in channel 0's pixels
    (`dualfit.paired_to_localizations`), turned into channel 1's by the registration's local map and onto
    the chip; plus the move the fit was run with (``base``), which the link already held."""
    px, py = (pixel_nm, pixel_nm) if np.isscalar(pixel_nm) else pixel_nm
    d_ref = np.c_[np.asarray(locs["dx_nm_ch1"], float) / px, np.asarray(locs["dy_nm_ch1"], float) / py]
    frame = np.asarray(locs["frame"], float)
    err = np.asarray(locs["xy_err_nm"] if "xy_err_nm" in locs else locs["x_err_nm"], float) / px
    ok = np.all(np.isfinite(d_ref), 1) & np.isfinite(err) & (err > 0)
    # the registration's local map at the channel's centre: an affine map's own, a warp's near enough
    w, h = layout.sizes[0]
    c = np.array([[w / 2, h / 2], [w / 2 + 1, h / 2], [w / 2, h / 2 + 1]], float)
    m = layout.to_channel(1, c)
    jac = np.c_[m[1] - m[0], m[2] - m[0]]                          # columns: d/dx, d/dy
    layer = d_ref[ok] @ jac.T
    shift = to_chip(layer, layout)
    if base is not None:
        shift = shift + base.at(frame[ok])
    return frame[ok], shift, err[ok]


def constant(shift, err, sigmas: float = ROBUST_SIGMAS, rounds: int = ROBUST_ROUNDS):
    """The error-weighted mean of the pairs' shifts, those beyond ``sigmas`` robust sds left out and
    the mean taken again, ``rounds`` times: ``(move (2,), its standard error (2,), pairs kept)``."""
    s = np.asarray(shift, float)
    w = 1.0 / np.asarray(err, float) ** 2
    keep = np.ones(len(s), bool)
    mean = np.median(s, 0)
    for _ in range(rounds):
        r = s - mean
        sd = 1.4826 * np.median(np.abs(r[keep]), 0)
        keep = np.all(np.abs(r) <= sigmas * np.maximum(sd, 1e-6), 1)
        if keep.sum() < MIN_PAIRS:
            break
        mean = (w[keep, None] * s[keep]).sum(0) / w[keep].sum()
    spread = s[keep].std(0) if keep.sum() > 1 else np.full(2, np.nan)
    return mean, spread / np.sqrt(max(int(keep.sum()), 1)), int(keep.sum())


def over_time(frame, shift, err, first: float, last: float, knots: Optional[int] = None,
              sigmas: float = ROBUST_SIGMAS, rounds: int = ROBUST_ROUNDS):
    """A robust, error-weighted smoothing spline of the shifts over the frames: ``(frames, curve (m,
    2), pairs kept)``, the curve sampled at `CURVE_SAMPLES` frames.  One "bead" of `beads.fit_offsets_and_spline`
    -- its offset the move, its curve the change -- with the pairs beyond ``sigmas`` robust sds of the
    curve left out and the fit made again."""
    from .beads import fit_offsets_and_spline
    t = np.asarray(frame, float)
    s = np.asarray(shift, float)
    w = 1.0 / np.asarray(err, float) ** 2
    n_coeffs = knots or max(4, min(12, int(len(np.unique(t)) // 50) + 4))
    keep = np.ones(len(t), bool)
    curve = None
    for _ in range(rounds):
        offsets, curve = fit_offsets_and_spline(t[keep], s[keep], np.zeros(int(keep.sum()), int), 1,
                                                first, last, n_coeffs, weights=w[keep])
        fitted = offsets[0] + curve(t)
        r = s - fitted
        sd = 1.4826 * np.median(np.abs(r[keep]), 0)
        new = np.all(np.abs(r) <= sigmas * np.maximum(sd, 1e-6), 1)
        if new.sum() < MIN_PAIRS or np.array_equal(new, keep):
            break
        keep = new
    grid = np.linspace(first, last, CURVE_SAMPLES)
    return grid, offsets[0] + curve(grid), int(keep.sum())


# ------------------------------------------------------------------------------------- the cache
def cache_key(movie, calibration, settings: dict) -> str:
    """What a measurement depends on: the movie, the calibration's bytes, the measuring settings."""
    h = hashlib.sha256()
    for part in (str(Path(movie).resolve()), json.dumps(settings, sort_keys=True, default=str)):
        h.update(part.encode())
    if calibration and Path(calibration).is_file():
        h.update(hashlib.sha256(Path(calibration).read_bytes()).hexdigest().encode())
    return h.hexdigest()[:16]


def cache_path(movie) -> Path:
    """Beside the movie: ``<movie>.channel_move.json``, every measurement of it by its key."""
    p = Path(movie)
    return p.with_name(p.name + ".channel_move.json")


def cached(movie, key: str) -> Optional[ChannelMove]:
    try:
        found = json.loads(cache_path(movie).read_text(encoding="utf-8")).get(key)
    except (OSError, ValueError):
        return None
    return None if found is None else ChannelMove.from_dict(found)


def keep(movie, key: str, move: ChannelMove) -> None:
    """Kept beside the movie; a folder that cannot be written to keeps nothing, silently."""
    path = cache_path(movie)
    try:
        known = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        known = {}
    known[key] = move.to_dict()
    try:
        path.write_text(json.dumps(known, indent=1), encoding="utf-8")
    except OSError:
        pass


# ------------------------------------------------------------------------------------ the figure
def draw(move: ChannelMove, data: Optional[dict] = None):
    """A `plugins.Plot` of the measurement: each pair's shift against the frame with the move (rows and
    columns), against z where the fit had it, and the vote's histogram when it ran."""
    from .plugins import Plot
    data = data or {}

    def plot(figure):
        axes = figure.subplots(1, 3)
        frame, shift, z = data.get("frame"), data.get("shift"), data.get("z_nm")
        for k, (ax, name) in enumerate(zip(axes[:2], ("columns", "rows"))):
            if frame is not None and len(frame):
                ax.plot(frame, shift[:, k], ".", ms=2, alpha=0.3, color="grey")
            t = (np.asarray(move.frames) if move.over_time
                 else np.array([0.0, float(np.max(frame)) if frame is not None and len(frame) else 1.0]))
            ax.plot(t, move.at(t)[:, k], "-", lw=2)
            ax.set(xlabel="frame", ylabel=f"move, {name} (px)")
            if frame is not None and len(frame):
                lo, hi = np.percentile(shift[:, k], [2, 98])
                pad = max(hi - lo, 0.2) * 0.3
                ax.set_ylim(lo - pad, hi + pad)
        ax = axes[2]
        if z is not None and len(z):
            r = shift - move.at(frame)
            ax.plot(z, r[:, 0], ".", ms=2, alpha=0.3, label="columns")
            ax.plot(z, r[:, 1], ".", ms=2, alpha=0.3, label="rows")
            ax.axhline(0, color="k", lw=0.8)
            lo, hi = np.percentile(r, [2, 98])
            ax.set(xlabel="z (nm)", ylabel="left after the move (px)", ylim=(lo - 0.1, hi + 0.1))
            ax.legend(loc="upper right", fontsize="small")
        else:
            ax.axis("off")
    return Plot(draw=plot, panels=3, size=(10, 3.2))
