"""Link tracks: localizations of one particle in successive frames joined
into a track, ``track_id`` and ``track_length``.

The algorithm is `smappy.tracking.link` (its docstring has the decisions);
this is the thin wrapper.  SMAP's ``SimpleTracking`` (the Crocker-Grier
linker of ``mytrack.m``) is the reference for the settings: a maximum jump,
a memory of dark frames, a minimum length, 2D or 3D.  Where it departs:

* two linkers, an *algorithm* choice: the **simple** one (nearest first,
  the default -- on sparse tracks it finds the same links, ten times
  faster) and a **global assignment** per frame, which is Crocker and
  Grier's criterion solved by `linear_sum_assignment` rather than SMAP's
  enumeration of subnetworks, which gave up on "excessive combinatorics";
* the default maximum jump is **500 nm**, not 1000: a jump comparable to
  the distance between molecules is not one a fit of single molecules
  gives, and a large one only invites wrong links;
* each **file** is linked on its own, always, and each **channel** unless
  *per channel* is off (SMAP offered all, per layer or per channel);
* ``track_id`` is **0** for a localization in no track and for a track
  shorter than the minimum, rather than such tracks being dropped from the
  numbering and their localizations keeping an old id; ids are 1, 2, ... in
  the order the tracks start;
* the localizations linked are those the layer's filter and the ROI keep;
  the rest get 0.  Always the localizations, never merged blinks: merging
  would join a slow particle's whole track into one blink.

A track is a region: ``track_id`` is an id column like ``cluster_id``, so
every ``per_region`` plugin runs per track, and *Segment regions* (*from an
id column*) makes ROIs of tracks.  Both columns are recipes with the rule
``max`` (`mathparse.remember`), so a merged blink keeps its track.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import Context, FormState, Plot, Plugin, Result, Stopped, param, register
from ..locs import Localizations

# tracks drawn in the figure at most: more is a tangle and seconds to draw
FIGURE_TRACKS = 3000


@dataclass
class LinkTracksSettings:
    algorithm: str = param("simple", label="algorithm",
                           choices=(("simple", "simple"), ("global", "global assignment")),
                           help="simple: nearest first; global: least squared jumps "
                                "per frame")
    max_jump_nm: float = param(500.0, label="maximum jump", unit="nm", min=1.0,
                               help="the farthest a particle moves between the "
                                    "frames it is seen in")
    memory: int = param(2, label="memory", unit="frames", min=0,
                        help="dark frames a track survives")
    min_length: int = param(2, label="minimum length", min=1,
                            help="localizations in a track; shorter ones get "
                                 "track_id 0")
    use_z: bool = param(True, label="in 3D", help="jumps in x, y and z when the "
                                                  "table has z_nm")
    per_channel: bool = param(True, label="per channel",
                              help="link each channel on its own")


@register("Analysis/Tracking/Link tracks")
class LinkTracks(Plugin):
    """Links localizations in successive frames into tracks: track_id and
    track_length, bridging dark frames."""

    Settings = LinkTracksSettings
    changes_locs = True
    # tracks are of localizations: a merged blink is a piece of a track
    grouping = "ungrouped"
    version = "1"

    def describe(self, ctx: Optional[Context], settings: LinkTracksSettings
                 ) -> FormState:
        state = FormState()
        if ctx is not None and ctx.session is not None:
            locs = ctx.locs
            state.active["use_z"] = "z_nm" in locs
            state.active["per_channel"] = "channel" in locs
        return state

    def run(self, ctx: Context, settings: LinkTracksSettings) -> Result:
        from ..mathparse import remember
        from ..tracking import link_tracks
        locs = ctx.locs
        missing = [c for c in ("x_nm", "y_nm", "frame") if c not in locs]
        if missing:
            raise ValueError(f"linking needs {', '.join(missing)}: the table has "
                             f"{', '.join(sorted(locs.keys()))}")
        use = np.asarray(ctx.selection.mask, dtype=bool)
        n = len(locs)
        groups = (np.asarray(locs["filenumber"], np.int64) if "filenumber" in locs
                  else np.zeros(n, np.int64))
        if settings.per_channel and "channel" in locs:
            channel = np.asarray(locs["channel"], np.int64)
            _, channel = np.unique(channel, return_inverse=True)
            groups = groups * (int(channel.max(initial=0)) + 1) + channel
        groups = np.where(use, groups, -1)
        in_3d = settings.use_z and "z_nm" in locs

        def progress(fraction):
            if ctx.stopped:
                raise Stopped(f"linking stopped at {100 * fraction:.0f}%: nothing written")
            ctx.report(f"linking: {100 * fraction:.0f}%")

        track, length = link_tracks(
            locs["x_nm"], locs["y_nm"], locs["frame"],
            locs["z_nm"] if in_3d else None, max_jump_nm=settings.max_jump_nm,
            memory=settings.memory, min_length=settings.min_length,
            algorithm=settings.algorithm, groups=groups, progress=progress)
        out = Localizations({**locs.columns, "track_id": track, "track_length": length},
                            dict(locs.metadata))
        remember(out, "track_id", "", grouped="max")
        remember(out, "track_length", "", grouped="max")

        n_tracks = int(track.max(initial=0))
        tracked = int((track > 0).sum())
        lengths = np.bincount(track)[1:] if n_tracks else np.zeros(0, np.int64)
        mean = float(lengths.mean()) if n_tracks else 0.0
        text = (f"{n_tracks} tracks of {tracked} localizations (mean length "
                f"{mean:.1f}), {int(use.sum()) - tracked} in none; "
                f"{settings.algorithm}, maximum jump {settings.max_jump_nm:g} nm, "
                f"memory {settings.memory}" + (", 3D" if in_3d else ""))
        x = np.asarray(locs["x_nm"], float)
        y = np.asarray(locs["y_nm"], float)
        frame = np.asarray(locs["frame"], np.int64)
        return Result(locs=out, text=text, settings=settings,
                      data={"tracks": n_tracks, "tracked": tracked,
                            "mean_length": mean},
                      plot=Plot(lambda figure: draw_tracks(figure, x, y, frame, track),
                                panels=2, size=(8, 3.6)))


def draw_tracks(figure, x, y, frame, track) -> None:
    """The tracks as lines, coloured by id; and how long they are."""
    from matplotlib.collections import LineCollection
    left, right = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.3, 1]})
    track = np.asarray(track, np.int64)
    n_tracks = int(track.max(initial=0))
    shown = np.arange(1, n_tracks + 1)
    if n_tracks > FIGURE_TRACKS:
        shown = np.sort(np.random.default_rng(0).choice(shown, FIGURE_TRACKS,
                                                        replace=False))
    rows = np.flatnonzero(np.isin(track, shown))
    rows = rows[np.lexsort((frame[rows], track[rows]))]
    t = track[rows]
    same = t[1:] == t[:-1]
    segments = np.stack([np.column_stack([x[rows][:-1], y[rows][:-1]]),
                         np.column_stack([x[rows][1:], y[rows][1:]])], axis=1)[same]
    lines = LineCollection(segments, linewidths=0.8, cmap="tab20")
    lines.set_array(t[:-1][same] % 20)
    left.add_collection(lines)
    if len(x):
        left.set_xlim(np.nanmin(x), np.nanmax(x))
        left.set_ylim(np.nanmin(y), np.nanmax(y))
    left.set_aspect("equal")
    left.set_xlabel("x (nm)")
    left.set_ylabel("y (nm)")
    left.set_title(f"{n_tracks} tracks" + (f" ({FIGURE_TRACKS} drawn)"
                                           if n_tracks > FIGURE_TRACKS else ""),
                   fontsize=9)
    lengths = np.bincount(track)[1:] if n_tracks else np.zeros(0, int)
    if len(lengths):
        right.hist(lengths, bins=np.arange(1, lengths.max() + 2) - 0.5
                   if lengths.max() <= 60 else 50, color="0.6")
    right.set_xlabel("localizations per track")
    right.set_ylabel("tracks")
