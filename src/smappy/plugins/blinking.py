"""How a dye blinks: on-times, dark times, and blinks per molecule.

A port of the counting part of SMAP's ``Analyze/cluster/dyeblinking`` and of
the ``locsdf`` count of its counting plugins
(``Analyze/counting/private/cluster_mmaple_makehist.m``), kept simple: plain
functions on frames, which a script calls without a session.

**A blink** is what the layer merges: the localizations that share a
``group_id`` when the layer shows merged blinks, else -- as in SMAP -- a run
of localizations in consecutive frames.  It is on from its first frame to
its last, so its **on-time** is ``last - first + 1`` frames: a frame the
fit missed in the middle of a blink was not dark.  The **dark time** before
a blink is the number of frames between it and the end of the one before:
``first - last_before - 1``.  Two blinks that overlap in time (two
fluorophores of one region, on together) have no dark time between them.

**Bursts** (SMAP's ``locsdf``): the blinks of a region joined wherever the
dark time between them is at most *dark time* T, a new burst wherever it is
longer.  With T longer than a dye's dark times and shorter than the time
between two of its molecules coming on, a burst is a molecule -- the count
SMAP fitted copy numbers to.  SMAP counts ``diff(frame) > T``, which is a
dark time of ``T`` frames or more; here a dark time longer than T splits,
the same rule as *dark frames* in merging, so T is the dark frames a burst
bridges.

**A molecule**, on the view, is the localizations linked with that long
dark time: the merging walk (`group.connect`) with *dark time* T and a box
of *within*, per file and channel -- SMAP regrouped the same way.  Within
each, its blinks, the dark times between them and their number give the
three histograms of SMAP's ``dyeblinking``.  Per region there is no
linking: a region is taken to be one structure, and its blinks are split
into bursts by time alone.

**Seconds** are frames times the file's frame time (`metadata.frame_time_ms`:
the cycle time the file recorded, else the exposure), or the *frame time*
set here.

Not here: a counting fit (copy numbers from the blinks per molecule), the
clusters to count in (Find clusters, Segment regions, Filter ROIs) and
qPAINT.  SMAP's ``dyeblinking`` also drew the photons, the on-time and the
blinks against a series of dark times, the on-fraction and the duration of a
molecule; those are left out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import NO_FRAME_TIME, Context, Plot, Plugin, Result, param, register

#: the most bins a time histogram is drawn with; more frames than this and a
#: bin is several frames wide
MAX_BINS = 60


# ------------------------------------------------------------- the algorithm

@dataclass
class Blinks:
    """One row per blink, sorted by molecule, then by first frame."""
    molecule: np.ndarray     # which molecule (region: all 0)
    first: np.ndarray        # first frame
    last: np.ndarray         # last frame
    n: np.ndarray            # localizations in it

    def __len__(self) -> int:
        return len(self.first)

    @property
    def on_frames(self) -> np.ndarray:
        """The on-time of each blink: first to last frame, both counted."""
        return self.last - self.first + 1


def blinks_of(frame, blink=None, molecule=None) -> Blinks:
    """The blinks among localizations in ``frame``.

    ``blink`` is the blink each localization belongs to (``group_id``);
    without it a blink is a run of consecutive frames.  ``molecule`` keeps
    blinks of different molecules apart; without it they are all one.
    """
    frame = np.asarray(frame, np.int64)
    n = len(frame)
    molecule = (np.zeros(n, np.int64) if molecule is None
                else np.asarray(molecule, np.int64))
    if n == 0:
        empty = np.zeros(0, np.int64)
        return Blinks(empty, empty, empty, empty)
    if blink is None:
        order = np.lexsort((frame, molecule))
        f, m = frame[order], molecule[order]
        new = np.ones(n, bool)
        new[1:] = (m[1:] != m[:-1]) | (f[1:] - f[:-1] > 1)
    else:
        blink = np.asarray(blink, np.int64)
        order = np.lexsort((frame, blink, molecule))
        f, m, b = frame[order], molecule[order], blink[order]
        new = np.ones(n, bool)
        new[1:] = (m[1:] != m[:-1]) | (b[1:] != b[:-1])
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], n) - 1
    out = Blinks(m[starts], f[starts], f[ends], ends - starts + 1)
    # a molecule's blinks in order of time: with ids, one blink may begin
    # before another that has a smaller id
    order = np.lexsort((out.first, out.molecule))
    return Blinks(out.molecule[order], out.first[order], out.last[order],
                  out.n[order])


def dark_frames(blinks: Blinks) -> np.ndarray:
    """The dark time before each blink, in frames: NaN for a molecule's first
    blink, and for one that began before every earlier blink had ended."""
    n = len(blinks)
    if n == 0:
        return np.zeros(0)
    m, first, last = blinks.molecule, blinks.first, blinks.last
    rank = np.unique(m, return_inverse=True)[1].reshape(-1).astype(np.int64)
    # the latest end so far within each molecule: one running maximum over
    # the whole table, each molecule lifted above the one before
    low = int(first.min())
    lift = int(last.max()) - low + 2
    ended = np.maximum.accumulate(rank * lift + (last - low)) - rank * lift + low
    gap = np.full(n, np.nan)
    gap[1:] = first[1:] - ended[:-1] - 1
    starts = np.ones(n, bool)
    starts[1:] = m[1:] != m[:-1]
    gap[starts] = np.nan
    gap[gap < 0] = np.nan
    return gap


def bursts(blinks: Blinks, dark_time: float,
           gaps: Optional[np.ndarray] = None) -> np.ndarray:
    """The burst each blink is in, 0-based: blinks joined across dark times of
    at most ``dark_time`` frames, a new burst after a longer one or at a new
    molecule.  The number of bursts is SMAP's ``locsdf``."""
    if len(blinks) == 0:
        return np.zeros(0, np.int64)
    gaps = dark_frames(blinks) if gaps is None else gaps
    new = np.zeros(len(blinks), bool)
    new[0] = True
    new[1:] = blinks.molecule[1:] != blinks.molecule[:-1]
    with np.errstate(invalid="ignore"):
        new |= gaps > dark_time
    return np.cumsum(new) - 1


def molecules(x, y, frame, dark_time: int, distance_nm: float,
              blocks=None) -> np.ndarray:
    """The molecule of each localization, 1-based: linked by the merging walk
    (`group.connect`) across up to ``dark_time`` dark frames, within a box of
    ``distance_nm`` -- blinks merged with a long dark time."""
    from ..group import connect
    if len(frame) == 0:
        return np.zeros(0, np.int64)
    return connect(x, y, frame, dx=float(distance_nm), dt=int(dark_time),
                   blocks=blocks)


def region_blinking(frame, blink=None, dark_time: float = 100,
                    frame_time_ms: Optional[float] = None) -> Dict[str, float]:
    """The numbers of one region: localizations, blinks, bursts at
    ``dark_time``, and the mean on-time and dark time (within bursts), in
    frames and, with a frame time, in seconds (NaN without)."""
    found = blinks_of(frame, blink)
    gaps = dark_frames(found)
    burst = bursts(found, dark_time, gaps)
    with np.errstate(invalid="ignore"):
        within = gaps[np.isfinite(gaps) & (gaps <= dark_time)]
    on = found.on_frames
    mean_on = float(on.mean()) if len(on) else float("nan")
    mean_dark = float(within.mean()) if len(within) else float("nan")
    seconds = (frame_time_ms / 1000.0 if frame_time_ms and frame_time_ms > 0
               else float("nan"))
    return {"n_localizations": int(len(np.asarray(frame))),
            "n_blinks": int(len(found)),
            "n_bursts": int(burst[-1] + 1) if len(burst) else 0,
            "mean_on_frames": mean_on, "mean_on_s": mean_on * seconds,
            "mean_dark_frames": mean_dark, "mean_dark_s": mean_dark * seconds}


@dataclass
class MoleculeBlinking:
    """What the molecules of one file did: the histograms' raw numbers."""
    on_frames: np.ndarray            # per blink
    dark_frames: np.ndarray          # per dark time between blinks of a molecule
    blinks_per_molecule: np.ndarray  # per molecule
    frame_time_ms: Optional[float] = None
    name: str = ""

    def summary(self) -> str:
        def both(values) -> str:
            if not len(values):
                return "-"
            mean = float(np.mean(values))
            text = f"{mean:.2f} frames"
            if self.frame_time_ms:
                text += f" = {mean * self.frame_time_ms / 1000:.3g} s"
            return text
        per = self.blinks_per_molecule
        return (f"{len(per)} molecules, {float(per.mean()) if len(per) else 0:.2f} "
                f"blinks each; on-time {both(self.on_frames)}, "
                f"dark time {both(self.dark_frames)}")


def molecule_blinking(frame, x, y, dark_time: int, distance_nm: float,
                      blink=None, blocks=None,
                      frame_time_ms: Optional[float] = None, name: str = ""
                      ) -> MoleculeBlinking:
    """On-times, dark times and blinks per molecule of one file's
    localizations, a molecule being what `molecules` links."""
    mol = molecules(x, y, frame, dark_time, distance_nm, blocks)
    found = blinks_of(frame, blink, mol)
    gaps = dark_frames(found)
    per = (np.bincount(np.unique(found.molecule, return_inverse=True)[1].reshape(-1))
           if len(found) else np.zeros(0, np.int64))
    return MoleculeBlinking(found.on_frames, gaps[np.isfinite(gaps)], per,
                            frame_time_ms, name)


def view_values(found: Sequence[MoleculeBlinking], n_localizations: int,
                frame_time_ms: Optional[float] = None) -> Dict[str, float]:
    """The numbers of `region_blinking` for the view, from its molecules: a
    burst is a molecule, and its blinks and dark times are its own -- one
    time line for the whole field would join the blinks of molecules that
    were on together."""
    def mean(key):
        values = [getattr(f, key) for f in found]
        values = np.concatenate(values) if values else np.zeros(0)
        return float(values.mean()) if len(values) else float("nan")
    seconds = (frame_time_ms / 1000.0 if frame_time_ms and frame_time_ms > 0
               else float("nan"))
    on, dark = mean("on_frames"), mean("dark_frames")
    return {"n_localizations": int(n_localizations),
            "n_blinks": int(sum(len(f.on_frames) for f in found)),
            "n_bursts": int(sum(len(f.blinks_per_molecule) for f in found)),
            "mean_on_frames": on, "mean_on_s": on * seconds,
            "mean_dark_frames": dark, "mean_dark_s": dark * seconds}


# --------------------------------------------------------------------- drawing

def integer_edges(values, most: int = MAX_BINS, start: int = 0) -> np.ndarray:
    """Bin edges between whole numbers from ``start`` to the 99.5th
    percentile, each bin a whole number of them, at most ``most`` bins."""
    values = np.asarray(values, float)
    top = int(np.ceil(np.percentile(values, 99.5))) if len(values) else start + 1
    top = max(top, start + 1)
    width = max(int(np.ceil((top - start + 1) / most)), 1)
    return np.arange(start, top + width + 1, width) - 0.5


def draw_histograms(figure, found: Sequence[MoleculeBlinking],
                    seconds: bool = False, log: bool = True) -> None:
    """Three panels -- on-times, dark times, blinks per molecule -- with a
    line per file, each the fraction of its own total."""
    axes = figure.subplots(3, 1, squeeze=False).ravel()
    unit = "s" if seconds else "frames"
    panels = ((axes[0], "on_frames", f"on-time ({unit})", 1, True),
              (axes[1], "dark_frames", f"dark time ({unit})", 0, True),
              (axes[2], "blinks_per_molecule", "blinks per molecule", 1, False))
    for ax, key, label, start, timed in panels:
        everything = np.concatenate([getattr(f, key) for f in found]) if found else []
        edges = integer_edges(everything, start=start)
        for f in found:
            values = np.asarray(getattr(f, key), float)
            if not len(values):
                continue
            counts, _ = np.histogram(values, edges)
            scale = (f.frame_time_ms / 1000.0) if (seconds and timed) else 1.0
            ax.stairs(counts / len(values), edges * scale, label=f.name or None)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("fraction")
    if any(f.name for f in found) and len(found) > 1:
        axes[0].legend(fontsize=7, frameon=False)
    if len(found) == 1:
        axes[0].set_title(found[0].summary(), fontsize=7.5, color="0.25")


# ---------------------------------------------------------------- the plugin

@dataclass
class BlinkingSettings:
    dark_time: int = param(100, label="dark time", unit="frames", min=0,
                           help="the longest dark time within a molecule: "
                                "longer starts a new burst (a region) or a new "
                                "molecule (the view)")
    distance_nm: float = param(50.0, label="within", unit="nm", min=1.0,
                               help="how far a molecule's localizations may lie "
                                    "apart to be linked (the view only)")
    time_axis: str = param("frames", label="time in",
                           choices=(("frames", "frames"), ("seconds", "seconds")),
                           help="the unit of the on-time and dark-time axes")
    log_y: bool = param(True, label="log scale",
                        help="the fractions on a logarithmic axis")
    frame_time_ms: Optional[float] = param(None, label="frame time", unit="ms",
                                           min=0.001, advanced=True,
                                           help="auto: the file's, else its "
                                                "exposure")


def _localizations(ctx: Context):
    """The localizations of what the layer shows and selects, and the blink
    of each (``group_id`` on a merged layer, else None)."""
    table, selection = ctx.table()
    shown = selection.apply(table)
    ungrouped = ctx.ungrouped()[0]
    if table is ungrouped:
        return shown, None
    if "group_id" not in shown or "group_id" not in ungrouped:
        raise ValueError("the merged blinks have no group_id to find their "
                         "localizations by")
    wanted = np.isin(np.asarray(ungrouped["group_id"]), np.asarray(shown["group_id"]))
    locs = ungrouped[wanted]
    return locs, np.asarray(locs["group_id"])


@register("Analysis/Measure/Blinking")
class Blinking(Plugin):
    """On-times, dark times and blinks per molecule; per region, the blinks
    and the bursts at a dark time."""

    Settings = BlinkingSettings
    name = "Blinking statistics"
    changes_locs = False
    per_region = True
    version = "1"

    def run(self, ctx: Context, settings: BlinkingSettings) -> Result:
        from ..metadata import file_frame_time, table_frame_time
        locs, blink = _localizations(ctx)
        if "frame" not in locs:
            raise ValueError("the table has no frame column: "
                             f"it has {', '.join(sorted(locs.keys()))}")
        frame = np.asarray(locs["frame"])
        if ctx.per_region:
            values = region_blinking(frame, blink, settings.dark_time,
                                     settings.frame_time_ms or table_frame_time(locs))
            text = (f"{values['n_localizations']} localizations, "
                    f"{values['n_blinks']} blinks, {values['n_bursts']} bursts at "
                    f"a dark time of {settings.dark_time} frames; mean on-time "
                    f"{values['mean_on_frames']:.2f} frames, mean dark time "
                    f"{values['mean_dark_frames']:.2f} frames")
            return Result(text=text, values=values, settings=settings)
        if not len(locs):
            raise ValueError("no localizations to measure")

        from ..render import positions
        x, y = positions(locs)
        files = (np.asarray(locs["filenumber"]) if "filenumber" in locs
                 else np.zeros(len(locs), np.int64))
        names = list(getattr(ctx.session, "file_names", lambda: [])() or [])
        found: List[MoleculeBlinking] = []
        for number in np.unique(files):
            here = files == number
            ft = settings.frame_time_ms or file_frame_time(locs.metadata, int(number))
            if settings.time_axis == "seconds" and not ft:
                raise ValueError(NO_FRAME_TIME)
            channel = (np.asarray(locs["channel"])[here] if "channel" in locs
                       else None)
            name = (names[int(number)] if 0 <= int(number) < len(names)
                    else f"file {int(number) + 1}")
            found.append(molecule_blinking(
                frame[here], np.asarray(x)[here], np.asarray(y)[here],
                settings.dark_time, settings.distance_nm,
                blink=None if blink is None else blink[here], blocks=channel,
                frame_time_ms=ft, name=name))
            if ctx.stopped:
                break
        values = view_values(found, len(locs),
                             settings.frame_time_ms or table_frame_time(locs))
        lines = [f"{len(locs)} localizations; molecules linked across "
                 f"{settings.dark_time} dark frames within "
                 f"{settings.distance_nm:g} nm"]
        lines += [f"{f.name}: {f.summary()}" for f in found]
        seconds = settings.time_axis == "seconds"

        def plot(figure) -> None:
            draw_histograms(figure, found, seconds=seconds, log=settings.log_y)

        return Result(text="\n".join(lines), values=values,
                      data={"files": found}, settings=settings,
                      plot=Plot(draw=plot, panels=3, size=(5.5, 6.3)))
