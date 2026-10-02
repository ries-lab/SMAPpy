"""The image tags -- what the microscope recorded per frame -- against frame.

A fit keeps every tag that changed during the acquisition
(`smappy.frametags`): the piezo position, the UV pulse length, the focus
lock's signals, the camera's clock.  This draws them, one tab each, with the
number of localizations per frame beside them on a second axis, because the
question is nearly always how the two go together: did the density follow
the UV, did the focus lock hold while the localizations thinned out.

The piezo position is the one looked at most -- a focus lock that ran into
its limit or jumped shows there before it shows in the data -- so the fit
draws it by itself when it finishes (`OutputSettings.show_tags`), with the
same function as here.

Ported from SMAP's ``Analyze/other/DisplayImageTags``.  SMAP plots tag k for
the frames where its value is not zero, to hide the frames it did not read;
here a frame that was not read is not in the table, so a real zero -- a UV
pulse that is off -- is drawn as one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from . import Context, Plugin, Result, param, register
from ..frametags import ACQUISITION, KEY


def names(tags: Dict[str, np.ndarray], wanted: str = "") -> List[str]:
    """The tags to show: those matching ``wanted``, or all of them.

    ``wanted`` is a comma-separated list of names or parts of names, without
    regard to case -- ``piezo, trigger`` finds nothing on a setup whose piezo
    is called ``PIZStage``, but ``PIZ`` does -- so that a name does not have
    to be typed out in full, units in brackets and all.
    """
    there = [n for n in tags if n != "frame"]
    parts = [w.strip().lower() for w in wanted.split(",") if w.strip()]
    if not parts:
        return there
    return [n for n in there if any(p in n.lower() for p in parts)]


def per_frame(frames: np.ndarray, first: int, last: int,
              bins: int = 500) -> tuple:
    """Localizations per frame, averaged over at most ``bins`` bins.

    ``(centres, rate)``: a 90 000-frame acquisition has a few localizations
    per frame and a per-frame count is noise; the average over a bin is the
    rate, still in localizations per frame.
    """
    frames = np.asarray(frames, np.int64)
    span = max(last - first + 1, 1)
    width = max(int(np.ceil(span / bins)), 1)
    counts = np.bincount((frames[(frames >= first) & (frames <= last)] - first)
                         // width, minlength=int(np.ceil(span / width)))
    centres = first + (np.arange(len(counts)) + 0.5) * width
    return centres, counts / width


def summary(tags: Dict[str, np.ndarray], shown: Sequence[str]) -> str:
    """One line per tag: its range, or its states."""
    lines = []
    for name in shown:
        values = np.asarray(tags[name])
        if values.dtype == object:
            states = list(dict.fromkeys(str(v) for v in values))
            lines.append(f"{name}: {' -> '.join(states)}"
                         if len(states) <= 4 else f"{name}: {len(states)} states")
            continue
        finite = values[np.isfinite(values)]
        if not len(finite):
            lines.append(f"{name}: not read")
            continue
        lo, hi = float(finite.min()), float(finite.max())
        lines.append(f"{name}: {finite[0]:.6g} at the start, {finite[-1]:.6g} at "
                     f"the end, {lo:.6g} to {hi:.6g} (range {hi - lo:.4g})")
    return "\n".join(lines)


def draw_tag(ax, tags: Dict[str, np.ndarray], name: str,
             locs_frames: Optional[np.ndarray] = None) -> None:
    """One tag against frame; the localization rate on a second axis."""
    frames = np.asarray(tags["frame"])
    values = np.asarray(tags[name])
    if values.dtype == object:
        # a state: drawn as steps between its values, in order of appearance
        states = list(dict.fromkeys(str(v) for v in values))
        index = {s: i for i, s in enumerate(states)}
        ax.step(frames, [index[str(v)] for v in values], where="post",
                color="C0", lw=1)
        ax.set_yticks(range(len(states)))
        ax.set_yticklabels(states)
    else:
        ax.plot(frames, values, color="C0", lw=0.8)
    ax.set_xlabel("frame")
    ax.set_ylabel(name, color="C0")
    if locs_frames is not None and len(locs_frames) and len(frames):
        centres, rate = per_frame(locs_frames, int(frames.min()), int(frames.max()))
        twin = ax.twinx()
        twin.plot(centres, rate, color="C1", lw=0.8, alpha=0.8)
        twin.set_ylabel("localizations per frame", color="C1")
        twin.set_ylim(bottom=0)
    ax.set_xlim(frames.min(), frames.max())


def plots(tags: Dict[str, np.ndarray], shown: Sequence[str],
          locs_frames: Optional[np.ndarray] = None) -> Dict[str, Callable]:
    """A drawing per tag, by name -- closures, drawn when their tab is opened."""
    def one(name):
        return lambda ax: draw_tag(ax, tags, name, locs_frames)
    return {name: one(name) for name in shown}


@dataclass
class ImageTagsSettings:
    tags: str = param("", label="tags",
                      help="names or parts of names, comma-separated; "
                           "empty: every tag the fit kept")
    rate: bool = param(True, label="localizations per frame",
                       help="draw the rate of the selected localizations "
                            "on a second axis")


@register("Analysis/Process/Image Tags")
class ImageTags(Plugin):
    """The per-frame image tags of the acquisition (piezo z, UV pulse, ...)
    against frame."""

    Settings = ImageTagsSettings
    version = "1"

    def run(self, ctx: Context, settings: ImageTagsSettings) -> Result:
        tags = (ctx.locs.metadata or {}).get(KEY) or {}
        if not tags or len(tags) < 2:
            raise ValueError(
                "this table carries no image tags: they are recorded by the "
                "fit, from a Micro-Manager acquisition whose frames say "
                "something that changes")
        shown = names(tags, settings.tags)
        if not shown:
            raise ValueError(f"no tag matches '{settings.tags}'; there are: "
                             + ", ".join(names(tags)))
        frames = None
        if settings.rate and "frame" in ctx.locs:
            frames = np.asarray(ctx.selection.apply(ctx.locs)["frame"])
        text = summary(tags, shown)
        comment = ((ctx.locs.metadata or {}).get(ACQUISITION) or {}).get("comment")
        if comment:
            text = f"comment: {' | '.join(comment.splitlines())}\n{text}"
        return Result(text=text, plots=plots(tags, shown, frames),
                      data={"tags": {n: tags[n] for n in ["frame", *shown]}},
                      settings=settings)
