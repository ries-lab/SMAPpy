"""Make the files of the session one acquisition.

Files opened together are one table but stay separate files: each keeps its
own frame numbers and its own ids, and everything that works along time
works per file (`docs/plugin-reference.md`, "Frames").  That is right for
files that are different acquisitions, and wrong for one acquisition saved
in pieces -- a movie split into several TIFFs, a measurement stopped and
restarted -- where drift, linking and the blinking statistics should see one
time line.  This plugin turns the second case into one file:

* **frames continue**: file ``k``'s frames move up by the frames of the
  files before it (each file's last frame + 1, plus *gap*), so they never
  overlap and keep their order;
* **time continues** the same way: a time column (`time_s`) moves up by the
  duration of the files before it -- their frames times their frame time,
  or their last time stamp when that is later or the frame time unknown;
* **ids stay apart**: an id column (``*_id``: `track_id`, `cluster_id`) is
  numbered on from the files before it, 0 staying 0 ("in none"), so track 5
  of one file and track 5 of the next do not become one track.  MINFLUX's
  trace id `tid` likewise, all of it, since its 0 is a trace.  `roi_id` and
  the columns written from regions' results are an ROI's number, the same
  in every file, and stay;
* **linking goes**: `group_id` / `n_in_group` said which localizations of
  one file are a blink; across the old boundary there may now be one, so
  the next grouping links again (`Session.add_file` drops them the same
  way when a file is added);
* **one file entry**: `filenumber` is 0 everywhere, and the table's
  ``metadata["files"]`` is one entry, named after the files and keeping each
  one's entry (``metadata["combined_from"]``), which the session takes as
  its file list (`Session._follow_files`).

What makes files incompatible is said, not refused: a different pixel size,
frame time, format or set of columns (`FileInfo.columns`) -- each a reason
to doubt they are one acquisition, none a reason the numbers cannot be
computed.  The first file's metadata (its camera, its fit) becomes the
table's.  The algorithm is `combine` on a table; `run` is the wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..group import GROUP_COLUMNS
from ..locs import Localizations
from ..metadata import file_frame_time
from . import Context, Plugin, Result, param, register

# Time columns and what one of their units is in seconds.  `time_s` is the
# one there is (MINFLUX's clock, a fit's frame times); `time_ms` in case.
TIME_COLUMNS: Dict[str, float] = {"time_s": 1.0, "time_ms": 1e-3}

# Integer ids whose every value is an id -- 0 included -- rather than 0
# meaning "in none" as for the ``*_id`` columns.
ALL_VALUE_IDS = ("tid",)

# An ROI's own number: the same ROI whichever file its rows came from.
REGION_NUMBER = "roi_id"


@dataclass
class CombineSettings:
    gap: int = param(0, label="gap", unit="frames", min=0, max=1_000_000,
                     help="empty frames left between one file and the next")


def file_entries(locs: Localizations, n_files: int) -> List[dict]:
    """Each file's entry from the table's ``metadata["files"]``; a bare
    entry for a file the metadata does not describe."""
    entries = locs.metadata.get("files")
    entries = entries if isinstance(entries, list) else []
    return [dict(entries[k]) if k < len(entries) and isinstance(entries[k], dict)
            else {"name": f"file {k + 1}", "path": "", "format": ""}
            for k in range(n_files)]


def id_columns(locs: Localizations) -> List[str]:
    """The integer columns whose numbers are per file and must be kept apart."""
    written = set(locs.metadata.get("region_columns") or {})
    out = []
    for name in locs.columns:
        if name in GROUP_COLUMNS or name == REGION_NUMBER or name in written:
            continue
        if not (name.endswith("_id") or name in ALL_VALUE_IDS):
            continue
        if np.asarray(locs[name]).dtype.kind in "iu":
            out.append(name)
    return out


def incompatibilities(entries: Sequence[dict], frame_times: Sequence[Optional[float]]
                      ) -> List[str]:
    """What makes these files look like more than one acquisition."""
    out = []
    names = [str(e.get("name") or f"file {k + 1}") for k, e in enumerate(entries)]

    def differ(values, what, unit=""):
        known = {round(float(v), 6) for v in values if v is not None}
        if len(known) > 1:
            listed = ", ".join(f"{n}: {'?' if v is None else f'{float(v):g}{unit}'}"
                               for n, v in zip(names, values))
            out.append(f"the files differ in {what} ({listed})")

    differ([e.get("pixelsize_nm") for e in entries], "pixel size", " nm")
    differ(frame_times, "frame time", " ms")
    formats = {str(e.get("format") or "") for e in entries} - {""}
    if len(formats) > 1:
        out.append(f"the files are of different formats ({', '.join(sorted(formats))})")
    columns = [e.get("columns") for e in entries]
    if all(c is not None for c in columns):
        every = set().union(*map(set, columns))
        for name, own in zip(names, columns):
            missing = sorted(every - set(own))
            if missing:
                out.append(f"{name} was read without {', '.join(missing)}: "
                           f"its rows hold padding there")
    return out


def combine(locs: Localizations, gap: int = 0) -> Tuple[Localizations, List[str]]:
    """The table as one file, and what made its files look incompatible.

    Files are taken in the order of their numbers.  A table of one file (or
    without `filenumber`) is refused: there is nothing to combine.
    """
    if "filenumber" not in locs or not len(locs):
        raise ValueError("one file only: there is nothing to combine")
    if "frame" not in locs:
        raise ValueError("combining continues the frames, and the table has no "
                         "frame column; it has " + ", ".join(sorted(locs.keys())))
    files = np.asarray(locs["filenumber"], np.int64)
    numbers = [int(k) for k in np.unique(files)]
    if len(numbers) < 2:
        raise ValueError("one file only: there is nothing to combine")
    n_files = max(numbers) + 1
    entries = file_entries(locs, n_files)
    frame_times = [file_frame_time(locs.metadata, k) for k in range(n_files)]
    warnings = incompatibilities([entries[k] for k in numbers],
                                 [frame_times[k] for k in numbers])

    frame = np.asarray(locs["frame"], np.int64)
    new_frame = frame.copy()
    times = {name: np.asarray(locs[name], np.float64).copy()
             for name in TIME_COLUMNS if name in locs}
    ids = {name: np.asarray(locs[name]).copy() for name in id_columns(locs)}
    frames_before = 0
    seconds_before = 0.0
    ids_before = {name: 0 for name in ids}
    for k in numbers:
        rows = files == k
        new_frame[rows] += frames_before
        frames = int(frame[rows].max()) + 1 + int(gap)
        # the file's duration: its frames at its frame time, or its last time
        # stamp if that is later (or the frame time unknown) -- so the next
        # file never starts inside this one
        duration = frames * frame_times[k] * 1e-3 if frame_times[k] else 0.0
        for name in times:
            own = np.asarray(locs[name], np.float64)[rows]
            finite = own[np.isfinite(own)]
            if len(finite):
                duration = max(duration, float(finite.max()) * TIME_COLUMNS[name])
        for name, values in times.items():
            values[rows] += seconds_before / TIME_COLUMNS[name]
        for name, values in ids.items():
            own = values[rows]
            if name in ALL_VALUE_IDS:
                shifted = own + ids_before[name]
                top = int(own.max()) + 1 if len(own) else 0
            else:                       # 0 is "in none" and stays so
                shifted = np.where(own != 0, own + ids_before[name], 0)
                top = int(own.max()) if len(own) else 0
            values[rows] = shifted
            ids_before[name] += max(top, 0)
        frames_before += frames
        seconds_before += duration

    columns = {name: v for name, v in locs.columns.items() if name not in GROUP_COLUMNS}
    dtype = np.asarray(locs["frame"]).dtype
    fits = dtype.kind in "iu" and new_frame.max() <= np.iinfo(dtype).max
    columns["frame"] = new_frame.astype(dtype) if fits else new_frame
    for name, values in times.items():
        columns[name] = values.astype(np.asarray(locs[name]).dtype)
    for name, values in ids.items():
        columns[name] = values
    columns["filenumber"] = np.zeros(len(locs), np.asarray(locs["filenumber"]).dtype)

    used = [entries[k] for k in numbers]
    first = used[0]
    known = [frame_times[k] for k in numbers]
    entry = {
        "name": " + ".join(str(e.get("name") or "") for e in used),
        "path": first.get("path", ""),
        "format": (first.get("format", "")
                   if len({e.get("format") for e in used}) == 1 else "combined"),
        "n": int(len(locs)),
        "pixelsize_nm": first.get("pixelsize_nm"),
        "frame_time_ms": known[0] if len(set(known)) == 1 else None,
    }
    if all(e.get("columns") is not None for e in used):
        entry["columns"] = sorted(set().union(*(e["columns"] for e in used)))
    entry["metadata"] = {**(first.get("metadata") or {}), "combined_from": used}
    metadata = dict(locs.metadata)
    metadata["files"] = [entry]
    return Localizations(columns, metadata), warnings


@register("Analysis/Process/Combine files")
class CombineFiles(Plugin):
    """Make the files of the session one: frames and time continue, ids are
    kept apart, `filenumber` is 0 throughout."""

    description = ("Make the open files one acquisition: frames and time "
                   "continue from one file to the next, ids stay apart, and "
                   "every localization gets file number 0.")
    Settings = CombineSettings
    changes_locs = True
    # the frames of every localization move, whatever the layer shows
    grouping = "ungrouped"
    version = "1"

    def run(self, ctx: Context, settings: CombineSettings) -> Result:
        combined, warnings = combine(ctx.locs, gap=settings.gap)
        for line in warnings:
            ctx.warn(line)
        entry = combined.metadata["files"][0]
        parts = entry["metadata"]["combined_from"]
        frames = int(np.asarray(combined["frame"]).max()) + 1
        text = (f"{len(parts)} files combined into one of {len(combined):,} "
                f"localizations over {frames:,} frames")
        if warnings:
            text += "; " + "; ".join(warnings)
        return Result(locs=combined, text=text, settings=settings,
                      data={"files": len(parts), "frames": frames,
                            "warnings": warnings})
