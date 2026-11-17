"""``smappy info FILE ...``: what a localization file holds, without a window.

Read the way the GUI reads it -- through a `Session`, so a SMAP or smappy
file of several files is split into them and its ROIs are found -- and said
in a few lines: the files, their camera, the columns, what was measured.
It is the check one otherwise makes by opening the file and looking.
"""
import argparse
import sys
import textwrap
from typing import Dict, List

from . import command_name

# the camera's fields worth a line, in the order a reader looks for them
_CAMERA = (("conversion", "conversion", "e-/ADU"), ("offset", "offset", "ADU"),
           ("emgain", "EM gain", ""), ("exposure_ms", "exposure", "ms"),
           ("frame_time_ms", "frame time", "ms"), ("roi", "chip ROI", ""),
           ("camera_name", "camera", ""))


def _number(value) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value:,}" if isinstance(value, int) else f"{value:.4g}"


def _camera_line(camera: Dict) -> str:
    parts = []
    for key, label, unit in _CAMERA:
        value = camera.get(key)
        if value is None or value == "":
            continue
        if key == "emgain" and camera.get("em_on") is False:
            continue                     # a gain that was not used says nothing
        if isinstance(value, (list, tuple)):
            text = " ".join(_number(v) for v in value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            text = _number(value)
        else:
            text = str(value)
        parts.append(f"{label} {text}{' ' + unit if unit else ''}")
    return ", ".join(parts) or "not recorded"


def _ranges(numbers: List[int]) -> str:
    """``files 0-14, 16`` for 0, 1, ..., 14, 16: the numbers of `filenumber`."""
    runs: List[List[int]] = []
    for k in numbers:
        if runs and k == runs[-1][1] + 1:
            runs[-1][1] = k
        else:
            runs.append([k, k])
    text = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)
    return ("file " if len(numbers) == 1 else "files ") + text


def summary(session) -> List[str]:
    """The lines `main` prints for the file ``session`` has open."""
    import numpy as np
    locs, files = session.locs, session.files
    from ..io.formats import reader_for
    path = session.path
    lines = [f"{path.name}  ({reader_for(path).name})" if path else "(no file)"]
    n = len(locs)
    frames = ""
    if n and "frame" in locs:
        frame = np.asarray(locs["frame"])
        frames = f", frames {_number(int(frame.min()))}-{_number(int(frame.max()))}"
    lines.append(f"  {_number(n)} localizations"
                 + (f" in {len(files)} files" if len(files) > 1 else "") + frames)
    counts = (np.bincount(np.asarray(locs["filenumber"], np.int64), minlength=len(files))
              if len(files) > 1 and "filenumber" in locs else [n])
    cameras = []                         # [camera, [file numbers]], first seen first
    for i, f in enumerate(files):
        camera = f.metadata.get("camera") or (locs.metadata.get("camera") if i == 0 else None)
        camera = camera or {}
        for seen in cameras:
            if seen[0] == camera:
                seen[1].append(i)
                break
        else:
            cameras.append([camera, [i]])
    if len(files) > 1:
        width = max(len(f.name) for f in files)
        lines.append(f"  {'#':>3}  {'file':<{width}}  {'localizations':>13}  {'pixel':>8}  "
                     f"{'frame time':>10}")
        for i, (f, count) in enumerate(zip(files, counts)):
            pixel = f"{_number(f.pixelsize_nm)} nm" if f.pixelsize_nm else "?"
            ftime = f"{_number(f.frame_time_ms)} ms" if f.frame_time_ms else "?"
            lines.append(f"  {i:>3}  {f.name:<{width}}  {_number(int(count)):>13}  {pixel:>8}  "
                         f"{ftime:>10}")
    elif files and files[0].pixelsize_nm:
        assumed = " (assumed)" if locs.metadata.get("pixelsize_assumed") else ""
        lines.append(f"  pixel size {_number(files[0].pixelsize_nm)} nm{assumed}")
    for camera, numbers in cameras:
        which = "" if len(cameras) == 1 else f" of {_ranges(numbers)}"
        lines += textwrap.wrap(_camera_line(camera), 88, initial_indent=f"  camera{which}: ",
                               subsequent_indent="    ")
    columns = list(locs.keys())
    lines += textwrap.wrap(" ".join(columns), 88, initial_indent=f"  {len(columns)} columns: ",
                           subsequent_indent="    ", break_on_hyphens=False)
    rois = session.rois.rois
    if rois:
        lines.append(f"  {len(rois)} ROI{'s' if len(rois) > 1 else ''}")
    measured = sorted(session.results)
    if measured:
        lines.append(f"  results: {', '.join(measured)}")
    for column, tables in session.id_regions.items():
        lines.append(f"  per {column}: {', '.join(sorted(tables))}")
    images = [im for im in session.images if im.kind != "empty"]
    if images:
        lines.append(f"  {len(images)} images: "
                     + ", ".join(im.name or im.kind for im in images))
    return lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog=command_name("smappy-info"),
                                description="What a localization file holds: its files, "
                                            "camera, columns, ROIs and results.")
    p.add_argument("files", nargs="+", metavar="FILE",
                   help="localization files: smappy HDF5, SMAP, Picasso, MINFLUX, csv, ...")
    args = p.parse_args(argv)
    from pathlib import Path
    from ..session import Session
    failed = 0
    for i, path in enumerate(args.files):
        if i:
            print()
        if not Path(path).exists():
            print(f"{path}: no such file", file=sys.stderr)
            failed += 1
            continue
        session = Session()
        try:
            session.load(path)
        except Exception as e:           # one unreadable file does not stop the rest
            print(f"{path}: {e}", file=sys.stderr)
            failed += 1
            continue
        print("\n".join(summary(session)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
