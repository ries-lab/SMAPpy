"""Acquisitions written as one TIFF file per frame.

Micro-Manager 1.4 in "separate image files" mode, and a good deal of older
data, writes a folder of ``img_000000000_Default_000.tif`` rather than one
growing OME series, with the whole acquisition's metadata in a sibling
``metadata.txt``.  There is no container to open, only a naming convention, so
this module is what turns such a folder back into one
:class:`~smappy.io.tiff.ImageSource`.

Reading needs nothing new: the base class already walks ``files`` page by
page, and every file here has exactly one page.  What this module adds is
*finding* the set -- from the folder, or from any single image in it, which is
what makes picking one file in a file dialog select the acquisition -- and
reading the camera metadata out of ``metadata.txt``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import tifffile

from .tiff import ImageSource

METADATA_NAME = "metadata.txt"
# MM 1.4: img_<frame>_<channel>_<slice>.tif
MM_SINGLE = re.compile(r"^img_(\d+)_(.*)_(\d+)\.tiff?$", re.IGNORECASE)
DIGITS = re.compile(r"(\d+)")
OME_SERIES = re.compile(r"\.ome\.tiff?$", re.IGNORECASE)


def _order(path: Path):
    """Sort by the numbers in the name, so frame 9 comes before frame 10."""
    match = MM_SINGLE.match(path.name)
    if match:
        return (0, int(match.group(1)), int(match.group(3)), match.group(2))
    parts = DIGITS.split(path.name)
    return (1, tuple(int(p) if p.isdigit() else p for p in parts))


def single_image_files(folder) -> List[Path]:
    """The TIFFs of a one-file-per-frame acquisition, in acquisition order.

    Empty when ``folder`` is not one: an OME series, an NDTiff dataset or a
    directory with a single image in it are all read by something else, and
    claiming them here would take those paths away from the readers that
    understand their metadata.
    """
    folder = Path(folder)
    if not folder.is_dir() or (folder / "NDTiff.index").exists():
        return []
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in (".tif", ".tiff")]
    if len(files) < 2 or any(OME_SERIES.search(p.name) for p in files):
        return []
    files = sorted(files, key=_order)
    # one page per file is what *makes* a one-file-per-frame acquisition; a
    # folder of several ordinary stacks (run1.tif, run2.tif) is not one, and
    # reading it as one would silently fuse unrelated acquisitions.  Both ends
    # are checked, which costs two file opens however many images there are.
    for end in (files[0], files[-1]):
        try:
            with tifffile.TiffFile(end) as tf:
                if len(tf.pages) != 1:
                    return []
        except Exception:
            return []
    return files


def is_single_image_set(path) -> bool:
    """Whether ``path`` is such a folder, or one image inside one."""
    return bool(single_image_files(_folder(path)))


def _folder(path) -> Path:
    path = Path(path)
    return path if path.is_dir() else path.parent


def read_metadata_txt(folder) -> Tuple[Dict[str, object], Dict[str, object]]:
    """``(summary, first plane)`` from ``metadata.txt``, empty when absent.

    The file is one JSON object of a ``Summary`` and a ``FrameKey-f-c-s`` per
    plane.  An acquisition that was interrupted leaves it unterminated, so a
    plain `json.loads` is not enough: the summary and the first frame are read
    on their own, which is all the camera metadata needs.
    """
    path = Path(folder) / METADATA_NAME
    if not path.is_file():
        return {}, {}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return {}, {}
    try:
        whole = json.loads(text)
    except ValueError:
        whole = None
    if isinstance(whole, dict):
        summary = whole.get("Summary") or {}
        frames = [v for k, v in whole.items()
                  if k.startswith("FrameKey") and isinstance(v, dict)]
        return (summary if isinstance(summary, dict) else {},
                frames[0] if frames else {})
    return _first_objects(text)


def _first_objects(text: str) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Summary and first FrameKey out of a truncated metadata.txt."""
    out = []
    for key in ('"Summary"', '"FrameKey'):
        at = text.find(key)
        if at < 0:
            out.append({})
            continue
        start = text.find("{", text.find(":", at))
        depth, end = 0, -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        try:
            out.append(json.loads(text[start:end]) if end > 0 else {})
        except ValueError:
            out.append({})
    return out[0], out[1]


@dataclass
class SingleImageSource(ImageSource):
    """A folder of one-image-per-frame TIFFs, read as one acquisition.

    ``pages_per_file`` is how many planes each file holds -- one, in every
    real dataset -- and is what lets a frame in the middle be read by opening
    one file instead of walking forty thousand of them.  Set to 0 when the
    files disagree, which falls back to the base class's honest page walk.
    """

    folder: Optional[Path] = None
    pages_per_file: int = 1

    def _at(self, index: int) -> Tuple[Path, int]:
        return (self.files[index//self.pages_per_file],
                index % self.pages_per_file)

    def frame(self, index: int) -> np.ndarray:
        if not self.pages_per_file:
            return super().frame(index)
        if not 0 <= index < self.n_frames:
            raise IndexError(f"frame {index} beyond end of stack ({self.n_frames})")
        path, page = self._at(index)
        with tifffile.TiffFile(path) as tf:
            return tf.pages[page].asarray()

    def frames(self, chunk: int = 100, start: int = 0,
               stop: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
        if not self.pages_per_file:
            yield from super().frames(chunk, start, stop)
            return
        stop = self.n_frames if stop is None else min(stop, self.n_frames)
        for first in range(max(start, 0), stop, chunk):
            last = min(first+chunk, stop)
            yield first, np.stack([self.frame(i) for i in range(first, last)])

    def watch(self, chunk: int = 100, **kwargs) -> Iterator:
        raise NotImplementedError(
            "watching a one-file-per-frame folder is not supported: the "
            "watcher follows a growing file, and here each frame is its own")


def open_singles(path) -> SingleImageSource:
    """Open the acquisition ``path`` belongs to, folder or any image in it."""
    folder = _folder(path)
    files = single_image_files(folder)
    if not files:
        raise FileNotFoundError(f"no one-file-per-frame TIFFs in {folder}")
    summary, plane = read_metadata_txt(folder)

    # One page per file is the convention, but it is the frame count that the
    # rest of the program trusts, so check the ends rather than assume.
    with tifffile.TiffFile(files[0]) as tf:
        page = tf.pages[0]
        shape, dtype, per_file = tuple(page.shape), page.dtype, len(tf.pages)
        if not plane:
            tag = page.tags.get("MicroManagerMetadata")
            if tag is not None and isinstance(tag.value, dict):
                plane = tag.value
    with tifffile.TiffFile(files[-1]) as tf:
        last = len(tf.pages)
    n_frames = per_file*len(files)
    uniform = per_file
    if last != per_file:                      # a stopped acquisition, usually
        n_frames = per_file*(len(files)-1)+last
        uniform = 0                           # no index arithmetic, then

    declared = summary.get("Frames")
    return SingleImageSource(
        files=files, shape=shape, dtype=dtype, n_frames=n_frames,
        n_frames_declared=int(declared) if declared else None,
        mm_metadata=plane, summary=summary, folder=folder,
        pages_per_file=uniform,
    )
