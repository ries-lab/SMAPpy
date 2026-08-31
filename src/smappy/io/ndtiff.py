"""Reading NDTiff datasets (Micro-Manager / pycro-manager NDTiffStorage).

An NDTiff dataset is a directory holding ``NDTiff.index`` and one or more
``*NDTiffStack*.tif`` files.  The index is a flat binary table with one record
per image giving the file it sits in and the byte offset of its pixels, so an
image is read by seeking and the TIFF page structure is never walked: opening
costs about 7 us per frame against the ~120 us a page walk takes.

This is a port of SMAP's MATLAB loader (``shared/imageloaders/``:
`readNDTiffIndex.m` and `imageloaderNDTiff.m`), which is where the awkward parts
were worked out: the table is zero-padded at the end, the pixel size is more
reliably derived from where the metadata starts than from the declared type, and
an interrupted acquisition leaves index records describing images that were
never written.

Because the index grows one record per finished image, it is also the natural
thing to watch during an acquisition: `NDTiffSource.watch` re-reads it and hands
on whatever has appeared, with none of the care a growing TIFF page chain needs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .tiff import ImageSource, _parse_roi

INDEX_NAME = "NDTiff.index"
#: the summary JSON sits in the header of every stack file, after this marker
SUMMARY_OFFSET = 20
SUMMARY_MARKER = 2355492


def is_ndtiff(path) -> bool:
    """Whether ``path`` is (or is inside) an NDTiff dataset directory."""
    return _dataset_folder(path) is not None


# ------------------------------------------------------------------- the index
@dataclass
class NDTiffIndex:
    """One row per image: where its pixels are, and how big they are."""

    folder: Path
    files: List[str]
    file_index: np.ndarray      # into ``files``
    pixel_offset: np.ndarray    # byte offset of the pixels within that file
    width: np.ndarray
    height: np.ndarray
    bytes_per_pixel: np.ndarray
    md_offset: np.ndarray
    md_length: np.ndarray
    axes: List[dict]

    def __len__(self) -> int:
        return len(self.pixel_offset)

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.height[0]), int(self.width[0])

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.uint8 if self.bytes_per_pixel[0] == 1 else np.uint16)

    def path(self, i: int) -> Path:
        return self.folder / self.files[int(self.file_index[i])]


def read_index(folder) -> NDTiffIndex:
    """Parse ``NDTiff.index``.

    Each record is, little-endian: ``axesLength``, ``axes`` (JSON),
    ``filenameLength``, ``filename``, then eight ``uint32`` -- ``pixelOffset``,
    ``width``, ``height``, ``pixelType``, ``pixelCompression``,
    ``metadataOffset``, ``metadataLength``, ``metadataCompression``.  The table
    is zero-padded, so parsing stops at the first ``axesLength == 0``.
    """
    folder = Path(folder)
    raw = (folder / INDEX_NAME).read_bytes()
    n = len(raw)

    files: List[str] = []
    by_name: Dict[str, int] = {}
    file_index, offsets, widths, heights = [], [], [], []
    types, md_offsets, md_lengths, axes = [], [], [], []

    p = 0
    while p + 4 <= n:
        alen = int(np.frombuffer(raw, np.uint32, 1, p)[0]); p += 4
        if alen == 0 or p + alen > n:       # the zero padding at the end
            break
        axes_text = raw[p:p + alen].decode("utf-8", "replace"); p += alen
        if p + 4 > n:
            break
        flen = int(np.frombuffer(raw, np.uint32, 1, p)[0]); p += 4
        if flen == 0 or p + flen > n:
            break
        name = raw[p:p + flen].decode("utf-8", "replace"); p += flen
        if p + 32 > n:
            break
        v = np.frombuffer(raw, np.uint32, 8, p); p += 32

        index = by_name.get(name)
        if index is None:
            index = by_name[name] = len(files)
            files.append(name)
        file_index.append(index)
        offsets.append(int(v[0])); widths.append(int(v[1]))
        heights.append(int(v[2])); types.append(int(v[3]))
        md_offsets.append(int(v[5])); md_lengths.append(int(v[6]))
        axes.append(_axes(axes_text))
        if v[4] or v[7]:
            raise ValueError(f"{folder / INDEX_NAME}: compressed NDTiff images "
                             f"are not supported")

    if not offsets:
        raise ValueError(f"no images listed in {folder / INDEX_NAME}")

    index = NDTiffIndex(
        folder=folder, files=files,
        file_index=np.array(file_index, np.int64),
        pixel_offset=np.array(offsets, np.int64),
        width=np.array(widths, np.int64), height=np.array(heights, np.int64),
        bytes_per_pixel=_bytes_per_pixel(np.array(types, np.int64),
                                         np.array(offsets, np.int64),
                                         np.array(md_offsets, np.int64),
                                         np.array(widths, np.int64),
                                         np.array(heights, np.int64)),
        md_offset=np.array(md_offsets, np.int64),
        md_length=np.array(md_lengths, np.int64),
        axes=axes,
    )
    index = _sorted_by_time(index)
    return _truncated_to_what_was_written(index)


def _axes(text: str) -> dict:
    try:
        axes = json.loads(text)
        return axes if isinstance(axes, dict) else {}
    except ValueError:
        return {}


def _bytes_per_pixel(pixel_type, offset, md_offset, width, height) -> np.ndarray:
    """Two, unless the metadata starts exactly one byte per pixel later.

    The declared ``pixelType`` has been seen to disagree with what was actually
    written; the distance to the metadata, which directly follows the pixels,
    has not.
    """
    guess = np.where(pixel_type == 0, 1, 2).astype(np.int64)
    gap = int(md_offset[0]) - int(offset[0])
    pixels = int(width[0]) * int(height[0])
    if gap == pixels:
        guess[:] = 1
    elif gap == 2 * pixels:
        guess[:] = 2
    return guess


def _sorted_by_time(index: NDTiffIndex) -> NDTiffIndex:
    """Frames in acquisition order, if the index says what that is."""
    times = np.array([a.get("time", -1) for a in index.axes], np.int64)
    if (times < 0).any() or np.all(np.diff(times) >= 0):
        return index
    order = np.argsort(times, kind="stable")
    return _take(index, order)


def _truncated_to_what_was_written(index: NDTiffIndex) -> NDTiffIndex:
    """Drop records whose pixels are not (yet, or ever) in the file.

    An acquisition stopped mid-write, or one that ran into NDTiff's 4 GB
    per-file limit, leaves the index describing more images than exist.  The
    same test makes reading a dataset that is still being written safe: a record
    is only used once the bytes it points at are there.
    """
    sizes = {name: (index.folder / name).stat().st_size
             if (index.folder / name).exists() else 0 for name in index.files}
    limit = np.array([sizes[index.files[i]] for i in index.file_index], np.int64)
    end = index.pixel_offset + index.width * index.height * index.bytes_per_pixel
    complete = end <= limit
    if complete.all():
        return index
    n = int(np.argmin(complete))          # keep the leading run, not the gaps
    return _take(index, np.arange(n))


def _take(index: NDTiffIndex, which) -> NDTiffIndex:
    which = np.asarray(which, np.int64)
    return NDTiffIndex(
        folder=index.folder, files=index.files,
        file_index=index.file_index[which], pixel_offset=index.pixel_offset[which],
        width=index.width[which], height=index.height[which],
        bytes_per_pixel=index.bytes_per_pixel[which],
        md_offset=index.md_offset[which], md_length=index.md_length[which],
        axes=[index.axes[i] for i in which],
    )


# ------------------------------------------------------------------ the source
@dataclass
class NDTiffSource(ImageSource):
    """An NDTiff dataset, read through its index.

    The same interface as :class:`~smappy.io.tiff.ImageSource`, so nothing
    downstream knows the difference.
    """

    index: NDTiffIndex = None
    _maps: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    # ----------------------------------------------------------------- reading
    def frame(self, i: int) -> np.ndarray:
        if not 0 <= i < len(self.index):
            raise IndexError(f"frame {i} beyond end of stack ({self.n_frames})")
        data = self._map(self.index.path(i))
        start = int(self.index.pixel_offset[i])
        h, w = int(self.index.height[i]), int(self.index.width[i])
        stop = start + h * w * int(self.index.bytes_per_pixel[i])
        dtype = np.uint8 if self.index.bytes_per_pixel[i] == 1 else np.uint16
        return data[start:stop].view(dtype).reshape(h, w)

    def frames(self, chunk: int = 100, start: int = 0,
               stop: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
        stop = len(self.index) if stop is None else min(stop, len(self.index))
        for first in range(start, stop, chunk):
            last = min(first + chunk, stop)
            yield first, np.stack([self.frame(i) for i in range(first, last)])

    def watch(self, chunk: int = 100, settings=None, start: int = 0,
              stop: Optional[int] = None, stop_event=None, on_wait=None
              ) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield blocks as the acquisition writes them.

        The index gains a record only once an image is complete, so following it
        needs none of the care a growing TIFF does -- re-read it, and read what
        is new.  It stops when nothing has arrived for ``settings.timeout``,
        which is how an acquisition ends.
        """
        from .watch import WatchSettings, _sleep, _stopped

        settings = settings or WatchSettings()
        index = start
        last_new = time.monotonic()
        while not _stopped(stop_event):
            self.reload()
            available = len(self.index) if stop is None else min(stop, len(self.index))
            while index < available and not _stopped(stop_event):
                last = min(index + chunk, available)
                yield index, np.stack([self.frame(i) for i in range(index, last)])
                index = last
                last_new = time.monotonic()
            if stop is not None and index >= stop:
                break
            waited = time.monotonic() - last_new
            if waited >= settings.timeout:
                break
            if on_wait is not None:
                on_wait(waited)
            _sleep(settings.poll, stop_event)

    # -------------------------------------------------------------- refreshing
    def reload(self) -> "NDTiffSource":
        """Re-read the index; new frames appear in ``n_frames``."""
        try:
            self.index = read_index(self.index.folder)
        except (OSError, ValueError):
            return self         # being written to: what we have still stands
        self.n_frames = len(self.index)
        self.files = [self.index.folder / name for name in self.index.files]
        return self

    def _map(self, path: Path) -> np.ndarray:
        """A memory map of one stack file, kept for as long as the source is.

        Mapping is per file, not per frame, and a map made before the file grew
        stays valid for the bytes it already covered -- so a file that is still
        being written is remapped when the index says it has grown.
        """
        key = str(path)
        mapped = self._maps.get(key)
        if mapped is None or len(mapped) < path.stat().st_size:
            mapped = self._maps[key] = np.memmap(path, np.uint8, mode="r")
        return mapped


def open_ndtiff(path) -> NDTiffSource:
    """Open an NDTiff dataset directory (or any file inside one)."""
    folder = _dataset_folder(path)
    if folder is None:
        raise FileNotFoundError(f"no {INDEX_NAME} at {path}")
    index = read_index(folder)
    source = NDTiffSource(
        files=[folder / name for name in index.files],
        shape=index.shape, dtype=index.dtype, n_frames=len(index),
        n_frames_declared=None, mm_metadata={}, summary={}, index=index,
    )
    source.summary = _summary(source.files[0])
    source.mm_metadata = _image_metadata(source, index)
    declared = source.summary.get("Frames")
    source.n_frames_declared = int(declared) if declared else None
    return source


def _dataset_folder(path) -> Optional[Path]:
    path = Path(path)
    for folder in (path, path.parent):
        if (folder / INDEX_NAME).is_file():
            return folder
    return None


# ---------------------------------------------------------------- the metadata
def _summary(path: Path) -> dict:
    """The summary JSON from the header of a stack file."""
    try:
        with open(path, "rb") as fh:
            fh.seek(SUMMARY_OFFSET)
            marker, length = np.frombuffer(fh.read(8), np.uint32, 2)
            if marker != SUMMARY_MARKER or not length:
                return {}
            return _json(fh.read(int(length)))
    except (OSError, ValueError):
        return {}


def _image_metadata(source: NDTiffSource, index: NDTiffIndex) -> dict:
    """The first image's metadata, flattened into Micro-Manager's key space.

    `metadata_from_stack` reads plane metadata keys like ``Core-Camera`` and
    ``<device>-Offset``, so an NDTiff dataset has to present the same flat
    dictionary a Micro-Manager TIFF does.  The summary fills in behind it.
    """
    flat = dict(_flatten(_summary_defaults(source.summary)))
    flat.update(_flatten(_frame_metadata(source, index, 0)))
    flat.setdefault("Format", "NDTiff")
    if "ROI" in flat:
        flat["ROI"] = _parse_roi(flat["ROI"]) or flat["ROI"]
    return flat


def _summary_defaults(summary: dict) -> dict:
    """Summary keys that the plane metadata of an MM TIFF would carry."""
    out = dict(summary)
    for key, alias in (("PixelSize_um", "PixelSizeUm"),
                       ("PixelSizeUm", "PixelSizeUm"),
                       ("Core-Camera", "Core-Camera")):
        if key in summary:
            out.setdefault(alias, summary[key])
    return out


def _frame_metadata(source: NDTiffSource, index: NDTiffIndex, i: int) -> dict:
    """The per-image JSON of frame ``i``."""
    length = int(index.md_length[i])
    if length <= 0:
        return {}
    data = source._map(index.path(i))
    start = int(index.md_offset[i])
    return _json(bytes(data[start:start + length]))


def _json(raw: bytes) -> dict:
    try:
        value = json.loads(raw.decode("utf-8", "replace").rstrip("\x00 \t\r\n"))
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _flatten(d: dict, prefix: str = "") -> Iterator[Tuple[str, object]]:
    """Nested JSON as flat ``device-property`` keys, the way MM writes them."""
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _flatten(value, f"{name}-")
        else:
            yield name, value
