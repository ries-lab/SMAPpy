"""Reading Micro-Manager / OME TIFF stacks and their metadata.

Micro-Manager writes one acquisition as a series of files
(``..._MMStack_Default.ome.tif``, ``..._1.ome.tif``, ...) and records in the
summary metadata how many frames were *planned*.  Acquisitions are usually
stopped early, so the declared frame count is larger than what was written.
We therefore never trust the declared length: frames are taken from the pages
actually present in each file, in acquisition order.

The loader yields chunks of frames so that downstream stages can work on
batches, and never requires the whole stack to be in memory or the source to
be finite -- the same interface works for online analysis later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import tifffile

from ..camera_db import Resolution
from ..frametags import tags_of_page
from ..metadata import CameraMetadata


@dataclass
class ImageSource:
    """A Micro-Manager acquisition, possibly split over several files."""

    files: List[Path]
    shape: Tuple[int, int]  # (height, width) of one frame
    dtype: np.dtype
    n_frames: int  # frames actually written
    n_frames_declared: Optional[int]  # what MM planned, if it said
    mm_metadata: Dict[str, object]  # per-plane metadata of the first frame
    summary: Dict[str, object]
    #: a `smappy.frametags.FrameTags` that `frames` hands each frame's
    #: metadata to, when set -- read with the pixels, so it costs no pass
    tags: Optional[object] = field(default=None, repr=False)
    #: per file, its frames' page offsets from Micro-Manager's index map (`_frame_offsets`), or None
    #: where it has none: a frame is then read where it is instead of after walking every page before it
    offsets: Optional[List[Optional[np.ndarray]]] = field(default=None, repr=False)

    def __len__(self) -> int:
        return self.n_frames

    # ------------------------------------------------------------------ frames
    def frames(self, chunk: int = 100, start: int = 0,
               stop: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(first_frame_index, block)`` with block shape (n, y, x).

        Frame indices are zero-based and continuous across the files of the
        acquisition.
        """
        buffer: List[np.ndarray] = []
        buffer_start = start

        stop = self.n_frames if stop is None else min(stop, self.n_frames)
        for page, index in self._pages(start, stop):
            buffer.append(page.asarray())
            if self.tags is not None:
                self.tags.add(index, tags_of_page(page))
            if len(buffer) == chunk:
                yield buffer_start, np.stack(buffer)
                buffer_start += len(buffer)
                buffer = []

        if buffer:
            yield buffer_start, np.stack(buffer)

    def _pages(self, start: int, stop: int):
        """``(page, frame index)`` of frames ``start`` to ``stop``: by the index map's offsets where
        `open_stack` found one -- a file before ``start`` not even opened, a page read where it is --
        else by walking each file's pages from its first."""
        index = 0
        for f, path in enumerate(self.files):
            if index >= stop:
                return
            offsets = self.offsets[f] if self.offsets is not None else None
            if offsets is not None and index + len(offsets) <= start:
                index += len(offsets)
                continue
            with tifffile.TiffFile(path) as tf:
                if offsets is not None:
                    first = max(0, start - index)
                    for k in range(first, min(len(offsets), stop - index)):
                        yield _page_at(tf, offsets[k], k), index + k
                    index += len(offsets)
                    continue
                for page in tf.pages:
                    if index >= stop:
                        break
                    if index >= start:
                        yield page, index
                    index += 1

    def watch(self, chunk: int = 100, **kwargs) -> Iterator[Tuple[int, np.ndarray]]:
        """Like `frames`, but for a file the microscope is still writing.

        Yields the frames already there and then whatever arrives, stopping
        when nothing new has for a while.  See
        :func:`smappy.io.watch.watch_stack` for the timing and the options;
        it is imported here rather than at the top because it is the one part
        of reading a stack that waits.
        """
        from .watch import watch_stack
        return watch_stack(self.files[0], chunk=chunk, **kwargs)

    def frame(self, index: int) -> np.ndarray:
        """Read a single frame (convenience for previews and tests)."""
        for start, block in self.frames(chunk=1, start=index, stop=index + 1):
            return block[0]
        raise IndexError(f"frame {index} beyond end of stack ({self.n_frames})")


def open_stack(path) -> ImageSource:
    """Open an acquisition: a TIFF series, an NDTiff dataset, or single images.

    Which one it is follows from what is there -- an NDTiff dataset is a
    directory with an ``NDTiff.index``, a one-file-per-frame acquisition is a
    folder of numbered single-page TIFFs -- so nothing above this has to know
    which format the microscope wrote.  In every case *any* file of the set
    stands for the set: picking one image in a file dialog opens the whole
    acquisition, which is the only thing a user could mean by it.
    """
    from .ndtiff import is_ndtiff, open_ndtiff   # NDTiff imports this module
    from .singles import is_single_image_set, open_singles
    from ..simulate.source import is_simulation, open_simulation

    if is_simulation(path):          # a recipe: frames made as they are read
        return open_simulation(path)
    from .dcimg import is_dcimg, open_dcimg
    from .quadrants import is_quadrant_mat, open_quadrant_frames
    if is_dcimg(path):               # Hamamatsu's own format: 4Pi microscopes
        return open_dcimg(path)
    if is_quadrant_mat(path):        # 4Pi quadrants, already cut, side by side
        return open_quadrant_frames(path)
    if is_ndtiff(path):
        return open_ndtiff(path)
    if is_single_image_set(path):
        return open_singles(path)

    path = Path(path)
    if path.is_dir():
        candidates = sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
        if not candidates:
            raise FileNotFoundError(f"no TIFF files in {path}")
        path = _first_of_series(candidates)

    files = _series_files(path)

    with tifffile.TiffFile(files[0]) as tf:
        page = tf.pages[0]
        shape, dtype = tuple(page.shape), page.dtype
        summary = {}
        mm = tf.micromanager_metadata or {}
        if isinstance(mm.get("Summary"), dict):
            summary = mm["Summary"]
        plane = {}
        tag = page.tags.get("MicroManagerMetadata")
        if tag is not None and isinstance(tag.value, dict):
            plane = tag.value

    # Counted from Micro-Manager's index map where there is one: walking the pages to count them read
    # every page's header, 28 s for a 34 GB acquisition of 314 000 frames on iCloud Drive (M2, 2026-09-17),
    # against 0.02 s for the eight maps
    n_frames, offsets = 0, []
    for k, f in enumerate(files):
        with tifffile.TiffFile(f) as tf:
            found = _frame_offsets(tf)
            n = len(found) if found is not None else len(tf.pages)
            # An acquisition stopped while writing leaves a last page that is not a frame (a DNA-PAINT
            # movie of M2's, 2026-09-17: shape ()); counted, it broke every block that reached it
            while k == len(files) - 1 and n and tuple(
                    (_page_at(tf, found[n - 1], n - 1) if found is not None else tf.pages[n - 1]).shape) != shape:
                n -= 1
            offsets.append(found[:n] if found is not None else None)
            n_frames += n

    declared = summary.get("Frames")
    return ImageSource(
        files=files, shape=shape, dtype=dtype, n_frames=n_frames,
        n_frames_declared=int(declared) if declared else None,
        mm_metadata=plane, summary=summary,
        offsets=offsets if any(o is not None for o in offsets) else None,
    )


def _frame_offsets(tf) -> Optional[np.ndarray]:
    """The page offsets of a Micro-Manager file's frames, in the order written, from the index map in its
    header; None when it has none or the map does not start at the first page.  Micro-Manager allocates
    the map for more frames than a file gets: the entries never written are 0."""
    if not getattr(tf, "is_micromanager", False):
        return None
    try:
        index = (tf.micromanager_metadata or {}).get("IndexMap")
    except Exception:
        return None
    if index is None or np.ndim(index) != 2 or not len(index):
        return None
    offsets = np.sort(np.asarray(index)[:, -1].astype(np.int64))
    offsets = offsets[offsets > 0]
    if not len(offsets) or int(offsets[0]) != int(tf.pages.first.offset):
        return None
    return offsets


def _page_at(tf, offset, index: int):
    """The page whose header is at ``offset`` -- what ``tf.pages[index]`` is, without reaching it through
    every page before it."""
    tf.filehandle.seek(int(offset))
    return tifffile.TiffPage(tf, index=int(index))


STACK_SUFFIXES = (".tif", ".tiff")      # ".ome.tif" ends in one of them


def is_image_stack(path) -> bool:
    """Whether `open_stack` would take ``path``, judged without opening it.

    By name for a file -- a TIFF or a simulation recipe -- and by what is in
    it for a folder: an NDTiff dataset (or any file of one), or TIFFs.  Cheap
    on purpose: a file dropped on a window is sorted with it before anything
    is read.
    """
    from .ndtiff import is_ndtiff
    from ..simulate.source import is_simulation

    path = Path(path)
    if path.is_dir():
        return is_ndtiff(path) or any(next(path.glob(f"*{s}"), None) is not None
                                      for s in STACK_SUFFIXES)
    return (path.name.lower().endswith(STACK_SUFFIXES) or is_simulation(path)
            or is_ndtiff(path))


def _first_of_series(paths: List[Path]) -> Path:
    """Pick the base file of an MM series (the one without a ``_<n>`` suffix)."""
    base = [p for p in paths if not re.search(r"_\d+\.ome\.tiff?$", p.name)]
    return (base or paths)[0]


def _series_files(first: Path) -> List[Path]:
    """All files of a Micro-Manager series, in acquisition order."""
    name = first.name
    m = re.match(r"(?P<stem>.+?)(?:_(?P<n>\d+))?\.ome\.tiff?$", name)
    if not m:
        return [first]

    stem = m.group("stem")
    found = {}
    for candidate in first.parent.glob(f"{stem}*.ome.tif*"):
        mm = re.match(rf"{re.escape(stem)}(?:_(\d+))?\.ome\.tiff?$", candidate.name)
        if mm:
            found[int(mm.group(1) or 0)] = candidate
    return [found[k] for k in sorted(found)] or [first]


# --------------------------------------------------------------------- metadata
def camera_tags(source: ImageSource) -> Dict[str, str]:
    """Every metadata key this acquisition carries.

    The summary under the plane's own tags: Micro-Manager writes the device
    properties once, at the start, and repeats a handful of them per image.
    The camera database looks things up by tag and does not care which of the
    two a tag came from -- but it does care that both are looked in, since the
    serial number that identifies the camera is only in the summary.
    """
    tags: Dict[str, str] = {}
    for source_tags in (source.summary or {}, source.mm_metadata or {}):
        for key, value in source_tags.items():
            # scalars and the odd already-parsed tuple (an NDTiff ROI); a
            # nested structure is not something a tag can be compared against
            if isinstance(value, (str, int, float, bool, tuple, list)):
                tags[str(key)] = value
    return tags


def _generic_values(tags: Dict[str, str]) -> Dict[str, object]:
    """What any Micro-Manager file says, without knowing the camera.

    The fallback for a camera that is not in the database, and the source of
    the values a camera entry does not describe.  It has to guess which key
    means what -- ``Gain`` is the EM gain on one camera and the pre-amp gain
    on another -- which is exactly what the database exists to stop doing, so
    the guesses lose to it wherever it has an answer.
    """
    device = str(tags.get("Core-Camera") or tags.get("Camera") or "").strip()

    def dev(key, default=None):
        return tags.get(f"{device}-{key}", default) if device else default

    port = dev("Port") or dev("Output_Amplifier")
    em_on = bool(port is not None
                 and str(port).strip() not in ("Normal", "Conventional"))
    pixelsize = _to_float(tags.get("PixelSizeUm"))
    exposure = _to_float(tags.get("Exposure-ms")) or _to_float(dev("Exposure"))
    # the frame time: the camera's own measure of its cycle, else the
    # interval a time-lapse was set to -- which Micro-Manager also writes,
    # as 0 or below the exposure, when it ran as fast as it could, and then
    # says nothing (the exposure stands in for it: `metadata.frame_time_ms`)
    interval = _to_float(tags.get("Interval_ms"))
    if not (interval and exposure and interval > exposure):
        interval = None
    return {
        "em_on": em_on,
        "emgain": (_to_float(dev("MultiplierGain")) or _to_float(dev("Gain"))
                   if em_on else 1.0),
        "offset": _to_float(dev("Offset")),
        # MM reports 0.0 for a pixel size nobody calibrated
        "pixelsize_um": pixelsize or None,
        "roi": _parse_roi(tags.get("ROI")),
        "exposure_ms": exposure,
        "frame_time_ms": _to_float(dev("ActualInterval-ms")) or interval,
        "camera_name": device or None,
    }


def _could_be(scope, tags: Dict[str, str], cameras) -> bool:
    """Whether the microscope's camera could have taken a file whose camera is not identified:
    recognised or chosen, yes; a stand-in, only if the file's Micro-Manager camera device is the
    one that camera is driven by (an iXon's ``Andor``), so a Prime95B is never read as an iXon."""
    if not scope.fallback:
        return True
    entry = cameras.get(scope.camera)
    device = str(tags.get("Core-Camera") or "").strip()
    return entry is not None and bool(device) and device == entry.prefix


def with_microscope(resolution: Resolution, found, pixel: bool = True) -> Resolution:
    """`resolution` with the file's microscope (a `MicroscopeResolution`) attached, and its
    pixel laid over the file's where the entry gives one for the devices' state (and `pixel`)."""
    from dataclasses import replace

    from ..camera_db import Source
    resolution = replace(resolution, microscope=found)
    pixel = found.pixel_nm if pixel else None
    if pixel is None:
        return resolution
    source = found.sources["pixel_nm"]
    return resolution.overlaid_with(
        {"pixelsize_um": [pixel[0] / 1000.0, pixel[1] / 1000.0]},
        Source("microscope", f"{found.name}, {source.detail}" if source.detail else found.name))


# where a simulation's camera came from: the camera window names it rather
# than calling a camera it knows exactly "not recognised"
SIMULATED = ("metadata", "the simulation")


def resolve_camera(source: ImageSource, camera: str = "",
                   overrides=None, microscope: str = "", fallback: str = "") -> Resolution:
    """Everything known about the camera behind this stack, and from where.

    Four layers, each winning over the one before it:

    1. what the camera database says this camera's parameters are -- the
       conversion for the readout mode it recognises, the pixel size, the
       baseline an iXon never reports;
    2. what the file itself says, because a value the acquisition recorded
       beats a value someone stored months ago;
    3. the pixel the microscope's entry gives for the state its devices were
       in (`microscope_db`): Micro-Manager's pixel size is not tied to the
       cylindrical lens, so M2's files say 127 nm square with the lens in,
       where the pixel is 127 x 116;
    4. what the user set, which beats all of them and is the end of the
       argument.

    `camera` names a database entry to use instead of identifying one, which
    is what a file whose camera carries no identifying tag needs;
    `microscope` names a microscope the same way; `fallback` is one to stand in
    when none is named or recognised (the last one used), marked as such
    (`MicroscopeResolution.fallback`).  A recognised (or named)
    microscope brings its camera: a file whose camera says nothing that
    identifies it is read with the camera the microscope is set up with.

    Every camera smappy resolves is resolved here: the fitters through
    `CameraSettings.resolution`, a script through `camera_metadata`.
    """
    from ..camera_db import Source, database

    tags = camera_tags(source)
    simulated = getattr(source, "camera", None)
    scope = None
    if not simulated:
        from ..microscope_db import database as microscopes
        known = camera or getattr(database().identify(tags), "name", "")
        scope = microscopes().resolve(tags, microscope=microscope, camera=known, channels=2,
                                      fallback=fallback)
        if scope.microscope is None:
            scope = None
        elif not camera and not known and _could_be(scope, tags, database()):
            camera = scope.camera          # the microscope's camera, for a file that names none
    resolution = database().resolve(tags, camera=camera)

    generic = _generic_values(tags)
    if resolution.camera is None:
        resolution = resolution.overlaid_with(
            generic, Source("metadata", "read without a camera entry"))
    else:
        # only over what the database supplied from its own store: a camera
        # entry that names a tag has already read the file, and better.  The
        # device name is not a parameter and always comes from the file.
        resolution = resolution.overlaid_with(
            {k: v for k, v in generic.items() if k != "camera_name"},
            Source("metadata", "from the file"), only_over=("fixed", "missing"))
        resolution = resolution.overlaid_with(
            {"camera_name": generic.get("camera_name")},
            Source("metadata", "the Micro-Manager device"))
    if simulated:
        # a simulation knows its camera exactly, which beats any guess
        resolution = resolution.overlaid_with(simulated, SIMULATED)
    elif scope is not None:
        # a stand-in microscope (the last one used) speaks for the pixel only if its camera
        # could have taken the file; else it names the optics and leaves the file's pixel
        resolution = with_microscope(resolution, scope,
                                     pixel=not scope.fallback or _could_be(scope, tags, database()))
    if overrides:
        values = overrides if isinstance(overrides, dict) else overrides.to_dict()
        resolution = resolution.overlaid_with(
            {k: v for k, v in values.items() if v is not None},
            Source("user", "your setting"))
    return resolution


def camera_metadata(source: ImageSource, overrides=None,
                    require: bool = True, camera: str = "",
                    microscope: str = "", fallback: str = "") -> CameraMetadata:
    """Camera metadata for a stack: the database, the file, then the user.

    `resolve_camera`'s values, for a script.  ``overrides`` may be a
    :class:`~smappy.metadata.CameraMetadata`, a dict, or a path to a YAML
    file of the same fields (``smappy-fit --camera CONFIG.yaml``), and is the
    last word on every value it sets -- so a camera that is in no database
    and a file that says nothing can still be fitted by stating the three
    numbers.  ``camera`` names a database entry to use instead of identifying
    one; a camera is added to the database with the camera editor.

    With ``require=True`` the result is checked for completeness, so a missing
    pixel size fails here rather than silently producing wrong nm coordinates.
    """
    if overrides is not None and not isinstance(overrides, (dict, CameraMetadata)):
        overrides = CameraMetadata.from_yaml(overrides)
    if isinstance(overrides, CameraMetadata):
        overrides = {k: v for k, v in overrides.to_dict().items() if v is not None}
    meta = resolve_camera(source, camera=camera, overrides=overrides,
                          microscope=microscope, fallback=fallback).camera_metadata()
    if require:
        meta.require()
    return meta


def _to_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_roi(value) -> Optional[Tuple[int, int, int, int]]:
    """Micro-Manager writes the ROI as ``"x-y-width-height"``.

    The separator is a hyphen, so the numbers must not be read as signed.
    """
    if value is None:
        return None
    parts = re.findall(r"\d+", str(value))
    if len(parts) != 4:
        return None
    return tuple(int(p) for p in parts)
