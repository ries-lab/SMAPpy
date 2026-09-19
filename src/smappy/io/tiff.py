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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import tifffile

from ..camera_db import Resolution
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

    def __len__(self) -> int:
        return self.n_frames

    # ------------------------------------------------------------------ frames
    def frames(self, chunk: int = 100, start: int = 0,
               stop: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(first_frame_index, block)`` with block shape (n, y, x).

        Frame indices are zero-based and continuous across the files of the
        acquisition.
        """
        index = 0
        buffer: List[np.ndarray] = []
        buffer_start = start

        for path in self.files:
            with tifffile.TiffFile(path) as tf:
                for page in tf.pages:
                    if stop is not None and index >= stop:
                        break
                    if index < start:
                        index += 1
                        continue
                    buffer.append(page.asarray())
                    if len(buffer) == chunk:
                        yield buffer_start, np.stack(buffer)
                        buffer_start += len(buffer)
                        buffer = []
                    index += 1
            if stop is not None and index >= stop:
                break

        if buffer:
            yield buffer_start, np.stack(buffer)

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

    n_frames = 0
    for f in files:
        with tifffile.TiffFile(f) as tf:
            n_frames += len(tf.pages)

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

    declared = summary.get("Frames")
    return ImageSource(
        files=files, shape=shape, dtype=dtype, n_frames=n_frames,
        n_frames_declared=int(declared) if declared else None,
        mm_metadata=plane, summary=summary,
    )


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
    return {
        "em_on": em_on,
        "emgain": (_to_float(dev("MultiplierGain")) or _to_float(dev("Gain"))
                   if em_on else 1.0),
        "offset": _to_float(dev("Offset")),
        # MM reports 0.0 for a pixel size nobody calibrated
        "pixelsize_um": pixelsize or None,
        "roi": _parse_roi(tags.get("ROI")),
        "exposure_ms": (_to_float(tags.get("Exposure-ms"))
                        or _to_float(dev("Exposure"))),
        "camera_name": device or None,
    }


def resolve_camera(source: ImageSource, camera: str = "", presets=None,
                   overrides=None) -> Resolution:
    """Everything known about the camera behind this stack, and from where.

    Three layers, each winning over the one before it:

    1. what the camera database says this camera's parameters are -- the
       conversion for the readout mode it recognises, the pixel size, the
       baseline an iXon never reports;
    2. what the file itself says, because a value the acquisition recorded
       beats a value someone stored months ago;
    3. what the user set, which beats both and is the end of the argument.

    `camera` names a database entry to use instead of identifying one, which
    is what a file whose camera carries no identifying tag needs.
    """
    from ..camera_db import Source, database

    tags = camera_tags(source)
    known = database()
    if presets is not None:
        known = known.joined_with(_as_database(presets))
    resolution = known.resolve(tags, camera=camera)

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
    if overrides:
        values = overrides if isinstance(overrides, dict) else overrides.to_dict()
        resolution = resolution.overlaid_with(
            {k: v for k, v in values.items() if v is not None},
            Source("user", "your setting"))
    return resolution


def _as_database(presets):
    """A camera database from whatever was passed: one, or a SMAP ``.mat``."""
    from ..camera_db import CameraDatabase
    if isinstance(presets, CameraDatabase):
        return presets
    path = Path(presets)
    if path.suffix.lower() == ".mat":
        from .cameras_mat import to_database
        return to_database(path)
    return CameraDatabase.load(path)


def metadata_from_stack(source: ImageSource, presets=None) -> CameraMetadata:
    """What the file and the camera database know about this acquisition.

    ``presets`` adds a camera database beyond the shipped one -- another JSON
    file, or a SMAP ``*_cameras.mat``, which is converted on the spot.
    """
    return resolve_camera(source, presets=presets).camera_metadata()


def camera_metadata(source: ImageSource, presets=None, overrides=None,
                    require: bool = True, camera: str = "") -> CameraMetadata:
    """Camera metadata for a stack: the database, the file, then the user.

    ``overrides`` may be a :class:`~smappy.metadata.CameraMetadata`, a dict, or
    a path to a YAML file, and is the last word on every value it sets -- so a
    camera that is in no database and a file that says nothing can still be
    fitted by stating the three numbers.  ``presets`` adds a camera database
    beyond the shipped one (another JSON file, or a SMAP ``*_cameras.mat``),
    and ``camera`` names an entry to use instead of identifying one.

    With ``require=True`` the result is checked for completeness, so a missing
    pixel size fails here rather than silently producing wrong nm coordinates.
    """
    if overrides is not None and not isinstance(overrides, (dict, CameraMetadata)):
        overrides = CameraMetadata.from_yaml(overrides)
    if isinstance(overrides, CameraMetadata):
        overrides = {k: v for k, v in overrides.to_dict().items() if v is not None}
    meta = resolve_camera(source, camera=camera, presets=presets,
                          overrides=overrides).camera_metadata()
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
