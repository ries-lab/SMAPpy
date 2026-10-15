"""Movies and z-stacks: the view rendered while one coordinate moves in steps.

A movie and a z-stack are one operation (*File/Export/Stack*): the picture
*Export Image* draws (`picture.render_picture`), once per step, with every
layer narrowed to a range of one column -- frames for a movie, z for a
stack of slices.  Nothing is rendered a second way, so a movie's frame is
the picture of the view with the layers' filters set to that frame range.

Along frames there are SMAP's *MovieRunningWindow*'s three numbers: a
*step* from one movie frame to the next, a *window* of camera frames each
one shows, and *cumulative*, from the first frame on.  A layer drawn in the
*tracks* mode can have a *tail* instead of the window: the frames before
"now" a track is drawn for, so each track grows up to the current frame
rather than being cut to the window.  Along z, slices of a thickness every
step.

The camera frames beside the picture are SMAP's *BlinkingMoviePresentation*:
the frames a fit kept (`smappy.rawframes`) on one side, the localizations
on the other.  A fit keeps a few dozen frames, not every one, so the camera
side shows the kept frame nearest to "now" and changes less often than the
other side.

Each movie frame is scaled by its own brightness, as SMAP's movie is: a
frame equals the picture of its window, and a cumulative movie stays
visible from its first frame rather than starting black.

Written as a TIFF stack (lossless, with the pixel size and the step, which
Fiji reads), an MP4 with H.264 (through ``imageio-ffmpeg``, a dependency) or a GIF (Pillow), frame by frame, so a movie longer than memory
can be written.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .layer_state import LayerSnapshot
from .picture import _join, render_picture, stamp
from .render import FieldOfView

FORMATS = {".tif": "tiff", ".tiff": "tiff", ".mp4": "mp4", ".gif": "gif"}
# H.264's constant rate factor per quality: 18 is close to visually
# lossless, 23 is x264's default, 28 a small file
QUALITY = {"high": 18, "medium": 23, "low": 28}
# bytes per pixel and frame, measured on movies of a simulated structure
# (600 x 600 pixels, 96 frames, a running window and a cumulative one: the
# MP4's 0.015 / 0.010 / 0.007 and 0.009 / 0.006 / 0.004, the GIF's 0.036 and
# 0.063).  Mostly black, so far below a photograph's; a dense picture is
# larger, which is why the form says "about"
MP4_BYTES_PER_PIXEL = {18: 0.015, 23: 0.010, 28: 0.006}
GIF_BYTES_PER_PIXEL = 0.05


@dataclass(frozen=True)
class Step:
    """One frame of a stack: the range of the column it shows, ``hi``
    exclusive; ``now`` is its end, where a track's tail is measured from."""

    lo: float
    hi: float

    @property
    def now(self) -> float:
        return self.hi


def frame_steps(first: int, last: int, step: int, window: Optional[int] = None,
                cumulative: bool = False) -> List[Step]:
    """The windows of a movie over frames ``first`` to ``last``: one every
    ``step`` frames, ``window`` frames long (the step when None), until a
    window reaches the last frame; from ``first`` on when ``cumulative``."""
    step = max(int(step), 1)
    window = step if window is None else max(int(window), 1)
    first, last = int(first), int(last)
    n = max(1, int(np.ceil((last + 1 - first - window) / step)) + 1)
    out = []
    for k in range(n):
        lo = first + k * step
        out.append(Step(float(first if cumulative else lo), float(lo + window)))
    return out


def z_steps(z_min: float, z_max: float, step: float,
            thickness: Optional[float] = None) -> List[Step]:
    """Slices ``thickness`` thick (the step when None), one every ``step``,
    from ``z_min`` until a slice reaches ``z_max``."""
    if not step > 0:
        raise ValueError("the z step must be above 0")
    thickness = step if thickness is None else float(thickness)
    n = max(1, int(np.ceil((z_max - z_min - thickness) / step - 1e-9)) + 1)
    return [Step(z_min + k * step, z_min + k * step + thickness) for k in range(n)]


def restrict(layers: Sequence[LayerSnapshot], column: str, step: Step,
             tail: Optional[float] = None) -> Tuple[LayerSnapshot, ...]:
    """The layers with only the localizations of ``step`` shown: their
    filters, and ``lo <= column < hi``.  A layer in the *tracks* mode shows
    ``now - tail <= column < now`` instead, when a ``tail`` is given.
    Image layers are left as they are."""
    out = []
    for layer in layers:
        if not layer.visible or layer.is_image or layer.locs is None:
            out.append(layer)
            continue
        if column not in layer.locs:
            raise KeyError(f"layer {layer.name!r} has no column {column!r}; it has "
                           f"{', '.join(layer.locs.columns)}")
        values = np.asarray(layer.locs[column], np.float64)
        lo = step.lo
        if tail is not None and layer.settings.mode == "tracks":
            lo = step.now - tail
        keep = (values >= lo) & (values < step.hi)
        mask = layer.mask
        mask = keep if mask is None else np.asarray(mask, bool) & keep[:len(mask)]
        out.append(replace(layer, mask=mask, known_indices=None, memo={}))
    return tuple(out)


def shown_range(layers: Sequence[LayerSnapshot], column: str) -> Tuple[float, float]:
    """The lowest and highest ``column`` among what the visible layers show."""
    lo, hi = np.inf, -np.inf
    for layer in layers:
        if not layer.visible or layer.is_image or layer.locs is None:
            continue
        if column not in layer.locs:
            raise KeyError(f"layer {layer.name!r} has no column {column!r}")
        values = np.asarray(layer.locs[column], np.float64)
        if layer.mask is not None:
            values = values[np.asarray(layer.mask, bool)[:len(values)]]
        values = values[np.isfinite(values)]
        if len(values):
            lo, hi = min(lo, values.min()), max(hi, values.max())
    if not np.isfinite(lo):
        raise ValueError("nothing shown to make a stack of")
    return float(lo), float(hi)


# --------------------------------------------------------------- the frames
def raw_plane(images, frame: float) -> Optional[Tuple[object, int]]:
    """The kept camera frame nearest to ``frame`` among ``images``
    (`ImageData` of kind "raw"), as ``(image, plane)``; never the average."""
    best = None
    for image in images:
        if image.frames is None:
            continue
        frames = np.asarray(image.frames)
        planes = np.flatnonzero(frames >= 0)
        if not len(planes):
            continue
        k = planes[np.argmin(np.abs(frames[planes] - frame))]
        distance = abs(frames[k] - frame)
        if best is None or distance < best[0]:
            best = (distance, image, int(k))
    return None if best is None else best[1:]


def raw_panel(image, plane: int, fov: FieldOfView) -> np.ndarray:
    """A camera frame on ``fov`` in grey, from its 1st to its 99.9th
    percentile: the background dark, the brightest spots just saturated."""
    values = image.resample(fov, plane).weight
    inside = values[values != 0]
    if not len(inside):
        return np.zeros(values.shape + (3,), np.uint8)
    lo, hi = np.percentile(inside, [1.0, 99.9])
    grey = np.clip((values - lo) / max(hi - lo, 1e-9), 0, 1)
    grey = np.round(grey * 255).astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=2)


def stack_frames(layers: Sequence[LayerSnapshot], fov: FieldOfView, steps: Sequence[Step],
                 column: str = "frame", what: str = "view", panels: Sequence = ("overlay",),
                 stacked: bool = False, tail: Optional[float] = None,
                 raw: Sequence = (), scale_bar: bool = True,
                 scale_bar_length: Optional[float] = None, colour_bars: bool = True,
                 labels: bool = True, text: Optional[Callable[[Step], str]] = None
                 ) -> Iterator[np.ndarray]:
    """Each step's picture as 8-bit RGB, in order: `render_picture` of the
    layers narrowed to the step (`restrict`), as the view or the composite,
    with the kept camera frame beside it when ``raw`` holds the images to
    take it from, and ``text(step)`` written in a corner."""
    white = any(l.display.white_background for l in layers if l.visible)
    for step in steps:
        shown = restrict(layers, column, step, tail)
        picture, = render_picture(shown, fov, what=what, panels=panels, stacked=stacked,
                                  scale_bar=scale_bar, scale_bar_length=scale_bar_length,
                                  colour_bars=colour_bars, labels=labels)
        image = picture.image
        if raw:
            found = raw_plane(raw, step.now - 1)
            camera = (raw_panel(found[0], found[1], fov) if found is not None
                      else np.zeros(image.shape, np.uint8))
            if labels:
                from .picture import _decorate
                camera, _ = _decorate(camera, fov, "camera", 0.0, picture.unit, (), False)
            image = _join([camera, image], False)
        if text is not None:
            image = stamp(image, text(step), white and not raw)
        yield image


# ---------------------------------------------------------------- the file
def kind_of(path) -> str:
    kind = FORMATS.get(Path(path).suffix.lower())
    if kind is None:
        raise ValueError(f"{Path(path).name}: the extension says the format -- "
                         f"{', '.join(sorted(FORMATS))}")
    return kind


def has_encoder() -> bool:
    """Whether MP4 can be written: ``imageio-ffmpeg``, with its ffmpeg."""
    try:
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
        return True
    except Exception:
        return False


def estimate_size(kind: str, height: int, width: int, n: int,
                  quality: str = "medium") -> float:
    """About how many bytes the file will be: exact for a TIFF, a measured
    rate for MP4 and GIF, which depends on what is drawn."""
    pixels = float(height) * float(width) * max(int(n), 1)
    if kind == "tiff":
        return 3 * pixels
    if kind == "gif":
        return GIF_BYTES_PER_PIXEL * pixels
    return MP4_BYTES_PER_PIXEL[QUALITY[quality]] * pixels


def write_stack(path, frames: Iterable[np.ndarray], n: int, fps: float = 10.0,
                quality: str = "medium", pixel_nm: Optional[float] = None,
                interval: Optional[float] = None, along: str = "frames",
                description: str = "") -> Path:
    """Write ``n`` frames to ``path``, the format by its extension.

    A TIFF is an ImageJ hyperstack -- frames along T, or slices along Z
    with their spacing ``interval`` (nm) -- with the pixel size and, along
    frames, ``interval`` seconds between frames when it is known; the
    description goes into ImageJ's *Info*.  An MP4 is H.264 at a
    ``quality`` of `QUALITY`, its sides made even; a GIF loops.
    """
    path = Path(path)
    kind = kind_of(path)
    frames = iter(frames)
    try:
        first = next(frames)
    except StopIteration:
        raise ValueError("no frames to write") from None

    def all_frames():
        yield first
        yield from frames

    if kind == "tiff":
        import tifffile
        meta = {"axes": ("TYXS" if along == "frames" else "ZYXS"), "unit": "um"}
        if description:
            meta["Info"] = description
        if interval:
            meta["finterval" if along == "frames" else "spacing"] = (
                interval if along == "frames" else interval / 1000.0)
        extra = {}
        if pixel_nm:
            per_um = 1000.0 / pixel_nm
            extra["resolution"] = (per_um, per_um)
        tifffile.imwrite(str(path), data=all_frames(), shape=(n,) + first.shape,
                         dtype=np.uint8, imagej=True, photometric="rgb",
                         metadata=meta, **extra)
    elif kind == "mp4":
        _write_mp4(path, all_frames(), first.shape, fps, quality, description)
    else:
        from PIL import Image
        images = [Image.fromarray(f).convert("P", palette=Image.Palette.ADAPTIVE)
                  for f in all_frames()]
        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=max(int(round(1000.0 / fps)), 10), loop=0,
                       optimize=False, disposal=1,
                       comment=description.encode() if description else b"")
    return path


def _write_mp4(path: Path, frames, shape, fps, quality, description) -> None:
    try:
        import imageio_ffmpeg
    except ImportError:
        raise ImportError("an MP4 is written by imageio-ffmpeg: pip install "
                          "imageio-ffmpeg") from None
    h, w = shape[:2]
    H, W = h + h % 2, w + w % 2           # H.264 in yuv420p wants even sides
    params = ["-crf", str(QUALITY[quality]), "-preset", "medium",
              "-movflags", "+faststart"]
    if description:
        params += ["-metadata", f"comment={description}"]
    writer = imageio_ffmpeg.write_frames(str(path), (W, H), fps=float(fps), codec="libx264",
                                         pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                         macro_block_size=1, quality=None,
                                         output_params=params)
    writer.send(None)
    try:
        for frame in frames:
            if (H, W) != (h, w):
                padded = np.zeros((H, W, 3), np.uint8)
                padded[:h, :w] = frame
                frame = padded
            writer.send(np.ascontiguousarray(frame, np.uint8).tobytes())
    finally:
        writer.close()
