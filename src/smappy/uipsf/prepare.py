"""What uiPSF is handed: photons, in channels, cut as it wants them.

smappy reads the data rather than uiPSF's loader, so that a bead stack's z
positions, the camera's gain and offset and the split of a two-colour camera
are found exactly as they are for smappy's own calibration and fit -- and the
transformation uiPSF learns can be put back into chip coordinates, which only
works if this side knows exactly how the halves were cut.

uiPSF's arrays:

* beads, one channel: ``(stacks, z, y, x)``; two: ``(2, stacks, z, y, x)``
* blinking, one channel: ``(frames, y, x)``; two: ``(2, frames, y, x)``

with the main channel first (``ref_channel = 0``) and the secondary flipped
back to the main channel's orientation when the splitter mirrors it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class Split:
    """How a split camera frame was cut into the two channels uiPSF sees.

    The halves are cut to the same length either side of ``split_position``
    -- ``[split-m, split)`` and ``[split, split+m)`` along the split axis --
    because uiPSF stacks the two channels into one array; the secondary is
    flipped along that axis when the layout is mirrored.  `to_chip` undoes
    exactly this, which is what turns uiPSF's channel transformation back
    into chip pixels.
    """
    layout: str
    main_channel: str
    split_position: int
    image_shape: tuple          # (height, width) of the whole frame
    length: int                 # m, each half's extent along the split axis
    origin: tuple = (0, 0)      # the frame's (x, y) on the chip, or (0, 0)

    @property
    def axis(self) -> int:
        """The array axis of a (y, x) frame that is split."""
        return 1 if "right-left" in self.layout else 0

    @property
    def mirrored(self) -> bool:
        return "mirrored" in self.layout

    @property
    def main_first(self) -> bool:
        return self.main_channel in ("left", "upper")

    def starts(self):
        """Where (main, secondary) begin along the split axis."""
        low, high = self.split_position - self.length, self.split_position
        return (low, high) if self.main_first else (high, low)

    def cut(self, images: np.ndarray) -> np.ndarray:
        """``(..., y, x)`` -> ``(2, ..., y', x')``, main first, secondary unmirrored."""
        out = []
        for channel, start in enumerate(self.starts()):
            index = [slice(None)] * images.ndim
            index[images.ndim - 2 + self.axis] = slice(start, start + self.length)
            half = images[tuple(index)]
            if channel == 1 and self.mirrored:
                half = np.flip(half, axis=images.ndim - 2 + self.axis)
            out.append(half)
        return np.stack(out)

    def to_chip(self, channel: int, yx) -> np.ndarray:
        """Pixel (y, x) positions in channel ``channel``'s array -> chip (x, y)."""
        yx = np.array(yx, dtype=float, copy=True)
        along = yx[:, self.axis]
        if channel == 1 and self.mirrored:
            along = self.length - 1 - along
        yx[:, self.axis] = along + self.starts()[channel]
        return yx[:, ::-1] + np.asarray(self.origin, float)

    def geometry(self, sources: Sequence[dict] = ()) -> dict:
        """The split as `smappy.calibrate.dual` records it with a calibration."""
        coord_axis = 1 - self.axis
        return {"layout": self.layout, "main_channel": self.main_channel,
                "split_position": int(self.split_position),
                "image_shape": list(self.image_shape),
                "coordinate_system": "camera-chip" if any(
                    s.get("roi") is not None for s in sources) else "roi-local",
                "sources": list(sources),
                "mirror_axis_xy": coord_axis if self.mirrored else None,
                "convention": "zero-based pixel centers; split is first index of second half"}


def make_split(shape, dual: dict, origin=(0, 0)) -> Split:
    """The `Split` of a frame of ``shape`` for a profile's ``dual`` section."""
    from ..calibrate.dual import DualColorSettings
    settings = DualColorSettings(layout=dual.get("layout", "up-down mirrored"),
                                 main_channel=dual.get("main_channel", "upper"),
                                 split_position=dual.get("split_position"))
    settings.validate()
    shape = tuple(int(n) for n in shape[-2:])
    axis = 1 if "right-left" in settings.layout else 0
    split = settings.split_position or shape[axis] // 2
    if not 0 < split < shape[axis]:
        raise ValueError(f"the split at {split} px is outside the {shape[axis]} px frame")
    return Split(settings.layout, settings.main_channel, int(split), shape,
                 min(split, shape[axis] - split), tuple(origin))


def channel_shift(channels: np.ndarray) -> list:
    """How far the secondary channel's image lies from the main's, [y, x] px.

    uiPSF pairs the two channels' beads from a first guess at this shift,
    which it finds by matching bead coordinates -- and on a regular pattern
    of beads that locked onto the neighbouring bead, 30 px off, leaving one
    pair.  The cross-correlation of the two channels' projections peaks where
    *all* the beads overlap, which a neighbour's shift cannot match.  Per
    channel, target minus reference, [y, x]: what uiPSF's multi-channel data
    calls ``shiftxy``, and what `worker.py` sets before uiPSF pairs the beads.
    """
    from scipy import ndimage
    ny, nx = channels.shape[-2:]
    a, b = (np.asarray(c, float).reshape(-1, ny, nx).max(axis=0) for c in channels)
    a, b = a - np.median(a), b - np.median(b)
    # b(r + d) = a(r) where this peaks: the secondary is the main shifted by d
    correlation = np.fft.ifft2(np.conj(np.fft.fft2(a)) * np.fft.fft2(b)).real
    correlation = np.fft.fftshift(ndimage.gaussian_filter(np.fft.fftshift(correlation), 1))
    dy, dx = np.unravel_index(np.argmax(correlation), correlation.shape)
    dy = dy - correlation.shape[0] if dy > correlation.shape[0] // 2 else dy
    dx = dx - correlation.shape[1] if dx > correlation.shape[1] // 2 else dx
    return [[0.0, 0.0], [float(dy), float(dx)]]


def to_photons(adu: np.ndarray, camera) -> np.ndarray:
    """Camera counts -> photons, as the fitter converts them."""
    return ((np.asarray(adu, np.float32) - float(camera.offset))
            * float(camera.adu_to_photons)).astype(np.float32)


def resolve_camera(camera_settings, source, profile):
    """The camera, with the profile's pixel size where nothing else gave one."""
    from ..io.tiff import camera_metadata
    from ..metadata import CameraMetadata
    overrides = camera_settings.overrides() if camera_settings is not None else {}
    if camera_settings is not None and camera_settings.preset:
        from dataclasses import asdict
        preset = asdict(CameraMetadata.from_yaml(camera_settings.preset))
        overrides = {**{k: v for k, v in preset.items() if v is not None}, **overrides}
    name = camera_settings.camera if camera_settings is not None else ""
    if source is not None:
        meta = camera_metadata(source, overrides=overrides, camera=name, require=False)
    else:
        meta = CameraMetadata.from_dict(overrides)
    if meta.pixelsize_um is None and profile.pixelsize_xy_um is not None:
        meta.pixelsize_um = list(profile.pixelsize_xy_um)
    meta.require()
    return meta


def pixel_size_xy_um(camera) -> tuple:
    p = camera.pixelsize_um
    if isinstance(p, (list, tuple, np.ndarray)):
        return float(p[0]), float(p[-1])
    return float(p), float(p)


@dataclass
class BeadInput:
    images: np.ndarray              # photons, as uiPSF wants them
    dz_nm: float
    camera: object
    sources: list                   # per stack: path, roi, shape, z
    split: Optional[Split] = None


def read_beads(paths, camera_settings, profile, dual: bool, dz_nm: Optional[float] = None,
               progress=None) -> BeadInput:
    """Bead z-stacks, from files or `BeadStack`s, as uiPSF's photons.

    Stacks of different lengths are cut to the shortest, about their centres:
    uiPSF stacks them into one array, and fits each bead's z itself.
    """
    from ..calibrate.input import BeadStack, discover_acquisitions, read_bead_stacks
    from ..io.tiff import open_stack
    report = progress or (lambda text: None)
    if isinstance(paths, (str, BeadStack)):
        paths = [paths]
    stacks, first_path = [], None
    for item in paths:
        if isinstance(item, BeadStack):
            stacks.append(item)
            continue
        for acquisition in discover_acquisitions([item]):
            first_path = first_path or acquisition
            stacks.extend(read_bead_stacks(acquisition, dz_nm))
    if not stacks:
        raise ValueError("no bead stacks: choose the files of a z-stack acquisition")
    report(f"{len(stacks)} bead stacks")
    shapes = {s.images.shape[-2:] for s in stacks}
    if len(shapes) > 1:
        raise ValueError(f"the bead stacks differ in frame size {sorted(shapes)}; "
                         "uiPSF needs them alike: calibrate them separately")
    steps = np.array([s.dz for s in stacks])
    if dz_nm is None and np.ptp(steps) > 0.01 * np.median(steps):
        raise ValueError(f"the stacks have different z steps {sorted(set(steps.round(1)))} nm")
    dz = float(dz_nm or np.median(steps))
    nz = min(len(s.images) for s in stacks)
    cut = [s.images[(len(s.images) - nz) // 2:(len(s.images) - nz) // 2 + nz] for s in stacks]
    source = None
    if first_path is not None:
        try:
            source = open_stack(first_path)
        except Exception:               # noqa: BLE001 -- the user's camera fields remain
            source = None
    camera = resolve_camera(camera_settings, source, profile)
    if camera.roi is None and stacks[0].roi is not None:
        camera.roi = stacks[0].roi
    images = to_photons(np.stack(cut), camera)
    sources = [{"path": str(s.source), "roi": list(s.roi) if s.roi is not None else None,
                "shape": list(s.images.shape), "z_nm": np.asarray(s.z_nm).tolist(),
                "axes": s.axes} for s in stacks]
    split = None
    if dual:
        rois = {tuple(s.roi) for s in stacks if s.roi is not None}
        if len(rois) > 1:
            raise ValueError("the bead stacks were taken with different camera ROIs")
        origin = tuple(next(iter(rois))[:2]) if rois else (0, 0)
        split = make_split(images.shape, profile.dual or {}, origin)
        images = split.cut(images)
    return BeadInput(subtract_background(images), dz, camera, sources, split)


def subtract_background(images: np.ndarray) -> np.ndarray:
    """Each bead stack less its median, per channel.

    uiPSF finds beads above a threshold relative to the brightest maximum of
    the z projection *including* the background: with a background a fifth of
    the peak -- two-colour beads, half the light each -- the noise of the
    projection passed it everywhere and nine beads in ten were lost as too
    close to a spurious neighbour.  uiPSF fits a background per bead anyway,
    so taking a constant off changes the model it learns nothing.
    """
    images = np.asarray(images, np.float32)
    return images - np.median(images, axis=(-3, -2, -1), keepdims=True)


@dataclass
class MovieInput:
    images: np.ndarray
    camera: object
    source: dict
    split: Optional[Split] = None


def read_movie(path, camera_settings, profile, dual: bool, start: int = 0,
               stop: Optional[int] = None, progress=None) -> MovieInput:
    """Frames of a blinking acquisition, as uiPSF's photons."""
    from ..io.tiff import open_stack
    report = progress or (lambda text: None)
    source = open_stack(path)
    camera = resolve_camera(camera_settings, source, profile)
    stop = len(source) if stop is None else min(stop, len(source))
    if stop - start < 100:
        raise ValueError(f"frames {start} to {stop}: too few to learn a PSF from")
    blocks = []
    for first, block in source.frames(chunk=500, start=start, stop=stop):
        blocks.append(to_photons(block, camera))
        report(f"read {first + len(block) - start} of {stop - start} frames")
    images = np.concatenate(blocks)
    roi = list(camera.roi) if camera.roi is not None else None
    info = {"path": str(path), "roi": roi, "shape": list(images.shape),
            "frames": [int(start), int(stop)]}
    split = None
    if dual:
        origin = tuple(roi[:2]) if roi is not None else (0, 0)
        split = make_split(images.shape, profile.dual or {}, origin)
        images = split.cut(images)
    return MovieInput(images, camera, info, split)
