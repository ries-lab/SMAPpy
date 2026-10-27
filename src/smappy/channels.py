"""Several channels on one frame: where each is, and how it maps onto the first.

This is the one layer that knows the geometry: it sits right after the
loader (`ChannelSource`), and everything after it -- detection, the fit,
registration, drift -- sees only the channels, (n, C, h, w), and maps
between them.  Three parameters, each with its own owner and lifetime:

* **the geometry** -- each channel's rectangle *on the camera chip*, in whole
  pixels, and its mirror -- belongs to the microscope (`microscope_db`) and
  is coarse.  On the chip rather than on the frame, it holds for any camera
  ROI: the layer takes the ROI's corner off (`ChannelLayout.on_camera`);
* **the registration** -- a projective map of each channel's pixels onto
  channel 0's, after the cut and the flip -- belongs to a calibration (the
  model, or beads, `register_channels`) and is fine.  After the flip it is
  close to the identity: a shift of a pixel or two, a scale and a rotation a
  few per mille from 1 and 0 (`describe_map`, which reads it about the
  channel's centre, where the shift does not mix with the rest).  A
  rectangle placed a pixel off costs nothing: the registration's shift takes
  it up, which is what lets the geometry be coarse and stable;
* **the drift** -- a shift of each channel over time -- belongs to the
  acquisition (`multichannel.ChannelDrift`).

A dual-colour splitter is the same three: two rectangles, one flipped, a map
near the identity (`docs/4pi.md`, M8).

A 4Pi microscope records four phase images of every molecule, a two-colour
splitter two: the same field of view seen more than once, in rectangles of
the camera frame -- or in separate files, which the readers put side by side
in one frame (`smappy.io.quadrants`), so that everything that handles frames
(the camera, the frame view, the raw frames kept with the table) keeps
handling one frame.  A `ChannelLayout` says, for each channel,

* the rectangle of the frame it occupies, ``(x, y, width, height)``;
* whether it is mirrored, in x or y, against the first -- undone when the
  rectangle is cut, which is exact: a flip moves pixels, it does not
  resample them; and
* the map from its pixels (after the flip) onto channel 0's: a 3x3
  projective matrix on column vectors ``(x, y, 1)``, identity for channel 0.
  After the flip what is left is nearly a translation, which is what the
  fitter's diagonal link represents well (`smappy.dualfit._linearise`).

Channel 0 is the reference: positions are reported in its pixels, and the
other channels' ROIs are cut where the map puts the molecule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ChannelLayout:
    regions: List[Tuple[int, int, int, int]]          # (x, y, width, height) per channel
    flips: List[Tuple[bool, bool]] = None             # (flip x, flip y) per channel
    maps: List[np.ndarray] = None                     # channel c -> channel 0, 3x3
    name: str = ""
    _inverse: list = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.regions = [tuple(int(v) for v in r) for r in self.regions]
        n = len(self.regions)
        if not 1 <= n <= 4:
            raise ValueError("a layout has one to four channels")
        sizes = {(w, h) for _, _, w, h in self.regions}
        if len(sizes) != 1:
            raise ValueError(f"every channel's rectangle must have one size; got {sorted(sizes)}")
        self.flips = [tuple(bool(v) for v in f) for f in (self.flips or [(False, False)] * n)]
        maps = self.maps if self.maps is not None else [np.eye(3)] * n
        self.maps = [np.asarray(m, float).reshape(3, 3) for m in maps]
        if len(self.flips) != n or len(self.maps) != n:
            raise ValueError("one flip and one map per channel")
        self._inverse = [np.linalg.inv(m) for m in self.maps]

    @property
    def n_channels(self) -> int:
        return len(self.regions)

    @property
    def size(self) -> Tuple[int, int]:
        """One channel's (width, height)."""
        return self.regions[0][2:]

    # ---------------------------------------------------------------- cutting
    def cut(self, frames: np.ndarray) -> np.ndarray:
        """(n, H, W) frames -> (n, C, h, w) channels, each flipped to match
        channel 0."""
        frames = np.asarray(frames)
        if frames.ndim == 2:
            frames = frames[None]
        out = []
        for (x, y, w, h), (fx, fy) in zip(self.regions, self.flips):
            if y + h > frames.shape[-2] or x + w > frames.shape[-1]:
                raise ValueError(f"the channel at x={x}, y={y} ({w} x {h}) does not fit "
                                 f"in the {frames.shape[-1]} x {frames.shape[-2]} frame")
            part = frames[:, y:y + h, x:x + w]
            if fx:
                part = part[:, :, ::-1]
            if fy:
                part = part[:, ::-1, :]
            out.append(part)
        return np.stack(out, axis=1)

    def fits(self, shape: Tuple[int, int]) -> bool:
        """Whether every rectangle lies inside a frame of ``shape`` (height, width)."""
        return all(y + h <= shape[0] and x + w <= shape[1] for x, y, w, h in self.regions)

    # ---------------------------------------------------------------- mapping
    def to_reference(self, channel: int, xy) -> np.ndarray:
        """Channel ``channel``'s pixel positions (n, 2), in channel 0's."""
        return _apply(self.maps[channel], xy)

    def to_channel(self, channel: int, xy) -> np.ndarray:
        """Channel 0's pixel positions (n, 2), in channel ``channel``'s."""
        return _apply(self._inverse[channel], xy)

    def on_frame(self, channel: int, xy) -> np.ndarray:
        """A channel's pixel positions back on the frame, the flip undone:
        where to draw them over the raw frame."""
        xy = np.asarray(xy, float).reshape(-1, 2).copy()
        x0, y0, w, h = self.regions[channel]
        fx, fy = self.flips[channel]
        if fx:
            xy[:, 0] = w - 1 - xy[:, 0]
        if fy:
            xy[:, 1] = h - 1 - xy[:, 1]
        return xy + (x0, y0)

    # ---------------------------------------------------------------- record
    def to_dict(self) -> dict:
        return {"name": self.name, "regions": [list(r) for r in self.regions],
                "flips": [list(f) for f in self.flips],
                "maps": [m.tolist() for m in self.maps]}

    @classmethod
    def from_dict(cls, record: dict) -> "ChannelLayout":
        return cls([tuple(r) for r in record["regions"]],
                   [tuple(f) for f in record.get("flips") or []] or None,
                   [np.asarray(m) for m in record.get("maps") or []] or None,
                   record.get("name", ""))

    def with_maps(self, maps: Sequence[np.ndarray]) -> "ChannelLayout":
        return ChannelLayout(self.regions, self.flips, list(maps), self.name)

    def on_camera(self, roi_offset=(0, 0)) -> "ChannelLayout":
        """The layout on the frames of a camera ROI whose corner is at
        ``roi_offset`` on the chip: every rectangle moved by it.  The maps
        are between channel images and do not change."""
        ox, oy = (int(round(v)) for v in roi_offset)
        if not (ox or oy):
            return self
        regions = [(x - ox, y - oy, w, h) for x, y, w, h in self.regions]
        if any(x < 0 or y < 0 for x, y, _, _ in regions):
            raise ValueError(f"the camera ROI, at ({ox}, {oy}) on the chip, cuts off a channel "
                             f"of {self.name or 'the layout'}")
        return ChannelLayout(regions, self.flips, self.maps, self.name)


def describe_map(matrix, size: Tuple[int, int]) -> dict:
    """A channel's map read about its centre: the shift there (px), the
    scale along x and y, the rotation (mrad) and the projective part's
    largest effect at a corner (px).  What a registration is worth looking
    at, in the units it is near: a map read about the corner would mix a
    rotation of a few mrad into a shift of a pixel."""
    w, h = size
    c = np.array([(w - 1) / 2, (h - 1) / 2])
    m = np.asarray(matrix, float) / matrix[2, 2]
    centre = _apply(m, c[None])[0]
    step = 1.0
    jx = (_apply(m, (c + [step, 0])[None])[0] - centre) / step
    jy = (_apply(m, (c + [0, step])[None])[0] - centre) / step
    affine = np.eye(3)
    affine[:2, :2] = np.c_[jx, jy]
    affine[:2, 2] = centre - affine[:2, :2] @ c
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float)
    projective = np.abs(_apply(m, corners) - _apply(affine, corners)).max()
    return {"shift_px": (centre - c).tolist(), "scale": [float(np.hypot(*jx)), float(np.hypot(*jy))],
            "rotation_mrad": float(1000 * np.arctan2(jx[1], jx[0])),
            "projective_px": float(projective)}


def _apply(matrix: np.ndarray, xy) -> np.ndarray:
    xy = np.asarray(xy, float).reshape(-1, 2)
    h = np.c_[xy, np.ones(len(xy))] @ matrix.T
    return h[:, :2] / h[:, 2:3]


def side_by_side(channels: int, width: int, height: int, name: str = "") -> ChannelLayout:
    """Channels of ``width`` x ``height`` placed left to right, unflipped:
    what the readers of separate channel images make (`smappy.io.quadrants`)."""
    return ChannelLayout([(c * width, 0, width, height) for c in range(channels)], name=name)


def from_corners(x: Sequence[int], y: Sequence[int], width: int, height: int,
                 flip_x: Sequence[bool] = (), flip_y: Sequence[bool] = (),
                 name: str = "") -> ChannelLayout:
    """A layout from each channel's corner and one size, as SMAP's 4Pi
    settings give it (``x4pi``, ``y4pi``, ``width4pi``, ``height4pi``,
    ``mirror4pi``)."""
    n = len(x)
    fx = list(flip_x) + [False] * (n - len(flip_x))
    fy = list(flip_y) + [False] * (n - len(flip_y))
    return ChannelLayout([(int(x[c]), int(y[c]), width, height) for c in range(n)],
                         list(zip(fx, fy)), name=name)


#: The Bewersdorf-design 4Pi-SMS of the example data (zenodo 14287640): four
#: 168-pixel images side by side on a 2048 x 168 strip, the last two mirrored
#: in x -- SMAP's ``Yale4Pi_settings_3D.txt`` (``x4pi=[151 389 1474 1713]``,
#: ``mirror4pi=[0 0 1 1]``).
YALE_4PI = from_corners([151, 389, 1474, 1713], [0, 0, 0, 0], 168, 168,
                        flip_x=[False, False, True, True], name="4Pi-SMS (Yale)")


def maps_from_uipsf_v1(T, centre, transpose: bool = True) -> List[np.ndarray]:
    """uiPSF v1's channel affines as this module's maps, channel c -> 0.

    v1 keeps, for channels 1.., a 3x3 ``T`` acting on row vectors
    ``(a, b, 1)`` about the image centre: channel 0's position mapped into
    channel c is ``(p0 - centre) @ T + centre``.  Its (a, b) are v1's (row,
    column), which for a ``.mat`` it read transposed are this package's
    (x, y) -- ``transpose`` -- and its rows and columns otherwise.  Channel 0
    is the identity; the matrices returned go the other way, c -> 0.
    """
    T = np.asarray(T, float)
    c = np.asarray(centre, float).reshape(-1)[:2]
    if not transpose:
        swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], float)
    maps = [np.eye(3)]
    for t in T:
        # row-vector affine about the centre -> column-vector homogeneous matrix
        forward = np.eye(3)
        forward[:2, :2] = t[:2, :2].T
        forward[:2, 2] = t[2, :2] + c - t[:2, :2].T @ c
        if not transpose:
            forward = swap @ forward @ swap
        maps.append(np.linalg.inv(forward))
    return maps


def local_scale(layout: ChannelLayout, channel: int, x, y, step: float = 1.0):
    """d(x_c)/d(x_0) and d(y_c)/d(y_0) at channel-0 positions: the diagonal
    the fitter's link can carry (`smappy.dualfit._linearise`)."""
    from .dualfit import _linearise
    return _linearise(lambda p: layout.to_channel(channel, p), np.asarray(x, float),
                      np.asarray(y, float), step)


def warp_onto_reference(image: np.ndarray, layout: ChannelLayout, channel: int,
                        grid: Optional[np.ndarray] = None) -> np.ndarray:
    """A channel's images (n, h, w) resampled onto channel 0's pixels, for
    detection only -- never for the fit, which cuts each channel at its own
    pixels (`smappy.dualfit`).  Bilinear; outside the channel, zero."""
    from scipy.ndimage import map_coordinates
    image = np.asarray(image, np.float32)
    h, w = image.shape[-2:]
    if np.allclose(layout.maps[channel], np.eye(3)):
        return image
    if grid is None:
        grid = channel_grid(layout, channel, (h, w))
    out = np.empty_like(image)
    for k in range(len(image)):
        out[k] = map_coordinates(image[k], grid, order=1, mode="constant", cval=0.0)
    return out


def channel_grid(layout: ChannelLayout, channel: int, shape: Tuple[int, int]) -> np.ndarray:
    """Where each of channel 0's pixels lies in channel ``channel``, as
    (row, column) coordinates for `scipy.ndimage.map_coordinates`."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    there = layout.to_channel(channel, np.c_[xx.ravel(), yy.ravel()])
    return np.stack([there[:, 1].reshape(h, w), there[:, 0].reshape(h, w)])


# ---------------------------------------------------------------- from beads
#
# A layout without a model's maps -- a model learned without them, a camera
# moved since -- is registered from beads.  A bead looks different in every
# phase channel at any one z: the interference moves its light between them.
# Summed over the whole stack -- every z step, every phase step -- the
# interference averages out and each channel shows the bead's envelope, whose
# centre is where the bead is.  The centres are fitted with a Gaussian, the
# channels are put roughly on top of each other by the projections' cross-
# correlation, paired, and the map of each channel onto channel 0 is the
# robust projective fit the dual-colour calibration uses
# (`calibrate.dual.fit_dual_transform`).  With fewer pairs than that needs --
# the Ries-lab bead scans have one bead per file -- the map is a translation,
# their mean offset.  What a centroid of the envelope cannot see -- where the
# model's PSF of each channel is centred -- the fit's own measurement of the
# channels' offsets takes up (`multichannel.measure_channel_drift`).

@dataclass
class Registration:
    """Maps of each channel onto channel 0, and how well the beads agree."""
    maps: List[np.ndarray]
    pairs: List[int]                 # per channel
    rms_px: List[float]              # residual of the pairs about the map, per channel
    kind: List[str]                  # "projective" or "translation", per channel


def bead_positions(image: np.ndarray, border: int = 6, sigma: float = 1.5,
                   threshold: float = 0.2, roisize: int = 11) -> np.ndarray:
    """Sub-pixel (x, y) of the beads in one projection: maxima of the
    smoothed image above ``threshold`` of its brightest, each fitted with a
    free-width Gaussian in a ``roisize`` box."""
    from scipy.ndimage import gaussian_filter, maximum_filter
    from . import _fit3d
    image = np.asarray(image, np.float64)
    smooth = gaussian_filter(image - np.median(image), sigma)
    peaks = (smooth == maximum_filter(smooth, size=2 * border + 1)) & \
        (smooth > threshold * smooth.max())
    half = roisize // 2
    h, w = image.shape
    ys, xs = np.nonzero(peaks)
    keep = (xs >= max(half, border)) & (xs < w - max(half, border)) & \
        (ys >= max(half, border)) & (ys < h - max(half, border))
    xs, ys = xs[keep], ys[keep]
    if not len(xs):
        return np.empty((0, 2))
    rois = np.stack([image[y - half:y + half + 1, x - half:x + half + 1] for x, y in zip(xs, ys)])
    rois = rois - rois.min(axis=(1, 2), keepdims=True)
    # scaled to a few thousand photons, so the Poisson fit weighs the pixels sensibly
    rois = (rois / rois.max(axis=(1, 2), keepdims=True) * 1000).astype(np.float32)
    theta = _fit3d.fit_gauss_free(np.ascontiguousarray(rois), 1.5, 50, 0)[0]
    return np.c_[xs - half + theta[:, 0], ys - half + theta[:, 1]]


def _shift(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Where ``other`` is relative to ``reference``, (dx, dy), by the peak of
    their cross-correlation, to a fraction of a pixel."""
    from scipy.signal import fftconvolve
    a = reference - reference.mean()
    b = other - other.mean()
    c = fftconvolve(b, a[::-1, ::-1], mode="same")
    iy, ix = np.unravel_index(np.argmax(c), c.shape)

    def vertex(m, i):
        if 0 < i < len(m) - 1:
            d = m[i - 1] - 2 * m[i] + m[i + 1]
            return 0.5 * (m[i - 1] - m[i + 1]) / d if d != 0 else 0.0
        return 0.0
    return np.array([ix - c.shape[1] // 2 + vertex(c[iy, :], ix),
                     iy - c.shape[0] // 2 + vertex(c[:, ix], iy)])


def register_channels(projections, settings=None, match_px: float = 3.0) -> Registration:
    """Maps of every channel onto channel 0 from bead projections.

    ``projections`` is one (C, h, w) array per bead acquisition: each
    channel's image summed over the stack (`bead_projections`).  Beads of
    every acquisition are pooled.
    """
    from .calibrate.dual import bead_registration, fit_dual_transform
    from .dualfit import _match
    settings = settings or bead_registration()
    projections = [np.asarray(p, np.float64) for p in projections]
    C = projections[0].shape[0]
    found = [[bead_positions(p[c]) for c in range(C)] for p in projections]
    maps, pairs, rms, kinds = [np.eye(3)], [0], [0.0], ["identity"]
    for c in range(1, C):
        # the coarse offset from every acquisition's correlation, summed
        coarse = np.median([_shift(p[0], p[c]) for p in projections], axis=0)
        source, target = [], []
        for beads in found:
            ref, other = beads[0], beads[c]
            a, b = _match(ref, other - coarse, match_px)
            source.append(other[b])
            target.append(ref[a])
        source, target = np.concatenate(source), np.concatenate(target)
        if len(source) == 0:
            raise ValueError(f"no bead of channel {c + 1} pairs with one of channel 1: "
                             "check the layout, or the beads' brightness")
        if len(source) >= settings.min_pairs:
            matrix = fit_dual_transform(source, target, settings).transformation
            kind = "projective"
        else:
            matrix = np.eye(3)
            matrix[:2, 2] = np.mean(target - source, axis=0)
            kind = "translation"
        residual = _apply(matrix, source) - target
        maps.append(np.asarray(matrix, float))
        pairs.append(len(source))
        rms.append(float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))))
        kinds.append(kind)
    return Registration(maps, pairs, rms, kinds)


def bead_projections(paths, layout: ChannelLayout) -> list:
    """Each bead acquisition at ``paths`` (files or folders, anything
    `open_stack` reads; a folder is searched for them) summed over all its
    frames and cut into its channels: (C, h, w) each."""
    from pathlib import Path
    from .io.tiff import open_stack
    files = []
    for path in [paths] if isinstance(paths, (str, Path)) else paths:
        path = Path(path)
        if path.is_dir():
            files += sorted(p for p in path.rglob("*")
                            if p.suffix.lower() in (".mat", ".dcimg", ".tif", ".tiff"))
        else:
            files.append(path)
    out, seen = [], set()
    for f in files:
        source = open_stack(f)
        key = tuple(str(p) for p in source.files)
        if key in seen:                # a series opened through another of its files
            continue
        seen.add(key)
        total = None
        for _, block in source.frames(chunk=200):
            s = np.asarray(block, np.float64).sum(axis=0)
            total = s if total is None else total + s
        lay = getattr(source, "layout", None) or layout
        out.append(lay.cut(total[None])[0])
    if not out:
        raise ValueError(f"no bead acquisitions at {paths}")
    return out



# ---------------------------------------------------------------- the layer

from .io.tiff import ImageSource  # noqa: E402  (tiff imports this module only in functions)


@dataclass
class ChannelSource(ImageSource):
    """An acquisition read as its channels: blocks of (n, C, h, w).

    Wraps any `ImageSource` -- a DCIMG file, quadrant files, a TIFF -- with
    the layout on its frames (`ChannelLayout.on_camera`, the camera ROI taken
    off).  ``shape`` is one channel's, the space every position is reported
    in.  ``base`` is the source as the camera wrote it.
    """
    base: ImageSource = None
    layout: ChannelLayout = None

    @classmethod
    def wrap(cls, base: ImageSource, layout: ChannelLayout) -> "ChannelSource":
        if not layout.fits(base.shape):
            raise ValueError(f"the channels ({layout.regions}) do not fit the "
                             f"{base.shape[1]} x {base.shape[0]} frames")
        w, h = layout.size
        return cls(files=list(base.files), shape=(h, w), dtype=base.dtype,
                   n_frames=base.n_frames, n_frames_declared=base.n_frames_declared,
                   mm_metadata=base.mm_metadata, summary=base.summary, base=base,
                   layout=layout)

    @property
    def n_channels(self) -> int:
        return self.layout.n_channels

    def frames(self, chunk: int = 100, start: int = 0, stop: Optional[int] = None):
        self.base.tags = self.tags
        for first, block in self.base.frames(chunk=chunk, start=start, stop=stop):
            yield first, self.layout.cut(block)

    def frame(self, index: int) -> np.ndarray:
        """One frame's channels, (C, h, w)."""
        return self.layout.cut(self.base.frame(index)[None])[0]

    def watch(self, chunk: int = 100, **kwargs):
        for first, block in self.base.watch(chunk=chunk, **kwargs):
            yield first, self.layout.cut(block)


# ---------------------------------------------------------------- the file

REGISTRATION_FORMAT = "smappy-channel-registration"


def save_registration(path, layout: ChannelLayout, registration: Optional[Registration] = None,
                      source: str = "") -> None:
    """A registration as JSON: the geometry it was measured on, each
    channel's map onto channel 0, and -- for reading by eye -- each map
    about its channel's centre (`describe_map`) and the beads' agreement."""
    import datetime
    import json
    from pathlib import Path
    record = {"format": REGISTRATION_FORMAT, "version": 1,
              "date": datetime.date.today().isoformat(), "source": str(source),
              "layout": layout.to_dict(),
              "maps_about_centre": [describe_map(m, layout.size) for m in layout.maps]}
    if registration is not None:
        record["pairs"] = list(registration.pairs)
        record["rms_px"] = list(registration.rms_px)
        record["kind"] = list(registration.kind)
    Path(path).write_text(json.dumps(record, indent=1))


def load_registration(path) -> ChannelLayout:
    """The layout and maps a registration file holds."""
    import json
    from pathlib import Path
    record = json.loads(Path(path).read_text())
    if record.get("format") != REGISTRATION_FORMAT:
        raise ValueError(f"{Path(path).name} is not a channel registration")
    return ChannelLayout.from_dict(record["layout"])


def registration_path(beads) -> "Path":
    """Where a registration measured on ``beads`` is kept: beside the bead
    file, or in the bead folder."""
    from pathlib import Path
    beads = Path(beads)
    return (beads / "channel_registration.json" if beads.is_dir()
            else beads.with_name(beads.stem + "_channels.json"))
