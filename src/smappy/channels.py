"""Several channels on one frame: where each is, and how it maps onto the first.

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
