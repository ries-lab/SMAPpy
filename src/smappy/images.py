"""Pixel images shown alongside localizations: a wide-field channel, a
rendered super-resolution TIFF, a transmitted-light picture.

An `ImageData` sits at ``(x0, y0)`` in the coordinates of the table with a
pixel of ``pixelsize`` (nm), and can be resampled onto any `FieldOfView`, so
a viewer treats it like a rendered layer: the same LUT, contrast and gamma.

Its **placement** moves it further: an affine map (3x3, nm) applied after
that grid, which *Transform* sets (`smappy.transformation`) when an image
rather than the localizations should move -- rotated, scaled, mirrored or
shifted.  The pixels are never resampled for it: the view looks each of its
pixels up through the inverse, so a move is lossless and undone exactly.
``x0``, ``y0`` and the pixel size stay what the file says, and the Render
tab's fields still set them.

A stack says what its planes are -- ``axis`` "z" (a z-stack, with ``z0_nm``
and ``z_step_nm``), "t" (a time series, ``frame_time_s``) or "" (not known,
or the frames a fit kept) -- in its metadata, so it is saved with the image
and survives every copy.  Channels are never an axis: a file's channels are
split into one image each when it is opened (`load_images`, `assign_axes`),
so that each is a layer with its own LUT, as localization channels are.  A
layer shows one plane, or a projection over a range of them (`shown_plane`);
the 3D view draws a z-stack as a volume (`view3d.render_volume_3d`).

The camera frames a fit keeps (`smappy.rawframes`) are one of these too, with
``kind = "raw"`` and each plane's frame number in ``frames``: the session
lists them beside any TIFF it has opened, and an image layer shows whichever
one is chosen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .render import PER_UM2, FieldOfView, RenderedImage


@dataclass
class ImageData:
    data: np.ndarray                 # (ny, nx) or (nz, ny, nx)
    pixelsize: float                 # data units (nm) per pixel
    x0: float = 0.0                  # outer edge of pixel (0, 0)
    y0: float = 0.0
    name: str = ""
    path: Optional[str] = None
    frame: int = 0                   # for a stack: which plane is shown
    metadata: Dict = field(default_factory=dict)
    pixelsize_y: Optional[float] = None   # None: square pixels
    # "image" (a file opened as one), "raw" (the frames a fit kept) or
    # "empty" (`ImageData.empty`)
    kind: str = "image"
    # per plane: its camera frame number, -1 for an average; None for a file
    frames: Optional[np.ndarray] = None
    # an affine map of the table's x, y (3x3, nm) after the grid above; None
    # is where the grid puts it.  Set by Transform, never by resampling.
    placement: Optional[np.ndarray] = None

    @classmethod
    def empty(cls) -> "ImageData":
        """What an image layer shows before there is an image to show.

        A layer asked for while a live fit runs has no frames yet -- the fit
        hands them over when it ends -- and asking for a TIFF instead was the
        wrong answer to that.  The layer opens on this, draws nothing, and the
        session points it at the first image that arrives
        (`Session.changed`).  It is no source: it is never listed, saved or
        counted in the view's extent.
        """
        return cls(np.zeros((1, 1), np.float32), 1.0, name="no image yet",
                   kind="empty")

    @property
    def is_empty(self) -> bool:
        return self.kind == "empty"

    @property
    def pixelsize_xy(self) -> Tuple[float, float]:
        return (self.pixelsize, self.pixelsize if self.pixelsize_y is None
                else self.pixelsize_y)

    @property
    def n_frames(self) -> int:
        return self.data.shape[0] if self.data.ndim == 3 else 1

    # ---------------------------------------------------------------- planes
    # What the planes are, kept in the metadata (saved with the image, shared
    # by the snapshots' copies): "z", "t" or "" for not known.
    @property
    def axis(self) -> str:
        return str(self.metadata.get("axis", "")) if self.n_frames > 1 else ""

    @axis.setter
    def axis(self, value: str) -> None:
        self.metadata["axis"] = str(value or "")

    @property
    def z0_nm(self) -> float:
        """The z of the first plane."""
        return float(self.metadata.get("z0_nm", 0.0) or 0.0)

    @z0_nm.setter
    def z0_nm(self, value: float) -> None:
        self.metadata["z0_nm"] = float(value)

    @property
    def z_step_nm(self) -> Optional[float]:
        value = self.metadata.get("z_step_nm")
        return float(value) if value else None

    @z_step_nm.setter
    def z_step_nm(self, value: Optional[float]) -> None:
        self.metadata["z_step_nm"] = None if not value else float(value)

    @property
    def frame_time_s(self) -> Optional[float]:
        value = self.metadata.get("frame_time_s")
        return float(value) if value else None

    @frame_time_s.setter
    def frame_time_s(self, value: Optional[float]) -> None:
        self.metadata["frame_time_s"] = None if not value else float(value)

    @property
    def is_zstack(self) -> bool:
        """Planes at known heights: what the 3D view can draw as a volume."""
        return self.axis == "z" and self.z_step_nm is not None

    @property
    def axes_unknown(self) -> bool:
        """Several planes, and the file did not say what they are: what
        File > Open asks about (`assign_axes`)."""
        return bool(self.metadata.get("axes_unknown")) and self.n_frames > 1

    def plane_z(self, index) -> np.ndarray:
        """The height of plane ``index`` (nm); the plane is centred on it."""
        return self.z0_nm + np.asarray(index, np.float64) * (self.z_step_nm or 0.0)

    def z_range(self) -> Tuple[float, float]:
        """What the planes cover in z: half a step beyond the first and the
        last, whichever way the stack was taken."""
        a, b = sorted((float(self.plane_z(0)), float(self.plane_z(self.n_frames - 1))))
        half = 0.5 * abs(self.z_step_nm or 0.0)
        return a - half, b + half

    def index_at_z(self, z: float) -> Optional[int]:
        """The plane at height ``z``; None outside the stack, or if it is not
        a z-stack."""
        if not self.is_zstack:
            return None
        lo, hi = self.z_range()
        if not lo <= z <= hi:
            return None
        i = int(np.round((z - self.z0_nm) / self.z_step_nm))
        return int(np.clip(i, 0, self.n_frames - 1))

    @property
    def plane(self) -> np.ndarray:
        return self.plane_at(self.frame)

    def volume(self) -> np.ndarray:
        """Every plane as one float32 array, read once and kept: a stack kept
        in a file is read a plane at a time otherwise (`io.hdf5.StoredStack`)."""
        cache = self._cache()
        found = cache.get("volume")
        if found is None or found[0] is not self.data:
            data = np.asarray(self.data, np.float32)
            found = cache["volume"] = (self.data, data if data.ndim == 3 else data[None])
        return found[1]

    def background(self) -> float:
        """The stack's median: its background, where most voxels of a
        sparse stack -- beads, a few cells -- are.  From a sample of a
        million voxels at most, which agrees with the whole to a fraction of
        an ADU and costs a hundredth of it."""
        cache = self._cache()
        found = cache.get("background")
        if found is None or found[0] is not self.data:
            flat = self.volume().reshape(-1)
            step = max(1, flat.size // 1_000_000)
            found = cache["background"] = (self.data, float(np.median(flat[::step])))
        return found[1]

    def projection(self, kind: str = "max", planes: Optional[Tuple[int, int]] = None
                   ) -> np.ndarray:
        """The planes ``planes`` (first and last, both kept; None: all) taken
        together: "max" (a maximum-intensity projection, MIP), "mean" or "sum".
        Computed once per range and kept, as a slider over a range would
        otherwise redo it on every move."""
        n = self.n_frames
        lo, hi = (0, n - 1) if planes is None else sorted(int(v) for v in planes)
        lo, hi = int(np.clip(lo, 0, n - 1)), int(np.clip(hi, 0, n - 1))
        key = ("projection", kind, lo, hi)
        cache = self._cache()
        found = cache.get(key)
        if found is not None and found[0] is self.data:
            return found[1]
        if n == 1:
            out = np.asarray(self.plane_at(0), np.float32)
        else:
            planes_ = [np.asarray(self.plane_at(i), np.float32) for i in range(lo, hi + 1)]
            stack = np.stack(planes_)
            out = {"max": stack.max, "mean": stack.mean, "sum": stack.sum}[kind](axis=0)
        if len([k for k in cache if k[0] == "projection"]) > 8:
            for k in [k for k in cache if k[0] == "projection"]:
                del cache[k]
        cache[key] = (self.data, out.astype(np.float32))
        return cache[key][1]

    def shown_plane(self, show: str = "slice", frame: Optional[int] = None,
                    planes: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """What a layer shows: one plane (``show`` "slice"), or a projection
        of a range of them."""
        if show == "slice" or self.n_frames == 1:
            return self.plane_at(self.frame if frame is None else frame)
        return self.projection(show, planes)

    def _cache(self) -> dict:
        # kept in the metadata's shadow, not in it: never saved, and shared
        # by the shallow copies the render threads are handed
        cache = self.__dict__.get("_computed")
        if cache is None:
            cache = self.__dict__["_computed"] = {}
        return cache

    def plane_at(self, index: int) -> np.ndarray:
        if self.data.ndim != 3:
            return self.data
        return self.data[int(np.clip(index, 0, self.n_frames - 1))]

    def label(self, index: int) -> str:
        """What plane ``index`` is, in words: the slider says it beside it."""
        if self.is_zstack:
            return f"z {float(self.plane_z(index)):.0f} nm"
        if self.axis == "t" and self.frame_time_s:
            return f"t {index * self.frame_time_s:.3g} s"
        if self.frames is None or not 0 <= index < len(self.frames):
            return f"plane {index + 1} of {self.n_frames}" if self.n_frames > 1 else ""
        number = int(self.frames[index])
        if number < 0:
            n = self.metadata.get("n_averaged")
            return f"average of {n} frames" if n else "average"
        return f"frame {number}"

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.data.shape[-2:])

    @property
    def grid(self) -> np.ndarray:
        """Pixel (column, row) -- the outer edge of pixel (0, 0) at (0, 0) --
        to table x, y: the grid and then the placement, as one 3x3."""
        px, py = self.pixelsize_xy
        m = np.array([[px, 0.0, self.x0], [0.0, py, self.y0], [0.0, 0.0, 1.0]])
        return m if self.placement is None else np.asarray(self.placement) @ m

    def place(self, matrix: np.ndarray) -> None:
        """Move the image by ``matrix`` (3x3 affine, nm) after where it is."""
        current = np.eye(3) if self.placement is None else np.asarray(self.placement)
        moved = np.asarray(matrix, float) @ current
        self.placement = None if np.allclose(moved, np.eye(3), atol=1e-12) else moved

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(x0, y0, x1, y1) covered by the pixels -- of the placed corners,
        when the placement turns the image."""
        ny, nx = self.shape
        px, py = self.pixelsize_xy
        if self.placement is None:
            return (self.x0, self.y0, self.x0 + nx * px, self.y0 + ny * py)
        corners = self.grid @ np.array([[0, nx, 0, nx], [0, 0, ny, ny], [1, 1, 1, 1]],
                                       float)
        return (float(corners[0].min()), float(corners[1].min()),
                float(corners[0].max()), float(corners[1].max()))

    def resample(self, fov: FieldOfView, frame: Optional[int] = None,
                 plane: Optional[np.ndarray] = None) -> RenderedImage:
        """Nearest-neighbour onto ``fov``; outside the image is zero.

        ``frame`` is the plane of a stack, by default the image's own
        ``frame``: a layer keeps its own, so that two layers can show the
        average and one frame of the same stack.

        Averaging when the view pixel is coarser than the image is left for
        later; nearest is right whenever one zooms in, which is the usual case
        for an overlay under localizations.
        """
        if plane is None:
            plane = self.plane_at(self.frame if frame is None else frame)
        ny, nx = plane.shape
        px, py = self.pixelsize_xy
        xs = fov.x0 + (np.arange(fov.nx) + 0.5) * fov.pixelsize
        ys = fov.y0 + (np.arange(fov.ny) + 0.5) * fov.pixelsize
        if self.placement is not None:
            return RenderedImage(fov, self._placed(plane, xs, ys), None, n_locs=0,
                                 count_area=PER_UM2)
        ix = np.floor((xs - self.x0) / px).astype(np.int64)
        iy = np.floor((ys - self.y0) / py).astype(np.int64)
        okx, oky = (ix >= 0) & (ix < nx), (iy >= 0) & (iy < ny)
        out = np.zeros((fov.ny, fov.nx), np.float32)
        if okx.any() and oky.any():
            sub = plane[np.ix_(iy[oky], ix[okx])].astype(np.float32)
            out[np.ix_(oky, okx)] = sub
        # a camera's values are as they are at every zoom: an absolute
        # maximum is one of them, not a density (`render.absolute_imax`)
        return RenderedImage(fov, out, None, n_locs=0, count_area=PER_UM2)

    def _placed(self, plane: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Nearest neighbour through the placement: each view pixel's centre
        taken back to the image's pixels, the grid no longer separable."""
        return self.values_at(xs[None, :], ys[:, None], plane=plane)

    def values_at(self, x, y, frame: Optional[int] = None,
                  plane: Optional[np.ndarray] = None) -> np.ndarray:
        """The pixel each table point ``(x, y)`` lies in, nearest neighbour,
        zero outside the image; ``x`` and ``y`` broadcast against each other.

        For points that are no grid -- the 3D view's, where the image is a
        plane seen at an angle and every screen pixel lands somewhere else
        on it."""
        if plane is None:
            plane = self.plane_at(self.frame if frame is None else frame)
        ny, nx = plane.shape
        back = np.linalg.inv(self.grid)
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        u0 = back[0, 0] * x + back[0, 1] * y + back[0, 2]
        v0 = back[1, 0] * x + back[1, 1] * y + back[1, 2]
        ix, iy = np.floor(u0).astype(np.int64), np.floor(v0).astype(np.int64)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        out = np.zeros(ix.shape, np.float32)
        if ok.any():
            # only the rows that are needed are read: a plane may be on disk
            rows = np.unique(iy[ok])
            sub = np.asarray(plane[rows.min():rows.max() + 1], np.float32)
            out[ok] = sub[iy[ok] - rows.min(), ix[ok]]
        return out


def load_image(path, pixelsize: Optional[float] = None, x0: float = 0.0,
               y0: float = 0.0) -> ImageData:
    """The first image of a file (`load_images`): a script's one-liner for a
    file with one channel."""
    return load_images(path, pixelsize, x0, y0)[0]


# The letters tifffile names a series' axes with that are neither the image
# (Y, X), its colour samples (S), channels (C), z (Z) nor time (T): "I" and
# "Q" are planes it cannot place, the rest are rare.  They are pooled into
# one axis of unknown planes, which File > Open asks about.
_KNOWN_AXES = "TZCYXS"


def load_images(path, pixelsize: Optional[float] = None, x0: float = 0.0,
                y0: float = 0.0) -> List[ImageData]:
    """A TIFF (or anything tifffile / Pillow read) as `ImageData`, one per
    channel -- and per time point, for a stack in both z and t.

    What the planes are comes from the file: an ImageJ hyperstack (its
    slices, frames and channels, its *spacing* and *finterval*), OME-TIFF
    (the dimension order, PhysicalSizeZ, TimeIncrement) or Micro-Manager
    (its z step).  A stack of plain pages says nothing: it comes back as one
    image with ``axes_unknown`` set, for `assign_axes` to split once someone
    has said what the pages are.

    The pixel size comes from the TIFF resolution tags (ImageJ or plain), in
    nm; pass ``pixelsize`` when the file has none.  RGB is turned to grey.
    """
    path = Path(path)
    meta: Dict = {}
    axes, z_step, dt = "", None, None
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            data, axes = series.asarray(), series.axes
            page = tf.pages[0]
            found = _pixelsize_from_tags(page, tf)
            if found:
                meta["pixelsize_from_file"] = found
            if tf.imagej_metadata:
                ij = tf.imagej_metadata
                meta["imagej"] = {k: v for k, v in ij.items()
                                  if isinstance(v, (int, float, str))}
                for k in ("x0", "y0"):
                    if k in ij:
                        meta[k] = float(ij[k])
                if ij.get("spacing"):
                    z_step = float(ij["spacing"]) * (_unit_nm(ij.get("unit")) or 1000.0)
                if ij.get("finterval"):
                    dt = float(ij["finterval"])
            if tf.is_ome and tf.ome_metadata:
                z_step = _ome_z_step(tf.ome_metadata) or z_step
            if tf.is_micromanager and tf.micromanager_metadata:
                summary = (tf.micromanager_metadata or {}).get("Summary") or {}
                step = summary.get("z-step_um", summary.get("z_step_um"))
                if step:
                    z_step = z_step or float(step) * 1000.0
    except (ImportError, Exception):
        from PIL import Image
        data = np.asarray(Image.open(path))
        found = None
        axes = "YXS"[:data.ndim] if data.ndim == 3 and data.shape[-1] in (3, 4) else ""
    data = np.asarray(data)
    if not axes or len(axes) != data.ndim:
        # no names: the last two are the image, anything before them planes
        axes = ("YXS" if data.ndim == 3 and data.shape[-1] in (3, 4)
                else "I" * (data.ndim - 2) + "YX")
    if "S" in axes:                                             # RGB(A) -> grey
        i = axes.index("S")
        data = np.take(data, range(min(3, data.shape[i])), axis=i).astype(np.float32).mean(axis=i)
        axes = axes.replace("S", "")
    if pixelsize is None:
        pixelsize = found
    if pixelsize is None:
        raise ValueError(f"{path.name} records no pixel size; pass pixelsize (nm)")
    data, axes = _standard_axes(data, axes)
    nt, nz, nc, ni = (data.shape[i] for i in range(4))
    base = ImageData(np.zeros((1, 1), np.float32), float(pixelsize),
                     float(meta.get("x0", x0)), float(meta.get("y0", y0)),
                     name=path.name, path=str(path), metadata=meta)
    if ni > 1:
        # pages nothing names (beside channels the file may name): one stack
        # of unknown planes per channel, for File > Open to ask about
        planes = np.moveaxis(data, 2, 3)                      # T, Z, I, C, Y, X
        planes = planes.reshape(nt * nz * ni, nc, *data.shape[-2:])
        return _split(base, planes, "", None, None, unknown=True)
    if nt > 1 and nz == 1:                                     # a time series
        return _split(base, data[:, 0, :, 0], "t", None, dt)
    images = []
    for t in range(nt):           # a z-stack (or one plane) per time point
        part = _split(base, data[t, :, :, 0], "z" if nz > 1 else "", z_step, dt)
        if nt > 1:
            for im in part:
                im.name = f"{im.name} t{t + 1}"
        images.extend(part)
    return images


def _standard_axes(data: np.ndarray, axes: str) -> Tuple[np.ndarray, str]:
    """``data`` as (T, Z, C, I, Y, X): every axis present, in that order,
    with the unnamed ones pooled into I."""
    axes = "".join(a if a in _KNOWN_AXES else "I" for a in axes)
    while axes.count("I") > 1:                     # pool the unnamed ones
        i = axes.index("I")
        j = axes.index("I", i + 1)
        data = np.moveaxis(data, j, i + 1)
        data = data.reshape(*data.shape[:i], -1, *data.shape[i + 2:])
        axes = axes[:i + 1] + axes[i + 1:].replace("I", "", 1)
    for a in "TZCI":
        if a not in axes:
            data, axes = data[None], a + axes
    order = [axes.index(a) for a in "TZCIYX"]
    return np.transpose(data, order), "TZCIYX"


def _split(base: ImageData, stack: np.ndarray, axis: str, z_step: Optional[float],
           dt: Optional[float], unknown: bool = False) -> List[ImageData]:
    """One image per channel of ``stack`` (planes, channels, y, x)."""
    import copy as _copy
    n_channels = stack.shape[1]
    out = []
    for c in range(n_channels):
        im = _copy.copy(base)
        im.metadata = dict(base.metadata)
        im.__dict__.pop("_computed", None)
        planes = np.ascontiguousarray(stack[:, c])
        im.data = planes[0] if len(planes) == 1 else planes
        if n_channels > 1:
            im.name = f"{base.name}: C{c + 1}"
            im.metadata["channel"] = c
        if axis:
            im.axis = axis
        if axis == "z" and z_step:
            im.z_step_nm = z_step
        if dt:
            im.frame_time_s = dt
        if unknown and len(planes) > 1:
            im.metadata["axes_unknown"] = True
        out.append(im)
    return out


def assign_axes(image: ImageData, axis: str, n_channels: int = 1,
                interleaved: bool = True, z_step_nm: Optional[float] = None,
                frame_time_s: Optional[float] = None) -> List[ImageData]:
    """Say what the planes of ``image`` are, and split its channels.

    ``axis`` is "z" or "t" for the planes of each channel, "" for neither.
    ``n_channels`` planes in a row are one of each channel when
    ``interleaved`` (ImageJ's order: c1 z1, c2 z1, ..., c1 z2), otherwise the
    first block of planes is the first channel.  A number of planes that
    ``n_channels`` does not divide is refused rather than guessed at.
    """
    data = np.asarray(image.data)
    data = data if data.ndim == 3 else data[None]
    n = data.shape[0]
    n_channels = max(int(n_channels), 1)
    if n % n_channels:
        raise ValueError(f"{n} planes do not make {n_channels} channels of equal length")
    per = n // n_channels
    if interleaved:
        stack = data.reshape(per, n_channels, *data.shape[1:])
    else:
        stack = data.reshape(n_channels, per, *data.shape[1:]).swapaxes(0, 1)
    base = _copy_image(image)
    base.metadata.pop("axes_unknown", None)
    base.metadata.pop("axis", None)
    return _split(base, stack, axis if per > 1 else "", z_step_nm, frame_time_s)


def _copy_image(image: ImageData) -> ImageData:
    import copy as _copy
    out = _copy.copy(image)
    out.metadata = dict(image.metadata)
    out.__dict__.pop("_computed", None)
    return out


def _unit_nm(unit) -> Optional[float]:
    """nm per one of ``unit``, as ImageJ and OME write them."""
    return {"nm": 1.0, "nanometer": 1.0, "um": 1e3, "µm": 1e3, "\\u00B5m": 1e3,
            "micron": 1e3, "micrometer": 1e3, "mm": 1e6, "cm": 1e7, "m": 1e9,
            "inch": 2.54e7}.get(str(unit).strip() if unit is not None else None)


def _ome_z_step(xml: str) -> Optional[float]:
    """PhysicalSizeZ of the first image of an OME-XML, in nm."""
    import re
    m = re.search(r'PhysicalSizeZ="([0-9.eE+-]+)"', xml)
    if not m:
        return None
    unit = re.search(r'PhysicalSizeZUnit="([^"]+)"', xml)
    return float(m.group(1)) * (_unit_nm(unit.group(1) if unit else "µm") or 1e3)


def _pixelsize_from_tags(page, tf) -> Optional[float]:
    """nm per pixel from XResolution + the unit (ImageJ 'unit' or ResolutionUnit)."""
    tag = page.tags.get("XResolution")
    if tag is None:
        return None
    num, den = tag.value
    if not num:
        return None
    per_unit = num / den                       # pixels per unit
    unit = None
    if tf.imagej_metadata and "unit" in tf.imagej_metadata:
        unit = tf.imagej_metadata["unit"]
    else:
        ru = page.tags.get("ResolutionUnit")
        unit = {2: "inch", 3: "cm"}.get(ru.value if ru else None)
    to_nm = {"nm": 1.0, "um": 1e3, "µm": 1e3, "micron": 1e3, "mm": 1e6, "cm": 1e7,
             "inch": 2.54e7, "m": 1e9}.get(unit)
    if to_nm is None:
        return None
    return to_nm / per_unit
