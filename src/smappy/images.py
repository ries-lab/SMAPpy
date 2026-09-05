"""Pixel images shown alongside localizations: a wide-field channel, a
rendered super-resolution TIFF, a transmitted-light picture.

An `ImageData` sits at ``(x0, y0)`` in the coordinates of the table with a
pixel of ``pixelsize`` (nm), and can be resampled onto any `FieldOfView`, so
a viewer treats it like a rendered layer: the same LUT, contrast and gamma.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .render import FieldOfView, RenderedImage


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

    @property
    def n_frames(self) -> int:
        return self.data.shape[0] if self.data.ndim == 3 else 1

    @property
    def plane(self) -> np.ndarray:
        return self.data[min(self.frame, self.n_frames - 1)] if self.data.ndim == 3 else self.data

    @property
    def shape(self) -> Tuple[int, int]:
        return self.plane.shape

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(x0, y0, x1, y1) covered by the pixels."""
        ny, nx = self.shape
        return (self.x0, self.y0, self.x0 + nx * self.pixelsize, self.y0 + ny * self.pixelsize)

    def resample(self, fov: FieldOfView) -> RenderedImage:
        """Nearest-neighbour onto ``fov``; outside the image is zero.

        Averaging when the view pixel is coarser than the image is left for
        later; nearest is right whenever one zooms in, which is the usual case
        for an overlay under localizations.
        """
        plane = self.plane
        ny, nx = plane.shape
        xs = fov.x0 + (np.arange(fov.nx) + 0.5) * fov.pixelsize
        ys = fov.y0 + (np.arange(fov.ny) + 0.5) * fov.pixelsize
        ix = np.floor((xs - self.x0) / self.pixelsize).astype(np.int64)
        iy = np.floor((ys - self.y0) / self.pixelsize).astype(np.int64)
        okx, oky = (ix >= 0) & (ix < nx), (iy >= 0) & (iy < ny)
        out = np.zeros((fov.ny, fov.nx), np.float32)
        if okx.any() and oky.any():
            sub = plane[np.ix_(iy[oky], ix[okx])].astype(np.float32)
            out[np.ix_(oky, okx)] = sub
        return RenderedImage(fov, out, None, n_locs=0)


def load_image(path, pixelsize: Optional[float] = None, x0: float = 0.0,
               y0: float = 0.0) -> ImageData:
    """A TIFF (or anything tifffile / Pillow read) as an `ImageData`.

    The pixel size comes from the TIFF resolution tags (ImageJ or plain), in
    nm; pass ``pixelsize`` when the file has none.  RGB is turned to grey.
    """
    path = Path(path)
    meta: Dict = {}
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            data = tf.asarray()
            page = tf.pages[0]
            found = _pixelsize_from_tags(page, tf)
            if found:
                meta["pixelsize_from_file"] = found
            if tf.imagej_metadata:
                meta["imagej"] = {k: v for k, v in tf.imagej_metadata.items()
                                  if isinstance(v, (int, float, str))}
                for k in ("x0", "y0"):
                    if k in tf.imagej_metadata:
                        meta[k] = float(tf.imagej_metadata[k])
    except (ImportError, Exception):
        from PIL import Image
        data = np.asarray(Image.open(path))
        found = None
    if data.ndim == 3 and data.shape[-1] in (3, 4):          # RGB(A) -> grey
        data = data[..., :3].astype(np.float32).mean(axis=-1)
    if data.ndim > 3:
        data = data.reshape(-1, *data.shape[-2:])
    if pixelsize is None:
        pixelsize = found
    if pixelsize is None:
        raise ValueError(f"{path.name} records no pixel size; pass pixelsize (nm)")
    return ImageData(np.asarray(data), float(pixelsize),
                     float(meta.get("x0", x0)), float(meta.get("y0", y0)),
                     name=path.name, path=str(path), metadata=meta)


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
