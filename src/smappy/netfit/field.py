"""A field-dependent PSF in netfit: where on the chip each pixel is, as two input maps (`docs/decode/fd-plan.md`).

A calibration learnt with *field dependence* holds a spline per node of a grid over the chip
(`smappy.field_splines`), and a spot is drawn with the four nodes around it blended
(`render.SplineRenderer`'s ``field``).  The network is told where each pixel is by two maps beside the
frames, FD-DeepLoc's way (Fu et al. 2023):

    u = (clip(chip_x, lo_x, hi_x) - centre_x) * pixel_x / scale

and v alike in rows -- uiPSF's `FieldPolynomial` coordinates (``scale`` 50 µm, the centre the middle of
the beads' span, `calibrate.pupil.field_centre`), clipped to the nodes' span: beyond the beads the PSF is
the grid's edge's, in the drawing and in Spline 3D alike, so the maps stop changing where the PSF does,
and a network never sees a value it was not trained on.  ``lo``, ``hi`` and the centre are the nodes'
(their span is the beads', `calibrate.pupil.field_splines`), ``pixel`` the training camera's.  All of it
is kept in the model file (`FieldRecord.to_dict`), and inference makes the maps from it and the frames'
corner on the chip -- one network for any camera ROI.

Positions are columns and rows with pixel centres at whole numbers, as everywhere in SMAPpy; the chip's
pixel of a frame's pixel ``(r, c)`` is ``(corner_x + c, corner_y + r)``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

SCALE_UM = 50.0                 # `uipsf.model.field.FieldPolynomial`'s default


@dataclass
class FieldRecord:
    """What the maps are made from, as a model file keeps it."""
    centre_px: tuple            # (x, y) chip pixels: the middle of the nodes' span
    lo_px: tuple                # (x, y) the nodes' lowest
    hi_px: tuple                # (x, y) and highest
    pixel_nm: tuple             # (x, y) the training camera's
    scale_um: float = SCALE_UM
    grid: tuple = (1, 1)        # (gy, gx) the node grid, a record only

    @classmethod
    def from_splines(cls, field, pixel_nm) -> FieldRecord:
        """From a calibration's `field_splines.FieldSplines` and the camera's pixel (x, y) nm."""
        lo = (float(field.xs[0]), float(field.ys[0]))
        hi = (float(field.xs[-1]), float(field.ys[-1]))
        return cls(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2), lo, hi,
                   tuple(float(v) for v in pixel_nm), SCALE_UM, tuple(int(v) for v in field.shape))

    @classmethod
    def from_dict(cls, d: dict) -> FieldRecord:
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    def uv(self, chip_x, chip_y):
        """u, v of chip positions (numpy or torch alike): clipped to the nodes' span, from the centre, in
        units of ``scale_um`` on the sample."""
        f = 1.0 / (1000.0 * self.scale_um)
        clip = np.clip if isinstance(chip_x, np.ndarray) else (lambda a, lo, hi: a.clamp(lo, hi))
        u = (clip(chip_x, self.lo_px[0], self.hi_px[0]) - self.centre_px[0]) * (self.pixel_nm[0] * f)
        v = (clip(chip_y, self.lo_px[1], self.hi_px[1]) - self.centre_px[1]) * (self.pixel_nm[1] * f)
        return u, v

    def maps(self, corner, rows: int, columns: int) -> np.ndarray:
        """(2, rows, columns) float32: u and v at each pixel of frames whose pixel (0, 0) is chip pixel
        ``corner`` (x, y) -- the camera ROI's offset, plus whatever was cropped from its top left."""
        x = float(corner[0]) + np.arange(columns, dtype=np.float64)
        y = float(corner[1]) + np.arange(rows, dtype=np.float64)
        u, v = self.uv(x, y)
        return np.stack([np.broadcast_to(u[None, :], (rows, columns)),
                         np.broadcast_to(v[:, None], (rows, columns))]).astype(np.float32)

    def maps_torch(self, corners, size: int):
        """(n, 2, size, size): the maps of ``n`` square crops whose corners on the chip are ``corners``
        (n, 2) x, y -- a torch tensor, on its device."""
        import torch
        a = torch.arange(size, device=corners.device, dtype=torch.float32)
        u, v = self.uv(corners[:, 0:1].float() + a[None], corners[:, 1:2].float() + a[None])     # (n, size)
        n = len(corners)
        return torch.stack([u[:, None, :].expand(n, size, size), v[:, :, None].expand(n, size, size)], 1)

    def corners(self, n: int, size: int, g, device):
        """``n`` crops' corners on the chip (n, 2) x, y, whole pixels: the crop's centre uniform over the
        nodes' span widened by half a crop each side, so that the network sees the field's corners and a
        little of the edge's PSF beyond them."""
        import torch
        out = []
        for k in range(2):
            lo, hi = self.lo_px[k] - size / 2.0, self.hi_px[k] + size / 2.0         # the centre's range
            centre = lo + (hi - lo) * torch.rand(n, generator=g, device=device)
            out.append(torch.floor(centre - (size - 1) / 2.0 + 0.5))
        return torch.stack(out, 1)

    def beyond(self, corner, rows: int, columns: int) -> float:
        """How far frames at ``corner`` (x, y) of ``rows`` x ``columns`` reach beyond the nodes' span, in
        pixels (0 inside): where the PSF is the grid's edge's."""
        x0, y0 = float(corner[0]), float(corner[1])
        return max(0.0, self.lo_px[0] - x0, self.lo_px[1] - y0,
                   x0 + columns - 1 - self.hi_px[0], y0 + rows - 1 - self.hi_px[1])


def resolve(mode: str, cal) -> bool:
    """Whether a training on ``cal`` (a `SplineCalibration`) draws the field: ``mode`` "off", "on" (refused
    for a calibration without one) or "auto" (on when it has one)."""
    has = getattr(cal, "field_splines", None) is not None
    if mode in ("off", "", None, False):
        return False
    if mode == "auto":
        return has
    if mode in ("on", True):
        if not has:
            raise ValueError("field 'on', but the calibration has no field dependence (no node splines)")
        return True
    raise ValueError(f"field {mode!r}: off, on or auto")
