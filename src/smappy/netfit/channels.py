"""Two channels on one grid: a split camera's halves as netfit reads them (`docs/decode/plex-plan.md` §16).

A dual calibration (`calibrate.dual.DualColorCalibration`, a ``_2ct.h5``) says where each half lies on
the chip, how it is flipped, and how a position in the main half maps into the other
(`DualColorCalibration.layout`).  The network's grid is channel 0's pixels.  Channel 1 is brought onto
it by a whole-pixel shift ``s``, the registration's median offset over the halves' overlap, so that its
image is cut, not resampled: a molecule at ``p`` in channel 0 is at ``p + d(p)`` in channel 1's
shifted image, ``d`` the registration's remainder.  Near the overlap's middle ``d`` is a fraction of a
pixel; at its corners it is what the map's turn and scale add up to (biplane, Fig. 1c's beads: 0.036
of shear, about 9 px over 256).  The network reads ``d`` at every pixel's centre as two input maps
(`offsets`), DECODE-Plex's offset maps; channel 0's would be zero and are left out.

Only where both channels hold the camera's pixels is read (`TwoChannels.box`): the overlap of channel 0
with channel 1 moved by ``s``.  Padding the rest, as DECODE-Plex's inference does with zeros, makes the
network invent molecules (plex-plan §15).

Positions are columns and rows with pixel centres at whole numbers, as everywhere in SMAPpy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TwoChannels:
    """A dual calibration's halves on channel 0's grid, for frames whose corner is at ``corner`` on the
    chip (the frames' camera ROI)."""
    layout: object                 # `channels.ChannelLayout`, regions in frame pixels
    shift: tuple                   # (columns, rows): channel 1's own pixels are channel 0's plus this, plus d
    box: tuple                     # (row0, row1, col0, col1) of channel 0 where both channels are read

    @classmethod
    def from_calibration(cls, cal, corner=(0, 0)) -> TwoChannels:
        from ..channels import ChannelLayout
        chip = cal.layout()
        x0, y0 = (int(v) for v in corner)
        regions = [(x - x0, y - y0, w, h) for x, y, w, h in chip.regions]
        layout = ChannelLayout(regions, chip.flips, chip.maps, name=chip.name, warps=chip.warps)
        (w0, h0), (w1, h1) = layout.sizes
        rows, cols = np.mgrid[0:h0, 0:w0]
        xy = np.c_[cols.ravel(), rows.ravel()].astype(float)
        off = layout.to_channel(1, xy) - xy
        inside = ((off[:, 0] + xy[:, 0] >= 0) & (off[:, 0] + xy[:, 0] <= w1 - 1)
                  & (off[:, 1] + xy[:, 1] >= 0) & (off[:, 1] + xy[:, 1] <= h1 - 1))
        sx, sy = (int(np.rint(np.median(off[inside, k]))) for k in (0, 1))
        box = (max(0, -sy), min(h0, h1 - sy), max(0, -sx), min(w0, w1 - sx))
        if box[1] <= box[0] or box[3] <= box[2]:
            raise ValueError("the two channels do not overlap")
        return cls(layout, (sx, sy), box)

    @property
    def size(self) -> tuple:
        """(rows, columns) of the box."""
        r0, r1, c0, c1 = self.box
        return r1 - r0, c1 - c0

    def remainder(self, xy) -> np.ndarray:
        """``d``: channel 1's position of channel-0 positions ``xy`` (n, 2), less the position and the shift."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        return self.layout.to_channel(1, xy) - xy - np.asarray(self.shift, float)

    def offsets(self, box=None) -> np.ndarray:
        """``d`` at the centre of each pixel of ``box`` (row0, row1, col0, col1; the whole box by
        default): (2, rows, columns), x then y, float32."""
        r0, r1, c0, c1 = box or self.box
        rows, cols = np.mgrid[r0:r1, c0:c1]
        d = self.remainder(np.c_[cols.ravel(), rows.ravel()])
        return d.T.reshape(2, r1 - r0, c1 - c0).astype(np.float32)

    def cut(self, frames, box=None) -> np.ndarray:
        """Frames (n, H, W) as the camera wrote them -> (n, 2, rows, columns) on channel 0's grid over
        ``box`` (the whole box by default): channel 0 as it is, channel 1 flipped as the layout says and
        moved by the shift."""
        r0, r1, c0, c1 = box or self.box
        both = self.layout.cut(frames)                                    # (n, 2, h, w), the flips done
        sx, sy = self.shift
        return np.stack([both[:, 0, r0:r1, c0:c1], both[:, 1, r0 + sy:r1 + sy, c0 + sx:c1 + sx]], 1)

    def on_frame(self, xy) -> np.ndarray:
        """Channel-0 positions (n, 2) back on the frame the camera wrote: where to report them."""
        return self.layout.on_frame(0, xy)

    def to_dict(self) -> dict:
        return {"shift": list(self.shift), "box": list(self.box)}


def splines(cal, kind: str):
    """The two splines the channels are drawn with, each in its channel's orientation on the grid
    (`DualColorCalibration.layer_splines`: the secondary flipped back along the mirror): biplane's as
    stored, each plane's share in it; colour's each scaled to one at focus, a molecule's photons split
    between them by its share (as the C = 2 sets draw them, `sets_c2.drawn_splines`)."""
    from dataclasses import replace
    main, second = cal.layer_splines()
    if kind == "colour":
        unit = lambda s: replace(s, coeff=(s.coeff / float(s.psf[round(s.z0)].sum())).astype(np.float32),
                                 psf=s.psf / float(s.psf[round(s.z0)].sum()))
        main, second = unit(main), unit(second)
    return main, second


class Grid:
    """`TwoChannels`' remainder over its box, on a device, for drawing training crops."""

    def __init__(self, two: TwoChannels, device="cpu"):
        import torch
        self.two = two
        self.d = torch.as_tensor(two.offsets(), device=device)            # (2, H, W) over the box
        self.device = torch.device(device)

    def origins(self, n: int, size: int, g):
        """``n`` crops' corners (rows, columns) in the box, uniform where a ``size`` square fits."""
        import torch
        H, W = self.d.shape[1:]
        if H < size or W < size:
            raise ValueError(f"the channels' overlap ({H} x {W}) is smaller than a {size}-pixel frame")
        r = torch.randint(0, H - size + 1, (n,), generator=g, device=self.device)
        c = torch.randint(0, W - size + 1, (n,), generator=g, device=self.device)
        return r, c

    def maps(self, r, c, size: int):
        """The crops' offset maps, (n, 2, size, size)."""
        import torch
        a = torch.arange(size, device=self.device)
        rows = (r[:, None] + a[None])[:, :, None]                        # (n, size, 1)
        cols = (c[:, None] + a[None])[:, None, :]                        # (n, 1, size)
        return self.d[:, rows, cols].permute(1, 0, 2, 3)

    def at(self, sample, x, y, r, c):
        """``d`` at molecules' positions (x, y in their crop), bilinear between the pixels' centres, the
        box's edge carried on beyond it: (m, 2)."""
        import torch
        H, W = self.d.shape[1:]
        gx = (x + c[sample]).clamp(0, W - 1)
        gy = (y + r[sample]).clamp(0, H - 1)
        x0, y0 = gx.floor().long().clamp(max=W - 2), gy.floor().long().clamp(max=H - 2)
        fx, fy = gx - x0, gy - y0
        d = self.d
        out = (d[:, y0, x0] * (1 - fx) * (1 - fy) + d[:, y0, x0 + 1] * fx * (1 - fy)
               + d[:, y0 + 1, x0] * (1 - fx) * fy + d[:, y0 + 1, x0 + 1] * fx * fy)
        return out.T.contiguous() if torch.is_tensor(out) else out
