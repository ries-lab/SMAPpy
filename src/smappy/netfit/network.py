"""The network: backbones and the heads that turn features into per-pixel outputs.

**Version 1, ``unet2``** -- DECODE's architecture as published (Speiser et al. 2021, Fig. 1): each
frame of the window through one U-Net (the same weights for every frame), their features
concatenated (the ``frame_stack`` time model) and through a second U-Net, then the heads.  A U-Net
here: two 3 x 3 convolutions a level, ``features * 2^level`` channels, a stride-2 convolution down,
nearest-neighbour upsampling and a 3 x 3 convolution up, the skip concatenated.

**The outputs**, per pixel (``output`` part ``offsets``), ten channels in scaled units (`Scaling`):

====  ==========================================================  ===========
 0    p, that a molecule is near this pixel                        sigmoid
 1    photons / phot_max                                           sigmoid
 2-3  x and y offset from the pixel's centre, pixels               tanh (+-1)
 4    z / z_max                                                    tanh
 5-8  the uncertainties of 1-4, in the same units                  sigmoid * 3 + 0.001
 9    background / bg_max                                          sigmoid
====  ==========================================================  ===========

x is along the columns, y along the rows, as everywhere in SMAPpy.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

SIGMA_SCALE, SIGMA_FLOOR = 3.0, 1e-3


@dataclass
class Scaling:
    """Network units: the input centred and scaled, the outputs as fractions of their range."""
    input_offset: float        # photons: the background's mean
    input_scale: float         # photons: a fiftieth of the mean flux
    phot_max: float            # mean flux + 8 sd
    z_max: float               # 1.2 * the largest |z|
    bg_max: float              # 1.2 * the largest background

    @classmethod
    def for_simulation(cls, sim) -> Scaling:
        return cls(input_offset=float(sum(sim.background) / 2), input_scale=float(sim.photons_mean / 50),
                   phot_max=float(sim.photons_mean + 8 * sim.photons_sd),
                   z_max=float(1.2 * max(abs(v) for v in sim.z_nm)), bg_max=float(1.2 * max(sim.background)))


_ACTIVATIONS = {"relu": nn.ReLU, "elu": nn.ELU}


def _block(c_in, c_out, act):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), act(), nn.Conv2d(c_out, c_out, 3, padding=1), act())


class UNet(nn.Module):
    def __init__(self, c_in: int, features: int, depth: int, act):
        super().__init__()
        f = [features * 2 ** level for level in range(depth + 1)]
        self.first = _block(c_in, f[0], act)
        self.down = nn.ModuleList(nn.Sequential(nn.Conv2d(f[i], f[i], 2, stride=2), act()) for i in range(depth))
        self.enc = nn.ModuleList(_block(f[i], f[i + 1], act) for i in range(depth))
        self.up = nn.ModuleList(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                              nn.Conv2d(f[i + 1], f[i], 3, padding=1), act())
                                for i in reversed(range(depth)))
        self.dec = nn.ModuleList(_block(2 * f[i], f[i], act) for i in reversed(range(depth)))
        self.depth = depth

    def forward(self, x):
        skips = [self.first(x)]
        for down, enc in zip(self.down, self.enc):
            skips.append(enc(down(skips[-1])))
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return x


class Head(nn.Module):
    def __init__(self, features, c_out, act):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(features, features, 3, padding=1), act(), nn.Conv2d(features, c_out, 1))

    def forward(self, x):
        return self.net(x)


class TwoStageUNet(nn.Module):
    """``unet2``: a U-Net per frame, shared, then one over their concatenated features, then heads."""

    def __init__(self, window: int = 3, features: int = 48, depth_frame: int = 2, depth_union: int = 2,
                 activation: str = "relu"):
        super().__init__()
        act = _ACTIVATIONS[activation]
        self.window = window
        self.frame = UNet(1, features, depth_frame, act)
        self.union = UNet(window * features, features, depth_union, act)
        self.heads = nn.ModuleDict({"p": Head(features, 1, act), "phot_xyz": Head(features, 4, act),
                                    "sigma": Head(features, 4, act), "bg": Head(features, 1, act)})
        self.fold = 2 ** max(depth_frame, depth_union)        # frames must be a multiple of this

    def forward(self, x):
        """``x`` (b, window, rows, columns) in network units -> (b, 10, rows, columns)."""
        b, t, h, w = x.shape
        feats = self.frame(x.reshape(b * t, 1, h, w)).reshape(b, -1, h, w)
        u = self.union(feats)
        values = self.heads["phot_xyz"](u)
        return torch.cat([torch.sigmoid(self.heads["p"](u)), torch.sigmoid(values[:, :1]), torch.tanh(values[:, 1:]),
                          torch.sigmoid(self.heads["sigma"](u)) * SIGMA_SCALE + SIGMA_FLOOR,
                          torch.sigmoid(self.heads["bg"](u))], 1)


BACKBONES = {"unet2": TwoStageUNet}


def build(parts, window: int) -> nn.Module:
    """The network the parts name (version 1: ``unet2`` with the ``frame_stack`` time model)."""
    if parts.time_model.name != "frame_stack":
        raise ValueError(f"time model {parts.time_model.name!r}: only frame_stack so far")
    return BACKBONES[parts.backbone.name](window=window, **parts.backbone.settings)


def parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
