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

import math
from dataclasses import dataclass

import torch
from torch import nn

SIGMA_SCALE, SIGMA_FLOOR = 3.0, 1e-3
P0 = 0.03                     # p a pixel starts at: about 50 molecules on 40 x 40 pixels


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
    """``upsample_kernel`` 3 (an activation after it) or 1 (DECODE's: linear); ``pool_act`` an activation
    after the stride-2 convolution (DECODE has none); ``out_conv`` a linear 1 x 1 convolution at the end,
    as DECODE's U-Nets end, so the features handed on are not cut at zero."""

    def __init__(self, c_in: int, features: int, depth: int, act, upsample_kernel: int = 3,
                 pool_act: bool = True, out_conv: bool = False):
        super().__init__()
        f = [features * 2 ** level for level in range(depth + 1)]
        self.first = _block(c_in, f[0], act)
        self.down = nn.ModuleList(nn.Sequential(nn.Conv2d(f[i], f[i], 2, stride=2), *([act()] if pool_act else []))
                                  for i in range(depth))
        self.enc = nn.ModuleList(_block(f[i], f[i + 1], act) for i in range(depth))
        if upsample_kernel == 3:
            up = lambda i: nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                         nn.Conv2d(f[i + 1], f[i], 3, padding=1), act())
        else:
            up = lambda i: nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(f[i + 1], f[i], 1))
        self.up = nn.ModuleList(up(i) for i in reversed(range(depth)))
        self.dec = nn.ModuleList(_block(2 * f[i], f[i], act) for i in reversed(range(depth)))
        self.out = nn.Conv2d(f[0], f[0], 1) if out_conv else nn.Identity()
        self.depth = depth

    def forward(self, x):
        skips = [self.first(x)]
        for down, enc in zip(self.down, self.enc):
            skips.append(enc(down(skips[-1])))
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.out(x)


class Head(nn.Module):
    def __init__(self, features, c_out, act):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(features, features, 3, padding=1), act(), nn.Conv2d(features, c_out, 1))

    def forward(self, x):
        return self.net(x)


class TwoStageUNet(nn.Module):
    """``unet2``: a U-Net per frame, shared, then one over their concatenated features, then heads."""

    def __init__(self, window: int = 3, features: int = 48, depth_frame: int = 2, depth_union: int = 2,
                 activation: str = "relu", upsample_kernel: int = 3, pool_act: bool = True, out_conv: bool = False,
                 init: str = "default", p_clamp: float | None = None):
        """``upsample_kernel``, ``pool_act``, ``out_conv`` (`UNet`), ``init`` "kaiming" (every convolution
        Kaiming-normal, fan in, as DECODE's ``init_custom``; "default" PyTorch's, about 2.4 times smaller)
        and ``p_clamp`` (p's logit clamped, DECODE: 8) are DECODE's details, there to find which of them
        pixelation depends on (2026-11-12); the defaults are version 1's."""
        super().__init__()
        act = _ACTIVATIONS[activation]
        self.window = window
        style = {"upsample_kernel": upsample_kernel, "pool_act": pool_act, "out_conv": out_conv}
        self.frame = UNet(1, features, depth_frame, act, **style)
        self.union = UNet(window * features, features, depth_union, act, **style)
        self.p_clamp = p_clamp
        self.heads = nn.ModuleDict({"p": Head(features, 1, act), "phot_xyz": Head(features, 4, act),
                                    "sigma": Head(features, 4, act), "bg": Head(features, 1, act)})
        self.fold = 2 ** max(depth_frame, depth_union)        # frames must be a multiple of this
        if init == "kaiming":
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        elif init != "default":
            raise ValueError(f"init {init!r}: default or kaiming")
        self._init_heads()

    def _init_heads(self, p0: float = P0):
        """Every head's last layer at zero output -- offsets 0, the scaled values and sigmas at their
        sigmoid's middle -- and p at ``p0``.  With PyTorch's default the y offset's tanh saturated at
        -1 in the first epochs of a training without gradient clipping and stayed there for 100 epochs
        (every position on the pixel's edge, KS 0.5; 2026-11-12); DECODE initialises its heads
        itself (``init_custom``)."""
        for head in self.heads.values():
            last = head.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        nn.init.constant_(self.heads["p"].net[-1].bias, math.log(p0 / (1 - p0)))

    def forward(self, x):
        """``x`` (b, window, rows, columns) in network units -> (b, 10, rows, columns)."""
        b, t, h, w = x.shape
        feats = self.frame(x.reshape(b * t, 1, h, w)).reshape(b, -1, h, w)
        u = self.union(feats)
        values = self.heads["phot_xyz"](u)
        logit = self.heads["p"](u)
        if self.p_clamp is not None:
            logit = logit.clamp(-self.p_clamp, self.p_clamp)
        return torch.cat([torch.sigmoid(logit), torch.sigmoid(values[:, :1]), torch.tanh(values[:, 1:]),
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
