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

With ``sigma_rep`` (step 4, an honest sigma) three more: 10 - 12, the *reported* uncertainties of
x, y, z (same units, sigmoid * 3 + 0.001), from a head of their own on the features with their
gradient stopped -- the loss term that trains them (`loss.reported_term`) cannot move anything else.

**Slots** (``slots`` K > 1; step 4, close molecules): each cell holds K molecules, not one -- channels
``9 k`` to ``9 k + 8`` are slot k's p, photons, offsets, z and their four uncertainties, channel ``9 K``
the background (one slot is the table above).  Trained by the matched loss (`loss.matched_terms`),
which gives each true molecule one slot of its own, so that a cell can say "two" where DECODE's map
cannot (two molecules 0 - 100 nm apart share a pixel 40 % of the time, 2026-11-15).  Not with
``sigma_rep``: the matched loss trains each slot's sigma on its own molecule's error already.

**A finer output grid** (``out_scale`` 2, 2026-11-14): the union U-Net's features upsampled once more
and through one more block, so the maps have ``out_scale`` cells a pixel along each axis and every
molecule is placed from a cell whose centre is at most a quarter pixel away -- the pixelation (the
positions pulled toward the centres of the cells they are placed from, min/max of their histogram
0.73 - 0.83 in every variant of DECODE's details, 2026-11-14) is the grid's.  The offsets are then
+-1 cell, handed on in pixels; `centres` is where each cell's centre is, in pixels.

**Time models** (the ``time_model`` part; step 4, 2026-11-15) join the frames' features before the union
U-Net.  ``frame_stack`` (DECODE's) concatenates them, so the union's first convolution weighs each
frame by a fixed weight whatever it holds.  ``attention`` (as LUNAR's, Fu et al. 2026, along time per
pixel): the frames' features at a pixel are tokens, with a learned embedding of each frame's place in
the window, through a transformer layer (`TimeAttention`); the middle frame's token goes on -- so a
neighbouring frame counts as much as it shows the same molecule, which a fixed weight cannot say.
``convgru``: a convolutional GRU (`TimeGRU`) run forward to the middle frame and backward to it, the
two states going on -- the recurrent form, which suits live fitting.  "Time is not a shift"
(`docs/decode.md` §3.3): molecules do not move, so convolutions stay in space and time gets its own.

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


class ResBlock(nn.Module):
    """``block`` "residual" (step 4's screen B): the same two 3 x 3 convolutions, their sum with the
    input (through a 1 x 1 convolution where the channels change) before the last activation."""

    def __init__(self, c_in, c_out, act):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), act(), nn.Conv2d(c_out, c_out, 3, padding=1))
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.act = act()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class LayerNorm2d(nn.Module):
    """LayerNorm over the channels of each pixel, as ConvNeXt's."""

    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """``block`` "convnext" (step 4's screen B; Liu et al. 2022, doi:10.48550/arXiv.2201.03545): a 1 x 1
    convolution where the channels change, then a 7 x 7 depthwise convolution, LayerNorm, a 1 x 1
    expansion to four times the channels, GELU and back, added to its input.  The activation the
    network asks for is not used: the block's own is GELU."""

    def __init__(self, c_in, c_out, act=None, expand: int = 4, kernel: int = 7):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.dw = nn.Conv2d(c_out, c_out, kernel, padding=kernel // 2, groups=c_out)
        self.norm = LayerNorm2d(c_out)
        self.mlp = nn.Sequential(nn.Conv2d(c_out, expand * c_out, 1), nn.GELU(), nn.Conv2d(expand * c_out, c_out, 1))

    def forward(self, x):
        x = self.proj(x)
        return x + self.mlp(self.norm(self.dw(x)))


def _dilated_block(c_in, c_out, act):
    """``block`` "dilated": the plain block with its second convolution dilated 2 -- a receptive field
    of 7 x 7 a block at the same weights (step 4's screen B, large kernels or dilations)."""
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), act(), nn.Conv2d(c_out, c_out, 3, padding=2, dilation=2), act())


BLOCKS = {"plain": _block, "residual": ResBlock, "convnext": ConvNeXtBlock, "dilated": _dilated_block}


class Blur(nn.Module):
    """A fixed binomial low-pass, [1 2 1] x [1 2 1] / 16 per channel, before a stride-2 convolution
    (``blur``, BlurPool -- Zhang 2019, doi:10.48550/arXiv.1904.11486): what is downsampled has no
    frequencies above the coarser grid's, so the features move with the molecule rather than jump
    with the pixel grid -- aimed at pixelation (step 4's screen A)."""

    def __init__(self, channels):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        self.register_buffer("kernel", (k[:, None] * k[None, :] / 16).expand(channels, 1, 3, 3).clone(), persistent=False)
        self.channels = channels

    def forward(self, x):
        x = nn.functional.pad(x, (1, 1, 1, 1), mode="reflect")
        return nn.functional.conv2d(x, self.kernel.to(x.dtype), groups=self.channels)


class UNet(nn.Module):
    """``upsample_kernel`` 3 (an activation after it) or 1 (DECODE's: linear); ``pool_act`` an activation
    after the stride-2 convolution (DECODE has none); ``out_conv`` a linear 1 x 1 convolution at the end,
    as DECODE's U-Nets end, so the features handed on are not cut at zero."""

    def __init__(self, c_in: int, features: int, depth: int, act, upsample_kernel: int = 3,
                 pool_act: bool = True, out_conv: bool = False, blur: bool = False, block: str = "plain"):
        super().__init__()
        f = [features * 2 ** level for level in range(depth + 1)]
        make = BLOCKS[block]
        self.first = make(c_in, f[0], act)
        self.down = nn.ModuleList(nn.Sequential(*([Blur(f[i])] if blur else []), nn.Conv2d(f[i], f[i], 2, stride=2),
                                                *([act()] if pool_act else []))
                                  for i in range(depth))
        self.enc = nn.ModuleList(make(f[i], f[i + 1], act) for i in range(depth))
        if upsample_kernel == 3:
            up = lambda i: nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                         nn.Conv2d(f[i + 1], f[i], 3, padding=1), act())
        else:
            up = lambda i: nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(f[i + 1], f[i], 1))
        self.up = nn.ModuleList(up(i) for i in reversed(range(depth)))
        self.dec = nn.ModuleList(make(2 * f[i], f[i], act) for i in reversed(range(depth)))
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


def centres(n: int, scale: int = 1, device=None, dtype=torch.float32):
    """The centres of ``n`` output cells along an axis, in pixels (a pixel's centre at its index):
    ``(i + 0.5) / scale - 0.5`` -- the pixels' own at ``scale`` 1."""
    return (torch.arange(n, device=device, dtype=dtype) + 0.5) / scale - 0.5


def grid(h: int, w: int, scale: int = 1, device=None, dtype=torch.float32):
    """(rows, columns) of the cell centres of an (h, w) output, in pixels."""
    return torch.meshgrid(centres(h, scale, device, dtype), centres(w, scale, device, dtype), indexing="ij")


class Head(nn.Module):
    def __init__(self, features, c_out, act):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(features, features, 3, padding=1), act(), nn.Conv2d(features, c_out, 1))

    def forward(self, x):
        return self.net(x)


class TimeAttention(nn.Module):
    """``attention``: per pixel, the window's ``t`` feature vectors as tokens (plus a learned embedding of
    each one's place), one pre-norm transformer layer -- multi-head self-attention, then an MLP of
    twice the width, each added to its input -- and the middle frame's token out.  ``drop`` (training
    only, LUNAR's temporal dropout): each frame but the middle one hidden from the attention with this
    probability, so that the network does not lean on a neighbour being there."""

    def __init__(self, features: int, window: int, heads: int = 4, drop: float = 0.0):
        super().__init__()
        self.place = nn.Parameter(torch.zeros(window, features))
        self.norm1, self.norm2 = nn.LayerNorm(features), nn.LayerNorm(features)
        self.attn = nn.MultiheadAttention(features, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(features, 2 * features), nn.GELU(), nn.Linear(2 * features, features))
        self.drop = float(drop)
        self.out_channels = features

    def forward(self, f):
        """``f`` (b, t, F, h, w) -> (b, F, h, w)."""
        b, t, c, h, w = f.shape
        x = f.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c) + self.place
        hide = None
        if self.training and self.drop > 0:
            hide = torch.rand(b, 1, 1, t, device=f.device) < self.drop
            hide[..., t // 2] = False
            hide = hide.expand(b, h, w, t).reshape(b * h * w, t)
        y = self.norm1(x)
        x = x + self.attn(y, y, y, key_padding_mask=hide, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x[:, t // 2].reshape(b, h, w, c).permute(0, 3, 1, 2)


class ConvGRUCell(nn.Module):
    """A GRU whose gates are 3 x 3 convolutions (Ballas et al. 2016, doi:10.48550/arXiv.1511.06432)."""

    def __init__(self, features: int):
        super().__init__()
        self.gates = nn.Conv2d(2 * features, 2 * features, 3, padding=1)
        self.cand = nn.Conv2d(2 * features, features, 3, padding=1)

    def forward(self, x, h):
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)
        return (1 - z) * h + z * torch.tanh(self.cand(torch.cat([x, r * h], 1)))


class TimeGRU(nn.Module):
    """``convgru``: one cell run from the first frame to the middle one and another from the last frame
    back to it; the two states, (b, 2 F, h, w), go on.  The middle frame is the last either reads, so
    both end on it."""

    def __init__(self, features: int, window: int):
        super().__init__()
        self.fwd, self.bwd = ConvGRUCell(features), ConvGRUCell(features)
        self.out_channels = 2 * features

    def forward(self, f):
        b, t, c, h, w = f.shape
        mid = t // 2
        hf = f.new_zeros(b, c, h, w)
        for i in range(mid + 1):
            hf = self.fwd(f[:, i], hf)
        hb = f.new_zeros(b, c, h, w)
        for i in range(t - 1, mid - 1, -1):
            hb = self.bwd(f[:, i], hb)
        return torch.cat([hf, hb], 1)


class TwoStageUNet(nn.Module):
    """``unet2``: a U-Net per frame, shared, then one over their concatenated features, then heads."""

    def __init__(self, window: int = 3, features: int = 48, depth_frame: int = 2, depth_union: int = 2,
                 activation: str = "relu", upsample_kernel: int = 3, pool_act: bool = True, out_conv: bool = False,
                 init: str = "default", p_clamp: float | None = None, out_scale: int = 1,
                 blur: bool = False, block: str = "plain", offset_range: float = 1.0, sigma_rep: bool = False,
                 slots: int = 1, time_model: str = "frame_stack", time_settings: dict | None = None):
        """``upsample_kernel``, ``pool_act``, ``out_conv`` (`UNet`), ``init`` "kaiming" (every convolution
        Kaiming-normal, fan in, as DECODE's ``init_custom``; "default" PyTorch's, about 2.4 times smaller)
        and ``p_clamp`` (p's logit clamped, DECODE: 8) are DECODE's details; ``out_scale`` the output
        grid's cells a pixel (the module's docstring); ``blur`` (`Blur`), ``block`` (`BLOCKS`) and
        ``offset_range`` (the offsets' reach, in cells: tanh times it) step 4's.  The first five are there to find which of them
        pixelation depends on (2026-11-12); the defaults here are version 1's, so that a model saved
        without them is built as it was trained -- a new training's are DECODE's (`settings.Parts`)."""
        super().__init__()
        act = _ACTIVATIONS[activation]
        self.window = window
        style = {"upsample_kernel": upsample_kernel, "pool_act": pool_act, "out_conv": out_conv, "blur": blur,
                 "block": block}
        self.offset_range = float(offset_range)
        self.frame = UNet(1, features, depth_frame, act, **style)
        self.time_model = time_model
        if time_model == "frame_stack":
            self.time, joined = None, window * features
        elif time_model in TIME_MODELS:
            self.time = TIME_MODELS[time_model](features, window, **(time_settings or {}))
            joined = self.time.out_channels
        else:
            raise ValueError(f"time model {time_model!r}: frame_stack, {', '.join(TIME_MODELS)}")
        self.union = UNet(joined, features, depth_union, act, **style)
        self.p_clamp = p_clamp
        self.out_scale = int(out_scale)
        if self.out_scale not in (1, 2):
            raise ValueError(f"out_scale {out_scale}: 1 or 2")
        # nearest, then two 3 x 3 convolutions: each cell learns where it is in its pixel from its
        # neighbours, as the U-Net's own upsampling does
        self.refine = (nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), _block(features, features, act))
                       if self.out_scale == 2 else nn.Identity())
        self.slots = int(slots)
        if self.slots < 1 or (self.slots > 1 and sigma_rep):
            raise ValueError(f"slots {slots}: at least 1, and 1 with sigma_rep")
        k = self.slots
        self.heads = nn.ModuleDict({"p": Head(features, k, act), "phot_xyz": Head(features, 4 * k, act),
                                    "sigma": Head(features, 4 * k, act), "bg": Head(features, 1, act)})
        self.sigma_rep = bool(sigma_rep)
        if self.sigma_rep:
            self.heads["sigma_rep"] = Head(features, 3, act)
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
        p0 = p0 / self.out_scale ** 2 / self.slots    # the same molecules over more cells and slots
        nn.init.constant_(self.heads["p"].net[-1].bias, math.log(p0 / (1 - p0)))

    def forward(self, x):
        """``x`` (b, window, rows, columns) in network units -> (b, 9 K + 1, rows * s, columns * s), ``s``
        the ``out_scale``, K the ``slots`` (13 channels with ``sigma_rep``)."""
        b, t, h, w = x.shape
        feats = self.frame(x.reshape(b * t, 1, h, w))
        feats = (feats.reshape(b, -1, h, w) if self.time is None
                 else self.time(feats.reshape(b, t, -1, h, w)))
        u = self.refine(self.union(feats))
        # the heads in full precision, whatever the backbone ran in: in bf16 an output of z over
        # +-1000 nm resolves only about 4 nm (`train`'s ``precision``)
        with torch.autocast(u.device.type, enabled=False):
            return self._heads(u.float())

    def _heads(self, u):
        values = self.heads["phot_xyz"](u)
        logit = self.heads["p"](u)
        if self.p_clamp is not None:
            logit = logit.clamp(-self.p_clamp, self.p_clamp)
        reach = self.offset_range / self.out_scale    # +-offset_range cells, in pixels
        sigma = torch.sigmoid(self.heads["sigma"](u)) * SIGMA_SCALE + SIGMA_FLOOR
        maps = []
        for k in range(self.slots):
            v = values[:, 4 * k:4 * k + 4]
            xyz = torch.tanh(v[:, 1:])
            if reach != 1:
                xyz = torch.cat([xyz[:, :2] * reach, xyz[:, 2:]], 1)
            maps += [torch.sigmoid(logit[:, k:k + 1]), torch.sigmoid(v[:, :1]), xyz, sigma[:, 4 * k:4 * k + 4]]
        maps.append(torch.sigmoid(self.heads["bg"](u)))
        if self.sigma_rep:
            maps.append(torch.sigmoid(self.heads["sigma_rep"](u.detach())) * SIGMA_SCALE + SIGMA_FLOOR)
        return torch.cat(maps, 1)


def slots_of(out) -> int:
    """The slots a network's output holds (`TwoStageUNet`'s layout): 9 channels each and the background,
    13 channels one slot with ``sigma_rep``'s three."""
    c = out.shape[1]
    return 1 if c == 13 else (c - 1) // 9


BACKBONES = {"unet2": TwoStageUNet}
TIME_MODELS = {"attention": TimeAttention, "convgru": TimeGRU}


def build(parts, window: int) -> nn.Module:
    """The network the parts name: a backbone (``unet2``) with a time model (``frame_stack``, DECODE's,
    or one of `TIME_MODELS`).  The time model's ``frames`` setting is a record only: the window is the
    simulation's."""
    tm = parts.time_model
    extra = {k: v for k, v in (tm.settings or {}).items() if k != "frames"}
    return BACKBONES[parts.backbone.name](window=window, time_model=tm.name, time_settings=extra or None,
                                          **parts.backbone.settings)


def parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
