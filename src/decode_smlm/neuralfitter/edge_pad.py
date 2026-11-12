"""Padding drawn into the training's frames, so that the network learns what padding is.

Not DECODE's: a SMAPpy addition (plan §6, item 17).  The U-Net takes frames of a multiple of 8
pixels, and the original cut the rest off (``AutoCenterCrop``), losing up to 7 edge rows and
columns.  Padding them instead with a value no frame holds costs the whole frame when the network
never saw it -- the pad reaches every pixel through the downsampled levels, and at input -1 the
inside's recall fell from 0.93 to 0.78 (``benchmarks/decode/edges``).  So the training shows it:
each side of a training frame is, with probability ``p``, replaced over a band of 1 -- ``max_px``
pixels by ``value`` (in the network's input, after the amplitude rescale), and the emitters under
a band are dropped from the targets -- the network is to find nothing in padding and to treat its
edge as the data's.  The test set is left as it is, so its metrics stay comparable.

Set by ``HyperParameter.edge_pad: {p: 0.2, max_px: 7, value: -1.0}`` in the job's method.yaml;
a model trained with it records it (``edge_pad`` in the model file), and the predictor pads with
its value instead of cropping.
"""
from __future__ import annotations

import torch


class EdgePad:
    def __init__(self, p: float, max_px: int, value: float):
        if not 0 <= p <= 1 or max_px < 1:
            raise ValueError(f"edge_pad: p in [0, 1] and max_px >= 1, not p={p}, max_px={max_px}")
        self.p, self.max_px, self.value = float(p), int(max_px), float(value)

    @classmethod
    def parse(cls, param):
        spec = getattr(param.HyperParameter, "edge_pad", None)
        if not spec:
            return None
        spec = spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
        return cls(p=spec.get("p", 0.2), max_px=spec.get("max_px", 7), value=spec.get("value", -1.0))

    def as_dict(self) -> dict:
        return {"p": self.p, "max_px": self.max_px, "value": self.value}

    def widths(self):
        """Bands (first-axis low, high, second-axis low, high), each 0 or 1 -- max_px."""
        on = torch.rand(4) < self.p
        width = torch.randint(1, self.max_px + 1, (4,))
        return [int(w) if o else 0 for o, w in zip(on, width)]

    def forward(self, frames: torch.Tensor, emitter):
        """``frames`` (C, H, W) of the network's input and their target emitters, padded alike."""
        lo0, hi0, lo1, hi1 = self.widths()
        if not (lo0 or hi0 or lo1 or hi1):
            return frames, emitter
        h, w = frames.shape[-2:]
        frames = frames.clone()
        frames[..., :lo0, :] = self.value
        frames[..., h - hi0:, :] = self.value
        frames[..., :, :lo1] = self.value
        frames[..., :, w - hi1:] = self.value
        if emitter is not None and len(emitter):
            x, y = emitter.xyz[:, 0], emitter.xyz[:, 1]          # px: x along the first axis, as the frames
            keep = (x >= lo0 - 0.5) & (x < h - hi0 - 0.5) & (y >= lo1 - 0.5) & (y < w - hi1 - 0.5)
            emitter = emitter[keep]
        return frames, emitter
