"""A trained model localizing frames: the interface `smappy.window_engine.WindowEngine` drives.

``Predictor(path)`` reads a model file of `train.py`; ``predictor(photons, first_frame)`` takes
frames in photons with ``window`` frames of context on either side and gives SMAPpy's pixel columns.
The frames are cropped to a multiple of the network's fold (its strides), centred -- the edge pixels
lost, as the DECODE fitter loses them (`docs/decode/poc-plan.md`, item 17) -- and mirrored along the
columns when the data's camera reads the chip the other way from the one the model was trained for.

A two-channel model (`docs/decode/plex-plan.md` §16) takes the channels already on its grid, ``(n, 2,
rows, columns)``, and their offset maps ``aux`` ``(2, rows, columns)`` -- `channels.TwoChannels.cut` and
``offsets`` -- and gives positions on channel 0's grid, with each channel's photons (colour) and
background beside them.

The DECODE fitter runs it (`smappy.training.predictors.load`, `plugins.fit_decode`) as it runs the
port's predictor: the same columns, ``window``, ``device``, ``raw_th`` and ``frames_mirrored``.  The
photons need no correction for the camera's ``qe``: the training drew frames with it and converted
them back with ``conversion`` alone (`simulate.camera`), as the fitter converts the data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..training import model_file
from . import extract, network
from .settings import ChannelSettings, Part, Parts


#: inference memory a pixel of a batch, the default network (48 features, 3 frames): 4 700 bytes on
#: Apple's GPU, the driver's pools included (8 frames of 256 x 224: 2.2 GB, 2026-11-22); DECODE's
#: SigmaMUNet takes 2 100 (`decode_smlm.predict.BYTES_PER_PIXEL`)
BYTES_PER_PIXEL = 5000
MEMORY_SHARE = 0.5          # of a CUDA device's free memory a batch may take
MPS_BUDGET = 3e9            # Apple's GPU shares the computer's memory: a fixed budget, as DECODE's
MAX_BATCH, CPU_BATCH = 256, 8


def auto_batch(device: str, rows: int, columns: int) -> int:
    """Frames of ``rows`` x ``columns`` a batch takes on ``device``: half of a CUDA device's free memory,
    at most `MPS_BUDGET` on Apple's GPU, `CPU_BATCH` on the CPU.  On Apple's GPU the frame rate still
    rises with the batch (9.8 frames a second at 4, 26.5 at 8, 47 at 16; 256 x 224)."""
    kind = str(device).split(":")[0]
    if kind == "cuda":
        budget = MEMORY_SHARE * torch.cuda.mem_get_info(torch.device(device))[0]
    elif kind == "mps":
        budget = min(MPS_BUDGET, MEMORY_SHARE * (torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()))
    else:
        return CPU_BATCH
    return max(1, min(MAX_BATCH, int(budget // (BYTES_PER_PIXEL * rows * columns))))


class Predictor:
    def __init__(self, path, device=None, *, frames_mirrored: bool = False, threshold: float | None = None,
                 extraction: str | None = None, batch: int = 32):
        tensors, meta = model_file.read(Path(path))
        self.metadata = meta
        s = meta["settings"]
        parts = Parts(**{k: Part(**v) for k, v in s["parts"].items()})
        self.device = device if device not in (None, "auto") else (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        ch = dict(s.get("channels") or {})
        self.channels = ChannelSettings(**{k: tuple(v) if k == "share" else v for k, v in ch.items()})
        self.model = network.build(parts, int(meta["window"]), self.channels).to(self.device).eval()
        self.model.load_state_dict({k: torch.as_tensor(np.asarray(v)) for k, v in tensors.items()})
        self.scaling = network.Scaling(**meta["scaling"])
        self.window = (int(meta["window"]) - 1) // 2
        self.threshold = threshold if threshold is not None else parts.schedule.settings.get("threshold", 0.5)
        self.extraction = extraction or parts.extraction.name
        self.frames_mirrored = frames_mirrored
        self.batch = batch                 # 0: `auto_batch`
        self.fold = self.model.fold
        self.scale = getattr(self.model, "out_scale", 1)

    @property
    def raw_th(self) -> float:
        """The threshold under DECODE's name, for the fitter's record."""
        return self.threshold

    def crop(self, h, w):
        """The rows and columns kept: (row0, col0, rows, columns)."""
        hh, ww = h - h % self.fold, w - w % self.fold
        return (h - hh) // 2, (w - ww) // 2, hh, ww

    def __call__(self, photons, first_frame: int = 0, aux=None) -> dict:
        photons = np.asarray(photons, np.float32)
        two = self.channels.kind != ""
        if two and (photons.ndim != 4 or aux is None):
            raise ValueError("a two-channel model takes frames (n, 2, rows, columns) and their offset maps")
        if two and self.frames_mirrored:
            raise ValueError("a two-channel model takes the channels as its calibration has them")
        k = self.window
        n = len(photons) - 2 * k
        r0, c0, hh, ww = self.crop(*photons.shape[-2:])
        x = torch.as_tensor(photons[..., r0:r0 + hh, c0:c0 + ww])
        if self.frames_mirrored:
            x = torch.flip(x, (-1,))
        x = (x - self.scaling.input_offset) / self.scaling.input_scale
        if two:
            x = x.transpose(0, 1)                                            # (2, n, h, w)
            maps = torch.as_tensor(np.asarray(aux, np.float32)[:, r0:r0 + hh, c0:c0 + ww]).to(self.device)
        layout = self.model.layout
        parts = []
        size = self.batch or auto_batch(self.device, hh * self.scale, ww * self.scale) // (2 if two else 1)
        size = max(1, size)
        with torch.no_grad():
            for a in range(0, n, size):
                b = min(a + size, n)
                # the windows a..b-1 as one movie: each frame through the per-frame U-Net once
                if two:
                    out = self.model(x[:, a:b + 2 * k][None].to(self.device), b - a, maps[None])
                else:
                    out = self.model(x[a:b + 2 * k][None].to(self.device), b - a)
                e = extract.extract(out, self.scaling, self.threshold, self.extraction, self.scale, layout=layout)
                e = {name: v.detach().cpu().numpy() for name, v in e.items()}
                e["frame"] = e["frame"] + a
                parts.append(e)
        names = ("frame", "x", "y", "z_nm", "photons", "prob", "photons_err", "x_err", "y_err", "z_err_nm", "background")
        names += tuple(c for c in ("photons_c", "photons_c_err", "background_c") if parts and c in parts[0])
        # a model with the reporting head reports its sigma; the mixture's is kept beside it, to compare
        mix = ("x_err_mix", "y_err_mix", "z_err_mix_nm") if parts and "x_err_mix" in parts[0] else ()
        cols = {name: np.concatenate([p[name] for p in parts]) if parts else np.zeros(0) for name in names + mix}
        x_pix = (ww - 1) - cols["x"] if self.frames_mirrored else cols["x"]
        return {"frame": cols["frame"].astype(np.int64) + int(first_frame), "x_pix": x_pix + c0, "y_pix": cols["y"] + r0,
                "z_nm": cols["z_nm"], "photons": cols["photons"], "background": cols["background"], "prob": cols["prob"],
                "x_err_pix": cols["x_err"], "y_err_pix": cols["y_err"], "z_err_nm": cols["z_err_nm"],
                "photons_err": cols["photons_err"],
                **{c: cols[c] for c in ("photons_c", "photons_c_err", "background_c") if c in cols},
                **({"x_err_mix_pix": cols["x_err_mix"], "y_err_mix_pix": cols["y_err_mix"],
                    "z_err_mix_nm": cols["z_err_mix_nm"]} if mix else {})}
