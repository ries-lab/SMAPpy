"""A trained model localizing frames: the interface `smappy.window_engine.WindowEngine` drives.

``Predictor(path)`` reads a model file of `train.py`; ``predictor(photons, first_frame)`` takes
frames in photons with ``window`` frames of context on either side and gives SMAPpy's pixel columns.
The frames are cropped to a multiple of the network's fold (its strides), centred -- the edge pixels
lost, as the DECODE fitter loses them (`docs/decode/poc-plan.md`, item 17) -- and mirrored along the
columns when the data's camera reads the chip the other way from the one the model was trained for.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..training import model_file
from . import extract, network
from .settings import Part, Parts


class Predictor:
    def __init__(self, path, device=None, *, frames_mirrored: bool = False, threshold: float | None = None,
                 extraction: str | None = None, batch: int = 32):
        tensors, meta = model_file.read(Path(path))
        self.metadata = meta
        s = meta["settings"]
        parts = Parts(**{k: Part(**v) for k, v in s["parts"].items()})
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = network.build(parts, int(meta["window"])).to(self.device).eval()
        self.model.load_state_dict({k: torch.as_tensor(np.asarray(v)) for k, v in tensors.items()})
        self.scaling = network.Scaling(**meta["scaling"])
        self.window = (int(meta["window"]) - 1) // 2
        self.threshold = threshold if threshold is not None else parts.schedule.settings.get("threshold", 0.5)
        self.extraction = extraction or parts.extraction.name
        self.frames_mirrored = frames_mirrored
        self.batch = batch
        self.fold = self.model.fold

    def crop(self, h, w):
        """The rows and columns kept: (row0, col0, rows, columns)."""
        hh, ww = h - h % self.fold, w - w % self.fold
        return (h - hh) // 2, (w - ww) // 2, hh, ww

    def __call__(self, photons, first_frame: int = 0) -> dict:
        photons = np.asarray(photons, np.float32)
        k = self.window
        n = len(photons) - 2 * k
        r0, c0, hh, ww = self.crop(*photons.shape[1:])
        x = torch.as_tensor(photons[:, r0:r0 + hh, c0:c0 + ww])
        if self.frames_mirrored:
            x = torch.flip(x, (-1,))
        x = (x - self.scaling.input_offset) / self.scaling.input_scale
        parts = []
        with torch.no_grad():
            for a in range(0, n, self.batch):
                b = min(a + self.batch, n)
                stack = torch.stack([x[i:i + 2 * k + 1] for i in range(a, b)]).to(self.device)
                e = extract.extract(self.model(stack), self.scaling, self.threshold, self.extraction)
                e = {name: v.detach().cpu().numpy() for name, v in e.items()}
                e["frame"] = e["frame"] + a
                parts.append(e)
        cols = {name: np.concatenate([p[name] for p in parts]) if parts else np.zeros(0) for name in
                ("frame", "x", "y", "z_nm", "photons", "prob", "photons_err", "x_err", "y_err", "z_err_nm", "background")}
        x_pix = (ww - 1) - cols["x"] if self.frames_mirrored else cols["x"]
        return {"frame": cols["frame"].astype(np.int64) + int(first_frame), "x_pix": x_pix + c0, "y_pix": cols["y"] + r0,
                "z_nm": cols["z_nm"], "photons": cols["photons"], "background": cols["background"], "prob": cols["prob"],
                "x_err_pix": cols["x_err"], "y_err_pix": cols["y_err"], "z_err_nm": cols["z_err_nm"],
                "photons_err": cols["photons_err"]}
