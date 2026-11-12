"""A fixed test set per context, drawn once and kept: every training and every variant is scored on
the same samples (decided 2026-11-12).

    python -m smappy.netfit testset OUT.npz --calibration CAL.h5 [--settings S.json] [--samples 2048]

Samples are drawn as training draws them (`simulate.draw_batch`: 3 frames, molecules blinking over
them, the camera) from the settings' simulation and camera, but on frames of ``frame_px`` 64 whatever
a variant trains on -- the networks are convolutional -- with each molecule's CRLB (`metrics.crlbs_torch`)
kept beside its truth.  A training scores its first ``test_samples`` (512) every epoch, for Follow and
the early stop; `evaluate` scores a converged model on all of them: at 2048 samples of 64 x 64 and 50
molecules a 40 x 40 pixels, about 260 000 molecules, the KS distance's counting noise is about 0.002.
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from . import metrics
from .render import SplineRenderer
from .simulate import Batch, draw_batch

FRAME_PX = 64
SAMPLES = 2048


def draw(path, calibration, sim, cam, samples: int = SAMPLES, seed: int = 12345, device: str = "cpu"):
    from .train import _calibration, _sha256
    sim = replace(sim, frame_px=FRAME_PX)
    _, drawn = _calibration(calibration, bool(cam.em_gain))
    renderer = SplineRenderer(drawn, device)
    g = torch.Generator(device).manual_seed(seed)
    parts = [draw_batch(renderer, sim, cam, min(256, samples - a), g) for a in range(0, samples, 256)]
    k = max(p.xyz.shape[1] for p in parts)

    def pad(t, fill=0):
        out = torch.full((t.shape[0], k, *t.shape[2:]), fill, dtype=t.dtype, device=t.device)
        out[:, :t.shape[1]] = t
        return out
    frames = torch.cat([p.frames for p in parts]).cpu().numpy().astype(np.float32)
    xyz = torch.cat([pad(p.xyz) for p in parts]).cpu().numpy()
    photons = torch.cat([pad(p.photons) for p in parts]).cpu().numpy()
    mask = torch.cat([pad(p.mask, False) for p in parts]).cpu().numpy()
    background = torch.cat([p.background for p in parts]).cpu().numpy()
    frame = np.nonzero(mask)[0]
    crlb = metrics.crlbs_torch(renderer, photons[mask], background[frame], xyz[mask][:, 2], cam.pixel_nm, bool(cam.em_gain))
    meta = {"simulation": asdict(sim), "camera": asdict(cam), "calibration": str(calibration),
            "calibration_sha256": _sha256(calibration), "seed": seed, "samples": samples,
            "molecules": int(mask.sum())}
    np.savez(path, frames=frames, xyz=xyz, photons=photons, mask=mask, background=background, crlb=crlb,
             meta=json.dumps(meta))
    return meta


def load(path, device="cpu", samples: int | None = None):
    """(Batch, truth for `metrics.evaluate`, meta) of the first ``samples`` samples."""
    d = np.load(path)
    n = samples or len(d["frames"])
    t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a[:n]), dtype=dt, device=device)
    mask = np.asarray(d["mask"][:n])
    batch = Batch(t(d["frames"]), t(d["xyz"]), t(d["photons"]), t(d["mask"], torch.bool), t(d["background"]))
    frame = np.nonzero(mask)[0]
    truth = {"frame": frame, "xyz": np.asarray(d["xyz"][:n])[mask].astype(np.float64),
             "crlb": np.asarray(d["crlb"])[:int(mask.sum())]}
    return batch, truth, json.loads(str(d["meta"]))


def evaluate(model_path, path, device=None, threshold=None, extraction=None) -> dict:
    """A model scored on the whole test set (`metrics.evaluate`), as a training scores it per epoch."""
    from . import extract
    from .predict import Predictor
    p = Predictor(model_path, device, threshold=threshold, extraction=extraction)
    batch, truth, meta = load(path, p.device)
    pixel_nm = tuple(meta["camera"]["pixel_nm"])
    outs = []
    with torch.no_grad():
        for a in range(0, len(batch.frames), 128):
            x = (batch.frames[a:a + 128] - p.scaling.input_offset) / p.scaling.input_scale
            outs.append(p.model(x))
    e = extract.extract(torch.cat(outs), p.scaling, p.threshold, p.extraction)
    to = lambda v: v.detach().cpu().numpy().astype(np.float64)
    found = {"frame": to(e["frame"]).astype(np.int64), "xyz": np.c_[to(e["x"]), to(e["y"]), to(e["z_nm"])],
             "err": np.c_[to(e["x_err"]) * pixel_nm[0], to(e["y_err"]) * pixel_nm[1], to(e["z_err_nm"])]}
    out = metrics.evaluate(truth, found, pixel_nm)
    out.update(model=str(model_path), test_set=str(path), molecules=len(truth["frame"]), threshold=p.threshold,
               extraction=p.extraction)
    return out


def default_path(context: str) -> Path:
    import os
    data = Path(os.environ.get("DECODE_SMLM_DATA", Path.home() / ".cache" / "decode_smlm"))
    return data / "netfit" / "testsets" / f"{context}.npz"
