"""Training: fresh simulated batches, a fixed test set, a line of progress an epoch, a model file.

    python -m smappy.netfit train FOLDER --calibration CAL.h5 [--settings S.json] [--device cuda]

``FOLDER`` gets ``settings.json`` (what is trained), ``progress.jsonl`` (the Train window's format:
`smappy.training.progress`; the early stop reads it), ``checkpoint.pt`` (every epoch; a run started
again on the folder resumes from it) and ``model.safetensors`` (the latest epoch's network with its
settings, scaling, parameters and training time: what the predictor and the benchmark read).

**The test set** is drawn once from the seed, ``test_samples`` samples, and the CRLB of each of its
molecules computed once (`metrics.crlbs_torch`); every epoch the network localizes it at the schedule's ``threshold`` and
`metrics.evaluate` scores it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..training import model_file, progress
from . import extract, loss, metrics, network
from .render import SplineRenderer
from .settings import CameraModel, Part, Parts, SimulationSettings, TrainingSettings
from .simulate import draw_batch

METHOD = "netfit-v1"


def load_settings(path) -> TrainingSettings:
    d = json.loads(Path(path).read_text())
    parts = Parts(**{k: Part(**v) for k, v in d.get("parts", {}).items()})
    sim = SimulationSettings(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.get("simulation", {}).items()})
    cam = CameraModel(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.get("camera", {}).items()})
    return TrainingSettings(simulation=sim, camera=cam, parts=parts, calibration=d.get("calibration"), seed=d.get("seed", 0))


def _calibration(path, em: bool):
    from ..io.calibration import load_spline_calibration, matched_to_em
    cal = load_spline_calibration(path)
    # drawn as the camera the model is for reads the chip: an EM register mirrors it
    return cal, matched_to_em(cal, em)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _found(out, scaling, threshold, how, pixel_nm):
    e = extract.extract(out, scaling, threshold, how)
    to = lambda v: v.detach().cpu().numpy().astype(np.float64)
    return {"frame": to(e["frame"]).astype(np.int64), "xyz": np.c_[to(e["x"]), to(e["y"]), to(e["z_nm"])],
            "err": np.c_[to(e["x_err"]) * pixel_nm[0], to(e["y_err"]) * pixel_nm[1], to(e["z_err_nm"])]}


def train(folder, settings: TrainingSettings, device: str = "cpu", epochs: int | None = None, log=print,
          early_stop: bool = True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "settings.json").write_text(json.dumps(settings.to_dict(), indent=1))
    sim, cam, parts = settings.simulation, settings.camera, settings.parts
    sch = parts.schedule.settings
    epochs = epochs or sch["epochs"]
    torch.manual_seed(settings.seed)
    _, drawn = _calibration(settings.calibration, bool(cam.em_gain))
    renderer = SplineRenderer(drawn, device)
    scaling = network.Scaling.for_simulation(sim)
    model = network.build(parts, sim.window).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=sch["lr"], weight_decay=sch["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=sch["step_epochs"], gamma=sch["gamma"])
    # the test set: drawn once from the seed, its bounds once
    g_test = torch.Generator(device).manual_seed(settings.seed + 1)
    test = draw_batch(renderer, sim, cam, sch["test_samples"], g_test)
    m = test.mask.cpu().numpy()
    frame = np.nonzero(m)[0]
    xyz = test.xyz.cpu().numpy()[m].astype(np.float64)
    ph = test.photons.cpu().numpy()[m]
    # the bound of the PSF as drawn (a mirror changes no bound), in torch: no compiled SMAPpy needed
    bound = metrics.crlbs_torch(renderer, ph, test.background.cpu().numpy()[frame], xyz[:, 2], cam.pixel_nm,
                                bool(cam.em_gain))
    truth = {"frame": frame, "xyz": xyz, "crlb": bound}
    start, seconds = 0, 0.0
    ckpt = folder / "checkpoint.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start, seconds = state["epoch"] + 1, state["seconds"]
        torch.set_rng_state(state["rng"].cpu())       # map_location moved it with the rest
        progress.append(folder, "start", start, resumed=True, device=device)
    else:
        progress.append(folder, "start", 0, device=device, method=METHOD, parameters=network.parameters(model),
                        settings=settings.to_dict())
    g = torch.Generator(device).manual_seed(settings.seed + 1000 + start)
    steps = max(1, sch["samples_per_epoch"] // sch["batch"])
    meta_base = {"method": METHOD, "settings": settings.to_dict(), "scaling": asdict(scaling), "window": sim.window,
                 "parameters": network.parameters(model), "calibration_sha256": _sha256(settings.calibration),
                 "calibration_em_on": bool(cam.em_gain)}
    for epoch in range(start, epochs):
        t0 = time.perf_counter()
        model.train()
        total, parts_sum = 0.0, {}
        for _ in range(steps):
            batch = draw_batch(renderer, sim, cam, sch["batch"], g)
            x = (batch.frames - scaling.input_offset) / scaling.input_scale
            value, t = loss.loss(model(x), batch, scaling, parts.loss)
            opt.zero_grad(set_to_none=True)
            value.backward()
            if sch.get("clip_norm"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), sch["clip_norm"])
            opt.step()
            total += float(value)
            for k, v in t.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v
        sched.step()
        model.eval()
        with torch.no_grad():
            outs, test_loss = [], 0.0
            for a in range(0, len(test.frames), 128):
                sub = _slice(test, a, a + 128)
                out = model((sub.frames - scaling.input_offset) / scaling.input_scale)
                test_loss += float(loss.loss(out, sub, scaling, parts.loss)[0]) * len(sub.frames)
                outs.append(out)
            out = torch.cat(outs)
        found = _found(out, scaling, sch["threshold"], parts.extraction.name, cam.pixel_nm)
        score = metrics.evaluate(truth, found, cam.pixel_nm)
        score.update({f"loss_{k}": v / steps for k, v in parts_sum.items()})
        dt = time.perf_counter() - t0
        seconds += dt
        progress.append(folder, "epoch", epoch, loss_train=total / steps, loss_test=test_loss / len(test.frames),
                        lr=sched.get_last_lr()[0], seconds=round(dt, 3), metrics=score)
        log(f"epoch {epoch}: loss {total / steps:.2f} / {test_loss / len(test.frames):.2f}  J {score['jaccard']:.3f}  "
            f"lat {score.get('rmse_lat_nm', float('nan')):.1f}  ax {score.get('rmse_ax_nm', float('nan')):.1f}  "
            f"KS {score.get('pixel_ks') or float('nan'):.3f}  ({dt:.0f} s)")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "seconds": seconds, "rng": torch.get_rng_state()}, ckpt)
        tensors = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        model_file.write(folder / "model.safetensors", tensors,
                         {**meta_base, "epoch": epoch, "epochs": epochs, "train_seconds": round(seconds, 1),
                          "device": device})
        why = progress.converged(progress.read(folder)) if early_stop else None
        if why:
            progress.append(folder, "finished", epoch, reason=why)
            log(f"stopped: {why}")
            return
    progress.append(folder, "finished", epochs - 1, reason="epochs")


def _slice(batch, a, b):
    from .simulate import Batch
    return Batch(batch.frames[a:b], batch.xyz[a:b], batch.photons[a:b], batch.mask[a:b], batch.background[a:b])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m smappy.netfit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("folder")
    t.add_argument("--calibration")
    t.add_argument("--settings")
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.add_argument("--epochs", type=int)
    t.add_argument("--all-epochs", action="store_true", help="no early stop")
    a = ap.parse_args(argv)
    folder = Path(a.folder)
    if a.settings:
        settings = load_settings(a.settings)
    elif (folder / "settings.json").exists():
        settings = load_settings(folder / "settings.json")
    else:
        settings = TrainingSettings()
    if a.calibration:
        settings.calibration = str(Path(a.calibration).resolve())
    if not settings.calibration:
        ap.error("a calibration is needed: --calibration")
    train(folder, settings, a.device, a.epochs, early_stop=not a.all_epochs)
