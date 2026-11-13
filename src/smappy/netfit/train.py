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
import math
import queue
import threading
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
# SMAPpy's early stop (`training.progress`), and the pixelation too: version 1's KS kept falling for
# epochs after the loss, Jaccard and RMSEs had settled.  0.002 is the counting noise of the fixed test
# set's whole KS; over the 512 samples scored each epoch the set is fixed, so a change is the model's
STOP_RULE = {**progress.EARLY_STOP, "ks_tol": 0.002}
# The stop rule's epochs are of this many samples (the port's, and netfit's default since
# 2026-11-13): an epoch of another size has its window and minimum rescaled, so that the patience
# is the same number of samples.  At 10 000 a sample, 100 epochs had been 1 M samples, not 3.3 M.
REFERENCE_EPOCH = 32768
# The rate is the one for a batch of this many; another batch scales it (``lr_batch_scaling``).
REFERENCE_BATCH = 64
# Models kept, and scored on the whole fixed test set, once this many samples were seen: a training's
# quality at a budget and how soon it got there, from one run (2026-11-13).
SNAPSHOTS = (1e6, 2e6, 3e6, 4.5e6, 6e6, 10e6)


def stop_rule(samples_per_epoch: int) -> dict:
    f = REFERENCE_EPOCH / samples_per_epoch
    return {**STOP_RULE, "window": max(1, round(STOP_RULE["window"] * f)),
            "min_epochs": max(1, round(STOP_RULE["min_epochs"] * f))}


def auto_batch(device) -> int:
    """The batch for the device: on an H100 (2026-11-13) a sample took 0.387, 0.299, 0.281, 0.269 ms
    at batches of 64, 128, 256, 512 (1 frame; 3 frames 0.677, 0.600, 0.574, 0.561) -- most of the
    gain by 256, at 5 to 8 GB."""
    if not str(device).startswith("cuda"):
        return REFERENCE_BATCH
    gb = torch.cuda.get_device_properties(torch.device(device)).total_memory / 1e9
    return 256 if gb >= 16 else 128 if gb >= 8 else REFERENCE_BATCH


def batch_and_rate(sch: dict, device) -> tuple:
    """(batch, rate): ``batch`` a number or "auto", the rate scaled from the reference batch's by
    ``lr_batch_scaling`` -- "sqrt" (the default: Adam's steps are normalised, so the square root, not
    the linear rule of SGD), "linear" or "none"."""
    batch = auto_batch(device) if sch["batch"] == "auto" else int(sch["batch"])
    k = batch / REFERENCE_BATCH
    scale = {"none": 1.0, "sqrt": math.sqrt(k), "linear": k}[sch.get("lr_batch_scaling", "sqrt")]
    return batch, sch["lr"] * scale


def schedule(opt, sch: dict, batch: int):
    """The rate times gamma every ``decay_samples`` samples (by default every ``step_epochs`` epochs,
    as StepLR stepped once an epoch did), stepped every batch: a schedule of samples, whatever the
    epoch's size or the batch."""
    decay = sch.get("decay_samples") or sch["step_epochs"] * sch["samples_per_epoch"]
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda step: sch["gamma"] ** ((step * batch) // decay))


class _Prefetch:
    """The next batches drawn in a thread, on a CUDA stream of their own, while the network trains on
    this one: drawing is a quarter (1 frame) to a third (3 frames) of a step, and it waits on the GPU
    several times a batch (molecule counts), which held the network up.  The same generator draws
    the same batches in the same order as drawing them in the loop did."""

    def __init__(self, draw, device, depth: int = 2):
        self.draw = draw
        self.cuda = str(device).startswith("cuda")
        self.device = torch.device(device)
        if self.cuda and self.device.index is None:      # the thread sets its device by index
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.stream = torch.cuda.Stream(self.device) if self.cuda else None
        self.q = queue.Queue(depth)
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            if self.cuda:
                torch.cuda.set_device(self.device)
            while not self.stop.is_set():
                if self.cuda:
                    with torch.cuda.stream(self.stream):
                        b = self.draw()
                        done = torch.cuda.Event()
                        done.record(self.stream)
                else:
                    b, done = self.draw(), None
                self._put((b, done))
        except Exception as e:  # noqa: BLE001 -- handed to the loop, which raises it
            self.error = e
            self._put(None)

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.q.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def get(self):
        item = self.q.get()
        if item is None:
            raise RuntimeError("drawing a training batch failed") from self.error
        b, done = item
        if done is not None:
            here = torch.cuda.current_stream(self.device)
            here.wait_event(done)
            for t in (b.frames, b.xyz, b.photons, b.mask, b.background, b.photons_window, b.background_window):
                if t is not None:
                    t.record_stream(here)        # not freed for the drawing stream while used here
        return b

    def close(self):
        self.stop.set()
        self.thread.join()


def load_settings(path) -> TrainingSettings:
    d = json.loads(Path(path).read_text())
    parts = Parts(**{k: Part(**v) for k, v in d.get("parts", {}).items()})
    raw = dict(d.get("simulation", {}))
    if "emitters_per_frame" in raw:          # before the density (2026-11-12): per frame of frame_px
        raw["density_per_40px"] = raw.pop("emitters_per_frame") * (40.0 / raw.get("frame_px", 40)) ** 2
    sim = SimulationSettings(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
    cam = CameraModel(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.get("camera", {}).items()})
    return TrainingSettings(simulation=sim, camera=cam, parts=parts, calibration=d.get("calibration"),
                            test_set=d.get("test_set"), seed=d.get("seed", 0))


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
    batch_size, lr = batch_and_rate(sch, device)
    steps = max(1, sch["samples_per_epoch"] // batch_size)
    per_epoch = steps * batch_size
    if sch.get("max_samples"):
        epochs = min(epochs, math.ceil(sch["max_samples"] / per_epoch))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=sch["weight_decay"])
    sched = schedule(opt, sch, batch_size)
    bf16 = sch.get("precision", "fp32") == "bf16"
    if bf16 and not str(device).startswith("cuda"):
        raise ValueError("precision bf16 needs a CUDA device")
    snapshots = sorted(float(v) for v in sch.get("snapshots", SNAPSHOTS))
    rule = stop_rule(sch["samples_per_epoch"])
    if settings.test_set:
        # the context's fixed test set: the same samples for every variant (`testset.py`)
        from . import testset
        test, truth, _ = testset.load(settings.test_set, device, sch["test_samples"], window=sim.window)
        # its loss over the targets the training learns: those from target_photons_min (the metrics
        # count from metrics.PHOTONS_MIN, the same 100 where DECODE's settings are kept)
        test.mask = test.mask & (test.photons >= sim.target_photons_min)
    else:
        # drawn once from the seed, its bounds once
        g_test = torch.Generator(device).manual_seed(settings.seed + 1)
        test = draw_batch(renderer, sim, cam, sch["test_samples"], g_test)
        m = test.mask.cpu().numpy()
        frame = np.nonzero(m)[0]
        xyz = test.xyz.cpu().numpy()[m].astype(np.float64)
        ph = test.photons.cpu().numpy()[m]
        # the bound of the PSF as drawn (a mirror changes no bound), in torch: no compiled SMAPpy needed
        bound = metrics.crlbs_torch(renderer, ph, test.background.cpu().numpy()[frame], xyz[:, 2], cam.pixel_nm,
                                    bool(cam.em_gain))
        truth = {"frame": frame, "xyz": xyz, "photons": ph, "crlb": bound}
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
                        settings=settings.to_dict(), stop_rule=rule, epochs=epochs, batch=batch_size, lr=lr)
    g = torch.Generator(device).manual_seed(settings.seed + 1000 + start)
    meta_base = {"method": METHOD, "settings": settings.to_dict(), "scaling": asdict(scaling), "window": sim.window,
                 "parameters": network.parameters(model), "calibration_sha256": _sha256(settings.calibration),
                 "calibration_em_on": bool(cam.em_gain)}
    # on a CPU the thread would only take cores from the network
    feed = _Prefetch(lambda: draw_batch(renderer, sim, cam, batch_size, g), device) \
        if sch.get("prefetch", True) and str(device).startswith("cuda") else None
    try:
        _epochs_loop(folder, settings, model, opt, sched, feed, renderer, g, test, truth, scaling, start, epochs,
                     steps, batch_size, seconds, meta_base, device, bf16, snapshots, rule, early_stop, log)
    finally:
        if feed is not None:
            feed.close()


def _epochs_loop(folder, settings, model, opt, sched, feed, renderer, g, test, truth, scaling, start, epochs,
                 steps, batch_size, seconds, meta_base, device, bf16, snapshots, rule, early_stop, log):
    sim, cam, parts = settings.simulation, settings.camera, settings.parts
    sch = parts.schedule.settings
    ckpt = folder / "checkpoint.pt"
    per_epoch = steps * batch_size
    for epoch in range(start, epochs):
        t0 = time.perf_counter()
        model.train()
        total, parts_sum = 0.0, {}
        for _ in range(steps):
            batch = feed.get() if feed is not None else draw_batch(renderer, sim, cam, batch_size, g)
            x = (batch.frames - scaling.input_offset) / scaling.input_scale
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                out = model(x)
            value, t = loss.loss(out.float(), batch, scaling, parts.loss)
            opt.zero_grad(set_to_none=True)
            value.backward()
            if sch.get("clip_norm"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), sch["clip_norm"])
            opt.step()
            sched.step()
            total = total + value.detach()
            for k, v in t.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v
        total = float(total)
        parts_sum = {k: float(v) for k, v in parts_sum.items()}
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
        seen = (epoch + 1) * per_epoch
        for at in (a for a in snapshots if seen - per_epoch < a <= seen):
            _snapshot(folder, settings, epoch, at, seen, tensors, meta_base, seconds, device, log)
        why = progress.converged(progress.read(folder), rule) if early_stop else None
        if why:
            progress.append(folder, "finished", epoch, reason=why)
            log(f"stopped: {why}")
            return
    progress.append(folder, "finished", epochs - 1, reason="epochs")


def _snapshot(folder, settings, epoch, at, seen, tensors, meta_base, seconds, device, log):
    """The model as it is after ``at`` samples, kept as ``model_<at>M.safetensors`` and, with a fixed
    test set, scored on all of it (`testset.evaluate`): a progress line "snapshot"."""
    path = folder / f"model_{at / 1e6:g}M.safetensors"
    model_file.write(path, tensors, {**meta_base, "epoch": epoch, "samples": seen, "train_seconds": round(seconds, 1),
                                     "device": device})
    if not settings.test_set:
        return
    from . import testset
    r = testset.evaluate(path, settings.test_set, device)
    keys = ("jaccard", "recall", "precision", "rmse_lat_nm", "rmse_ax_nm", "pixel_ks", "crlb_width_lat",
            "crlb_width_ax", "core_rmse_lat_nm", "core_rmse_ax_nm", "core_pixel_ks")
    progress.append(folder, "snapshot", epoch, samples=seen, at=at, seconds=round(seconds, 1),
                    metrics={k: r[k] for k in keys if r.get(k) is not None})
    log(f"snapshot at {at / 1e6:g} M samples: J {r['jaccard']:.4f}  near focus {r.get('core_rmse_lat_nm', float('nan')):.1f} nm"
        f"  KS {r.get('pixel_ks') or float('nan'):.4f}")


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
    ts = sub.add_parser("testset", help="draw a context's fixed test set once")
    ts.add_argument("out")
    ts.add_argument("--calibration", required=True)
    ts.add_argument("--settings")
    ts.add_argument("--samples", type=int, default=2048)
    ev = sub.add_parser("evaluate", help="score a model on a whole test set")
    ev.add_argument("model")
    ev.add_argument("test_set")
    ev.add_argument("--device")
    ev.add_argument("--out", help="write the scores here as JSON")
    ev.add_argument("--shuffled", action="store_true", help="the neighbouring frames from other samples")
    a = ap.parse_args(argv)
    if a.cmd == "testset":
        from . import testset
        s = load_settings(a.settings) if a.settings else TrainingSettings()
        print(testset.draw(a.out, Path(a.calibration).resolve(), s.simulation, s.camera, a.samples))
        return
    if a.cmd == "evaluate":
        from . import testset
        r = testset.evaluate(a.model, a.test_set, a.device, shuffled=a.shuffled)
        if a.out:
            Path(a.out).write_text(json.dumps(r, indent=1))
        keys = ("jaccard", "recall", "precision", "rmse_lat_nm", "rmse_ax_nm", "pixel_ks", "crlb_width_lat",
                "crlb_width_ax", "core_rmse_lat_nm", "core_rmse_ax_nm", "core_pixel_ks")
        print({k: round(r[k], 4) for k in keys if r.get(k) is not None})
        for b in r.get("by_abs_z", []) + r.get("by_frames_on", []):
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items()})
        return
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
