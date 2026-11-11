"""Training from a job folder: milestone (f) of docs/decode/poc-plan.md.

`train(folder)` is DECODE's ``live_engine_setup``, restructured around the job folder SMAPpy
writes (`smappy.training.job`): its ``setup_random_simulation`` and ``setup_trainer`` are called as
they are, its loop is this module's.  What the loop adds:

* **A seeded test set, kept.**  The original drew its test set again at every start, a resume
  included, and saved no random state (plan §6, item 10), so a resumed run was validated on other
  frames.  Here the seed is the job's, the test set is drawn once and saved (``test_set.pt``), and
  the checkpoint holds every random state (torch's, CUDA's or MPS's, numpy's, Python's), taken
  before the next epoch's frames are drawn: stopped and resumed, a run on the CPU without workers
  has the same losses, bitwise, as one that ran through (a test).
* **Stopping**: a ``STOP`` file, read after every epoch, or SIGTERM -- both a stop at the
  checkpoint just written.  The checkpoint is written to a temporary file and renamed, so a stop
  or a crash never leaves half of one.
* **The model file every epoch** (``model.safetensors``, `smappy.training.model_file`), the
  weights and what a fitter needs, without pickle; and a ``progress.jsonl`` line
  (`smappy.training.progress`) with the epoch's losses, learning rate, time and the test set's
  metrics -- Jaccard, RMSEs, efficiencies, under the names the original's log has as well.
* **DECODE's restarts** (``GMMHeuristicCheck``, up to ``auto_restart_param.num_restarts``) as
  ``restart`` lines, each restart seeded anew from the job's seed and its number.
* **A lock** (``lock.json``: host, pid, started), so two trainings never share a folder, and a
  refusal of a job prepared by another SMAPpy version unless asked.

What it leaves out: TensorBoard and the figures DECODE logged every epoch (`log_dists` drew four
matplotlib figures to close them again when there was no TensorBoard); the progress file is the
log.  The test set's evaluation is DECODE's (``post_process_log_test`` without the figures).
"""
from __future__ import annotations

import json
import os
import random
import signal
import socket
import time
from pathlib import Path

import numpy as np
import torch

METHOD = "decode-v0.10.2"
CKPT, MODEL, TEST_SET, STOP, LOCK = "ckpt.pt", "model.safetensors", "test_set.pt", "STOP", "lock.json"
# the progress file's metric names, from DECODE's evaluation
METRICS = {"jac": "jaccard", "prec": "precision", "rec": "recall", "f1": "f1", "rmse_lat": "rmse_lat_nm",
           "rmse_ax": "rmse_ax_nm", "rmse_vol": "rmse_vol_nm", "effcy_lat": "eff_lat", "effcy_ax": "eff_ax",
           "effcy_vol": "eff_vol"}


class JobError(RuntimeError):
    """A job that cannot be trained as it stands; the message says why."""


# ----------------------------------------------------------------------------- parameters
def parameters(folder: Path, method_params: dict, device: str, epochs: int | None = None,
               num_workers: int | None = None):
    """The job's ``method.yaml``, filled from DECODE's reference, its paths made the folder's."""
    from .utils import hardware, param_io
    from .utils.types import RecursiveNamespace

    param = RecursiveNamespace(**param_io.autofill_dict(method_params, param_io.load_reference()))
    param = param_io.autoset_scaling(param)
    param.InOut.calibration_file = str(folder / param.InOut.calibration_file)
    param.InOut.experiment_out = str(folder)
    param.InOut.checkpoint_init = None
    device = hardware.default_device() if device in (None, "auto") else hardware.available_device(device)
    param.Hardware.device = param.Hardware.device_simulation = str(device)
    if epochs is not None:
        param.HyperParameter.epochs = int(epochs)
    if num_workers is not None:
        param.Hardware.num_worker_train = int(num_workers)
    return param


def _plain(ns):
    """A RecursiveNamespace as plain JSON-able values."""
    if hasattr(ns, "to_dict"):
        ns = ns.to_dict()
    if isinstance(ns, dict):
        return {k: _plain(v) for k, v in ns.items()}
    if isinstance(ns, (list, tuple)):
        return [_plain(v) for v in ns]
    if isinstance(ns, torch.Tensor):
        return ns.tolist()
    return ns


# ----------------------------------------------------------------------------- random state
def rng_state() -> dict:
    """Every random state the training draws from, as tensors and plain values (weights-only safe)."""
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    state = {"torch": torch.get_rng_state(), "numpy": [kind, torch.from_numpy(keys.astype(np.int64)), pos,
                                                       has_gauss, cached],
             "python": list(random.getstate()[1]), "python_version": random.getstate()[0]}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict):
    torch.set_rng_state(state["torch"])
    kind, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((kind, keys.numpy().astype(np.uint32), pos, has_gauss, cached))
    random.setstate((state["python_version"], tuple(state["python"]), None))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed % 2 ** 32)
    random.seed(seed)


# ----------------------------------------------------------------------------- files in the folder
def _save_atomic(obj, path: Path):
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_test_set(ds, path: Path):
    """The a-priori test set as drawn and processed: frames, emitters, targets."""
    _save_atomic({"frames": ds._frames, "emitter": ds._emitter.to_dict(), "target": list(ds._target),
                  "weight": ds._weight}, path)


def load_test_set(ds, path: Path):
    from .generic.emitter import EmitterSet
    d = torch.load(path, map_location="cpu", weights_only=True)
    ds._frames = d["frames"]
    ds._emitter = EmitterSet(**d["emitter"])
    ds._em_split = ds._emitter.split_in_frames(0, ds._frames.size(0) - 1)
    ds._target, ds._weight = tuple(d["target"]), d["weight"]


def write_model(path: Path, model, param, job, epoch: int, calibration_em_on=None):
    from smappy import __version__
    from smappy.training import model_file

    from .predict import trained_on
    tensors = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    meta = {"method": METHOD, "architecture": param.HyperParameter.architecture,
            "arch_param": _plain(param.HyperParameter.arch_param),
            "channels_in": param.HyperParameter.channels_in, "channels_out": 10,
            # the EM state of the calibration's beads: the fitter flips data taken the other way
            "calibration_em_on": None if calibration_em_on is None else bool(calibration_em_on),
            "scaling": _plain(param.Scaling), "camera": _plain(param.Camera),
            "post_processing": {"kind": param.PostProcessing, "raw_th": param.PostProcessingParam.raw_th},
            "calibration_sha256": job.calibration.get("sha256"), "ranges": job.ranges,
            "trained_on": trained_on(param),
            "base_model": job.base_model, "epoch": epoch, "epochs": param.HyperParameter.epochs,
            "batch_size": int(param.HyperParameter.batch_size), "learning_rate": float(param.HyperParameter.opt_param.lr),
            "seed": job.seed, "smappy_version": __version__, "decode_version": "0.10.2"}
    model_file.write(path, tensors, meta)


def _warm_up(optimizer, param, epoch: int) -> list:
    """The rates of the epoch's training ramped up from ``lr_warmup_from`` over the first
    ``lr_warmup_epochs`` (plan (f2)); the scheduler's rates, to be put back after the epoch, so
    that the scheduler, the checkpoint and a resume never see the ramp.  A larger batch's larger
    rate made the first epochs' loss blow up without it (5 restarts at batch 256, 2x the rate).
    The ramp starts by default at the rate the linear rule gives ``batch_base``."""
    hp = param.HyperParameter
    target = [group["lr"] for group in optimizer.param_groups]
    n = int(getattr(hp, "lr_warmup_epochs", None) or 0)
    if n <= 0 or epoch > n:
        return target
    start = getattr(hp, "lr_warmup_from", None)
    for group, lr in zip(optimizer.param_groups, target):
        lr0 = float(start) if start is not None else lr * int(getattr(hp, "batch_base", None) or 64) / int(hp.batch_size)
        group["lr"] = lr0 + (lr - lr0) * (epoch - 1) / n          # epoch 1 at the start, epoch n + 1 at the target
    return target


def _ckpt_batch(batch: dict) -> dict:
    """The batch's record as the checkpoint keeps it: plain numbers and strings (weights-only safe)."""
    return {"batch_size": int(batch["batch_size"]), "lr": float(batch["lr"]), "chosen": str(batch["chosen"])}


def load_weights(model, path: Path):
    """A model file's weights into ``model`` (a warm start)."""
    from smappy.training import model_file
    tensors, meta = model_file.read(path)
    if meta.get("method") not in (None, METHOD):
        raise JobError(f"{path.name} is a {meta['method']} model, not {METHOD}")
    model.load_state_dict({k: torch.from_numpy(v) for k, v in tensors.items()})


class _Lock:
    def __init__(self, folder: Path):
        self.path = folder / LOCK

    def __enter__(self):
        if self.path.exists():
            try:
                info = json.loads(self.path.read_text())
            except (ValueError, OSError):
                info = {}
            host, pid = info.get("host"), info.get("pid")
            if host == socket.gethostname() and pid and _alive(pid):
                raise JobError(f"{self.path.parent} is being trained already, by process {pid}")
            if host and host != socket.gethostname():
                raise JobError(f"{self.path.parent} is locked by a training on {host}; if that is gone, "
                               f"delete {self.path.name}")
        self.path.write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid(),
                                         "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
        return self

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _Capture:
    """Just enough of DECODE's logger for ``log_kpi``: this epoch's scalars."""

    def __init__(self):
        self.scalars = {}

    def add_scalar(self, name, value, step=None, *a, **k):
        self.scalars[name] = float(value)

    def add_scalar_dict(self, prefix, values, step=None, *a, **k):
        for name, value in values.items():
            self.add_scalar(prefix + name, value)

    def add_figure(self, *a, **k):
        pass

    add_histogram = add_graph = add_figure


def evaluate_test(test_out, ds_test, post_processor, matcher, val_loss) -> dict:
    """DECODE's ``post_process_log_test`` without its figures: the metrics of this epoch."""
    import warnings

    from .evaluation import evaluation
    from .evaluation.evaluation import WeightedErrors
    from .neuralfitter.utils import log_train_val_progress

    em_out = post_processor.forward(test_out.y_out)
    tp, fp, fn, tp_match = matcher.forward(em_out, ds_test.emitter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evaluation.SMLMEvaluation(weighted_eval=WeightedErrors(
            mode="crlb", reduction="gaussian")).forward(tp, fp, fn, tp_match)
    cap = _Capture()
    log_train_val_progress.log_kpi(loss_scalar=val_loss, loss_cmp=test_out.loss, eval_set=result._asdict(),
                                   logger=cap, step=0)
    metrics = {METRICS.get(k[5:], k[5:]): v for k, v in cap.scalars.items() if k.startswith("eval/")}
    metrics.update({k: v for k, v in cap.scalars.items() if k.startswith("loss_cmp/")})
    return metrics


# ----------------------------------------------------------------------------- the loop
def train(folder, *, device: str | None = None, epochs: int | None = None, num_workers: int | None = None,
          allow_other_version: bool = False, quiet: bool = False) -> str:
    """Train the job in ``folder`` from where it stands; returns ``"finished"`` or ``"stopped"``.

    ``device`` overrides the job's (``"auto"``: CUDA, else MPS, else the CPU), ``epochs`` its
    method's, ``num_workers`` the data loaders'.
    """
    from smappy import __version__
    from smappy.training import job as jobs
    from smappy.training import progress

    folder = Path(folder)
    problems = jobs.check(folder)
    if problems:
        raise JobError(f"{folder}: " + "; ".join(problems))
    job, method_params = jobs.read(folder)
    if job.method != METHOD:
        raise JobError(f"{folder} is a {job.method} job; this trains {METHOD}")
    if job.smappy_version != __version__ and not allow_other_version:
        raise JobError(f"{folder} was prepared by SMAPpy {job.smappy_version}, this is {__version__}: "
                       f"prepare it again, or allow another version")
    stop = {"why": None}

    def on_term(signum, frame):
        stop["why"] = "SIGTERM"
    old = signal.signal(signal.SIGTERM, on_term) if _main_thread() else None
    try:
        with _Lock(folder):
            return _run(folder, job, method_params, device or job.device, epochs, num_workers, stop, quiet)
    except Exception as e:
        progress.append(folder, "error", None, message=f"{type(e).__name__}: {e}")
        raise
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)


def _main_thread() -> bool:
    import threading
    return threading.current_thread() is threading.main_thread()


def _run(folder, job, method_params, device, epochs, num_workers, stop, quiet) -> str:
    import contextlib
    import io

    from smappy.training import progress

    from .neuralfitter import train_val_impl
    from .neuralfitter.train.live_engine import setup_dataloader, setup_trainer
    from .neuralfitter.train.random_simulation import setup_random_simulation
    from .neuralfitter.utils import logger as loggers
    from .neuralfitter.utils.progress import GMMHeuristicCheck

    param = parameters(folder, method_params, device, epochs, num_workers)
    device = param.Hardware.device
    torch.set_num_threads(param.Hardware.torch_threads)
    ckpt_path = folder / CKPT
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True) if ckpt_path.exists() else None
    restarts = ckpt["restarts"] if ckpt else 0
    # the batch (and its rate) once, before the first epoch: a resume keeps the checkpoint's, so
    # that a run is the same however often it was stopped (`batch.resolve`, plan (f2))
    if ckpt is not None and "batch" in ckpt:
        batch = dict(ckpt["batch"])
        param.HyperParameter.batch_size = int(batch["batch_size"])
        param.HyperParameter.opt_param.lr = float(batch["lr"])
    else:
        from . import batch as batches
        from .neuralfitter import models
        batch = batches.resolve(param, lambda: getattr(models, param.HyperParameter.architecture).parse(param))
    def quietly():
        # DECODE prints and draws tqdm bars (stderr); quiet keeps both out, the progress file is the log
        if not quiet:
            return contextlib.nullcontext()
        stack = contextlib.ExitStack()
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        return stack

    def setup(seed):
        seed_all(seed)
        with quietly():
            sim_train, sim_test = setup_random_simulation(param)
            out = setup_trainer(sim_train, sim_test, loggers.NoLog(), folder / "model.pt", ckpt_path, device, param)
        ds_train, ds_test, model, _, optimizer, criterion, lr_scheduler, grad_mod, post_processor, matcher, _ = out
        if (folder / TEST_SET).exists():
            load_test_set(ds_test, folder / TEST_SET)
        else:
            save_test_set(ds_test, folder / TEST_SET)
        dl_train, dl_test = setup_dataloader(param, ds_train, ds_test)
        return ds_train, ds_test, model, optimizer, criterion, lr_scheduler, grad_mod, post_processor, matcher, \
            dl_train, dl_test

    (ds_train, ds_test, model, optimizer, criterion, lr_scheduler, grad_mod, post_processor, matcher,
     dl_train, dl_test) = setup(job.seed + 1000 * restarts)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        set_rng_state(ckpt["rng"])
        with quietly():
            ds_train.sample(True)        # the frames the run would have drawn after that checkpoint
        first = ckpt["epoch"] + 1
    else:
        if job.base_model:
            load_weights(model, folder / job.base_model)
        first = 0
    n_epochs = param.HyperParameter.epochs
    from smappy.io.calibration import load_spline_calibration
    cal_em_on = load_spline_calibration(param.InOut.calibration_file).em_on
    progress.append(folder, "start", first, device=str(device), torch=torch.__version__, decode="0.10.2",
                    resumed=ckpt is not None, epochs=n_epochs, seed=job.seed, restarts=restarts, batch=batch)
    if first >= n_epochs:
        progress.append(folder, "finished", first - 1)
        return "finished"

    max_restarts = param.HyperParameter.auto_restart_param.num_restarts
    i = first
    while i < n_epochs:
        conv_check = GMMHeuristicCheck(ref_epoch=1, emitter_avg=ds_train.simulator.em_sampler.em_avg,
                                       threshold=param.HyperParameter.auto_restart_param.restart_treshold)
        t0 = time.perf_counter()
        loss_train = None
        with quietly():
            if i >= 1:
                target = _warm_up(optimizer, param, i)
                loss_train = train_val_impl.train(model=model, optimizer=optimizer, loss=criterion, dataloader=dl_train,
                                                  grad_rescale=param.HyperParameter.moeller_gradient_rescale,
                                                  grad_mod=grad_mod, epoch=i, device=torch.device(device),
                                                  logger=loggers.NoLog())
                lr_trained = optimizer.param_groups[0]["lr"]
                for group, lr in zip(optimizer.param_groups, target):
                    group["lr"] = lr          # the scheduler's own rate back before it steps
            val_loss, test_out = train_val_impl.test(model=model, loss=criterion, dataloader=dl_test, epoch=i,
                                                     device=torch.device(device))
        if not conv_check(test_out.loss[:, 0].mean(), i):
            restarts += 1
            per_emitter = float(test_out.loss[:, 0].mean() / conv_check.emitter_avg)
            if restarts > max_restarts:
                raise JobError(f"the loss stayed pathological through {max_restarts} restarts "
                               f"({per_emitter:.1f} per emitter against at most {conv_check.threshold:.1f})")
            progress.append(folder, "restart", i, why="pathological loss", loss_per_emitter=per_emitter,
                            threshold=float(conv_check.threshold), restart=restarts)
            (folder / TEST_SET).unlink(missing_ok=True)    # drawn again with the restart's seed, as DECODE did
            (ds_train, ds_test, model, optimizer, criterion, lr_scheduler, grad_mod, post_processor, matcher,
             dl_train, dl_test) = setup(job.seed + 1000 * restarts)
            i = 0
            continue
        metrics = evaluate_test(test_out, ds_test, post_processor, matcher, float(val_loss))
        lr = lr_trained if i >= 1 else optimizer.param_groups[0]["lr"]
        if i >= 1:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(val_loss)
            else:
                lr_scheduler.step()
        write_model(folder / MODEL, model, param, job, i, cal_em_on)
        _save_atomic({"epoch": i, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "lr_scheduler": lr_scheduler.state_dict(), "rng": rng_state(), "restarts": restarts,
                      "batch": _ckpt_batch(batch)}, ckpt_path)
        progress.append(folder, "epoch", i, loss_train=loss_train, loss_test=float(val_loss), lr=lr,
                        seconds=round(time.perf_counter() - t0, 3), metrics=metrics)
        if (folder / STOP).exists() or stop["why"]:
            why = stop["why"] or "STOP file"
            (folder / STOP).unlink(missing_ok=True)
            progress.append(folder, "stopped", i, why=why)
            return "stopped"
        i += 1
        if i < n_epochs:
            with quietly():
                ds_train.sample(True)
    progress.append(folder, "finished", n_epochs - 1)
    return "finished"


def request_stop(folder) -> Path:
    """Ask a running training to stop at its next checkpoint."""
    path = Path(folder) / STOP
    path.touch()
    return path
