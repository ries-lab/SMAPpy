"""Where a training epoch's time goes: the port's live engine, each phase timed with the device synchronised.

    python bench_training.py PACKAGE_DIR OUT_DIR --device cuda --epochs 4 [--workers 4] [--frames 32768]

PACKAGE_DIR is DECODE's experimental_data_workflow (its param_run.yaml and calibration), made
runnable as for the reference runs (StepLR, scripts/decode/make_params.py).  The phases of the
live engine (``decode_smlm/neuralfitter/train/live_engine.py``) are wrapped, not changed: the
training loop and, inside it, the time spent waiting for the data loader (its first batch apart:
the workers starting); the test loop; the test set's post-processing and matching; the model file
and the checkpoint; the next epoch's simulation.  One JSON line per epoch to stdout.
"""
import argparse
import json
import pathlib
import sys
import time

import torch
import yaml

T = {}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed(name, fn):
    def wrapper(*a, **k):
        _sync()
        t = time.perf_counter()
        out = fn(*a, **k)
        _sync()
        T[name] = T.get(name, 0.0) + time.perf_counter() - t
        return out
    return wrapper


def patch_loader():
    from torch.utils.data import dataloader as dl
    base = dl._BaseDataLoaderIter
    nxt = base.__next__

    def __next__(self):
        t = time.perf_counter()
        try:
            return nxt(self)
        finally:
            key = "data_first" if not getattr(self, "_bench_seen", False) else "data_wait"
            self._bench_seen = True
            T[key] = T.get(key, 0.0) + time.perf_counter() - t
    base.__next__ = __next__


W = {}
LAST = {}                         # the last training batch's network output and targets, for a dump


class Diverged(Exception):
    pass


def watch(out, epoch):
    """Per batch: the network output's range, the gradient's norm; stop at the first non-finite."""
    from decode_smlm.neuralfitter.loss import GaussianMMLoss

    forward = GaussianMMLoss.forward
    last = LAST

    def loss_forward(self, output, target, weight):
        with torch.no_grad():
            finite = torch.isfinite(output).flatten(2).all(-1).all(0)          # per channel
            stats = torch.stack([output[:, 0].sum((-1, -2)).min(), output[:, 5:9].amin(),
                                 output[:, 5:9].amax(), finite.all().float()]).tolist()
        if output.requires_grad:
            last.update(output=output.detach(), target=[t.detach() for t in target], batch=W.get("batches", 0))
        W["p_sum_min"] = min(W.get("p_sum_min", 1e30), stats[0])
        W["sigma_min"] = min(W.get("sigma_min", 1e30), stats[1])
        W["sigma_max"] = max(W.get("sigma_max", 0.0), stats[2])
        if not stats[3]:
            raise Diverged({"where": "network output", "nonfinite_channels": (~finite).nonzero().ravel().tolist(),
                            **_dump(out, last, epoch), **W})
        loss = forward(self, output, target, weight)
        lv = float(loss.detach().mean())
        if output.requires_grad:
            W["batches"] = W.get("batches", 0) + 1
            W["loss_last"] = lv
            W["loss_max"] = max(W.get("loss_max", -1e30), lv)
        if lv != lv or abs(lv) == float("inf"):
            raise Diverged({"where": "loss", "loss": lv, **_dump(out, last, epoch), **W})
        return loss
    GaussianMMLoss.forward = loss_forward

    step = torch.optim.AdamW.step

    def adamw_step(self, *a, **k):
        with torch.no_grad():
            grads = [q.grad for g in self.param_groups for q in g["params"] if q.grad is not None]
            norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads])))
        W["grad_norm_last"] = norm
        if norm != norm or norm == float("inf"):
            raise Diverged({"where": "gradient", "grad_norm": norm, **_dump(out, last, epoch), **W})
        W["grad_norm_max"] = max(W.get("grad_norm_max", 0.0), norm)
        return step(self, *a, **k)
    torch.optim.AdamW.step = adamw_step


def _dump(out, last, epoch):
    if last:
        torch.save({k: v for k, v in last.items()}, out / "diverged.pt")
    return {"batch": last.get("batch"), "saved": bool(last)}


def profiled(train, n_batches, out):
    """The first epoch's first ``n_batches`` under torch's profiler; its table to stderr."""
    from itertools import islice

    from torch.profiler import ProfilerActivity, profile

    done = {"yes": False}

    def wrapper(*a, **k):
        if done["yes"]:
            return train(*a, **k)
        done["yes"] = True
        loader = k["dataloader"]

        class Few:
            def __iter__(self):
                return islice(iter(loader), n_batches)

            def __len__(self):
                return n_batches
        k["dataloader"] = Few()
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
        with profile(activities=acts) as prof:
            result = train(*a, **k)
            _sync()
        for key in ("self_cpu_time_total", "self_device_time_total"):
            print(prof.key_averages().table(sort_by=key, row_limit=25), file=sys.stderr, flush=True)
        return result
    return wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--frames", type=int, default=None, help="pseudo_ds_size; the package's 32768")
    ap.add_argument("--tf32", choices=["on", "off"], default=None, help="cuDNN's TF32; torch's default is on")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--profile", type=int, default=0, help="profile this many training batches of the first epoch")
    ap.add_argument("--channels-last", action="store_true", help="the network in channels-last memory layout")
    ap.add_argument("--clip", action="store_true", help="DECODE's gradient clipping (grad_mod: clip_grad_norm_ at 0.03)")
    ap.add_argument("--watch", action="store_true",
                    help="each batch: the output's finiteness, the smallest p sum and sigma, the gradient's norm; "
                         "on the first non-finite one, the batch saved to OUT/diverged.pt and the run stopped")
    ap.add_argument("--fused", action="store_true", help="AdamW's fused implementation (opt_param fused: true)")
    ap.add_argument("--validate", choices=["on", "off"], default=None,
                    help="torch.distributions' argument checks (each a wait for the GPU); torch's default is on")
    a = ap.parse_args()
    if a.tf32:
        torch.backends.cudnn.allow_tf32 = a.tf32 == "on"
    if a.seed is not None:
        torch.manual_seed(a.seed)
    if a.validate:
        torch.distributions.Distribution.set_default_validate_args(a.validate == "on")

    p = yaml.safe_load((a.package / "param_run.yaml").read_text())
    p["InOut"].update(calibration_file=str(a.package / "spline_calibration_3dcal.mat"), experiment_out=str(a.out),
                      checkpoint_init=None)
    p["Hardware"].update(device=a.device, device_simulation=a.device)
    if a.workers is not None:
        p["Hardware"]["num_worker_train"] = a.workers
    p["HyperParameter"].update(epochs=a.epochs, learning_rate_scheduler="StepLR")
    p["HyperParameter"].pop("learning_rate_scheduler_param", None)
    if a.frames:
        p["HyperParameter"]["pseudo_ds_size"] = a.frames
    if a.fused:
        p["HyperParameter"]["opt_param"]["fused"] = True
    if a.clip:
        p["HyperParameter"]["grad_mod"] = True
    a.out.mkdir(parents=True, exist_ok=True)
    param_file = a.out / "param.yaml"
    param_file.write_text(yaml.safe_dump(p, sort_keys=False))

    from decode_smlm.neuralfitter import dataset, train_val_impl
    from decode_smlm.neuralfitter.train import live_engine
    from decode_smlm.neuralfitter.utils import log_train_val_progress
    from decode_smlm.utils import checkpoint, model_io

    train, test = train_val_impl.train, train_val_impl.test
    post = log_train_val_progress.post_process_log_test
    epoch = {"i": 0, "t": time.perf_counter()}

    def test_and_report(*args, **kw):
        out = timed("test", test)(*args, **kw)
        W["test_loss"] = float(out[0])
        return out

    def sample_and_report(self, *args, **kw):
        out = timed("simulate", sample)(self, *args, **kw)
        # the epoch ends with the next one's simulation
        total = time.perf_counter() - epoch["t"]
        print(json.dumps({"epoch": epoch["i"], "total_s": round(total, 2),
                          **{k: round(v, 2) for k, v in sorted(T.items())}}), flush=True)
        T.clear()
        epoch["i"] += 1
        epoch["t"] = time.perf_counter()
        return out

    sample = dataset.SMLMLiveDataset.sample
    train_val_impl.train = timed("train", train)
    if a.channels_last:
        setup = live_engine.setup_trainer

        def setup_channels_last(*args, **kw):
            out = setup(*args, **kw)
            out[2].to(memory_format=torch.channels_last)        # the model, in place
            return out
        live_engine.setup_trainer = setup_channels_last
    if a.profile:
        train_val_impl.train = timed("train", profiled(train, a.profile, a.out))
    train_val_impl.test = test_and_report
    log_train_val_progress.post_process_log_test = timed("post_process", post)
    checkpoint.CheckPoint.dump = timed("checkpoint", checkpoint.CheckPoint.dump)
    model_io.LoadSaveModel.save = timed("model_file", model_io.LoadSaveModel.save)
    dataset.SMLMLiveDataset.sample = sample_and_report
    patch_loader()
    if a.watch:
        watch(a.out, epoch)
    print(json.dumps({"torch": torch.__version__, "device": a.device,
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      "threads": torch.get_num_threads(), "tf32": torch.backends.cudnn.allow_tf32, "workers": p["Hardware"]["num_worker_train"],
                      "frames": p["HyperParameter"]["pseudo_ds_size"]}), flush=True)
    try:
        live_engine.live_engine_setup(str(param_file), no_log=True)
    except Diverged as d:
        print(json.dumps({"diverged": True, "epoch": epoch["i"], **d.args[0]}), flush=True)
        return 0
    except ValueError as e:          # torch.distributions' checks, if a NaN got past the watch
        (a.out / "error.txt").write_text(str(e))
        print(json.dumps({"diverged": True, "epoch": epoch["i"], "error": str(e)[:160], **_dump(a.out, LAST, epoch),
                          **W}), flush=True)
        return 0
    print(json.dumps({"finished": True, **W}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
