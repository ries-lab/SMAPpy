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
    print(json.dumps({"torch": torch.__version__, "device": a.device,
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      "threads": torch.get_num_threads(), "tf32": torch.backends.cudnn.allow_tf32, "workers": p["Hardware"]["num_worker_train"],
                      "frames": p["HyperParameter"]["pseudo_ds_size"]}), flush=True)
    live_engine.live_engine_setup(str(param_file), no_log=True)


if __name__ == "__main__":
    sys.exit(main())
