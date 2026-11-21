"""The study's architectures timed on one idle GPU (plex-plan §16): what a training epoch costs, what
drawing the batches alone costs, and what a fit costs, each architecture with the same script on the
same machine -- the queues' own times were taken on two GPUs, one shared at times.

    python timing.py OUT.jsonl JOB [JOB ...]       # from the study folder; SUITE and DECODE_SMLM_DATA set

Per job (its settings from ``jobs/<JOB>.json``, its trained model from ``runs/<JOB>``):

* **epoch**: `train` itself for three epochs in a scratch folder -- drawing, the steps and the epoch's
  test evaluation, as a real training spends them; the first epoch (compilation) left out, the others'
  ``seconds`` from its progress file.
* **drawing**: `Drawer` alone for one epoch's batches, synchronised: if it is near the epoch's time,
  the simulation, not the network, sets the pace (the training draws in a thread beside the steps).
* **fit**: the trained model on the context's density-1 set (646 frames; `run_c2.fit_netfit`) after an
  untimed fit of the density-15 set, ms a frame.

Training time for a variant is then its epochs to a target (from the queues' logs, machine-free:
every run draws the same samples an epoch) times its epoch here.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time


def gpu_busy() -> str:
    """Other processes on the GPU, as nvidia-smi lists them: the timing is void if any."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False).stdout.strip()
    mine = str(os.getpid())
    return "\n".join(line for line in out.splitlines() if not line.startswith(mine) and "mps-server" not in line)


def main():
    import torch
    from smappy.netfit.train import Drawer, load_settings, train
    out = pathlib.Path(sys.argv[1]).resolve()
    study = pathlib.Path.cwd()
    suite = pathlib.Path(os.environ["SUITE"])
    sys.path.insert(0, str(suite))
    for job in sys.argv[2:]:
        rec = {"job": job, "gpu": torch.cuda.get_device_name(0), "others_before": gpu_busy()}
        settings = load_settings(study / "jobs" / f"{job}.json")
        scratch = study / "timing_runs" / job
        shutil.rmtree(scratch, ignore_errors=True)
        train(scratch, settings, "cuda", epochs=3, early_stop=False, log=lambda *a, **k: None)
        epochs = [json.loads(x) for x in (scratch / "progress.jsonl").read_text().splitlines()]
        sec = [e["seconds"] for e in epochs if e["kind"] == "epoch"]
        rec.update(epoch_seconds=sec, epoch_s=sum(sec[1:]) / max(len(sec) - 1, 1))
        # one epoch's batches drawn alone
        sch = settings.parts.schedule.settings
        drawer = Drawer(settings, "cuda")
        g = torch.Generator("cuda").manual_seed(1)
        batch = sch["batch"]
        steps = sch["samples_per_epoch"] // batch
        drawer(batch, g)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps):
            drawer(batch, g)
        torch.cuda.synchronize()
        rec["draw_s"] = time.time() - t0
        del drawer
        # a fit of the trained model
        model = study / "runs" / job / "model.safetensors"
        if model.exists():
            ctx = "biplane_2c" if "_biplane_" in job else "colour_2a"
            here = os.getcwd()
            os.chdir(suite)
            import run_c2 as RC
            RC.fit_netfit(f"{ctx}_blink_density_15", scratch / "warm.npz", str(model), crlb=False)
            info = RC.fit_netfit(f"{ctx}_blink_density_1", scratch / "fit.npz", str(model), crlb=False)
            os.chdir(here)
            rec.update(fit_ms_per_frame=1000 * info["seconds"] / info["frames"], fit_seconds=info["seconds"])
        rec["others_after"] = gpu_busy()
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        shutil.rmtree(scratch, ignore_errors=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
