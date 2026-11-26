"""How long a job will train, and whether the hardware it would train on is up to it -- said before *Start*.

A training on the CPU runs for days where a GPU takes an hour, and nothing in the window said so until the
first epochs had crawled by.  The estimate is the epochs it will most likely run times the seconds an
epoch takes on that device, and both are read without torch, so the window can say it at once:

* **Seconds an epoch** scale with the samples an epoch draws and their pixels (``work``, in samples of
  40 x 40).  Where earlier jobs of the same method trained on the same device class and host, their
  epochs' ``seconds`` (`progress`) are the rate -- the machine's own number.  Otherwise the rates
  measured in docs/decode/poc-plan.md: DECODE's reference job (32 768 samples) 12.2 s an epoch on an
  H100 and 196 s on an M1 Max's GPU; an optimised model's 11.4 s on an H100 in fp32 (7.4 s with bf16 and
  compiling, CUDA only), its Apple GPU taken at DECODE's ratio.  No training on the CPU was timed: its
  inference was 20 times slower than the M1 Max's GPU (15 against 0.7 frames of 512 x 512 a second),
  and that ratio is used.  These are rough -- a 4090 took 26 s on the original's reference job --
  which is why the estimate says "about".
* **Epochs**: an optimised model trains to its samples (6 M by default, ~183 epochs).  DECODE with the
  early stop stopped at 327 -- 397 epochs on the seven trainings it was tuned on, so 400 (never more than
  *epochs*); without it, all *epochs*.

A warning is given for the CPU, and for anything longer than `TOO_LONG_H`, with what a GPU would take.

The device is what the training would pick (`decode_smlm.training`'s order: CUDA, the Apple GPU, the
CPU), guessed without importing torch: an Apple-silicon Mac has the Apple GPU, a machine with
``nvidia-smi`` CUDA.  A training host is taken to have a GPU (`remote.check` says whether it does)
unless its earlier jobs trained on the CPU.
"""
from __future__ import annotations

import json
import math
import platform
import shutil
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from . import job as jobfile
from . import progress

REFERENCE_SAMPLES = 32768          # the reference jobs' epoch, at which the rates below were measured
REFERENCE_PX = 40 * 40
#: seconds for an epoch of 32 768 samples of 40 x 40, per method and device class (module docstring)
RATES = {
    "decode": {"cuda": 12.2, "mps": 196.0, "cpu": 196.0 * 20},
    "netfit": {"cuda": 7.4, "mps": 11.4 * 196.0 / 12.2, "cpu": 11.4 * 196.0 / 12.2 * 20},
}
DECODE_CONVERGED_EPOCHS = 400      # the early stop's 327 -- 397 on the trainings it was tuned on
TOO_LONG_H = 12.0                  # longer than this is worth a word whatever the device
DEVICE_NAMES = {"cuda": "an NVIDIA GPU", "mps": "the Apple GPU", "cpu": "the CPU"}


def device_class(device) -> str:
    """``cuda:0`` -> ``cuda``; ``mps``; anything else the CPU."""
    d = str(device or "").lower()
    return "cuda" if d.startswith("cuda") else "mps" if d.startswith("mps") else "cpu"


def local_device(requested: str = "auto") -> str:
    """The device a training here would use, without importing torch."""
    if requested and requested != "auto":
        return device_class(requested)
    if shutil.which("nvidia-smi"):
        return "cuda"
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mps"
    return "cpu"


@dataclass
class Work:
    method: str                    # "decode" or "netfit"
    samples_per_epoch: int
    pixels: int                    # a sample's frame, in pixels
    max_epochs: int | None      # an optimised model's samples as epochs; None: *epochs* decides

    @property
    def units(self) -> float:
        """An epoch's work in reference epochs (32 768 samples of 40 x 40)."""
        return self.samples_per_epoch / REFERENCE_SAMPLES * self.pixels / REFERENCE_PX


def work(folder) -> Work | None:
    """What an epoch of the job in ``folder`` draws; None where its files do not read."""
    folder = Path(folder)
    try:
        method = json.loads((folder / jobfile.JOB).read_text(encoding="utf-8")).get("method")
        if method == jobfile.NETFIT:
            s = json.loads((folder / jobfile.NETFIT_SETTINGS).read_text(encoding="utf-8"))
            sch = ((s.get("parts") or {}).get("schedule") or {}).get("settings") or {}
            per = int(sch.get("samples_per_epoch", REFERENCE_SAMPLES))
            px = int((s.get("simulation") or {}).get("frame_px", 40)) ** 2
            most = sch.get("max_samples")
            return Work("netfit", per, px, math.ceil(float(most) / per) if most else None)
        from importlib.resources import files

        import yaml
        p = yaml.safe_load((folder / jobfile.METHOD).read_text(encoding="utf-8")) or {}
        ref = yaml.safe_load((files("decode_smlm") / "utils" / "reference_files" / "reference.yaml")
                             .read_text(encoding="utf-8"))
        per = int((p.get("HyperParameter") or {}).get("pseudo_ds_size")
                  or ref["HyperParameter"]["pseudo_ds_size"])
        size = (p.get("Simulation") or {}).get("img_size") or ref["Simulation"]["img_size"]
        return Work("decode", per, int(size[0]) * int(size[1]), None)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        return None


def expected_epochs(w: Work, epochs: int, until_converged: bool) -> int:
    """An optimised model's samples (or *epochs*, if fewer); DECODE's `DECODE_CONVERGED_EPOCHS` with the
    early stop, else all *epochs*."""
    if w.max_epochs is not None:
        return max(1, min(int(epochs), w.max_epochs))
    return max(1, min(int(epochs), DECODE_CONVERGED_EPOCHS) if until_converged else int(epochs))


def measured_rate(method: str, device: str, host: str | None, entries=None) -> float | None:
    """Seconds a reference epoch took in the listed jobs of this method, device class and host (the
    median over their epochs); None when none trained there."""
    from . import jobs
    seen = []
    for e in jobs.entries() if entries is None else entries:
        if (e.host or None) != (host or None):
            continue
        w = work(e.folder)
        if w is None or w.method != method or w.units <= 0:
            continue
        lines = progress.read(e.folder)
        starts = [ln for ln in lines if ln["kind"] == "start" and ln.get("device")]
        if not starts or device_class(starts[-1]["device"]) != device:
            continue
        seen += [ln["seconds"] / w.units for ln in lines
                 if ln["kind"] == "epoch" and isinstance(ln.get("seconds"), (int, float)) and ln["seconds"] > 0]
    return statistics.median(seen) if seen else None


def host_device(host: str, requested: str = "auto", entries=None) -> str:
    """A training host's device: as asked, else what its last job trained on, else a GPU."""
    if requested and requested != "auto":
        return device_class(requested)
    from . import jobs
    for e in reversed(jobs.entries() if entries is None else entries):
        if e.host == host:
            starts = [ln for ln in progress.read(e.folder) if ln["kind"] == "start" and ln.get("device")]
            if starts:
                return device_class(starts[-1]["device"])
    return "cuda"


@dataclass
class Estimate:
    seconds: float
    epochs: int
    device: str                    # cuda, mps or cpu
    where: str                     # "here" or the host's name
    measured: bool                 # the rate from earlier jobs there, not the defaults
    warning: str = ""              # why this hardware is not up to it; "" when it is

    def text(self) -> str:
        on = DEVICE_NAMES[self.device] + ("" if self.where == "here" else f" of {self.where}")
        how = "as earlier jobs there ran" if self.measured else "from benchmark rates"
        return f"about {duration(self.seconds)} for {self.epochs} epochs on {on} ({how})"


def duration(seconds: float) -> str:
    """A rough length: minutes, hours or days."""
    if seconds < 90 * 60:
        return f"{max(1, round(seconds / 60))} min"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def estimate(folder, run_on: str = "here", device: str = "auto", epochs: int = 1000,
             until_converged: bool = True, hosts_exist: bool = False, entries=None) -> Estimate | None:
    """How long the job in ``folder`` trains where *run on* says, and a warning when that is the CPU or
    longer than `TOO_LONG_H`; None where the job does not read."""
    w = work(folder)
    if w is None:
        return None
    host = None if run_on == "here" else run_on
    dev = local_device(device) if host is None else host_device(host, device, entries)
    rate = measured_rate(w.method, dev, host, entries)
    measured = rate is not None
    if rate is None:
        rate = RATES[w.method][dev]
    n = expected_epochs(w, epochs, until_converged)
    e = Estimate(rate * w.units * n, n, dev, run_on if host else "here", measured)
    gpu = RATES[w.method]["cuda"] * w.units * n
    if dev == "cpu":
        e.warning = (f"this trains on the CPU: {e.text()}, where a GPU takes about {duration(gpu)}. "
                     + ("Choose a training host with a GPU under run on."
                        if host is None and hosts_exist else
                        "Train on a computer with a GPU (Training hosts... adds one)." if host is None else
                        "Check the host's GPU in Preferences > Training hosts."))
    elif e.seconds > TOO_LONG_H * 3600:
        e.warning = (f"{e.text()}: a long training. On an NVIDIA GPU it would take about {duration(gpu)}"
                     + (" -- choose a training host under run on." if host is None and dev != "cuda" else "."))
    return e
