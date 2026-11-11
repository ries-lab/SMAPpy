"""The training's batch size fitted to its GPU: milestone (f2) of docs/decode/poc-plan.md.

**Measured, and not worth it for DECODE's training (2026-11-11):** on the reference job, batch 256
needed three to five times the epochs of 64 to reach the same test loss -- an epoch's frames are
a quarter of the steps -- and an epoch at 256 is at most 1.4 times as fast on an H100.  So
``auto`` is an option, never the default; the H100's spare capacity goes to several trainings at
once through MPS (the plan, §8).  What follows is how ``auto`` works.

The example package trains on batches of 64 frames of 40 x 40, which leave a large GPU partly
idle: on an H100 a frame's step (forward, backward, AdamW) costs 253 us at 64 and 159 us at 512,
the GPU's kernels too small to fill it.  With ``batch_size: auto`` in ``method.yaml``, the training
measures the network's step on its own device at ``batch_base`` (64) and its doublings before the
first epoch -- a few seconds -- and takes the smallest batch whose time a frame is within
``TOLERANCE`` of the best, so that a batch no larger than needed is used: a larger one takes
fewer steps an epoch, and the training then has fewer to learn by.  A size the device has no
memory for ends the measurement.  The CPU keeps ``batch_base``: there is nothing to fill.

**The learning rate follows the batch** (`scaled_lr`): DECODE's rate was chosen for 64, and the
same 32 768 frames an epoch then take fewer, larger steps.  ``lr_scaling`` is ``sqrt`` (the rate
times the square root of the batch's ratio to ``batch_base``), ``linear`` or ``none``; which
trains as well as 64 per epoch is the study of (f2).  A batch given as a number keeps the rate
as given.

**What is measured** is the network's step on random frames, not the loss: the mixture loss's
cost a frame is roughly constant, and the network is where the GPU's time goes (plan §8,
"Training speed").  The measurement draws from a generator of its own, and the training seeds
itself after it, so a run is the same whether it measured or not.
"""
from __future__ import annotations

import copy
import time

import torch

BASE = 64
DOUBLINGS = 4                 # up to 16 times the base: 1024 frames for the package
TOLERANCE = 0.10              # within 10 % of the best time a frame
RULES = ("sqrt", "linear", "none")


def choose(per_frame: dict, tolerance: float = TOLERANCE) -> int:
    """The smallest batch whose seconds a frame (``per_frame``: batch -> seconds) are within
    ``tolerance`` of the smallest."""
    if not per_frame:
        raise ValueError("no batch size was measured")
    best = min(per_frame.values())
    return min(b for b, t in per_frame.items() if t <= best * (1 + tolerance))


def scaled_lr(lr: float, batch: int, base: int = BASE, rule: str = "sqrt") -> float:
    """``lr`` chosen for batches of ``base``, for batches of ``batch``."""
    if rule not in RULES:
        raise ValueError(f"lr_scaling {rule!r}: one of {', '.join(RULES)}")
    ratio = batch / base
    return float(lr * (ratio ** 0.5 if rule == "sqrt" else ratio if rule == "linear" else 1.0))


def _synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _out_of_memory(error: Exception) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def measure(model: torch.nn.Module, channels_in: int, frame_shape, device, base: int = BASE,
            doublings: int = DOUBLINGS, steps: int = 8, warmup: int = 3) -> dict:
    """Seconds a frame of the network's training step at ``base`` and its doublings on
    ``device``: a copy of ``model`` with an AdamW of its own, so neither the model nor any random
    state the training uses is touched."""
    device = torch.device(device)
    net = copy.deepcopy(model).to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-6)
    gen = torch.Generator().manual_seed(0)
    out = {}
    for k in range(doublings + 1):
        b = base * 2 ** k
        try:
            x = torch.randn((b, channels_in, *frame_shape), generator=gen).to(device)
            for i in range(warmup + steps):
                if i == warmup:
                    _synchronize(device)
                    t0 = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                net(x).square().mean().backward()
                opt.step()
            _synchronize(device)
            out[b] = (time.perf_counter() - t0) / steps / b
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            if not _out_of_memory(error):
                raise
            break                                  # no memory for this one: the last that fitted is the largest
        finally:
            x = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
    del net, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def resolve(param, model_factory) -> dict:
    """Set ``param.HyperParameter.batch_size`` (and the rate) where it is ``auto``; what was done,
    for the progress file.  ``model_factory()`` builds the network the training will train."""
    hp = param.HyperParameter
    base = int(getattr(hp, "batch_base", None) or BASE)
    rule = str(getattr(hp, "lr_scaling", None) or "sqrt")
    if str(hp.batch_size) != "auto":
        return {"batch_size": int(hp.batch_size), "chosen": "given", "lr": float(hp.opt_param.lr)}
    device = torch.device(param.Hardware.device)
    if device.type == "cpu":
        hp.batch_size = base
        return {"batch_size": base, "chosen": "cpu", "lr": float(hp.opt_param.lr)}
    frame = tuple(int(v) for v in param.Simulation.img_size)
    per_frame = measure(model_factory(), int(hp.channels_in), frame, device, base=base)
    batch = choose(per_frame)
    lr0 = float(hp.opt_param.lr)
    hp.batch_size = batch
    hp.opt_param.lr = scaled_lr(lr0, batch, base, rule)
    return {"batch_size": batch, "chosen": "measured", "lr": hp.opt_param.lr, "lr_base": lr0, "lr_scaling": rule,
            "us_per_frame": {str(b): round(t * 1e6, 1) for b, t in per_frame.items()}}
