"""A training's ``progress.jsonl``: one JSON object a line, appended and flushed as it happens.

A reader -- the Train window, ``smappy-decode status``, ``tail -f`` over ssh -- follows the file
without talking to the training.  Every line has ``kind``, ``time`` (ISO, local, with its offset)
and ``epoch``.  The kinds (docs/decode/poc-plan.md §5): ``start`` (versions, device), ``epoch``
(``loss_train``, ``loss_test``, ``lr``, ``seconds`` and ``metrics``, the method's own values),
``sample`` (the path of an npz), ``snapshot`` (netfit: a model kept after ``samples`` samples
and its ``metrics`` on the whole fixed test set), ``restart`` (why), ``stopped``, ``finished`` and
``error`` (``message``).  A reader ignores keys it does not know, so a method adds values without the
window changing; a line cut short by a crash, the last, is skipped.
"""
from __future__ import annotations

import datetime
import json
import math
import os
from pathlib import Path

NAME = "progress.jsonl"
KINDS = ("start", "epoch", "sample", "snapshot", "restart", "stopped", "finished", "error")


def _clean(value):
    # JSON has no NaN or infinity: a diverged loss is written as null, not as invalid JSON
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def append(folder, kind: str, epoch: int | None, **fields) -> dict:
    """Append one line to ``folder/progress.jsonl`` and flush it to disk."""
    if kind not in KINDS:
        raise ValueError(f"progress kind {kind!r}: one of {', '.join(KINDS)}")
    line = {"kind": kind, "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "epoch": epoch, **_clean(fields)}
    path = Path(folder) / NAME
    text = json.dumps(line, allow_nan=False) + "\n"
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                text = "\n" + text                    # a line a crash cut short stays on its own
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    return line


def read(folder) -> list[dict]:
    """Every complete line of a folder's progress, in order; none when there is no file yet."""
    path = Path(folder) / NAME
    if not path.exists():
        return []
    out = []
    for text in path.read_text(encoding="utf-8").splitlines():
        try:
            line = json.loads(text)
        except json.JSONDecodeError:
            continue                    # a line being written, or cut short by a crash
        if isinstance(line, dict) and "kind" in line:
            out.append(line)
    return out


def state(lines: list[dict]) -> str:
    """``"new"``, ``"running"`` (or interrupted), ``"stopped"``, ``"finished"`` or ``"error"``: the last
    word the training had; a reader checks the lock file to tell running from interrupted."""
    for line in reversed(lines):
        if line["kind"] in ("stopped", "finished", "error"):
            return line["kind"]
        if line["kind"] in ("start", "epoch", "restart", "sample"):
            return "running"
    return "new"


#: When a training stops on its own (`converged`): over the last ``window`` epochs no quantity
#: improved by more than its tolerance on its best before them, after at least ``min_epochs``.
#: The quantities (`QUANTITIES`): the test loss, against how far it has fallen so far (its gain over
#: the window at most ``loss_tol`` of its fall from epoch ``fall_from`` to the best before the
#: window); the test set's Jaccard and its lateral and axial efficiencies (absolute); its lateral
#: and axial RMSE (relative).  Each is the running median of ``smooth`` epochs first: early on an
#: epoch's values scatter by about 1 in a loss of 300, 0.003 in Jaccard, 0.4 nm and 0.8 nm in the
#: RMSEs (the test set is fixed; it is the weights that move), and one lucky epoch would otherwise
#: set the bar.
#:
#: **The loss against its fall, not its value** (2026-11-12): a loss's size is the density's -- 24
#: at 25 molecules a 40 x 40 frame, 300 at 50 -- and 0.5 % of it kept a sparse training of 500
#: epochs going at epoch 436, its loss falling 3 % in 100 epochs while everything else had settled
#: since 400.  Replayed on the proof of concept's six trainings of 1000 epochs (the example's, the
#: pixelation study's, the SMLM challenge's) and that one, this rule stops at epochs 327 -- 397,
#: the Jaccard within 0.005 of the last epoch's.  A window of 50 epochs lost up to 0.012 in
#: Jaccard, of 75 up to 0.007.  An RMSE alone is no measure of progress: it rises while a training
#: learns, as it finds more of the harder molecules -- so it only keeps a training going while it
#: still falls.
EARLY_STOP = {"window": 100, "min_epochs": 150, "smooth": 11, "fall_from": 10, "loss_tol": 0.01,
              "jaccard_tol": 0.005, "eff_tol": 0.005, "rmse_tol": 0.01}
#: (the name a progress line has it under, what the status line calls it, better when lower, the
#: rule's tolerance, how a gain is measured: "fall" -- of the fall so far, "relative", "absolute")
QUANTITIES = (("loss_test", "loss", True, "loss_tol", "fall"),
              ("jaccard", "Jaccard", False, "jaccard_tol", "absolute"),
              ("eff_lat", "efficiency lateral", False, "eff_tol", "absolute"),
              ("eff_ax", "efficiency axial", False, "eff_tol", "absolute"),
              ("rmse_lat_nm", "RMSE lateral", True, "rmse_tol", "relative"),
              ("rmse_ax_nm", "RMSE axial", True, "rmse_tol", "relative"),
              # judged only when the rule has a tolerance for it (netfit's has: its KS kept falling
              # after the others settled; the DECODE port's rule leaves it out)
              ("pixel_ks", "pixelation", True, "ks_tol", "absolute"))


def rule(changes: dict | None = None) -> dict:
    """The rule a training stops by: `EARLY_STOP` with ``changes`` over it."""
    return {**EARLY_STOP, **(changes or {})}


def _running_median(values: list, n: int) -> list:
    import numpy as np
    v = np.asarray(values, float)
    out = []
    for i in range(len(v)):
        part = v[max(0, i - n + 1):i + 1]
        out.append(float(np.nanmedian(part)) if np.isfinite(part).any() else float("nan"))
    return out


def _epochs(lines: list) -> list:
    """The epoch lines since the last start from scratch: a restart's run begins again, and a
    resume repeats none."""
    epochs = []
    for line in lines:
        if line.get("kind") == "restart":
            epochs = []
        elif line.get("kind") == "epoch" and line.get("loss_test") is not None:
            epochs = [e for e in epochs if e["epoch"] < line["epoch"]] + [line]
    return epochs


def stop_status(lines: list, rule: dict | None = None) -> list:
    """Each quantity of the rule as it stands: ``{"name", "label", "gain", "tolerance", "kind",
    "settled", "level"}`` -- the gain over the window as the rule measures it, and ``level`` the
    median of the window's smoothed values, what the Follow curves draw.  Empty before there is a
    window to judge; a quantity a training does not report (none matched yet) is left out."""
    import numpy as np
    rule = {**EARLY_STOP, **(rule or {})}
    epochs = _epochs(lines)
    w = int(rule["window"])
    if len(epochs) <= w:
        return []
    out = []
    for name, label, lower, tol, kind in QUANTITIES:
        if rule.get(tol) is None:
            continue
        raw = [e.get(name) if name == "loss_test" else (e.get("metrics") or {}).get(name) for e in epochs]
        v = np.asarray(_running_median([np.nan if x is None else x for x in raw], int(rule["smooth"])))
        old, new = v[:-w], v[-w:]
        if not (np.isfinite(old).any() and np.isfinite(new).any()):
            continue
        before, last = (np.nanmin(old), np.nanmin(new)) if lower else (np.nanmax(old), np.nanmax(new))
        gain = (before - last) if lower else (last - before)
        if kind == "fall":
            start = v[min(int(rule["fall_from"]), len(old) - 1)]
            fell = start - before if lower else before - start
            # a loss that has not fallen yet is settled only if the window gained nothing either; the
            # start of a training is the minimum's to guard (`min_epochs`), not this
            if np.isfinite(fell) and fell > 0:
                gain = gain / fell
            else:
                gain = 0.0 if gain <= 0 else float("inf")
        elif kind == "relative":
            gain /= abs(before) or 1.0
        out.append({"name": name, "label": label, "gain": float(gain), "tolerance": float(rule[tol]),
                    "kind": kind, "settled": bool(gain <= float(rule[tol])),
                    "level": float(np.nanmedian(new))})
    return out


def converged(lines: list, rule: dict | None = None) -> str | None:
    """Why the training whose progress is ``lines`` may stop now, or None (`stop_status`)."""
    rule = {**EARLY_STOP, **(rule or {})}
    if len(_epochs(lines)) < max(int(rule["min_epochs"]), int(rule["window"]) + 1):
        return None
    status = stop_status(lines, rule)
    if not status or not all(q["settled"] for q in status):
        return None
    return f"no better over the last {int(rule['window'])} epochs: " + status_text(status)


def status_text(status: list) -> str:
    """The rule's state in a line: what it still waits for, with the numbers, and what settled."""
    def number(q):
        if q["kind"] == "absolute":
            return f"{q['gain']:+.3f}, needs \u2264 {q['tolerance']:.3f}"
        what = " of its fall" if q["kind"] == "fall" else ""
        return f"{100 * q['gain']:+.1f} %{what}, needs \u2264 {100 * q['tolerance']:.1f} %"
    waiting = [f"{q['label']} ({number(q)})" for q in status if not q["settled"]]
    settled = [q["label"] for q in status if q["settled"]]
    parts = (["waiting for: " + ", ".join(waiting)] if waiting else []) + \
            (["settled: " + ", ".join(settled)] if settled else [])
    return " \u00b7 ".join(parts)


# ----------------------------------------------------------------------------- how fast a training got there
#: (the quantity, its tolerance, absolute or relative): within it of its final value is "there"
CONVERGENCE = (("loss_test", 0.01, "relative"), ("jaccard", 0.005, "absolute"),
               ("rmse_lat_nm", 0.02, "relative"), ("rmse_ax_nm", 0.02, "relative"),
               ("pixel_ks", 0.002, "absolute"), ("core_pixel_ks", 0.002, "absolute"),
               ("crlb_width_lat", 0.02, "absolute"), ("crlb_width_ax", 0.02, "absolute"))


def convergence(lines: list, smooth: int = 11) -> dict:
    """For each quantity a training reports: its final value (the median of the last ``smooth``
    epochs' running medians), and the epoch, and the training seconds summed to it, from which on its
    running median stays within the tolerance of `CONVERGENCE` of that final value.  Two trainings
    compare by these -- how good, and how soon -- not by the final value alone."""
    import numpy as np
    epochs = _epochs(lines)
    if len(epochs) < smooth:
        return {}
    seconds = np.cumsum([float(e.get("seconds") or 0.0) for e in epochs])
    out = {}
    for name, tol, kind in CONVERGENCE:
        raw = [e.get(name) if name == "loss_test" else (e.get("metrics") or {}).get(name) for e in epochs]
        v = np.asarray(_running_median([np.nan if x is None else x for x in raw], smooth))
        if not np.isfinite(v[-smooth:]).any():
            continue
        final = float(np.nanmedian(v[-smooth:]))
        bound = tol * abs(final) if kind == "relative" else tol
        off = ~(np.abs(v - final) <= bound)                 # NaN counts as not there
        last_off = np.flatnonzero(off)
        i = int(last_off[-1]) + 1 if len(last_off) else 0
        if i >= len(v):
            i = len(v) - 1
        out[name] = {"final": final, "epoch": int(epochs[i]["epoch"]), "seconds": float(seconds[i]),
                     "tolerance": tol, "kind": kind}
    return out
