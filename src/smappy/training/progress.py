"""A training's ``progress.jsonl``: one JSON object a line, appended and flushed as it happens.

A reader -- the Train window, ``smappy-decode status``, ``tail -f`` over ssh -- follows the file
without talking to the training.  Every line has ``kind``, ``time`` (ISO, local, with its offset)
and ``epoch``.  The kinds (docs/decode/poc-plan.md §5): ``start`` (versions, device), ``epoch``
(``loss_train``, ``loss_test``, ``lr``, ``seconds`` and ``metrics``, the method's own values),
``sample`` (the path of an npz), ``restart`` (why), ``stopped``, ``finished`` and ``error``
(``message``).  A reader ignores keys it does not know, so a method adds values without the
window changing; a line cut short by a crash, the last, is skipped.
"""
from __future__ import annotations

import datetime
import json
import math
import os
from pathlib import Path

NAME = "progress.jsonl"
KINDS = ("start", "epoch", "sample", "restart", "stopped", "finished", "error")


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
#: The quantities (`QUANTITIES`): the test loss (relative), the test set's Jaccard and its lateral
#: and axial efficiencies (absolute), its lateral and axial RMSE (relative).  Each is the running
#: median of ``smooth`` epochs first: early on an epoch's values scatter by about 1 in a loss of
#: 300, 0.003 in Jaccard, 0.4 nm and 0.8 nm in the RMSEs (the test set is fixed; it is the
#: weights that move), and one lucky epoch would otherwise set the bar.  On the proof of concept's
#: six trainings of 1000 epochs (the example's, the pixelation study's, the SMLM challenge's)
#: this stopped at epochs 340 -- 576, the Jaccard and both efficiencies within 0.005 of the 1000th
#: epoch's.  The loss and the Jaccard settle last; the efficiencies and RMSEs are safeguards.  An
#: RMSE alone is no measure of progress: it rises while a training learns, as it finds more of the
#: harder molecules -- so it only keeps a training going while it still falls.
EARLY_STOP = {"window": 100, "min_epochs": 150, "smooth": 11, "loss_tol": 0.005, "jaccard_tol": 0.005,
              "eff_tol": 0.005, "rmse_tol": 0.01}
#: (the name a progress line has it under, better when lower, the rule's tolerance, relative)
QUANTITIES = (("loss_test", True, "loss_tol", True), ("jaccard", False, "jaccard_tol", False),
              ("eff_lat", False, "eff_tol", False), ("eff_ax", False, "eff_tol", False),
              ("rmse_lat_nm", True, "rmse_tol", True), ("rmse_ax_nm", True, "rmse_tol", True))


def _running_median(values: list, n: int) -> list:
    import numpy as np
    v = np.asarray(values, float)
    out = []
    for i in range(len(v)):
        part = v[max(0, i - n + 1):i + 1]
        out.append(float(np.nanmedian(part)) if np.isfinite(part).any() else float("nan"))
    return out


def converged(lines: list, rule: dict | None = None) -> str | None:
    """Why the training whose progress is ``lines`` may stop now, or None.  Only the epochs since
    the last start from scratch count: a restart's run begins again.  A quantity a training does
    not report (none matched yet) never keeps it going and never stops it."""
    import numpy as np
    rule = {**EARLY_STOP, **(rule or {})}
    epochs = []
    for line in lines:
        if line.get("kind") == "restart":
            epochs = []
        elif line.get("kind") == "epoch" and line.get("loss_test") is not None:
            epochs = [e for e in epochs if e["epoch"] < line["epoch"]] + [line]   # a resume repeats none
    w = int(rule["window"])
    if len(epochs) < max(int(rule["min_epochs"]), w + 1):
        return None
    said = []
    for name, lower, tol, relative in QUANTITIES:
        raw = [e.get(name) if name == "loss_test" else (e.get("metrics") or {}).get(name) for e in epochs]
        v = np.asarray(_running_median([np.nan if x is None else x for x in raw], int(rule["smooth"])))
        old, new = v[:-w], v[-w:]
        if not (np.isfinite(old).any() and np.isfinite(new).any()):
            continue
        before, last = (np.nanmin(old), np.nanmin(new)) if lower else (np.nanmax(old), np.nanmax(new))
        gain = (before - last) if lower else (last - before)
        if relative:
            gain /= abs(before) or 1.0
        if gain > float(rule[tol]):
            return None
        if name in ("loss_test", "jaccard"):
            said.append(f"{'test loss' if name == 'loss_test' else 'Jaccard'} {last:.4g} against {before:.4g}")
    return f"no better over the last {w} epochs: " + ", ".join(said)
