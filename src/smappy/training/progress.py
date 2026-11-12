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


#: When a training stops on its own (`converged`): the best test loss of the last ``window``
#: epochs no more than ``loss_tol`` of it better than the best before, and the best test-set
#: Jaccard no more than ``jaccard_tol`` higher, after at least ``min_epochs``.  On the seven
#: trainings of the proof of concept (1000 epochs each: the example's, the pixelation study's,
#: the SMLM challenge's) this stopped at epochs 300 -- 560, the Jaccard within 0.004 of the
#: 1000th epoch's and the loss within 1 %: half to two thirds of the time saved.  The RMSEs are
#: not read: they rise as the network finds more of the harder molecules.  A window of 50 and 1 %
#: stopped at 160 -- 355 and lost up to 0.019 in Jaccard.
EARLY_STOP = {"window": 100, "loss_tol": 0.005, "jaccard_tol": 0.005, "min_epochs": 150}


def converged(lines: list, rule: dict | None = None) -> str | None:
    """Why the training whose progress is ``lines`` may stop now, or None.  Only the epochs since
    the last start from scratch count: a restart's run begins again."""
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
    loss = [e["loss_test"] for e in epochs]
    jac = [(e.get("metrics") or {}).get("jaccard") or 0.0 for e in epochs]
    before, last = min(loss[:-w]), min(loss[-w:])
    if before - last > float(rule["loss_tol"]) * abs(before):
        return None
    if max(jac[-w:]) - max(jac[:-w]) > float(rule["jaccard_tol"]):
        return None
    return (f"no better over the last {w} epochs: test loss {last:.4g} against {before:.4g}, "
            f"Jaccard {max(jac[-w:]):.3f} against {max(jac[:-w]):.3f}")
