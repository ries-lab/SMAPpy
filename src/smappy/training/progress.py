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
