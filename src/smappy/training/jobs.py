"""The trainings this computer prepared or started, and where they run: the Train window's list.

A job is a folder (`training.job`), beside the movie it was prepared on, trained here or on a
workstation over ssh.  The window lists every one it knows, after a restart too, so the list is
links in the configuration folder (``training_jobs.json``), as `calibration_list` keeps calibrations:
a folder moved or deleted is shown as missing, never removed from disk.

**Where a job runs** is its entry's ``host``: None here, else the name of a training host
(`hosts`) -- an ssh alias, the folder jobs go to there, the command that trains -- with the job's
folder there in ``remote``.  The local folder stays the job's: the window copies the model and the
progress back into it, so a finished job reads the same wherever it ran.

**A job's state** (`status`) is the last word of its ``progress.jsonl`` (`progress.state`), told
apart from a training that died by its ``lock.json`` (written by the training, with its host and
process): "running" with no lock, or with the lock of a process here that is gone, is
"interrupted".  A job on a host is judged on the progress copied back; whether its process lives
is the window's to ask over ssh.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from . import progress

LIST_NAME = "training_jobs.json"
HOSTS_KEY = "training_hosts"
LOCK = "lock.json"                      # decode_smlm.training.LOCK: the training's own


@dataclass
class Entry:
    """One job: its folder here, and where it runs."""
    folder: str
    host: Optional[str] = None          # a training host's name (`hosts`); None: this computer
    remote: Optional[str] = None        # the job's folder there
    added: str = ""                     # ISO time it was listed

    @property
    def name(self) -> str:
        return Path(self.folder).name

    @property
    def where(self) -> str:
        return "this computer" if self.host is None else self.host


@dataclass
class Host:
    """A workstation trainings run on: what Preferences > Training hosts holds."""
    name: str                           # what the window shows
    ssh: str                            # the alias or user@host `ssh` is given (a key or a socket: BatchMode)
    folder: str = "~/smappy_training"   # where job folders go there
    command: str = "smappy-decode"      # what trains there
    extra: dict = field(default_factory=dict)


def list_path() -> Path:
    from .. import config
    return config.config_dir() / LIST_NAME


def entries() -> List[Entry]:
    """The list as saved; an unreadable one reads as empty (links, made again by use)."""
    try:
        raw = json.loads(list_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for item in raw.get("jobs", []) if isinstance(raw, dict) else []:
        try:
            out.append(Entry(**{k: item[k] for k in item if k in Entry.__dataclass_fields__}))
        except TypeError:
            continue
    return out


def _write(items: List[Entry]) -> None:
    path = list_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "jobs": [asdict(e) for e in items]}, indent=1), encoding="utf-8")


def add(folder, host: Optional[str] = None, remote: Optional[str] = None) -> Entry:
    """List the job at ``folder``; a folder listed before is replaced (where it runs may have changed)."""
    folder = Path(folder).expanduser().resolve()
    entry = Entry(str(folder), host=host, remote=remote,
                  added=_dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    _write([e for e in entries() if Path(e.folder) != folder] + [entry])
    return entry


def remove(folder) -> None:
    """Take a job off the list; its folder stays."""
    folder = Path(folder).expanduser().resolve()
    _write([e for e in entries() if Path(e.folder) != folder])


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OverflowError, ValueError):
        return True
    return True


@dataclass
class Status:
    """What the list shows of a job."""
    state: str                          # missing, new, running, interrupted, stopped, finished, error
    epoch: Optional[int] = None
    epochs: Optional[int] = None
    last: Optional[dict] = None         # the last epoch line
    message: str = ""


def status(entry: Entry) -> Status:
    folder = Path(entry.folder)
    if not folder.is_dir():
        return Status("missing", message=f"{folder} is not there any more")
    lines = progress.read(folder)
    state = progress.state(lines)
    epochs = next((line.get("epochs") for line in reversed(lines) if line["kind"] == "start"), None)
    last = next((line for line in reversed(lines) if line["kind"] == "epoch"), None)
    message = next((line.get("message", "") for line in reversed(lines) if line["kind"] == "error"), "")
    if state == "running" and entry.host is None:
        try:
            lock = json.loads((folder / LOCK).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            lock = None
        if lock is None or (lock.get("host") == socket.gethostname() and not _alive(lock.get("pid", 0))):
            state = "interrupted"
    return Status(state, epoch=None if last is None else last.get("epoch"), epochs=epochs, last=last, message=message)


def hosts() -> List[Host]:
    """The training hosts Preferences holds."""
    from .. import config
    out = []
    for item in config.get(HOSTS_KEY, []) or []:
        try:
            out.append(Host(**{k: item[k] for k in item if k in Host.__dataclass_fields__}))
        except TypeError:
            continue
    return out


def set_hosts(items: List[Host]) -> None:
    from .. import config
    config.set(HOSTS_KEY, [asdict(h) for h in items])


def host(name: str) -> Optional[Host]:
    return next((h for h in hosts() if h.name == name), None)
