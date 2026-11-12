"""A job trained on a workstation over ssh: sent there, started, followed, stopped, its model fetched.

The Train window's other *where* (`jobs.Host`): a computer with a GPU this one reaches with ``ssh``
without a password -- a key, or a socket an earlier login left (``BatchMode``: a prompt would hang
the window).  Nothing is shared but the job folder: the training simulates its own frames, so the
movie stays here, and a job folder is a few files (the calibration, ``method.yaml``, ``job.json``;
about 30 MB with the test set it writes there).

**Sent** as a tar stream into ``ssh HOST 'mkdir -p DIR && tar -xzf - -C DIR'``, made here by
`tarfile`, so this side needs no ``tar`` (Windows).  The remote folder is ``<host folder>/<job>``.
**Started** as ``nohup COMMAND train DIR --epochs N`` detached there, its output in ``DIR/train.log``;
COMMAND is the host's (``smappy-decode``, or a whole command line: an environment's python ``-m
decode_smlm.cli``).  **Followed** by copying ``progress.jsonl``, ``lock.json`` and the end of ``train.log`` back into the local
folder, so `jobs.status` reads a job on a host as one here, and the window draws it the same; the
lock's process is asked about there (``kill -0``), and a training that died says *interrupted*.
**Stopped** by the STOP file there.  **The model** is copied back when asked for (*Try*, *Use*)
and when the training finished -- 9 MB, not every few seconds.  The checkpoint stays there: a
resumed training resumes there.
"""
from __future__ import annotations

import io
import json
import shlex
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .jobs import LOCK, LOG, Entry, Host, add, host

PROGRESS, MODEL, STOP = "progress.jsonl", "model.safetensors", "STOP"
# what the training writes; never sent back over a running one
OUTPUTS = {PROGRESS, MODEL, LOCK, STOP, LOG, "ckpt.pt"}
SEPARATOR = "--- smappy: progress ---"
LOG_LINES = 50
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")
TIMEOUT_S = 60


class RemoteError(RuntimeError):
    """What ssh or the command said, for the window to show."""


def _quote_dir(path: str) -> str:
    """A remote path for the shell there: ``~`` left to it to expand, the rest quoted."""
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


def ssh(h: Host, command: str, data: Optional[bytes] = None, timeout: float = TIMEOUT_S,
        runner=None) -> bytes:
    """``command`` run by the shell on ``h``; its output, or a `RemoteError` with what it said."""
    runner = runner or subprocess.run       # looked up at the call: a test replaces it
    try:
        done = runner(["ssh", *SSH_OPTIONS, h.ssh, command], input=data, capture_output=True, timeout=timeout)
    except FileNotFoundError as e:
        raise RemoteError("no ssh on this computer") from e
    except subprocess.TimeoutExpired as e:
        raise RemoteError(f"{h.name} did not answer in {timeout:.0f} s") from e
    if done.returncode != 0:
        said = (done.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise RemoteError(f"{h.name}: " + (said[-1] if said else f"exit code {done.returncode}"))
    return done.stdout or b""


def remote_folder(h: Host, folder) -> str:
    return f"{h.folder.rstrip('/')}/{Path(folder).name}"


def pack(folder) -> bytes:
    """The job folder as a gzip tar, without what a training writes."""
    folder = Path(folder)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.name not in OUTPUTS:
                tar.add(path, arcname=path.name)
    return buffer.getvalue()


def send(folder, h: Host, runner=None) -> Entry:
    """The job folder copied to ``h`` and listed as running there."""
    remote = remote_folder(h, folder)
    ssh(h, f"mkdir -p {_quote_dir(remote)} && tar -xzf - -C {_quote_dir(remote)}", data=pack(folder),
        timeout=600, runner=runner)
    return add(folder, host=h.name, remote=remote)


def train_command(h: Host, remote: str, epochs: int, device: str = "auto", all_epochs: bool = False) -> str:
    """The shell line that starts the training there, detached: it outlives the ssh session."""
    d = _quote_dir(remote)
    train = f"{h.command} train {d} --epochs {int(epochs)}"
    if all_epochs:
        train += " --all-epochs"
    if device and device != "auto":
        train += f" --device {shlex.quote(device)}"
    if h.allow_other_version:
        train += " --allow-other-version"
    return f"cd {d} && rm -f {STOP} && (nohup {train} >> {LOG} 2>&1 < /dev/null &) && echo started"


def start(entry: Entry, epochs: int, device: str = "auto", runner=None, all_epochs: bool = False) -> None:
    h = _host_of(entry)
    ssh(h, train_command(h, entry.remote, epochs, device, all_epochs), runner=runner)


def stop(entry: Entry, runner=None) -> None:
    h = _host_of(entry)
    ssh(h, f"touch {_quote_dir(entry.remote)}/{STOP}", runner=runner)


def follow(entry: Entry, runner=None) -> bool:
    """The job's progress and lock copied back; False where the lock's training is gone there (the
    window says *interrupted*).  One ssh call: a JSON line of what is there, then the progress."""
    h = _host_of(entry)
    d = _quote_dir(entry.remote)
    script = (f"cd {d} || exit 3; "
              f"if [ -f {LOCK} ]; then pid=$(sed -n 's/.*\"pid\": *\\([0-9]*\\).*/\\1/p' {LOCK}); "
              f"if kill -0 $pid 2>/dev/null; then alive=1; else alive=0; fi; lock=$(cat {LOCK}); "
              f"else alive=0; lock=null; fi; "
              f"printf '{{\"alive\": %s, \"lock\": %s}}\\n' $alive \"$lock\"; "
              f"tail -n {LOG_LINES} {LOG} 2>/dev/null; echo {SEPARATOR}; cat {PROGRESS} 2>/dev/null; true")
    out = ssh(h, script, runner=runner).decode("utf-8", "replace")
    head, _, rest = out.partition("\n")
    log, _, progress = rest.partition(SEPARATOR + "\n")
    try:
        told = json.loads(head)
    except ValueError:
        told = {"alive": 0, "lock": None}
    folder = Path(entry.folder)
    if log.strip():                         # its end, for a training that would not start (`jobs.status`)
        (folder / LOG).write_text(log, encoding="utf-8")
    if progress.strip():
        (folder / PROGRESS).write_text(progress, encoding="utf-8")
    lock = folder / LOCK
    if told.get("lock") and told.get("alive"):      # a lock left by a dead training is no lock: `jobs.status`
        lock.write_text(json.dumps(told["lock"]), encoding="utf-8")
    elif lock.exists():
        lock.unlink()
    return bool(told.get("alive"))


def fetch_model(entry: Entry, runner=None) -> Path:
    """The newest model copied back into the local folder."""
    h = _host_of(entry)
    data = ssh(h, f"cat {_quote_dir(entry.remote)}/{MODEL}", timeout=300, runner=runner)
    if not data:
        raise RemoteError(f"{h.name}: no model yet")
    path = Path(entry.folder) / MODEL
    part = path.with_suffix(".part")
    part.write_bytes(data)
    part.replace(path)
    return path


def check(h: Host, runner=None) -> str:
    """What Preferences' *Test* says: that ssh reaches the host and its command answers."""
    out = ssh(h, f"{h.command} --help >/dev/null && echo ok && nvidia-smi --query-gpu=name --format=csv,noheader "
                 f"2>/dev/null | head -1; true", runner=runner).decode("utf-8", "replace").split()
    if not out or out[0] != "ok":
        raise RemoteError(f"{h.name}: {h.command} did not answer")
    gpu = " ".join(out[1:])
    return f"{h.name} answers; " + (f"its GPU: {gpu}" if gpu else "no NVIDIA GPU seen there")


def _host_of(entry: Entry) -> Host:
    found = host(entry.host) if entry.host else None
    if found is None or not entry.remote:
        raise RemoteError(f"{entry.name}: its host {entry.host!r} is not in Preferences > Training hosts")
    return found


# ----------------------------------------------------------------------------- jobs not on this list
@dataclass
class Found:
    """A job folder in a host's folder that this computer's list does not hold: one started from
    another computer (or taken off this list)."""
    host: str
    remote: str                       # its folder there
    state: str                        # as `jobs.Status.state`: new, running, interrupted, stopped, finished, error
    epoch: Optional[int] = None
    converged: bool = False

    @property
    def name(self) -> str:
        return self.remote.rstrip("/").rsplit("/", 1)[-1]


def list_jobs(h: Host, runner=None) -> list:
    """Every job folder in ``h``'s folder, with the last line of its progress and whether its lock's
    process lives: one ssh call."""
    folder = _quote_dir(h.folder.rstrip("/"))
    script = (f"for d in {folder}/*/; do [ -f \"$d/job.json\" ] || continue; alive=0; "
              f"if [ -f \"$d/{LOCK}\" ]; then pid=$(sed -n 's/.*\"pid\": *\\([0-9]*\\).*/\\1/p' \"$d/{LOCK}\"); "
              f"kill -0 $pid 2>/dev/null && alive=1; fi; "
              f"printf '%s\\t%s\\t%s\\n' \"${{d%/}}\" $alive \"$(tail -n 1 \"$d/{PROGRESS}\" 2>/dev/null)\"; done; true")
    out = ssh(h, script, runner=runner).decode("utf-8", "replace")
    found = []
    for line in out.splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        remote, alive = parts[0], parts[1] == "1"
        try:
            last = json.loads(parts[2]) if len(parts) > 2 and parts[2].strip() else None
        except ValueError:
            last = None
        kind = (last or {}).get("kind")
        if kind is None:
            state = "new"
        elif kind in ("stopped", "finished", "error"):
            state = kind
        else:
            state = "running" if alive else "interrupted"
        found.append(Found(h.name, remote, state, epoch=(last or {}).get("epoch"),
                           converged=bool((last or {}).get("converged"))))
    return found


def others(listed: list, runner=None) -> tuple:
    """The jobs on every training host that ``listed`` (this computer's entries) does not hold, and
    the hosts that could not be reached with why; for the window's thread, never raising."""
    from .jobs import hosts
    # by host and folder name: the list keeps ``~/...`` as sent, the shell there answers it expanded,
    # and a job's folder there is always ``<host folder>/<its name>`` (`remote_folder`)
    known = {(e.host, e.remote.rstrip("/").rsplit("/", 1)[-1]) for e in listed if e.host and e.remote}
    found, problems = [], {}
    for h in hosts():
        try:
            found += [f for f in list_jobs(h, runner=runner) if (f.host, f.name) not in known]
        except Exception as error:                   # an unreachable host is said, not raised
            problems[h.name] = str(error)
    return found, problems


def adopt(found: Found, root=None, runner=None) -> Entry:
    """A job found on a host put on this computer's list: its job files copied into
    ``<root>/<host>/<job>`` (the config folder's ``training_jobs`` by default), its progress
    followed from there as one sent from here."""
    from .. import config
    h = host(found.host)
    if h is None:
        raise RemoteError(f"no training host {found.host!r}")
    root = Path(root) if root is not None else config.config_dir() / "training_jobs"
    local = root / h.name / found.name
    local.mkdir(parents=True, exist_ok=True)
    data = ssh(h, f"cd {_quote_dir(found.remote)} && tar -czf - $(ls | grep -v -e ckpt.pt -e {MODEL} -e {STOP})",
               timeout=300, runner=runner)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(local, filter="data")
    entry = add(local, host=h.name, remote=found.remote.rstrip("/"))
    follow(entry, runner=runner)
    return entry
