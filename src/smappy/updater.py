"""Update and restart: the helper that installs a new SMAPpy once the old one has gone.

A running SMAPpy cannot replace itself on Windows -- its compiled extensions
and ``smappy-gui.exe`` are locked while it runs -- and anywhere else what it
has imported stays the old version until it starts again.  So the GUI writes
a plan (`write_plan`), starts this file as a process of its own and quits;
this waits for SMAPpy's process to end (and on Windows for the
``smappy-gui.exe`` launcher that started it), runs the install, writes what
happened where the next start reads it (`smappy.updates.last_update`), and
starts SMAPpy again -- whether or not the install worked, so a failure is
said in a window rather than nowhere.

This file is the whole helper and imports nothing but the standard library:
the GUI copies it out of the package and runs it with ``python -I``, so no
file of the installation being replaced is open or imported while pip works
(`start_helper`).  pip is retried once after a pause, for a file another
process let go of a moment late.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

DISTRIBUTION = "smappy-smlm"
WAIT_S = 120            # for SMAPpy to close: then the install is tried anyway
RETRY_PAUSE_S = 5
PLAN_PREFIX = "smappy-update-"


def commands(kind: str, python: str, uv: Optional[str] = None) -> Optional[List[List[str]]]:
    """The install, as argument lists, for a kind of installation
    (`smappy.updates.how_installed`); None where the GUI only gives the
    command -- a checkout, whose ``git pull`` may meet local changes, and
    ``uvx``, which fetches the newest on its own when asked for ``@latest``."""
    uv = uv if uv is not None else shutil.which("uv")
    if kind == "pip":
        return [[python, "-m", "pip", "install", "--upgrade", DISTRIBUTION]]
    # uv keeps PyPI's list of releases for the ten minutes PyPI allows, so
    # an update just after a release found nothing new (0.5.7, October
    # 2026); pip asks again for every list on its own
    if kind == "uv tool" and uv:
        return [[uv, "tool", "upgrade", "--refresh-package", DISTRIBUTION, DISTRIBUTION]]
    if kind == "uv" and uv:
        return [[uv, "pip", "install", "--upgrade", "--refresh-package", DISTRIBUTION,
                 "--python", python, DISTRIBUTION]]
    return None


def relaunch_command(python: str) -> List[str]:
    """SMAPpy again: through the same Python, which an upgrade keeps, else
    whatever ``smappy-gui`` the PATH finds."""
    if Path(python).exists():
        return [python, "-m", "smappy.cli.gui"]
    found = shutil.which("smappy-gui")
    return [found] if found else [python, "-m", "smappy.cli.gui"]


def pids_to_wait_for() -> List[int]:
    """This process, and on Windows the ``smappy-gui.exe`` launcher that
    started it, which pip must replace too and which stays until its child
    has gone.  Not any other parent: a terminal stays open for good."""
    pids = [os.getpid()]
    if sys.platform.startswith("win") and Path(sys.argv[0]).stem.lower() == "smappy-gui":
        pids.append(os.getppid())
    return pids


def write_plan(folder: Path, install: List[List[str]], relaunch: List[str],
               result: Path, log: Path, version: str) -> Path:
    plan = {"wait_for": pids_to_wait_for(), "commands": install, "relaunch": relaunch,
            "result": str(result), "log": str(log), "from_version": version}
    path = Path(folder) / "plan.json"
    path.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    return path


def _detached() -> Dict:
    """Popen's arguments for a process that outlives the one starting it and
    opens no window of its own."""
    if sys.platform.startswith("win"):
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                 | 0x01000000)                  # CREATE_BREAKAWAY_FROM_JOB
        return {"creationflags": flags, "close_fds": True}
    return {"start_new_session": True, "close_fds": True}


def spawn(argv: List[str], **kwargs) -> subprocess.Popen:
    options = dict(_detached(), stdin=subprocess.DEVNULL, **kwargs)
    try:
        return subprocess.Popen(argv, **options)
    except OSError:
        if "creationflags" not in options:
            raise
        # a job that forbids breaking away (some terminals run in one)
        options["creationflags"] &= ~0x01000000
        return subprocess.Popen(argv, **options)


def start_helper(plan: Path, python: Optional[str] = None) -> subprocess.Popen:
    """This file, copied beside the plan, run isolated (``-I``: no user site,
    no current directory on the path) so that it imports nothing of the
    installation it replaces."""
    script = Path(plan).parent / "smappy_updater.py"
    shutil.copyfile(__file__, script)
    return spawn([python or sys.executable, "-I", str(script), str(plan)],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def new_plan_folder() -> Path:
    return Path(tempfile.mkdtemp(prefix=PLAN_PREFIX))


# ------------------------------------------------------------ the helper

def wait_for_exit(pid: int, timeout_s: float = WAIT_S) -> bool:
    """True once the process is gone; False if it is still there at the end."""
    if sys.platform.startswith("win"):
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000, False, pid)       # SYNCHRONIZE
        if not handle:
            return True                                             # gone already
        try:
            return kernel32.WaitForSingleObject(handle, int(timeout_s * 1000)) == 0
        finally:
            kernel32.CloseHandle(handle)
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:         # someone else's process now: ours is gone
            return True
        if _zombie(pid):                # exited; its parent has not collected it yet
            return True
        time.sleep(0.2)
    return False


def _zombie(pid: int) -> bool:
    """A process that has exited still answers ``kill(pid, 0)`` until its
    parent waits for it.  Linux says so in ``/proc``; macOS, which has no
    ``/proc``, in ``ps``'s state column (``Z``)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        stat = None
    if stat is not None:
        return stat.rsplit(")", 1)[-1].split()[:1] == ["Z"]
    try:
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
                               text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return state.startswith("Z")


def _run(command: List[str], log) -> int:
    log.write(f"\n$ {' '.join(command)}\n")
    log.flush()
    try:
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL).returncode
    except OSError as error:
        log.write(f"could not run it: {error}\n")
        return 127


def run_plan(plan: Dict, retry_pause_s: float = RETRY_PAUSE_S) -> Dict:
    log_path = Path(plan["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"ok": True, "from_version": plan.get("from_version"), "log": str(log_path),
              "command": " ".join(plan["commands"][-1]) if plan["commands"] else ""}
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write(f"SMAPpy update, {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for pid in plan.get("wait_for", []):
            if not wait_for_exit(pid):
                log.write(f"process {pid} still running after {WAIT_S} s; installing anyway\n")
        for command in plan["commands"]:
            code = _run(command, log)
            if code != 0:
                log.write(f"exit code {code}; trying again in {retry_pause_s:g} s\n")
                log.flush()
                time.sleep(retry_pause_s)
                code = _run(command, log)
            if code != 0:
                result.update(ok=False, code=code, command=" ".join(command))
                break
        log.write("\nupdated\n" if result["ok"] else "\nthe update failed\n")
    result_path = Path(plan["result"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    plan = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    result = run_plan(plan)
    if plan.get("relaunch"):
        try:
            spawn(plan["relaunch"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as error:
            with open(plan["log"], "a", encoding="utf-8") as log:
                log.write(f"could not start SMAPpy again: {error}\n")
    folder = Path(argv[0]).parent
    if folder.name.startswith(PLAN_PREFIX):     # ours (`new_plan_folder`), and done with
        shutil.rmtree(folder, ignore_errors=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
