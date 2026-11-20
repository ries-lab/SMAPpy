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

import functools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DISTRIBUTION = "smappy-smlm"
WAIT_S = 120            # for SMAPpy to close: then the install is tried anyway
RETRY_PAUSE_S = 5
PLAN_PREFIX = "smappy-update-"


def commands(kind: str, python: str, uv: Optional[str] = None, pixi: Optional[str] = None,
             workspace: Optional[Tuple[Path, str]] = None, pip: bool = True,
             version: Optional[str] = None, prefix: Optional[Path] = None,
             torch: Optional[str] = "?") -> Optional[List[List[str]]]:
    """The install, as argument lists, for a kind of installation
    (`smappy.updates.how_installed`); None where the GUI only gives the
    command -- a checkout, whose ``git pull`` may meet local changes, and
    ``uvx``, which fetches the newest on its own when asked for ``@latest``
    -- or none would work: no pip, no uv, no pixi.  ``workspace`` is a pixi
    environment's manifest and name (`smappy.updates.pixi_environment`);
    the commands are `smappy.updates.update_command`'s, which says why.
    ``version`` is the release to install (the newest where None),
    ``prefix`` a uv tool's environment and ``torch`` the torch in it, None
    for none (`tool_install`); both this process's by default."""
    uv = uv if uv is not None else shutil.which("uv")
    if kind == "pixi":
        pixi = pixi if pixi is not None else find_pixi()
        if not (pixi and workspace):
            return None
        manifest, environment = workspace
        return [[pixi, "upgrade", "--manifest-path", str(manifest), DISTRIBUTION],
                [pixi, "install", "--manifest-path", str(manifest), "-e", environment]]
    if kind == "pip":
        return [[python, "-m", "pip", "install", "--upgrade", DISTRIBUTION]] if pip else None
    # uv keeps PyPI's list of releases for the ten minutes PyPI allows, so
    # an update just after a release found nothing new (0.5.7, October
    # 2026): --refresh-package; pip asks again for every list on its own
    if kind == "uv tool" and uv:
        install = tool_install(Path(sys.prefix if prefix is None else prefix), version, torch=torch)
        return [[uv, *install]] if install is not None else None
    if kind == "uv" and uv:
        return [[uv, "pip", "install", "--upgrade", "--refresh-package", DISTRIBUTION,
                 "--python", python, DISTRIBUTION]]
    return None


def find_pixi() -> Optional[str]:
    """pixi: the one that started SMAPpy (``pixi run`` says which in
    ``PIXI_EXE``), else the PATH's, else where pixi's installer puts it --
    SMAPpy started from the Dock or a shortcut has a PATH without
    ``~/.pixi/bin``."""
    found = os.environ.get("PIXI_EXE") or shutil.which("pixi")
    if found and Path(found).exists():
        return found
    home = Path(os.environ.get("PIXI_HOME") or Path.home() / ".pixi")
    installed = home / "bin" / ("pixi.exe" if sys.platform.startswith("win") else "pixi")
    return str(installed) if installed.exists() else None


# ------------------------------------------------- a uv tool: one recipe
#
# A uv tool's environment is rebuilt by uv from its receipt
# (``uv-receipt.toml``), and whatever the receipt leaves out is lost on the
# way.  ``uv tool upgrade`` removed what SMAPpy's *Install PyTorch* had put
# in with ``uv pip`` (the receipt never hears of it), fetched torch again
# from PyPI because uv does not record ``--torch-backend`` (PyPI's Windows
# torch is the CPU build), and left a version pinned at install where it
# was; its options and their names changed under us twice between uv
# versions (issue 189 and the commits before it, October-November 2026).
# So an update is never ``uv tool upgrade``: it is the one ``uv tool
# install`` that `tool_install` writes from what the environment holds.

# uv 0.9.19 is the first whose tool commands honour --torch-backend (its
# changelog: "Respect `--torch-backend` in `uv tool` commands"); an older
# one takes the option and quietly installs PyPI's torch
MIN_UV = (0, 9, 19)

# the local part of a torch wheel from PyTorch's own index, which is uv's
# --torch-backend: 2.7.0+cu126, +cpu, +rocm6.3, +xpu
TORCH_BACKEND = re.compile(r"^(cu\d+|cpu|rocm[\d.]+|xpu)$")


@functools.lru_cache(maxsize=None)
def uv_version(uv: str) -> Optional[Tuple[int, ...]]:
    """``uv --version`` as numbers, None where it does not answer."""
    try:
        text = subprocess.run([uv, "--version"], capture_output=True, text=True, errors="replace",
                              stdin=subprocess.DEVNULL, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    return tuple(int(n) for n in match.groups()) if match else None


def uv_too_old(uv: Optional[str], version: Optional[Tuple[int, ...]] = None) -> Optional[str]:
    """What is wrong with this uv for an update, in a sentence; None when
    nothing is (or it cannot be told: the install then says)."""
    version = version if version is not None else (uv_version(uv) if uv else None)
    if version is None or version >= MIN_UV:
        return None
    return (f"This uv is {'.'.join(map(str, version))}; updating SMAPpy needs uv "
            f"{'.'.join(map(str, MIN_UV))} or newer")


def installed_torch() -> Optional[str]:
    """The version of torch in this environment, None where there is none
    -- from its metadata, never imported (seconds, and gigabytes of DLLs)."""
    from importlib import metadata
    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        return None


def torch_backend(version: Optional[str]) -> Optional[str]:
    """uv's ``--torch-backend`` that installed this torch: its local part
    (``cu126`` of ``2.7.0+cu126``); None for PyPI's own build."""
    local = (version or "").partition("+")[2]
    return local if TORCH_BACKEND.match(local) else None


def _name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def extras_present(requires: Optional[List[str]] = None, has=None) -> List[str]:
    """SMAPpy's extras whose packages are all in this environment, however
    they came -- the extra at install, or *Install PyTorch* afterwards --
    so that an update keeps them.  An empty extra (kept for old commands)
    says nothing and is left out.  ``requires`` is SMAPpy's
    ``Requires-Dist`` and ``has`` whether a package is installed; this
    environment's by default."""
    from importlib import metadata
    if requires is None:
        try:
            requires = metadata.requires(DISTRIBUTION) or []
        except metadata.PackageNotFoundError:
            requires = []
    if has is None:
        def has(name):
            try:
                metadata.version(name)
                return True
            except metadata.PackageNotFoundError:
                return False
    needs: Dict[str, List[str]] = {}
    base = set()
    for line in requires:
        name = _name(re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", line).group(1))
        marker = re.search(r"""extra\s*==\s*["']([^"']+)["']""", line)
        if marker:
            needs.setdefault(marker.group(1), []).append(name)
        elif "extra" not in line:
            base.add(name)
    # an extra of base dependencies alone (`drift`: matplotlib) is always here
    return sorted(extra for extra, names in needs.items()
                  if set(names) - base and all(has(n) for n in names))


def read_receipt(prefix: Path) -> Optional[Dict]:
    """A uv tool's ``uv-receipt.toml``, its ``[tool]`` table; None where it
    cannot be read."""
    import tomllib
    try:
        with open(Path(prefix) / "uv-receipt.toml", "rb") as file:
            tool = tomllib.load(file).get("tool")
    except (OSError, ValueError):
        return None
    return tool if isinstance(tool, dict) else None


def _requirement(entry: Dict) -> str:
    """A receipt's requirement as uv takes it on the command line."""
    text = str(entry.get("name", ""))
    if entry.get("extras"):
        text += "[" + ",".join(entry["extras"]) + "]"
    source = entry.get("url") or entry.get("git") or entry.get("path") or entry.get("directory")
    if source:
        return f"{text} @ {source}"
    return text + str(entry.get("specifier") or "")


def _option_args(options: Dict) -> List[str]:
    """``[tool.options]`` of a receipt as the flags that wrote it: its keys
    are uv's option names (``extra-index-url``, ``prerelease``); an index
    is ``--index name=url``, or ``--default-index`` for the default."""
    args: List[str] = []
    for key, value in options.items():
        if key == "index":
            for index in value if isinstance(value, list) else []:
                if not isinstance(index, dict) or not index.get("url"):
                    continue
                if index.get("default"):
                    args += ["--default-index", str(index["url"])]
                else:
                    name = index.get("name")
                    args += ["--index", f"{name}={index['url']}" if name else str(index["url"])]
        elif value is True:
            args.append(f"--{key}")
        elif isinstance(value, (str, int, float)) and value is not False:
            args += [f"--{key}", str(value)]
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, (str, int, float)):
                    args += [f"--{key}", str(item)]
    return args


def tool_install(prefix: Path, version: Optional[str] = None, extras=None,
                 torch: Optional[str] = "?") -> Optional[List[str]]:
    """The ``uv tool install`` (its arguments after ``uv``) that puts
    SMAPpy ``version`` (the newest where None) into the uv tool at
    ``prefix`` and keeps everything else as it is; None where the tool has
    no receipt to read or SMAPpy did not come from an index.

    It repeats all the receipt says, since an install replaces the receipt
    and drops what it is not given -- other ``--with`` requirements, the
    Python, the indexes (tried with uv 0.12.4) -- and adds what the receipt
    cannot hold: the extras whose packages are here (`extras_present`) and
    the build of torch (`torch_backend`).  The version is a lower bound:
    a pin made at install no longer holds SMAPpy back, uv cannot settle for
    an old list of releases, and the receipt it leaves pins nothing, so a
    ``uv tool upgrade`` by hand still moves it later.  No ``--force``, which installs every package
    again: an existing tool is changed in place, SMAPpy and what its new
    version needs.  ``extras`` are SMAPpy's extras present (this
    environment's by default) and ``torch`` its torch version, None for
    none.
    """
    tool = read_receipt(prefix)
    if tool is None:
        return None
    requirements = [r for r in tool.get("requirements") or [] if isinstance(r, dict)]
    ours = [r for r in requirements if _name(r.get("name", "")) == DISTRIBUTION]
    if len(ours) != 1 or any(ours[0].get(k) for k in ("url", "git", "path", "directory")):
        return None
    wanted = sorted(set(ours[0].get("extras") or []) |
                    set(extras_present() if extras is None else extras))
    target = DISTRIBUTION + (f"[{','.join(wanted)}]" if wanted else "")
    target += f">={version}" if version else "@latest"
    args = ["tool", "install", "--refresh-package", DISTRIBUTION, target]
    for entry in requirements:
        if entry is not ours[0]:
            args += ["--with", _requirement(entry)]
    if tool.get("python"):
        args += ["--python", str(tool["python"])]
    options = tool.get("options")
    args += _option_args(options if isinstance(options, dict) else {})
    backend = torch_backend(installed_torch() if torch == "?" else torch)
    if backend is not None and "--torch-backend" not in args:
        args += ["--torch-backend", backend]
    return args


def torch_check(python: str, torch: Optional[str] = "?") -> List[List[str]]:
    """``import torch`` in SMAPpy's Python, where torch is there before the
    update: an update that loses it is said at the next start, not at the
    next uiPSF calibration, where a user found it (issue 189)."""
    torch = installed_torch() if torch == "?" else torch
    return [[python, "-I", "-c", "import torch"]] if torch else []


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
               result: Path, log: Path, version: str,
               checks: Optional[List[List[str]]] = None) -> Path:
    """``checks`` are commands that must still work after the install
    (`torch_check`); one that fails makes the update's result ``lost``."""
    plan = {"wait_for": pids_to_wait_for(), "commands": install, "relaunch": relaunch,
            "result": str(result), "log": str(log), "from_version": version,
            "checks": checks or []}
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


def helper_python() -> str:
    """The Python the helper runs on: the one SMAPpy's environment was made
    from, not the environment's own.  On Windows a venv's
    ``Scripts\\python.exe`` stays locked while the helper runs, and
    ``uv tool upgrade`` rebuilds a tool's environment where its Python has
    changed -- which then fails on that one file (0.5.8, October 2026).
    The base interpreter is outside every environment an install touches,
    and the helper needs nothing but the standard library."""
    base = getattr(sys, "_base_executable", None)
    if base and base != sys.executable and Path(base).exists():
        return base
    return sys.executable


def start_helper(plan: Path, python: Optional[str] = None) -> subprocess.Popen:
    """This file, copied beside the plan, run isolated (``-I``: no user site,
    no current directory on the path) so that it imports nothing of the
    installation it replaces, on a Python outside it (`helper_python`)."""
    script = Path(plan).parent / "smappy_updater.py"
    shutil.copyfile(__file__, script)
    return spawn([python or helper_python(), "-I", str(script), str(plan)],
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
        if result["ok"]:
            for command in plan.get("checks") or []:
                if _run(command, log) != 0:
                    log.write(f"after the update this fails: {' '.join(command[1:])}\n")
                    result.setdefault("lost", []).append(command[-1])
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
