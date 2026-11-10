"""Whether a newer SMAPpy is on PyPI, and the command that installs it here.

The GUI asks once a day at most (*Help > Check for updates...* asks at once),
and only says so: installing is left to the user, because a running SMAPpy
cannot replace itself everywhere -- on Windows its compiled extensions and
``smappy-gui.exe`` are locked while it runs.  The command depends on how this
copy was installed, which the package's own metadata records:

* a checkout installed editable (PEP 610's ``direct_url.json`` says
  ``editable``): ``git pull`` and install again, which rebuilds the C++ --
  by the tool that installed it, since neither a pixi environment nor a uv
  one has pip;
* a pixi environment (``conda-meta/pixi``): ``pixi upgrade`` in its
  workspace, which moves the manifest's range past a new minor version
  rather than leave the user on the old one, then ``pixi install`` of the
  environment, which ``upgrade`` installs only when it is the default;
* a ``uv tool install`` (``uv-receipt.toml`` beside the tool's Python):
  ``uv tool upgrade``;
* an ephemeral ``uvx`` environment (in uv's cache): ``uvx`` with ``@latest``;
* anything else installed by uv (``INSTALLER`` says ``uv``): ``uv pip``;
* otherwise pip, through this very Python, so it lands in this environment
  whatever ``pip`` on the PATH would be -- conda's and mamba's environments
  among them, which have pip beside their Python.

Where none of these fits and this Python has no pip either, there is no
command to give: the dialog says so rather than offer one that fails.

Qt-free, so the tests reach every decision without a window; the request
itself is the GUI's (`gui/updates.py`), made with QtNetwork, which uses the
system's certificates and proxy where Python's ``urllib`` often has neither.
"""
from __future__ import annotations

import json
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from . import config

DISTRIBUTION = "smappy-smlm"
PYPI_JSON = f"https://pypi.org/pypi/{DISTRIBUTION}/json"
PYPI_PAGE = f"https://pypi.org/project/{DISTRIBUTION}/"
REPOSITORY = "https://github.com/ries-lab/SMAPpy"
INTERVAL_S = 24 * 3600          # automatic checks: once a day at most


def installed() -> str:
    from .pipeline import _version
    return _version()


def parse(version: str) -> Tuple:
    """A version as something that sorts: the release numbers, then whether
    it is a final release, so ``0.5.0rc1 < 0.5.0 < 0.5.1``.

    Not `packaging`, which is not a dependency; PyPI's versions here are
    plain PEP 440 releases and this is all the comparison they need.  A
    local part (``+g1234``) is ignored.
    """
    text = version.strip().lower().split("+")[0].lstrip("v")
    match = re.match(r"(\d+(?:\.\d+)*)(.*)", text)
    if match is None:
        return ((), 0, text)
    release = tuple(int(n) for n in match.group(1).split("."))
    while len(release) > 1 and release[-1] == 0:      # 0.5 == 0.5.0
        release = release[:-1]
    rest = match.group(2)
    final = 1 if rest == "" or rest.startswith(".post") else 0
    return (release, final, rest)


def is_newer(candidate: str, than: str) -> bool:
    return parse(candidate) > parse(than)


def newest_from(payload: bytes | str) -> Optional[str]:
    """The newest release in PyPI's JSON answer, or None if it is not one.

    ``info.version`` is the newest final release, which is what to offer:
    a release candidate is for those who ask for it by name.
    """
    try:
        data = json.loads(payload)
        version = data["info"]["version"]
    except (ValueError, KeyError, TypeError):
        return None
    return version if isinstance(version, str) and version else None


def changes_url(installed: str, newest: str) -> str:
    """The commits between two releases, which are tagged ``v<version>``:
    there are no release notes, and PyPI's page is the README."""
    return f"{REPOSITORY}/compare/v{installed.split('+')[0]}...v{newest}"


# ------------------------------------------------------------ how installed

def _distribution():
    from importlib import metadata
    try:
        return metadata.distribution(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return None


def pixi_environment(prefix: Optional[str] = None) -> Optional[Tuple[Path, str]]:
    """The pixi workspace's manifest and the environment's name, where this
    is a pixi environment; None elsewhere.

    pixi records both in the environment's ``conda-meta/pixi``, which is
    there however SMAPpy was started -- a launcher has none of ``pixi run``'s
    variables -- and wherever the environment is, which need not be under
    the workspace (pixi's *detached environments*).  Without that file the
    environment's place says it: ``<workspace>/.pixi/envs/<name>``.
    """
    prefix = Path(prefix or sys.prefix)
    try:
        data = json.loads((prefix / "conda-meta" / "pixi").read_text(encoding="utf-8"))
        return Path(data["manifest_path"]), str(data.get("environment_name") or "default")
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if prefix.parent.name == "envs" and prefix.parent.parent.name == ".pixi":
        workspace = prefix.parent.parent.parent
        for name in ("pixi.toml", "pyproject.toml"):
            if (workspace / name).exists():
                return workspace / name, prefix.name
    return None


def how_installed(prefix: Optional[str] = None, dist=None) -> str:
    """``checkout``, ``pixi``, ``uv tool``, ``uvx``, ``uv`` or ``pip``; see
    the module."""
    prefix = Path(prefix or sys.prefix)
    dist = _distribution() if dist is None else dist
    direct = (dist.read_text("direct_url.json") if dist is not None else None) or ""
    try:
        if json.loads(direct).get("dir_info", {}).get("editable"):
            return "checkout"
    except ValueError:
        pass
    if pixi_environment(prefix) is not None:
        return "pixi"
    if (prefix / "uv-receipt.toml").exists():
        return "uv tool"
    installer = ((dist.read_text("INSTALLER") if dist is not None else None) or "").strip()
    if installer == "uv":
        # uvx builds its environments in uv's cache, under archive-v0 (or a
        # later version of that name); a tool or a venv lives anywhere else
        if any(part.startswith("archive-v") for part in prefix.parts):
            return "uvx"
        return "uv"
    return "pip"


def _checkout_dir(dist) -> Optional[Path]:
    """Where an editable install's source is, from ``direct_url.json``."""
    from urllib.parse import unquote, urlparse
    from urllib.request import url2pathname
    try:
        url = json.loads(dist.read_text("direct_url.json") or "")["url"]
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    return Path(url2pathname(unquote(parsed.path)))


def _quote(path, windows: bool) -> str:
    """A path as the terminal the install guide uses takes it: PowerShell on
    Windows, where a quoted program needs ``&`` to be run, and a POSIX shell
    elsewhere."""
    text = str(path)
    if not windows:
        return shlex.quote(text)
    return f'"{text}"' if " " in text else text


def _installer(dist) -> str:
    return ((dist.read_text("INSTALLER") if dist is not None else None) or "").strip()


def has_pip() -> bool:
    import importlib.util
    return importlib.util.find_spec("pip") is not None


def update_command(kind: Optional[str] = None, windows: Optional[bool] = None,
                   python: Optional[str] = None, dist=None, prefix: Optional[str] = None,
                   pip: Optional[bool] = None) -> str:
    """The command to run in a terminal, with SMAPpy closed, to update it;
    empty where there is none to give (the module's last paragraph).

    One line for PowerShell on Windows -- Windows PowerShell 5.1 has no
    ``&&``, so steps are joined with ``;`` there -- and for a POSIX shell
    elsewhere.  pip is run through this Python, so the update goes where
    SMAPpy is, whichever ``pip`` the PATH would find.  ``prefix`` is the
    environment (this one by default) and ``pip`` whether it has pip (asked
    of this Python by default).
    """
    windows = sys.platform.startswith("win") if windows is None else windows
    dist = _distribution() if dist is None else dist
    kind = kind or how_installed(prefix, dist=dist)
    python = python or sys.executable
    pip = has_pip() if pip is None else pip
    run = _quote(python, windows)
    if windows and run.startswith('"'):
        run = "& " + run
    then = "; " if windows else " && "
    pixi = pixi_environment(prefix)
    if kind == "checkout":
        source = _checkout_dir(dist) if dist is not None else None
        if pixi is not None:
            # a pixi environment has no pip, and pixi keeps its lock in step
            install = (f"pixi reinstall --manifest-path {_quote(pixi[0], windows)} "
                       f"-e {pixi[1]} {DISTRIBUTION}")
        elif _installer(dist) == "uv":
            install = f"uv pip install --python {_quote(python, windows)} -e ."
        elif pip:
            install = f"{run} -m pip install -e ."
        else:
            return ""
        steps = ([f"cd {_quote(source, windows)}"] if source else []) + ["git pull", install]
        return then.join(steps)
    if kind == "pixi" and pixi is not None:
        manifest = f"--manifest-path {_quote(pixi[0], windows)}"
        return then.join([f"pixi upgrade {manifest} {DISTRIBUTION}",
                          f"pixi install {manifest} -e {pixi[1]}"])
    if kind == "uv tool":
        # uv's copy of PyPI's list may predate the release, and how to ask
        # afresh depends on the uv (`smappy.updater.refresh_args`)
        import shutil
        from .updater import refresh_args
        uv = shutil.which("uv")
        refresh = refresh_args(uv) if uv else ["--refresh-package", DISTRIBUTION]
        return f"uv tool upgrade {' '.join(refresh)} {DISTRIBUTION}"
    if kind == "uvx":
        return f"uvx --from {DISTRIBUTION}@latest smappy-gui"
    if kind == "uv":
        return (f"uv pip install --upgrade --refresh-package {DISTRIBUTION} "
                f"--python {_quote(python, windows)} {DISTRIBUTION}")
    return f"{run} -m pip install --upgrade {DISTRIBUTION}" if pip else ""


# ------------------------------------------------------------- the schedule

def automatic() -> bool:
    """Whether the GUI checks on its own.  On by default: an old copy is how
    a fixed bug keeps being reported.  The request sends PyPI nothing but
    itself; Preferences > General turns it off."""
    return bool(config.get("check_for_updates", True))


def set_automatic(on: bool) -> None:
    config.set("check_for_updates", bool(on))


def due(now: Optional[float] = None) -> bool:
    """An automatic check is due: switched on, and a day since the last.

    A last check in the future -- a clock set back -- counts as long ago,
    or the checks would stop until the clock caught up.
    """
    if not automatic():
        return False
    now = time.time() if now is None else now
    try:
        last = float(config.get("updates_checked_at", 0) or 0)
    except (TypeError, ValueError):
        last = 0.0
    return not (0 <= now - last < INTERVAL_S)


def remember(newest: Optional[str], now: Optional[float] = None) -> None:
    """A check was made: when, and what it found, so the next start can say
    so without asking again."""
    data = dict(config.load())
    data["updates_checked_at"] = time.time() if now is None else now
    if newest:
        data["updates_newest"] = newest
    config.save(data)


def known_newer() -> Optional[str]:
    """A newer release that an earlier check found, if this copy is older."""
    newest = config.get("updates_newest")
    if isinstance(newest, str) and is_newer(newest, installed()):
        return newest
    return None


# ------------------------------------------------- update and restart's trace

def result_file() -> Path:
    return config.config_dir() / "update-result.json"


def log_file() -> Path:
    return config.config_dir() / "update.log"


def last_update() -> Optional[dict]:
    """What the helper of *Update and restart* left (`smappy.updater`), once:
    the file is removed as it is read, so the next start says nothing."""
    path = result_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        path.unlink()
    except OSError:
        pass
    return data if isinstance(data, dict) else None


def log_tail(lines: int = 15) -> str:
    try:
        text = log_file().read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.strip().splitlines()[-lines:])


def log_error() -> str:
    """What the install's output says went wrong: the last ``error:`` line
    (uv's; pip's is ``ERROR:``) and uv's ``Caused by:`` lines under it,
    which name the file and the OS error -- the end of the log is often only
    a hint or the retry's notice."""
    try:
        lines = log_file().read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    starts = [i for i, line in enumerate(lines) if line.strip().lower().startswith("error")]
    if not starts:
        return ""
    found = [lines[starts[-1]].strip()]
    for line in lines[starts[-1] + 1:]:
        if not line.strip().lower().startswith("caused by"):
            break
        found.append(line.strip())
    return "\n".join(found)


# What the install's output says, turned into what to do about it: the
# failures seen or expected, each a test on the log's text, lowercased.
UV_UPDATE = ("uv self update (if uv came from its installer; otherwise as it was "
             "installed: pip install -U uv, winget upgrade astral-sh.uv, brew upgrade uv)")
_ADVICE = (
    (lambda log: "uv" in log and "unexpected argument" in log,
     "The uv on this computer is too old for this update. Update uv with\n"
     f"{UV_UPDATE}\nand then use Update and restart again."),
    (lambda log: any(text in log for text in (
        "os error 32", "being used by another process", "access is denied",
        "winerror 5", "winerror 32", "permission denied", "permissionerror")),
     "A file of SMAPpy was still in use. Close every SMAPpy window, and anything "
     "else run from SMAPpy's Python (a notebook, a script), then run the command "
     "below by hand."),
    (lambda log: any(text in log for text in (
        "failed to fetch", "could not fetch", "dns error", "error sending request",
        "connection refused", "connection reset", "timed out", "proxyerror",
        "newconnectionerror", "certificate verify failed", "invalid peer certificate")),
     "PyPI could not be reached. Check the internet connection; behind a proxy, "
     "set HTTPS_PROXY (and, for a company certificate, UV_NATIVE_TLS=1 for uv) "
     "before running the command below."),
)


def failure_advice(ok: bool = False) -> str:
    """What to do about the last *Update and restart*, in a sentence or two
    a user can act on, from what its log says; empty where the failure is
    not one of the known ones -- the error line and the log say the rest.
    ``ok``: the install said it worked, but this is still the old version."""
    if ok:
        return ("PyPI may not have offered the new release yet; try again in ten "
                f"minutes. Or the new release needs a newer Python than this one "
                f"({sys.version.split()[0]}).")
    try:
        log = log_file().read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return ""
    for matches, advice in _ADVICE:
        if matches(log):
            return advice
    return ""
