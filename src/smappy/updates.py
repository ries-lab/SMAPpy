"""Whether a newer SMAPpy is on PyPI, and the command that installs it here.

The GUI asks once a day at most (*Help > Check for updates...* asks at once),
and only says so: installing is left to the user, because a running SMAPpy
cannot replace itself everywhere -- on Windows its compiled extensions and
``smappy-gui.exe`` are locked while it runs.  The command depends on how this
copy was installed, which the package's own metadata records:

* a checkout installed with ``pip install -e`` (PEP 610's ``direct_url.json``
  says ``editable``): ``git pull`` and install again, which rebuilds the C++;
* a ``uv tool install`` (``uv-receipt.toml`` beside the tool's Python):
  ``uv tool upgrade``;
* an ephemeral ``uvx`` environment (in uv's cache): ``uvx`` with ``@latest``;
* anything else installed by uv (``INSTALLER`` says ``uv``): ``uv pip``;
* otherwise pip, through this very Python, so it lands in this environment
  whatever ``pip`` on the PATH would be.

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


def how_installed(prefix: Optional[str] = None, dist=None) -> str:
    """``checkout``, ``uv tool``, ``uvx``, ``uv`` or ``pip``; see the module."""
    prefix = Path(prefix or sys.prefix)
    dist = _distribution() if dist is None else dist
    direct = (dist.read_text("direct_url.json") if dist is not None else None) or ""
    try:
        if json.loads(direct).get("dir_info", {}).get("editable"):
            return "checkout"
    except ValueError:
        pass
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


def update_command(kind: Optional[str] = None, windows: Optional[bool] = None,
                   python: Optional[str] = None, dist=None) -> str:
    """The command to run in a terminal, with SMAPpy closed, to update it.

    One line for PowerShell on Windows -- Windows PowerShell 5.1 has no
    ``&&``, so steps are joined with ``;`` there -- and for a POSIX shell
    elsewhere.  pip is run through this Python, so the update goes where
    SMAPpy is, whichever ``pip`` the PATH would find.
    """
    windows = sys.platform.startswith("win") if windows is None else windows
    dist = _distribution() if dist is None else dist
    kind = kind or how_installed(dist=dist)
    python = python or sys.executable
    run = _quote(python, windows)
    if windows and run.startswith('"'):
        run = "& " + run
    if kind == "checkout":
        source = _checkout_dir(dist) if dist is not None else None
        steps = ([f"cd {_quote(source, windows)}"] if source else []) + [
            "git pull", f"{run} -m pip install -e ."]
        return ("; " if windows else " && ").join(steps)
    if kind == "uv tool":
        return f"uv tool upgrade {DISTRIBUTION}"
    if kind == "uvx":
        return f"uvx --from {DISTRIBUTION}@latest smappy-gui"
    if kind == "uv":
        return f"uv pip install --upgrade --python {_quote(python, windows)} {DISTRIBUTION}"
    return f"{run} -m pip install --upgrade {DISTRIBUTION}"


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
