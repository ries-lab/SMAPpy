"""User configuration: a small YAML file, and where it lives.

Kept out of Qt on purpose -- the plugin scanner reads the extra plugin roots
from here and `smappy.plugins` must stay importable in a script (see GUI.md).
The GUI's tabs and values are not here: they live in GUI files the user saves
and loads (`smappy.workspace`), and this file only names the last one.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

_APP = "smappy"
_cache: Dict[str, Any] | None = None


def config_dir() -> Path:
    """The platform's per-user configuration directory for smappy.

    ``SMAPPY_CONFIG_DIR`` overrides it, which is how the tests get a directory
    of their own rather than writing into the developer's real settings.
    """
    override = os.environ.get("SMAPPY_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / _APP
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming"
        return Path(base) / _APP
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / _APP


def config_file() -> Path:
    return config_dir() / "config.yaml"


def load(reload: bool = False) -> Dict[str, Any]:
    """The configuration, cached.  A missing or unreadable file reads as {}."""
    global _cache
    if _cache is not None and not reload:
        return _cache
    path = config_file()
    data: Dict[str, Any] = {}
    if path.exists():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            # a corrupt config must not stop the program from starting; the
            # user sees the defaults and can fix or delete the file
            data = {}
    if not isinstance(data, dict):
        data = {}
    _cache = data
    return data


def save(data: Dict[str, Any]) -> Path:
    global _cache
    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    _cache = data
    return path


def get(key: str, default: Any = None) -> Any:
    return load().get(key, default)


def set(key: str, value: Any) -> None:      # noqa: A001 -- reads well as config.set
    data = dict(load())
    data[key] = value
    save(data)


# ------------------------------------------------------------- plugin roots

def user_plugin_dir() -> Path:
    """The user's own plugin folder, scanned whether or not it is listed.

    One folder a plugin can be put in without first naming it in the
    preferences, so that writing a small analysis -- by hand, or by describing
    it to an AI -- is "save a file here" and nothing else
    (`smappy.user_plugins`).  It was deliberately absent until 2026-10-18, on
    the grounds that what is loaded should be something the user wrote down;
    a folder inside the user's own configuration, empty until they put
    something in it, is that, and it was the step every new plugin tripped on.
    """
    return config_dir() / "plugins"


def plugin_roots() -> List[Path]:
    """Extra folders to scan for plugins, in the order the user listed them.

    The user's own folder (`user_plugin_dir`) is scanned before these and is
    not one of them.
    """
    raw = get("plugin_roots", []) or []
    if isinstance(raw, (str, Path)):
        raw = [raw]
    return [Path(str(p)).expanduser() for p in raw]


def set_plugin_roots(roots) -> None:
    set("plugin_roots", [str(Path(p).expanduser()) for p in roots])


# ---------------------------------------------------------- unsaved changes

def ask_before_quit() -> bool:
    """Whether quitting with a changed table asks to save it first.

    Off by default: most tables here are a file's analysis, written again by
    a chain or a click in seconds, and a question at every quit is one that
    gets answered without being read.
    """
    return bool(get("ask_before_quit_unsaved", False))


def set_ask_before_quit(on: bool) -> None:
    set("ask_before_quit_unsaved", bool(on))


# -------------------------------------------------------------- last folder

def last_folder() -> Path | None:
    """Where the last file was opened or saved, for the next dialog to start.

    A folder that has since gone (an unplugged drive, a renamed run) gives
    its nearest parent that is still there, which is usually still closer to
    the data than the home directory.
    """
    raw = get("last_folder")
    if not raw:
        return None
    folder = Path(str(raw)).expanduser()
    for candidate in (folder, *folder.parents):
        if candidate.is_dir():
            return candidate
    return None


def remember_folder(path, up: int = 0) -> None:
    """Remember the folder of ``path`` (a file, or a folder itself), ``up``
    levels higher.

    ``up=1`` is for a camera stack: acquisition software writes each stack
    into a folder of its own, so the next stack is in a sibling of this
    one's folder, and the folder above is where to start looking.
    """
    if not path:
        return
    folder = Path(str(path)).expanduser()
    if not folder.is_dir():
        folder = folder.parent
    for _ in range(up):
        if folder.parent != folder:
            folder = folder.parent
    folder = folder.resolve()
    if str(folder) != get("last_folder"):
        set("last_folder", str(folder))
