"""User configuration: a small YAML file, and where it lives.

Kept out of Qt on purpose -- the plugin scanner reads the extra plugin roots
from here and `smappy.plugins` must stay importable in a script (see GUI.md).
Qt's own `QSettings` still holds transient window state; anything a user would
want to look at or edit by hand belongs here instead.
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
            data = yaml.safe_load(path.read_text()) or {}
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
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    _cache = data
    return path


def get(key: str, default: Any = None) -> Any:
    return load().get(key, default)


def set(key: str, value: Any) -> None:      # noqa: A001 -- reads well as config.set
    data = dict(load())
    data[key] = value
    save(data)


# ------------------------------------------------------------- plugin roots

def plugin_roots() -> List[Path]:
    """Extra folders to scan for plugins, in the order the user listed them.

    There is deliberately no implicit ``~/.smappy/plugins``: a plugin folder is
    something the user names, so that what gets loaded is always something they
    wrote down.
    """
    raw = get("plugin_roots", []) or []
    if isinstance(raw, (str, Path)):
        raw = [raw]
    return [Path(str(p)).expanduser() for p in raw]


def set_plugin_roots(roots) -> None:
    set("plugin_roots", [str(Path(p).expanduser()) for p in roots])
