"""Named sets of settings kept as YAML files: presets, and saved list items.

A preset is the settings a plugin had, written under a name, to be chosen
again from a list (*Segment regions*' ring template at 55 nm; a simulated
scene).  A saved item is one entry of a list of parts on its own (a scene's
component: "NPC", "beads"), to be added to another list.  Both are the same
thing on disk -- a flat map of dotted names, as `plugins.settings_values`
writes it -- so one module keeps both.

Where they live is a list of folders, the user's first: the shipped ones
come after it and a user's file of the same name wins, so a lab can adjust
a shipped preset without touching the package.  Only the user's folder is
ever written to or deleted from.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

SUFFIX = ".yaml"


def flatten(values: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """A nested map as dotted names; a flat one comes back as it was."""
    out: Dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        else:
            out[name] = value
    return out


def names(folders: Sequence[Path]) -> List[str]:
    """Every name in the folders, once, sorted."""
    found = set()
    for folder in folders:
        try:
            found.update(p.name[:-len(SUFFIX)] for p in Path(folder).glob(f"*{SUFFIX}"))
        except OSError:
            continue
    return sorted(found, key=str.lower)


def path_of(folders: Sequence[Path], name: str):
    """The file ``name`` is read from: the first folder that has it, or None."""
    for folder in folders:
        path = Path(folder) / f"{name}{SUFFIX}"
        if path.is_file():
            return path
    return None


def read(folders: Sequence[Path], name: str) -> Dict[str, Any]:
    """``name``'s values as dotted names.  A file that does not read is an
    error that names it: a preset chosen from the list has to do something."""
    import yaml
    path = path_of(folders, name)
    if path is None:
        raise ValueError(f"no preset {name!r}: there are "
                         + (", ".join(names(folders)) or "none"))
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a preset is a mapping of settings")
    return flatten(data)


def is_users(folders: Sequence[Path], name: str) -> bool:
    """Whether ``name`` is read from the user's folder (the first): only
    those can be deleted."""
    path = path_of(folders, name)
    return path is not None and path.parent == Path(folders[0])


def write(folders: Sequence[Path], name: str, values: Dict[str, Any]) -> Path:
    """Write ``values`` under ``name`` in the user's folder (the first)."""
    import yaml
    name = name.strip()
    if not name or any(c in name for c in '/\\:*?"<>|'):
        raise ValueError(f"{name!r} cannot be a file name")
    folder = Path(folders[0])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}{SUFFIX}"
    path.write_text(yaml.safe_dump(dict(values), sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path


def delete(folders: Sequence[Path], name: str) -> None:
    """Remove the user's ``name``; a shipped one is refused, not hidden."""
    path = Path(folders[0]) / f"{name}{SUFFIX}"
    if not path.is_file():
        raise ValueError(f"{name!r} is not one of yours: a shipped preset is not deleted")
    path.unlink()
