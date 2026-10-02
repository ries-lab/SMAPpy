"""Where file dialogs start: the folder the user last opened or saved in.

The folder itself is kept in the configuration (`config.last_folder`), so it
survives a restart; this is the dialog end of it.  A dialog that has a better
idea -- the folder of the file a field already names, the table's own name
for an export -- keeps it, and one with nothing to go on starts here rather
than in whatever directory the program was launched from.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

from .. import config

STACK_SUFFIXES = (".tif", ".tiff")


def start(name: str = "") -> str:
    """The last folder, with ``name`` in it when a dialog suggests a file name."""
    folder = config.last_folder()
    if folder is None:
        return name
    return str(folder / name) if name else str(folder)


def remember(paths: Union[str, Path, Iterable], stack: bool = False) -> None:
    """Remember where ``paths`` (one, or the first of several) are.

    ``stack`` says a camera stack was picked: the folder above its own is
    remembered (see `config.remember_folder`).  Only a TIFF counts, so the
    same field handed a simulation recipe remembers that file's own folder.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    first = next(iter(paths), None)
    if not first:
        return
    up = 1 if stack and Path(str(first)).suffix.lower() in STACK_SUFFIXES else 0
    try:
        config.remember_folder(first, up=up)
    except OSError:
        pass        # a read-only configuration must not lose the user's file
