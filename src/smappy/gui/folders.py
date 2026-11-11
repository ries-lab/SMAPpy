"""Where file dialogs start: the folder the user last opened or saved in.

The folder itself is kept in the configuration (`config.last_folder`), so it
survives a restart; this is the dialog end of it.  A dialog that has a better
idea -- the folder of the file a field already names, the table's own name
for an export -- keeps it, and one with nothing to go on starts here rather
than in whatever directory the program was launched from.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

from .. import config

STACK_SUFFIXES = (".tif", ".tiff")


# The folder of the data being fitted, set when a fitter's movie is chosen (`use_data`): where a
# plugin's empty path field opens its dialog (`gui.params`), since what goes with a movie -- its
# bead movies, a transformation, a calibration measured that day -- usually lies beside it.  Kept
# apart from the last folder, which every dialog moves: picking a calibration from a library, or
# saving elsewhere, would otherwise lose it.  For this run of the program only.
_data: Optional[Path] = None


def data_folder_of(path) -> Optional[Path]:
    """The folder a movie's neighbours are in: its own, or one up for a movie in a folder of its
    own -- Micro-Manager writes ``Pos1_x/Pos1_x_MMStack_Pos0.ome.tif``, the file named after its
    folder, beside the other acquisitions of the day."""
    if not path:
        return None
    path = Path(str(path)).expanduser()
    folder = path if path.is_dir() else path.parent
    if not path.is_dir() and folder.name and path.name.startswith(folder.name):
        folder = folder.parent
    return folder if folder.is_dir() else None


def use_data(path) -> None:
    """``path`` is the data now being fitted: its folder is the data folder."""
    global _data
    folder = data_folder_of(path)
    if folder is not None:
        _data = folder


def data_start() -> str:
    """Where a plugin's empty path field opens: the data folder, else the last folder."""
    if _data is not None and _data.is_dir():
        return str(_data)
    return start()


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


def choose_folders(parent, title: str) -> list:
    """Several folders at once, starting in the last one used; [] when cancelled.

    `QFileDialog.getExistingDirectory` takes one, and the native dialog on
    macOS cannot be talked into more, so this is Qt's own dialog with its
    list and tree switched to extended selection (cmd- and shift-click).
    """
    from PySide6.QtWidgets import QAbstractItemView, QFileDialog, QListView, QTreeView
    dialog = QFileDialog(parent, title, start())
    dialog.setFileMode(QFileDialog.FileMode.Directory)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
    for view in dialog.findChildren(QListView) + dialog.findChildren(QTreeView):
        if view.objectName() in ("listView", "treeView"):
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    if not dialog.exec():
        return []
    paths = [p for p in dialog.selectedFiles() if p]
    remember(paths)
    return paths
