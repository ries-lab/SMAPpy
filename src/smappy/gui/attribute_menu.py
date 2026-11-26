"""*Rename...* and *Remove* on a right-click of an attribute in the Render tab.

An attribute is renamed or removed where it is looked at -- the filter's
drop-down, the colour-by list -- rather than by finding *Edit attributes*
in the Plugins menu.  The work is that plugin's, run on the session
(`Session.run`), not a second implementation: the undo, the log line, the
recipes rewritten and the layers' bounds and colour following a rename
(`Result.renamed_columns`) are the plugin's, as File > Save is the saver's.
The plugin stays for chains and scripts.

A column the program finds by its name (`table_edit.STRUCTURAL`: the
positions, the frame, the file, the grouping) is offered greyed, its reason
in the tooltip, rather than left out: the menu says why, not nothing.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QInputDialog, QMenu, QMessageBox, QWidget

from ..session import Session

EDIT = "Analysis/Process/Edit attributes"


def add_actions(menu: QMenu, session: Session, column: Optional[str],
                parent: QWidget) -> None:
    """*Rename* and *Remove* ``column`` at the end of ``menu``; nothing for no
    column or one the table does not have (a region table's, an image's)."""
    from ..table_edit import STRUCTURAL
    if not column or session.locs is None or column not in session.locs:
        return
    if not menu.isEmpty():
        menu.addSeparator()
    rename = menu.addAction(f"Rename {column}...")
    remove = menu.addAction(f"Remove {column}")
    menu.setToolTipsVisible(True)
    if column in STRUCTURAL:
        for action in (rename, remove):
            action.setEnabled(False)
            action.setToolTip(f"{column} is {STRUCTURAL[column]}")
        return
    rename.triggered.connect(lambda: rename_attribute(session, column, parent))
    remove.triggered.connect(lambda: remove_attribute(session, column, parent))


def show_menu(session: Session, column: Optional[str], parent: QWidget, at) -> None:
    """The menu of ``column`` alone, at ``at`` in ``parent``'s coordinates."""
    menu = QMenu(parent)
    add_actions(menu, session, column, parent)
    if not menu.isEmpty():
        menu.exec(parent.mapToGlobal(at))


def rename_attribute(session: Session, column: str, parent: QWidget,
                     name: Optional[str] = None) -> bool:
    """Ask for a new name (``name`` given: no dialog) and rename. Whether
    it was renamed."""
    if name is None:
        name, ok = QInputDialog.getText(parent, f"Rename {column}", "new name",
                                        text=column)
        if not ok or not name.strip() or name.strip() == column:
            return False
    return _run(session, parent, operation="rename", column=column, new_name=name)


def remove_attribute(session: Session, column: str, parent: QWidget) -> bool:
    """Remove the column -- undone by Edit > Undo, so no question first."""
    return _run(session, parent, operation="remove", remove=[column])


def _run(session: Session, parent: QWidget, **values) -> bool:
    from dataclasses import replace

    from .. import plugins
    cls = plugins.get(EDIT)
    plugin = cls()
    try:
        result = session.run(plugin, replace(cls.Settings(), **values))
    except Exception as error:          # a name taken or not a word: said, not raised
        QMessageBox.warning(parent, "Edit attributes", str(error))
        return False
    if "; " in (result.text or ""):     # the notes: recipes, bounds that moved or went
        QMessageBox.information(parent, "Edit attributes", result.text.replace("; ", "\n"))
    return True
