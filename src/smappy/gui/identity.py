"""Which session a window belongs to: its name and colour in every title.

Several sessions open at once -- two instances side by side on different
datasets, or a snapshot of a live fit beside the fit -- each put up the same
figure windows, the same "Statistics" and "COMET" titles, and nothing said
which was whose.  So every top-level window's title carries the session's
mark (a coloured circle, `session.COLOURS`) in front and its name behind:
``🔵 Statistics — NPC, drift corrected``.

One session per process is what the program runs -- a second session is a
second instance -- so this is installed once on the application, like the
icon, rather than threaded through every window class.  An event filter sees
each top-level window's title being set and adds the mark and the name; the
windows themselves set their titles as they always did.  The control and
render windows, whose titles are *about* the session already, say so with
`OWN_TITLE` and build theirs with `title`.
"""
from __future__ import annotations

import traceback
from typing import Optional

from ..session import COLOURS, Session

# a window whose title already names the session, and must not get it twice
OWN_TITLE = "smappy_own_title"
_BASE = "smappy_base_title"

_installed = None          # the application's filter, once installed


def mark(session: Session) -> str:
    """The session's coloured circle, or "" when it has no colour."""
    return COLOURS[session.colour][1] if session.colour in COLOURS else ""


def unsaved(session: Session) -> str:
    """"*" while the table has changed since it was saved or opened, the mark
    every editor puts in its title, for the main windows' titles."""
    return "*" if getattr(session, "unsaved", False) else ""


def colour_hex(session: Session) -> str:
    return COLOURS[session.colour][0] if session.colour in COLOURS else ""


def title(session: Session, text: str) -> str:
    """``text`` with the session's mark in front and its name behind."""
    name = session.display_name
    out = f"{mark(session)} {text}".strip()
    if name and name not in text:
        out += f" — {name}"
    return out


def install_identity(app, session: Session) -> None:
    """Tag every top-level window of ``app`` with ``session``.

    Safe to call again: the filter is kept and follows the latest session.
    """
    global _installed
    if _installed is None:
        _installed = _make_titles(app)
        app.installEventFilter(_installed)
    _installed.follow(session)


def uninstall_identity(app) -> None:
    """Stop tagging; the titles already set stay as they are."""
    global _installed
    if _installed is not None:
        app.removeEventFilter(_installed)
        if _installed.session is not None:
            _installed.session.off_change(_installed._on_session)
        _installed = None


def _make_titles(app):
    """The filter, its class made here so that importing this costs no Qt."""
    from PySide6.QtCore import QEvent, QObject
    from PySide6.QtWidgets import QWidget

    class Titles(QObject):
        def __init__(self, parent):
            super().__init__(parent)
            self.session: Optional[Session] = None

        def follow(self, session: Session) -> None:
            if self.session is not None:
                self.session.off_change(self._on_session)
            self.session = session
            session.on_change(self._on_session)
            self.retitle_all()

        def _on_session(self, what: str) -> None:
            # "locs": a file opened or added, which is the default name
            if what in ("name", "locs"):
                self.retitle_all()

        def retitle_all(self) -> None:
            from PySide6.QtWidgets import QApplication
            for window in QApplication.topLevelWidgets():
                base = window.property(_BASE)
                if base is not None:
                    window.setWindowTitle(self._tagged(str(base)))
                else:
                    self.retitle(window)

        def _tagged(self, base: str) -> str:
            return title(self.session, base) if self.session is not None else base

        def retitle(self, window) -> None:
            if window.property(OWN_TITLE) or self.session is None:
                return
            text = window.windowTitle()
            base = window.property(_BASE)
            if base is not None and text == self._tagged(str(base)):
                return                     # the change this made itself
            if not text:
                return
            window.setProperty(_BASE, text)
            window.setWindowTitle(self._tagged(text))

        def eventFilter(self, obj, event) -> bool:
            # Every event of the application passes here, so the type is
            # checked first and nothing else is done for the rest.  Never
            # raises: an exception left pending in a Qt override is what
            # crashes PySide later, somewhere else.
            try:
                if (event.type() == QEvent.Type.WindowTitleChange
                        and isinstance(obj, QWidget) and obj.isWindow()):
                    self.retitle(obj)
            except Exception:
                traceback.print_exc()
            return False

    return Titles(app)


class NameDialog:
    """Ask for the session's name and colour; `ask` returns them or None."""

    @staticmethod
    def ask(parent, session: Session):
        from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                                       QFormLayout, QLineEdit)
        dialog = QDialog(parent)
        dialog.setWindowTitle("Session name")
        form = QFormLayout(dialog)
        name = QLineEdit(session.name)
        name.setPlaceholderText(session.display_name if not session.name
                                else "the file's name")
        name.setToolTip("shown in the title of every window of this session; "
                        "empty: the file's name")
        form.addRow("name", name)
        colour = QComboBox()
        colour.addItem("none", "")
        for key, (_, circle) in COLOURS.items():
            colour.addItem(f"{circle} {key}", key)
        colour.setCurrentIndex(max(colour.findData(session.colour), 0))
        colour.setToolTip("a coloured mark in front of every window title")
        form.addRow("colour", colour)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return name.text(), colour.currentData()
