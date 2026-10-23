"""*Help > Check for updates...*, and the daily check that runs on its own.

The request is QtNetwork's, on the GUI thread: asynchronous without a thread
of our own (`collector`'s reasons to keep Python threads away from Qt
objects), and with the system's certificates and proxy, which a python.org
Python on a Mac does not have for ``urllib``.  A check that fails says so
when asked for and says nothing when it ran on its own -- offline is not an
error worth a dialog.

What it finds is offered, never installed: the dialog gives the command for
this installation (`smappy.updates.update_command`) and copies it.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton, QWidget

from .. import updates

START_DELAY_MS = 5000           # after the windows are up and a file has started loading
TIMEOUT_MS = 10000


class UpdateChecker(QObject):
    """One request to PyPI at a time; `answered` gets the newest release, or
    None and what went wrong."""

    answered = Signal(object, str)

    def __init__(self, parent: Optional[QObject] = None, url: str = updates.PYPI_JSON):
        super().__init__(parent)
        self.url = url
        self._manager = None
        self._reply = None

    def check(self) -> None:
        if self._reply is not None:
            return
        from PySide6.QtNetwork import (QNetworkAccessManager, QNetworkProxyFactory,
                                       QNetworkRequest)
        if self._manager is None:
            QNetworkProxyFactory.setUseSystemConfiguration(True)
            self._manager = QNetworkAccessManager(self)
        request = QNetworkRequest(QUrl(self.url))
        request.setTransferTimeout(TIMEOUT_MS)
        request.setRawHeader(b"User-Agent", f"SMAPpy/{updates.installed()}".encode())
        request.setRawHeader(b"Accept", b"application/json")
        self._reply = self._manager.get(request)
        self._reply.finished.connect(self._finished)

    def _finished(self) -> None:
        from PySide6.QtNetwork import QNetworkReply
        reply, self._reply = self._reply, None
        try:
            if reply.error() != QNetworkReply.NetworkError.NoError:
                self.answered.emit(None, reply.errorString())
                return
            newest = updates.newest_from(bytes(reply.readAll()))
            if newest is None:
                self.answered.emit(None, "PyPI's answer named no version")
                return
            try:
                updates.remember(newest)
            except OSError:
                pass                    # a read-only config costs the memory, not the answer
            self.answered.emit(newest, "")
        except Exception as error:      # a slot of a Qt signal must not raise
            print(f"update check: {error}")
        finally:
            reply.deleteLater()


def message(newest: Optional[str], error: str = "", installed: Optional[str] = None) -> str:
    """What the dialog says, as rich text."""
    installed = installed or updates.installed()
    if newest is None:
        return (f"Could not ask PyPI for the newest version: {error or 'no answer'}.<br>"
                f"You have SMAPpy {installed}; the newest is listed at "
                f'<a href="{updates.PYPI_PAGE}">{updates.PYPI_PAGE}</a>.')
    if not updates.is_newer(newest, installed):
        return f"SMAPpy {installed} is the newest version."
    return (f"<b>SMAPpy {newest} is available</b>; you have {installed}.<br><br>"
            "To update, close SMAPpy and run this in a terminal "
            "(Copy puts it on the clipboard):"
            f"<pre>{_escape(updates.update_command())}</pre>"
            f'<a href="{updates.changes_url(installed, newest)}">What changed since {installed}</a>')


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def show_result(parent: QWidget, newest: Optional[str], error: str = "") -> None:
    box = QMessageBox(parent)
    box.setWindowTitle("Check for updates")
    box.setTextFormat(Qt.RichText)
    box.setTextInteractionFlags(Qt.TextBrowserInteraction)   # links open; the command selects
    box.setText(message(newest, error))
    newer = newest is not None and updates.is_newer(newest, updates.installed())
    copy = box.addButton("Copy command", QMessageBox.AcceptRole) if newer else None
    box.addButton(QMessageBox.Close)
    box.exec()
    if copy is not None and box.clickedButton() is copy:
        QApplication.clipboard().setText(updates.update_command())


class Updates(QObject):
    """The menu item's check and the automatic one, for one window.

    A newer release, found now or by an earlier start, is a button at the
    right of the window's status bar -- there until the window closes, where
    a status message would be gone at the next one -- which opens the dialog.
    """

    def __init__(self, window, url: str = updates.PYPI_JSON):
        super().__init__(window)
        self.window = window
        self.checker = UpdateChecker(self, url)
        self.checker.answered.connect(self._answered)
        self.asked = False              # the menu's check: answer whatever it is
        self.button: Optional[QPushButton] = None
        self.newest: Optional[str] = None

    def check_now(self) -> None:
        self.asked = True
        self.window.statusBar().showMessage("asking PyPI for the newest version...", 5000)
        self.checker.check()

    def start(self) -> None:
        """At startup: say what an earlier check found, and check again if due."""
        newer = updates.known_newer()
        if newer:
            self._offer(newer)
        if updates.due():
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(START_DELAY_MS)
            timer.timeout.connect(self.checker.check)
            timer.start()

    def _answered(self, newest, error: str) -> None:
        asked, self.asked = self.asked, False
        if newest is not None and updates.is_newer(newest, updates.installed()):
            self._offer(newest)
        if asked:
            show_result(self.window, newest, error)

    def _offer(self, newest: str) -> None:
        self.newest = newest
        if self.button is None:
            self.button = QPushButton()
            self.button.setFlat(True)
            self.button.setStyleSheet("color: palette(link); padding: 0 6px")
            self.button.clicked.connect(lambda: show_result(self.window, self.newest))
            self.window.statusBar().addPermanentWidget(self.button)
        self.button.setText(f"SMAPpy {newest} is available")
        self.button.setToolTip("a newer version is on PyPI: click for how to update")
