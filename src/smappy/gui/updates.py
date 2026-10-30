"""*Help > Check for updates...*, and the daily check that runs on its own.

The request is QtNetwork's, on the GUI thread: asynchronous without a thread
of our own (`collector`'s reasons to keep Python threads away from Qt
objects), and with the system's certificates and proxy, which a python.org
Python on a Mac does not have for ``urllib``.  A check that fails says so
when asked for and says nothing when it ran on its own -- offline is not an
error worth a dialog.

What it finds is offered, never installed unasked: the dialog gives the
command for this installation (`smappy.updates.update_command`) and copies
it, and where the install is pip or uv, *Update and restart* hands it to
`smappy.updater`, which installs once SMAPpy has closed and starts it again;
the next start says how that went (`Updates.report_last_update`).
"""
from __future__ import annotations

import sys
from typing import List, Optional

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton, QWidget

from .. import updater, updates

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


def install_commands() -> Optional[List[List[str]]]:
    """What *Update and restart* would run here, or None where it is not offered."""
    return updater.commands(updates.how_installed(), sys.executable)


def message(newest: Optional[str], error: str = "", installed: Optional[str] = None,
            restart: bool = False) -> str:
    """What the dialog says, as rich text."""
    installed = installed or updates.installed()
    if newest is None:
        return (f"Could not ask PyPI for the newest version: {error or 'no answer'}.<br>"
                f"You have SMAPpy {installed}; the newest is listed at "
                f'<a href="{updates.PYPI_PAGE}">{updates.PYPI_PAGE}</a>.')
    if not updates.is_newer(newest, installed):
        return f"SMAPpy {installed} is the newest version."
    how = ("<i>Update and restart</i> closes SMAPpy (close any other SMAPpy windows "
           "first), installs it and starts it again. Or close SMAPpy and run this in a "
           "terminal" if restart else "To update, close SMAPpy and run this in a terminal")
    return (f"<b>SMAPpy {newest} is available</b>; you have {installed}.<br><br>"
            f"{how} (Copy puts it on the clipboard):"
            f"<pre>{_escape(updates.update_command())}</pre>"
            f'<a href="{updates.changes_url(installed, newest)}">What changed since {installed}</a>')


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def show_result(parent: QWidget, newest: Optional[str], error: str = "",
                on_update=None) -> None:
    """The answer; with a newer release, *Copy command*, and *Update and
    restart* when there is an `on_update` and the install can be run here."""
    newer = newest is not None and updates.is_newer(newest, updates.installed())
    restart = newer and on_update is not None and install_commands() is not None
    box = QMessageBox(parent)
    box.setWindowTitle("Check for updates")
    box.setTextFormat(Qt.RichText)
    box.setTextInteractionFlags(Qt.TextBrowserInteraction)   # links open; the command selects
    box.setText(message(newest, error, restart=restart))
    update = box.addButton("Update and restart", QMessageBox.AcceptRole) if restart else None
    copy = box.addButton("Copy command", QMessageBox.ActionRole) if newer else None
    box.addButton(QMessageBox.Close)
    if update is not None:
        box.setDefaultButton(update)
    box.exec()
    if copy is not None and box.clickedButton() is copy:
        QApplication.clipboard().setText(updates.update_command())
    elif update is not None and box.clickedButton() is update:
        on_update()


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
        """At startup: how an update before it went, what an earlier check
        found, and a check again if one is due."""
        self._after(0, self.report_last_update)
        newer = updates.known_newer()
        if newer:
            self._offer(newer)
        if updates.due():
            self._after(START_DELAY_MS, self.checker.check)

    def _after(self, ms: int, slot) -> None:
        """A timer's signal rather than `QTimer.singleShot` with a callable,
        which crashes PySide if the callable raises (CLAUDE.md)."""
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(ms)
        timer.timeout.connect(slot)
        timer.start()

    def update_and_restart(self) -> None:
        """Hand the install to `smappy.updater` and quit; it starts SMAPpy again.

        The window is asked first, as quitting asks (a changed table), and
        once it has agreed it is not asked again by the quit itself
        (`leaving`).  If the helper cannot be started, nothing has closed.
        """
        install = install_commands()
        if install is None or not self.window.may_close():
            return
        try:
            plan = updater.write_plan(updater.new_plan_folder(), install,
                                      updater.relaunch_command(sys.executable),
                                      updates.result_file(), updates.log_file(),
                                      updates.installed())
            updater.start_helper(plan)
        except OSError as error:
            QMessageBox.warning(self.window, "Update and restart",
                                f"Could not start the update: {error}\n\n"
                                f"Close SMAPpy and run:\n{updates.update_command()}")
            return
        self.window.leaving = True
        QApplication.quit()

    def report_last_update(self) -> None:
        """What the last *Update and restart* did: a line in the status bar
        when it worked, a window with the end of pip's output when not."""
        try:
            done = updates.last_update()
            if done is None:
                return
            before, now = done.get("from_version") or "?", updates.installed()
            if done.get("ok") and updates.is_newer(now, before):
                self.window.statusBar().showMessage(f"SMAPpy updated from {before} to {now}", 20000)
                if self.button is not None and not updates.known_newer():
                    self.button.hide()
                return
            what = ("The update ran, but this is still SMAPpy " + now
                    if done.get("ok") else "The update failed; this is still SMAPpy " + now)
            box = QMessageBox(QMessageBox.Warning, "Update and restart",
                              f"{what}.", QMessageBox.Close, self.window)
            # what to do first, then what the installer said, then the log
            ok = bool(done.get("ok"))
            parts = [updates.failure_advice(ok), "" if ok else updates.log_error(),
                     f"The whole output is in {done.get('log')}. To try by hand, close "
                     f"SMAPpy and run:\n{updates.update_command()}"]
            box.setInformativeText("\n\n".join(part for part in parts if part))
            box.setDetailedText(updates.log_tail())
            box.exec()
        except Exception as error:      # a timer's slot must not raise
            print(f"update report: {error}")

    def _answered(self, newest, error: str) -> None:
        asked, self.asked = self.asked, False
        if newest is not None and updates.is_newer(newest, updates.installed()):
            self._offer(newest)
        if asked:
            show_result(self.window, newest, error, self.update_and_restart)

    def _offer(self, newest: str) -> None:
        self.newest = newest
        if self.button is None:
            self.button = QPushButton()
            self.button.setFlat(True)
            self.button.setStyleSheet("color: palette(link); padding: 0 6px")
            self.button.clicked.connect(
                lambda: show_result(self.window, self.newest, on_update=self.update_and_restart))
            self.window.statusBar().addPermanentWidget(self.button)
        self.button.setText(f"SMAPpy {newest} is available")
        self.button.setToolTip("a newer version is on PyPI: click for how to update")
