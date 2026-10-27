"""A failure, said where one is looking.

A plugin that failed used to print "ValueError: no ROI: draw a line" into its
output box -- below the eye, with the exception's class in front of a message
written for the reader -- and say "failed" beside Run; a file that would not
open put up a modal box that had to be clicked away before anything else.
Now the message alone stands in red where the status is, and the traceback is
one click away (*details…*), with a bug report one more: the traceback is
what a report is asked for, and the reader of the message rarely needs it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

ERROR_STYLE = "color: #e0453a;"


def message_of(error: BaseException) -> str:
    """What the error says, without its class: the text a plugin wrote for
    the reader ("no ROI: draw a line").  An error with nothing to say is
    named by its class instead, which is then all there is."""
    if isinstance(error, KeyError) and len(error.args) == 1 \
            and not str(error.args[0]).count(" "):
        # a bare key is no sentence: "'x_nm'" says less than "missing x_nm"
        return f"missing {error.args[0]}"
    text = str(error).strip()
    return text or type(error).__name__


def save_bug_report(parent, session, details: str = "") -> Optional[Path]:
    """Ask for a note and a place, and write the bug report there
    (`diagnostics.bug_report`); ``details`` -- a traceback -- goes into the
    report beside the note, so it is there even without the log."""
    from datetime import datetime
    from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox
    from .. import diagnostics
    note, ok = QInputDialog.getMultiLineText(
        parent, "Bug report", "What happened, and what did you expect?  (optional)")
    if not ok:
        return None
    default = str(Path.home() / f"smappy-bug-{datetime.now():%Y%m%d-%H%M}.zip")
    path, _ = QFileDialog.getSaveFileName(parent, "Save bug report", default,
                                          "Zip (*.zip)")
    if not path:
        return None
    if details:
        note = f"{note}\n\nthe error:\n{details}".strip()
    written = diagnostics.bug_report(path, session, note)
    QMessageBox.information(
        parent, "Bug report",
        f"Written to\n{written}\n\nIt has the log, the environment and a "
        "description of the open table (its columns, files and history), "
        "not the localizations.  File names and folders are in it.")
    return written


def details_window(parent, title: str, message: str, details: str, session=None):
    """A window with the whole traceback, and *Save a bug report…*.

    Not modal: the traceback is often read beside the settings that caused
    it.  Returns the window, shown.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFontDatabase
    from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QPlainTextEdit,
                                   QPushButton, QVBoxLayout)
    window = QDialog(parent)
    window.setWindowTitle(title)
    layout = QVBoxLayout(window)
    said = QLabel(message)
    said.setWordWrap(True)
    said.setStyleSheet(ERROR_STYLE)
    said.setTextInteractionFlags(Qt.TextSelectableByMouse)
    layout.addWidget(said)
    text = QPlainTextEdit(readOnly=True)
    text.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
    text.setPlainText(details)
    text.setLineWrapMode(QPlainTextEdit.NoWrap)
    layout.addWidget(text, 1)
    row = QHBoxLayout()
    report = QPushButton("Save a bug report\u2026")
    report.setToolTip("the log, the environment, a description of the table "
                      "and this traceback, zipped for an issue")

    def save():
        try:
            save_bug_report(window, session, details)
        except Exception as e:            # a report that fails must say so,
            from PySide6.QtWidgets import QMessageBox    # not crash the slot
            QMessageBox.warning(window, "Bug report", f"not written: {e}")

    report.clicked.connect(save)
    close = QPushButton("Close")
    close.clicked.connect(window.close)
    row.addWidget(report)
    row.addStretch(1)
    row.addWidget(close)
    layout.addLayout(row)
    window.report_button = report
    window.text = text
    window.resize(640, 360)
    window.show()
    return window
