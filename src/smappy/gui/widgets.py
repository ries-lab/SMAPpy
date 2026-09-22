"""Small widgets the panels are built from."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QToolButton, QVBoxLayout,
                               QWidget)

# The control window's width.  A tab whose content would ask for more (a long
# file name in a wrapping label) caps its own hint at this, so one tab never
# widens the window for the other three.
CONTROL_WIDTH = 380


class CollapsibleSection(QWidget):
    """A title bar with an arrow; click it to show or hide the content.

    With ``detachable`` a small button on the right asks for the content to
    be moved to its own window (`detach_requested`); `detach` / `reattach`
    do the moving, so several sections can be open at once.
    """

    toggled = Signal(bool)
    detach_requested = Signal()
    starred = Signal(bool)

    def __init__(self, title: str, content: QWidget, expanded: bool = False,
                 detachable: bool = False, star: Optional[bool] = None, parent=None):
        super().__init__(parent)
        self.content = content
        self.button = QToolButton(text=title, checkable=True, checked=expanded)
        self.button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.button.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.button)
        header.addStretch(1)
        self.star_button = None
        if star is not None:
            self.star_button = QToolButton(checkable=True, checked=star, autoRaise=True)
            self.star_button.setToolTip("favourite: shown without 'all'")
            self.star_button.toggled.connect(self._on_star)
            self._on_star(star)
            header.addWidget(self.star_button)
        self.detach_button = None
        if detachable:
            self.detach_button = QToolButton(text="↗", autoRaise=True)
            self.detach_button.setToolTip("open in its own window")
            self.detach_button.clicked.connect(self.detach_requested)
            header.addWidget(self.detach_button)
        self.frame = QFrame()
        self.frame.setFrameShape(QFrame.StyledPanel)
        self._inner = QVBoxLayout(self.frame)
        self._inner.setContentsMargins(6, 4, 6, 4)
        self._inner.addWidget(content)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(header)
        layout.addWidget(self.frame)
        self.button.toggled.connect(self.set_expanded)
        self.set_expanded(expanded)

    def _on_star(self, on: bool) -> None:
        self.star_button.setText("★" if on else "☆")
        self.starred.emit(on)

    @property
    def title(self) -> str:
        return self.button.text().replace(" (window)", "")

    def set_expanded(self, on: bool) -> None:
        self.button.setChecked(on)
        self.button.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.frame.setVisible(on and self.content.parent() is self.frame)
        self.toggled.emit(on)

    def detach(self) -> QWidget:
        """Hand the content out; the section shows only its title meanwhile."""
        self._inner.removeWidget(self.content)
        self.content.setParent(None)
        self.frame.hide()
        self.button.setText(self.title + " (window)")
        self.button.setEnabled(False)
        if self.detach_button:
            self.detach_button.hide()
        return self.content

    def reattach(self) -> None:
        self._inner.addWidget(self.content)
        self.content.show()
        self.button.setText(self.title)
        self.button.setEnabled(True)
        if self.detach_button:
            self.detach_button.show()
        self.set_expanded(self.button.isChecked())


def detach_to_window(section: CollapsibleSection, parent=None) -> "FloatingWindow":
    """Move a section's content to its own window; closing it puts it back."""
    window = FloatingWindow(section.title, section.detach(), parent=parent)
    window.closed.connect(section.reattach)
    window.resize(max(section.content.sizeHint().width() + 20, 320), 400)
    window.show()
    section._window = window            # keep it alive
    return window


class FloatingWindow(QWidget):
    """A section's content in its own window; closing puts it back."""

    closed = Signal()

    def __init__(self, title: str, content: QWidget, parent=None):
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(content)
        self.content = content
        from .app import window_shortcuts
        window_shortcuts(self)

    def closeEvent(self, event) -> None:
        self.layout().removeWidget(self.content)
        self.content.setParent(None)
        self.closed.emit()
        super().closeEvent(event)


def place_beside(window: QWidget, other: QWidget, gap: int = 0) -> None:
    """Put ``window`` against ``other``'s right edge, on the same screen.

    Touching rather than overlapping: a second window dropped on top of the
    one it belongs to hides what it was opened to compare against.  Right
    first, then left, and if neither side has room the window is pushed back
    onto the screen rather than off it -- a window with its title bar past
    the edge cannot be moved back by hand on some desktops.
    """
    from PySide6.QtWidgets import QApplication

    anchor = other.frameGeometry()
    width = window.frameGeometry().width() or window.width()
    x, y = anchor.right() + 1 + gap, anchor.top()
    screen = other.screen() or window.screen() or QApplication.primaryScreen()
    if screen is not None:
        room = screen.availableGeometry()
        if x + width > room.right():
            left = anchor.left() - gap - width          # the other side
            x = left if left >= room.left() else max(room.left(), room.right() - width)
        y = min(max(y, room.top()), max(room.top(), room.bottom() - window.height()))
    window.move(x, y)


# What a field that can be typed into looks like.  Qt's native styles draw a
# line edit almost flush with the window on several platforms -- on macOS in
# particular a spin box and a label are hard to tell apart at a glance -- and
# the panels here are dense enough that "which of these can I change?" was a
# real question.  So every editable widget gets the base colour and a border,
# and a read-only one deliberately does not: the difference is the point.
#
# The colours are palette roles rather than literals, so this is a white field
# on a light desktop and a dark one on a dark desktop, and it follows a theme
# change without being told.
EDITABLE_STYLE = """
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox,
QComboBox:editable, QAbstractSpinBox {
    background-color: palette(base);
    border: 1px solid palette(mid);
    border-radius: 3px;
    selection-background-color: palette(highlight);
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QAbstractSpinBox:focus {
    border: 1px solid palette(highlight);
}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled,
QAbstractSpinBox:disabled {
    background-color: palette(window);
    color: palette(mid);
    border: 1px solid palette(window);
}
QLineEdit[readOnly="true"], QPlainTextEdit[readOnly="true"],
QTextEdit[readOnly="true"] {
    background-color: palette(window);
    border: 1px solid palette(window);
}
"""


def apply_style(app) -> None:
    """Mark the editable fields, keeping whatever style is already set."""
    existing = app.styleSheet() or ""
    if EDITABLE_STYLE not in existing:
        app.setStyleSheet(existing + EDITABLE_STYLE)
