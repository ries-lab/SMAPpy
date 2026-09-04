"""Small widgets the panels are built from."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QToolButton, QVBoxLayout,
                               QWidget)


class CollapsibleSection(QWidget):
    """A title bar with an arrow; click it to show or hide the content.

    With ``detachable`` a small button on the right asks for the content to
    be moved to its own window (`detach_requested`); `detach` / `reattach`
    do the moving, so several sections can be open at once.
    """

    toggled = Signal(bool)
    detach_requested = Signal()

    def __init__(self, title: str, content: QWidget, expanded: bool = False,
                 detachable: bool = False, parent=None):
        super().__init__(parent)
        self.content = content
        self.button = QToolButton(text=title, checkable=True, checked=expanded)
        self.button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.button.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.button)
        header.addStretch(1)
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

    def closeEvent(self, event) -> None:
        self.layout().removeWidget(self.content)
        self.content.setParent(None)
        self.closed.emit()
        super().closeEvent(event)
