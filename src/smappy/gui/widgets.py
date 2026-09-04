"""Small widgets the panels are built from."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """A title bar with an arrow; click it to show or hide the content."""

    toggled = Signal(bool)

    def __init__(self, title: str, content: QWidget, expanded: bool = False,
                 parent=None):
        super().__init__(parent)
        self.button = QToolButton(text=title, checkable=True, checked=expanded)
        self.button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.button.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.button.setSizePolicy(self.button.sizePolicy().horizontalPolicy(),
                                  self.button.sizePolicy().verticalPolicy())
        self.frame = QFrame()
        self.frame.setFrameShape(QFrame.StyledPanel)
        inner = QVBoxLayout(self.frame)
        inner.setContentsMargins(6, 4, 6, 4)
        inner.addWidget(content)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.button)
        layout.addWidget(self.frame)
        self.button.toggled.connect(self.set_expanded)
        self.set_expanded(expanded)

    @property
    def title(self) -> str:
        return self.button.text()

    def set_expanded(self, on: bool) -> None:
        self.button.setChecked(on)
        self.button.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.frame.setVisible(on)
        self.toggled.emit(on)
