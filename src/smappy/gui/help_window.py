"""The plugins' pages, in one window: a tree of them beside the page.

One window rather than one per plugin, because the pages link to each other
("see RCC") and reading one usually means reading the one it points to; it is
kept once made, so it reopens where it was.  The page itself is built by
`smappy.docs.render`, which knows nothing about Qt -- here its images are handed
to the document as resources rather than written anywhere.

Figures are drawn when a page is opened, by the plugin's own code; the first
opening of a page with figures costs a second or so, and the result is cached
(in memory, and under the config directory) until that code changes.
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import unquote

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication, QImage, QKeySequence, \
    QPalette, QShortcut, QTextDocument
from PySide6.QtWidgets import (QMainWindow, QSplitter, QTextBrowser,
                               QTreeWidget, QTreeWidgetItem)

from .. import docs, plugins

_WINDOW: Optional["HelpWindow"] = None


def show_help(path: Optional[str] = None, parent=None) -> "HelpWindow":
    """Open the Help window, on ``path``'s page if given."""
    global _WINDOW
    from shiboken6 import isValid
    # kept by its parent, so it goes when that window does: make another
    if _WINDOW is None or not isValid(_WINDOW):
        _WINDOW = HelpWindow(parent)
    if path:
        _WINDOW.show_page(path)
    elif _WINDOW.current is None:
        _WINDOW.show_index()
    _WINDOW.show()
    _WINDOW.raise_()
    _WINDOW.activateWindow()
    return _WINDOW


class HelpWindow(QMainWindow):
    """Every plugin's page, browsable."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("Plugin documentation")
        self.current: Optional[str] = None
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self._on_item)
        self.browser = QTextBrowser()
        self.browser.setOpenLinks(False)
        self.browser.anchorClicked.connect(self._on_link)
        self.browser.document().setDefaultStyleSheet(docs.STYLE)
        splitter = QSplitter()
        splitter.addWidget(self.tree)
        splitter.addWidget(self.browser)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 680])
        self.setCentralWidget(splitter)
        self.resize(940, 760)
        QShortcut(QKeySequence.Close, self, self.close)
        self._fill_tree()

    def _fill_tree(self) -> None:
        """From the scan, as the Plugins menu is: listing imports nothing."""
        self.tree.clear()
        self._items = {}
        groups = {}

        def group(path: str):
            if not path:
                return self.tree.invisibleRootItem()
            if path not in groups:
                head, _, leaf = path.rpartition("/")
                item = QTreeWidgetItem(group(head), [leaf])
                item.setFlags(Qt.ItemIsEnabled)
                item.setExpanded(True)
                groups[path] = item
            return groups[path]

        for path, ref in sorted(plugins.refs().items()):
            item = QTreeWidgetItem(group(ref.group), [ref.name])
            item.setData(0, Qt.UserRole, path)
            item.setToolTip(0, ref.description or path)
            if docs.page_file(path, ref.origin) is None:
                item.setForeground(0, self.palette().color(QPalette.PlaceholderText))
                item.setToolTip(0, (ref.description or path) +
                                "\n(no written page yet: settings only)")
            self._items[path] = item
        self.tree.expandAll()

    def _on_item(self, item: QTreeWidgetItem) -> None:
        path = item.data(0, Qt.UserRole)
        if path:
            self.show_page(path)

    def _on_link(self, url: QUrl) -> None:
        text = url.toString()
        if text.startswith("plugin:"):
            self.show_page(unquote(text[len("plugin:"):]))
        elif url.scheme() in ("http", "https", "mailto"):
            QDesktopServices.openUrl(url)

    def show_index(self) -> None:
        self.current = None
        self.browser.setHtml(
            "<h1>Plugin documentation</h1><p>Pick a plugin on the left.  Each "
            "page says what the plugin does, how it works and what every "
            "setting means; the settings table is read off the plugin itself, "
            "so it is always the one you are looking at.  A grey name has no "
            "written page yet, only that table.</p><p>The <b>?</b> in a "
            "plugin's title bar, or F1 in its panel, opens its page here.</p>")

    def show_page(self, path: str) -> None:
        """Render ``path``'s page.  Importing the plugin is the price of it."""
        document = self.browser.document()
        try:
            plugin_cls = plugins.get(path)
        except Exception as error:
            self.current = path
            self.browser.setHtml(f"<h1>{path}</h1><p>This plugin could not be "
                                 f"loaded: {error}</p>")
            return
        ref = plugins.refs().get(path)
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            color = self.palette().color(QPalette.Text).name()
            size = self.browser.font().pointSizeF()
            rendered = docs.render(plugin_cls, color=color,
                                   size_pt=size if size > 0 else 10.0,
                                   origin=ref.origin if ref else None)
        finally:
            QGuiApplication.restoreOverrideCursor()
        self.current = path
        document.clear()
        # the names repeat from page to page (math1.png, ...), so the resources
        # are replaced on every page rather than accumulated
        for name, png in rendered.images.items():
            image = QImage.fromData(png, "PNG")
            image.setDevicePixelRatio(docs.SCALE)
            document.addResource(QTextDocument.ImageResource, QUrl(name), image)
        self.browser.setHtml(rendered.html)
        self.setWindowTitle(f"{plugin_cls.name} - plugin documentation")
        item = self._items.get(path)
        if item is not None:
            self.tree.setCurrentItem(item)
