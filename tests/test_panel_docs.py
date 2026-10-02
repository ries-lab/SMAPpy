"""The pages of the tabs and windows that are not plugins.

A plugin's settings table is generated from its `param` specs; these panels
build their widgets by hand, so a page names its controls itself, one
``### label`` each under *Controls*.  What keeps that honest is here: every
label on the page must be a label the panel really shows, so a control that is
renamed or removed fails this test rather than leaving the page describing a
button nobody can find.
"""
import inspect
import re

import pytest

from smappy import docs

PANELS = {"Panels/Render tab", "Panels/ROI tab", "Panels/ROI manager",
          "Panels/3D view", "Panels/Bead calibration", "Panels/Batch window"}


def controls(page_text: str):
    """The ``### label`` headings of a page's *Controls* section."""
    match = re.search(r"^##\s+Controls\s*$", page_text, flags=re.M)
    if not match:
        return []
    rest = page_text[match.end():]
    after = re.search(r"^##\s+\S", rest, flags=re.M)
    section = rest[:after.start()] if after else rest
    return [m.strip().strip("*`") for m in re.findall(r"^###\s+(.+?)\s*$", section, flags=re.M)]


def normal(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("&", "")).strip().rstrip(":").strip().lower()


def shown_labels(widget):
    """Every text a user can read on the panel: labels, buttons, group and
    tab titles, menu entries."""
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QAbstractButton, QGroupBox, QLabel, QTabWidget
    texts = set()
    for child in widget.findChildren(QLabel):
        texts.add(child.text())
    for child in widget.findChildren(QAbstractButton):
        texts.add(child.text())
    for child in widget.findChildren(QGroupBox):
        texts.add(child.title())
    for child in widget.findChildren(QTabWidget):
        texts.update(child.tabText(i) for i in range(child.count()))
    for child in widget.findChildren(QAction):
        texts.add(child.text())
    return {normal(t) for t in texts if t}


def build(dotted: str):
    from smappy.session import Session
    cls = docs.resolve(dotted)
    wants = [p for p in inspect.signature(cls).parameters if p != "parent"]
    return cls(Session()) if wants and wants[0] == "session" else cls()


def test_every_panel_has_a_page():
    assert set(docs.panels()) == PANELS


@pytest.mark.parametrize("path", sorted(PANELS))
def test_a_panel_page_renders_with_its_figures(path):
    panel = docs.panels()[path]
    rendered = docs.render(panel)
    assert rendered.errors == []
    assert f"<h1>{panel.name}</h1>" in rendered.html
    assert "## Parameters" not in rendered.html and "setting</th>" not in rendered.html


@pytest.mark.parametrize("path", sorted(PANELS))
def test_every_control_a_page_names_is_on_the_panel(app, path):
    panel = docs.panels()[path]
    assert panel.widget, f"{path}: the front matter names no widget"
    page = docs.read_page(panel.file)
    named = controls(page.body)
    assert named, f"{path}: no ### controls under ## Controls"
    shown = shown_labels(build(panel.widget))
    missing = [label for label in named if normal(label) not in shown]
    assert not missing, f"{path}: not on the panel: {missing}"


def test_the_help_window_lists_the_panels_and_opens_one(app):
    from smappy.gui import help_window
    window = help_window.show_help("Panels/Render tab")
    assert window.current == "Panels/Render tab"
    assert "Controls" in window.browser.toPlainText()
    assert all(path in window._items for path in PANELS)


def test_the_render_tabs_sections_open_its_page(app):
    from smappy.gui import help_window
    from smappy.gui.render_tab import RenderTab
    from smappy.gui.widgets import CollapsibleSection
    from smappy.session import Session
    tab = RenderTab(Session())
    filter_section = next(s for s in tab.findChildren(CollapsibleSection)
                          if s.button.text() == "filter")
    filter_section.help_button.click()
    assert help_window._WINDOW.current == "Panels/Render tab"


@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_static_site_files_the_windows_first_and_links_home(tmp_path):
    """``python -m smappy.docs`` -- what is published beside the tutorials."""
    from smappy.docs.__main__ import export
    errors = export(tmp_path, ["Panels/Batch window", "Analysis/Process/History"],
                    report=lambda *_: None, home="../")
    assert errors == {}
    index = (tmp_path / "index.html").read_text()
    assert index.index("windows and tabs") < index.index("Analysis")
    assert 'href="../"' in index and "SMAPpy help pages" in index
    assert (tmp_path / "panels-batch-window.html").exists()
