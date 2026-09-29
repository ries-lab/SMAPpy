"""Every shipped plugin's panel fits the control column.

`ColumnScroll` holds a tab's content to the column's width, so a control
that is too wide no longer pushes the title bars' ? and detach arrow out of
view -- but it is clipped instead.  This keeps the clipping from happening:
a combo box with a long choice, a label, a row of buttons that outgrows the
column fails here, with the plugin named.
"""
import pytest

pytest.importorskip("PySide6")

from smappy import plugins                                  # noqa: E402
from smappy.session import Session                          # noqa: E402

# the tab's margins, the section's frame and a vertical scroll bar
SLACK = 40


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _panel(cls):
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.gui.widgets import CollapsibleSection
    # every part opened: a folded one hides what it would ask for
    panel = PluginPanel(cls, Session())
    for section in panel.findChildren(CollapsibleSection):
        section.set_expanded(True)
    return panel


def test_every_plugin_panel_fits_the_control_column(app):
    from smappy.gui.widgets import CONTROL_WIDTH
    too_wide = {}
    for path in sorted(plugins.refs()):
        cls = plugins.get(path)
        width = _panel(cls).minimumSizeHint().width()
        if width > CONTROL_WIDTH - SLACK:
            too_wide[path] = width
    assert not too_wide, (f"wider than the {CONTROL_WIDTH - SLACK} px the "
                          f"control column leaves a panel: {too_wide}")


def test_a_long_choice_does_not_widen_its_row(app):
    from dataclasses import dataclass
    from smappy.gui.params import SettingsForm
    from smappy.plugins import param, param_specs

    @dataclass
    class Settings:
        shape: str = param("short", choices=["short", "a choice " * 12])

    form = SettingsForm(Settings, param_specs(Settings))
    box = form.fields["shape"].widget
    assert form.minimumSizeHint().width() < 200
    # ...and when there is room it is as wide as its longest choice
    assert box.maximumWidth() > 400 and box.view().minimumWidth() > 400


def test_a_too_wide_control_leaves_the_title_bar_in_view(app):
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
    from smappy.gui.widgets import CollapsibleSection, ColumnScroll

    inner = QWidget()
    column = QVBoxLayout(inner)
    section = CollapsibleSection("wide", QLabel("x" * 300), expanded=True,
                                 detachable=True, helpable=True)
    column.addWidget(section)
    scroll = ColumnScroll(inner)
    scroll.resize(300, 400)
    scroll.show()
    app.processEvents()
    right = section.detach_button.mapTo(scroll.viewport(),
                                        section.detach_button.rect().topRight())
    assert right.x() < scroll.viewport().width()
    scroll.close()
