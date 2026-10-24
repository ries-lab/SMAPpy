"""A plugin's section in a window of its own, and back."""
import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_a_plugin_never_opened_detaches_with_its_panel_in_a_window_that_fits(app):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.gui.plugin_tab import PluginTab
    from smappy.session import Session
    from smappy.workspace import Instance, Tab

    tab = PluginTab(Tab(name="A", instances=[
        Instance(plugin="Localize/Spline 3D")]), Session())   # a long panel
    tab.show()
    section = tab.sections[0]
    assert not section.button.isChecked()          # collapsed: no panel made yet
    section.detach_requested.emit()
    window = section._window
    # the panel, not the empty slot it would have been
    assert window.content.findChild(PluginPanel) is not None
    # a window of its own, not a tool panel; no taller than the screen,
    # the rest scrolled
    assert window.windowFlags() & Qt.WindowType_Mask == Qt.Window
    room = (window.screen() or QApplication.primaryScreen()).availableGeometry()
    assert window.height() <= room.height() and window.width() <= room.width()
    window.close()
    assert window.content.parent() is not None      # back in its section
    assert section.button.isEnabled()
    tab.close()
