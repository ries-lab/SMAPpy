"""A launcher with nothing to set has its buttons in its section's title bar."""
import gc

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _tab(*paths):
    from smappy.gui.plugin_tab import PluginTab
    from smappy.session import Session
    from smappy.workspace import Instance, Tab
    session = Session()
    opened = []
    for name in ("open files", "add files", "psf calibration"):
        session.window_openers[name] = lambda settings, n=name: opened.append(n)
    tab = PluginTab(Tab(name="A", instances=[Instance(plugin=p) for p in paths]),
                    session)
    return tab, opened


def test_load_has_open_and_add_in_its_title_bar_and_does_not_unfold(app):
    tab, opened = _tab("File/Load/Load", "Localize/Gaussian 2D")
    tab.show()
    load, fitter = tab.sections
    assert [w.text() for w in load.title_widgets] == ["Open", "Add"]
    assert all(w.isVisible() for w in load.title_widgets)
    assert not load.button.isCheckable()
    load.button.click()
    tab.open_section(tab.tab.instances[0].id)
    assert not load.frame.isVisible() and tab.open_instance() is None
    assert load.detach_button.isHidden()
    load.title_widgets[1].click()
    load.title_widgets[0].click()
    assert opened == ["add files", "open files"]
    # a plugin with settings is a section as before, never imported
    assert fitter.title_widgets == [] and fitter.button.isCheckable()
    assert tab.slots[tab.tab.instances[1].id].panel is None
    tab.close()


def test_the_title_bar_buttons_survive_a_rebuild_and_are_the_panels(app):
    from PySide6.QtCore import QCoreApplication, QEvent
    tab, opened = _tab("Localize/PSF calibration")
    tab.show()
    panel = tab.slots[tab.tab.instances[0].id].panel
    tab.rebuild()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    gc.collect()
    section = tab.sections[0]
    assert section.title_widgets == [panel.run_button]
    panel.run_button.click()                  # what the tutorials and assistant press
    assert opened == ["psf calibration"]
    tab.close()


def test_a_launchers_title_bar_fits_the_control_column(app):
    from smappy import plugins
    from smappy.gui.widgets import CONTROL_WIDTH
    launchers = [p for p, r in plugins.refs().items()
                 if r.root == "builtin" and r.window]
    assert "File/Load/Load" in launchers
    tab, _ = _tab(*launchers)
    for section, path in zip(tab.sections, launchers):
        assert section.minimumSizeHint().width() <= CONTROL_WIDTH - 40, path
    tab.close()
