"""The plugin chooser: Plugins > Find a plugin..., and every tab's +."""
import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_search_finds_a_plugin_by_a_word_only_its_page_has(app):
    from smappy.gui.chooser import PluginChooser
    dialog = PluginChooser()
    # a reference cited in Localization Statistics' page, in no name or summary
    dialog.search.setText("Thompson")
    assert dialog.path == "Analysis/Measure/Localization Statistics"
    assert "In its page:" in dialog.about.text()
    assert "Thompson" in dialog.about.text()


def test_enter_in_the_search_opens_the_first_match(app):
    from smappy.gui.chooser import PluginChooser
    dialog = PluginChooser()
    dialog.search.setText("RCC")
    dialog.search.returnPressed.emit()
    from PySide6.QtWidgets import QDialog
    assert dialog.result() == QDialog.Accepted
    assert dialog.path == "Analysis/Drift/RCC"
