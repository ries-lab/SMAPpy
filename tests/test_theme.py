"""The look: Fusion, light or dark or as the system is, switched in every
open window at once."""
import pytest


@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    from smappy.gui import make_app, theme
    application = make_app()
    yield application
    # the application is shared by the worker's other tests: leave it light
    theme.set_theme("system")


def _pixel(widget, x, y):
    from PySide6.QtWidgets import QApplication
    for _ in range(3):
        QApplication.processEvents()
    return widget.grab().toImage().pixelColor(x, y)


def test_the_style_is_fusion_and_system_is_light_where_the_platform_cannot_tell(app):
    from PySide6.QtGui import QPalette

    from smappy.gui import theme
    sheet = app.styleSheet()          # under a style sheet the style has no name
    app.setStyleSheet("")
    try:
        assert app.style().name().lower() == "fusion"
    finally:
        app.setStyleSheet(sheet)
    assert theme.theme() == "system"
    assert not theme.system_is_dark(app)          # offscreen has no theme
    assert app.palette().color(QPalette.Window).name() == theme.LIGHT["Window"]


def test_a_saved_theme_that_is_not_one_reads_as_system(app):
    from smappy import config
    from smappy.gui import theme
    config.set(theme.KEY, "purple")
    assert theme.theme() == "system"
    with pytest.raises(ValueError):
        theme.set_theme("purple")


def test_dark_reaches_the_fields_of_a_window_already_open(app):
    """The fields' style sheet reads the palette when a widget is polished;
    without polishing again, a field stayed white on a dark window."""
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QLineEdit, QVBoxLayout, QWidget

    from smappy.gui import theme
    window = QWidget()
    field = QLineEdit()                      # empty: the middle is background
    QVBoxLayout(window).addWidget(field)
    window.resize(200, 60)
    window.show()
    inside = field.geometry().center()
    assert _pixel(window, inside.x(), inside.y()) == QColor(theme.LIGHT["Base"])

    theme.set_theme("dark")
    assert app.palette().color(QPalette.Window).name() == theme.DARK["Window"]
    assert _pixel(window, inside.x(), inside.y()) == QColor(theme.DARK["Base"])

    theme.set_theme("light")
    assert _pixel(window, inside.x(), inside.y()) == QColor(theme.LIGHT["Base"])
    window.close()


def test_a_note_made_in_light_takes_the_dark_grey_when_switched(app):
    from PySide6.QtWidgets import QLabel

    from smappy.gui import theme
    from smappy.gui.widgets import note_colour, note_style
    theme.set_theme("light")
    light_grey = note_colour()
    note = QLabel("a note")
    note.setStyleSheet(note_style("font-style: italic"))

    theme.set_theme("dark")
    dark_grey = note_colour()
    assert dark_grey != light_grey
    assert note.styleSheet() == f"color: {dark_grey}; font-style: italic"

    theme.set_theme("light")
    assert note.styleSheet() == f"color: {light_grey}; font-style: italic"


def test_the_choice_is_in_preferences_saved_and_shown_at_once(app):
    from PySide6.QtGui import QPalette

    from smappy.gui import theme
    from smappy.gui.preferences import GeneralPage
    page = GeneralPage()
    choice = page.appearance
    assert [choice.itemData(i) for i in range(choice.count())] == list(theme.THEMES)
    assert choice.currentData() == "system"

    choice.setCurrentIndex(choice.findData("dark"))
    assert theme.theme() == "dark"
    assert app.palette().color(QPalette.Window).name() == theme.DARK["Window"]
    assert GeneralPage().appearance.currentData() == "dark"     # read back
    page.deleteLater()
