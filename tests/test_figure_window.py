"""The window a result's figures live in: tabs, laziness, and the All page.

The rule the whole thing turns on is that a figure is drawn when it is looked
at and not before -- a plugin with six figures has to cost what a plugin with
one costs, or stepping through ROIs is unaffordable.
"""
import pytest

pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")

from smappy.plugins import Plot                                       # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def window(app, title="Demo"):
    from smappy.gui.figures import ResultWindow
    made = ResultWindow(title)
    made.show()
    return made


def counted(drawn, name, panels=1, size=None):
    def draw(target):
        drawn.append(name)
        if panels > 1:
            target.subplots(1, panels)
        else:
            target.plot([0, 1], [0, 1])
    return Plot(draw=draw, name=name, panels=panels, size=size)


def test_one_figure_is_a_window_with_a_figure_in_it(app):
    drawn = []
    made = window(app)
    made.show_plots([counted(drawn, "")])
    assert made.tabs.count() == 1
    assert not made.tabs.tabBar().isVisible()      # nothing to choose between
    assert drawn == [""]


def test_several_become_tabs_and_only_the_open_one_is_drawn(app):
    drawn = []
    made = window(app)
    made.show_plots([counted(drawn, ""), counted(drawn, "second"),
                     counted(drawn, "third")])
    assert [made.tabs.tabText(i) for i in range(made.tabs.count())] == [
        "figure", "second", "third", "All"]
    assert drawn == [""]

    made.tabs.setCurrentIndex(2)
    assert drawn == ["", "third"]
    made.tabs.setCurrentIndex(0)                  # drawn already
    assert drawn == ["", "third"]


def test_the_all_page_is_drawn_when_it_is_opened_and_not_before(app):
    drawn = []
    made = window(app)
    made.show_plots([counted(drawn, ""), counted(drawn, "second")])
    assert drawn == [""]                          # the page costs nothing yet

    made.tabs.setCurrentIndex(made.tabs.count() - 1)
    assert drawn == ["", "", "second"]            # both, onto one page
    page = made.tabs.currentWidget().figure
    assert len(page.subfigs) == 2                 # a subfigure each


def test_a_multi_panel_plot_composes_onto_the_page(app):
    """What `panels` is for: a plot that seizes the figure cannot share one."""
    drawn = []
    made = window(app)
    made.show_plots([counted(drawn, "", panels=3), counted(drawn, "second")])
    made.tabs.setCurrentIndex(made.tabs.count() - 1)
    page = made.tabs.currentWidget().figure
    assert len(page.subfigs) == 2
    assert len(page.axes) == 4                    # three panels and the other


def test_a_rerun_redraws_what_is_open_and_greys_the_rest(app):
    drawn = []
    made = window(app)
    plots = [counted(drawn, ""), counted(drawn, "second")]
    made.show_plots(plots)
    made.tabs.setCurrentIndex(1)                  # draw "second" too
    drawn.clear()

    made.show_plots(plots)
    assert drawn == ["second"]                    # only the tab on screen
    from PySide6.QtCore import Qt
    assert made.tabs.tabBar().tabTextColor(0) == Qt.GlobalColor.gray
    assert made.tabs.tabBar().tabTextColor(1) != Qt.GlobalColor.gray

    made.tabs.setCurrentIndex(0)
    assert drawn == ["second", ""]
    assert made.tabs.tabBar().tabTextColor(0) != Qt.GlobalColor.gray


def test_the_size_hint_sets_the_window_once(app):
    drawn = []
    made = window(app)
    made.show_plots([counted(drawn, "", size=(12, 4))])
    dpi = made.panes[""].figure.get_dpi()
    assert made.width() == int(12 * dpi)

    made.resize(400, 300)                         # the user's size wins now
    made.show_plots([counted(drawn, "", size=(12, 4))])
    assert made.width() == 400


def test_a_figure_that_is_gone_takes_its_tab_with_it(app):
    drawn = []
    made = window(app)
    made.show_plots([counted(drawn, ""), counted(drawn, "second")])
    made.show_plots([counted(drawn, "")])
    assert [made.tabs.tabText(i) for i in range(made.tabs.count())] == ["figure"]
    assert "second" not in made.panes


def test_nothing_is_drawn_while_the_window_is_hidden(app):
    """A result offered to a closed window waits until it is opened again."""
    from smappy.gui.figures import ResultWindow
    drawn = []
    made = ResultWindow("Demo")
    made.show_plots([counted(drawn, "")])
    assert drawn == []
    made.show()
    assert drawn == [""]
