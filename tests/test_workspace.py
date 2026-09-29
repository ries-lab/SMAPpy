"""Tabs as curated lists over one tree, and the GUI files that keep them."""
import pytest

from smappy import plugins
from smappy.workspace import Instance, Tab, Workspace, load

COMET = "Analysis/Drift/COMET"
RCC = "Analysis/Drift/RCC"
COLORS = "Analysis/Dual-Color/AssignColors"
TRUTH = "Analysis/Measure/Ground Truth"
PROFILE = "Analysis/Measure/Line Profile"
PRECISION = "Analysis/Measure/Localization Precision"
STATS = "Analysis/Measure/Localization Statistics"
HISTORY = "Analysis/Process/History"
MATH = "Analysis/Process/Math Parser"
REMOVE = "Analysis/Process/Remove Localizations"
REGISTER = "Analysis/Register/Calibrate transform"


# ---------------------------------------------------------------- the model

def test_an_instance_is_identified_apart_from_its_label():
    one = Instance(plugin=COMET)
    two = Instance(plugin=COMET)
    assert one.id and one.id != two.id          # the same plugin, twice
    assert one.title("COMET") == "COMET"
    one.label = "COMET (coarse)"
    assert one.title("COMET") == "COMET (coarse)"


def test_a_duplicate_diverges_from_its_original():
    one = Instance(plugin=COMET, values={"segmentation_var": 5})
    two = one.copy()
    assert two.id != one.id
    two.values["segmentation_var"] = 9
    assert one.values["segmentation_var"] == 5


def test_moving_stops_at_the_ends():
    tab = Tab(name="Analysis", instances=[Instance(plugin=COMET), Instance(plugin=RCC)])
    first, second = tab.instances
    assert not tab.move(first.id, -1)           # already at the top
    assert tab.move(first.id, 1)
    assert [i.id for i in tab.instances] == [second.id, first.id]
    assert not tab.move(first.id, 1)


def test_the_default_workspace_is_seeded_from_what_is_installed():
    ws = Workspace.default()
    assert [t.name for t in ws.tabs] == ["File", "Localize", "Render", "Analysis", "ROI"]
    analysis = next(t for t in ws.tabs if t.name == "Analysis")
    assert sorted(i.plugin for i in analysis.instances) == [
        COMET, RCC, COLORS, TRUTH, PROFILE, PRECISION, STATS, HISTORY, MATH, REMOVE,
        REGISTER]
    assert next(t for t in ws.tabs if t.name == "Render").kind == "render"
    # a tab the *user* adds starts empty; only the shipped ones are seeded
    assert Tab(name="Mine").instances == []


def test_a_tab_is_not_a_branch_of_the_tree():
    """The point of the model: any plugin can be pinned to any tab."""
    tab = Tab(name="Whatever")
    tab.instances.append(Instance(plugin=COMET))
    tab.instances.append(Instance(plugin="Localize/Spline 3D"))
    assert len({i.plugin.split("/")[0] for i in tab.instances}) == 2


# ----------------------------------------------------------------- the file

def test_a_round_trip_keeps_order_labels_and_values(tmp_path):
    ws = Workspace.default()
    analysis = next(t for t in ws.tabs if t.name == "Analysis")
    analysis.instances[0].label = "COMET (coarse)"
    analysis.instances[0].values = {"segmentation_var": 12}
    analysis.instances.append(analysis.instances[0].copy())
    ws.layout = {"active_tab": "Analysis", "open": {"Analysis": analysis.instances[0].id}}
    path = ws.save(tmp_path / "w.yaml")

    back = load(path)
    tab = next(t for t in back.tabs if t.name == "Analysis")
    assert [i.title() for i in tab.instances] == [
        "COMET (coarse)", "RCC", "AssignColors", "Ground Truth", "Line Profile",
        "Localization Precision", "Localization Statistics", "History",
        "Math Parser", "Remove Localizations", "Calibrate transform",
        "COMET (coarse)"]
    assert tab.instances[0].values == {"segmentation_var": 12}
    assert back.layout["active_tab"] == "Analysis"


def shape(ws):
    """Tabs and pins without the ids, which are deliberately random."""
    return [(t.name, t.kind, [i.plugin for i in t.instances]) for t in ws.tabs]


def test_a_missing_file_gives_the_default(tmp_path):
    assert shape(load(tmp_path / "nope.yaml")) == shape(Workspace.default())


def test_a_broken_file_gives_the_default_rather_than_no_tabs(tmp_path):
    path = tmp_path / "w.yaml"
    path.write_text("tabs: [oh dear\n")
    assert [t.name for t in load(path).tabs] == ["File", "Localize", "Render",
                                                 "Analysis", "ROI"]


def test_unrecognised_entries_are_dropped_not_raised(tmp_path):
    path = tmp_path / "w.yaml"
    path.write_text("""
version: 1
tabs:
  - name: Analysis
    instances:
      - plugin: Analysis/Drift/COMET
        something_new: 3
      - notaplugin: true
      - plugin: ''
  - nameless: true
""".lstrip())
    ws = load(path)
    assert [t.name for t in ws.tabs] == ["Analysis"]
    assert [i.plugin for i in ws.tabs[0].instances] == [COMET]


def test_pruning_reports_what_it_dropped():
    ws = Workspace.default()
    tab = next(t for t in ws.tabs if t.name == "Analysis")
    tab.instances.append(Instance(plugin="Gone/Away"))
    gone = ws.prune(list(plugins.refs()))
    assert gone == ["Gone/Away"]
    assert [i.plugin for i in tab.instances] == [
        COMET, RCC, COLORS, TRUTH, PROFILE, PRECISION, STATS, HISTORY, MATH, REMOVE,
        REGISTER]


# ------------------------------------------------------------------ the GUI

@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    from smappy import config
    config.load(reload=True)
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.session import Session
    session = Session()
    render = RenderWindow(session)
    return ControlWindow(session, render)


def tab_named(window, name):
    """By name, not by index: the strip's order is the user's to change."""
    for i in range(window.tabs.count()):
        if window.tabs.tabText(i) == name:
            return window.tabs.widget(i)
    raise AssertionError(f"no {name} tab in "
                         f"{[window.tabs.tabText(i) for i in range(window.tabs.count())]}")


def test_the_window_opens_the_shipped_tabs(window):
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == \
        ["File", "Localize", "Render", "Analysis", "ROI"]
    localize = tab_named(window, "Localize")
    assert [s.title for s in localize.sections] == \
        ["Gaussian 2D", "Gaussian 2D 2C", "Spline 3D", "Spline 3D 2C"]


def test_opening_a_section_is_what_imports_the_plugin(window):
    import sys
    tab = tab_named(window, "Analysis")
    instance = tab.tab.instances[0]
    assert tab.slots[instance.id].panel is None       # nothing built yet
    tab.open_section(instance.id)
    assert tab.slots[instance.id].panel is not None
    assert "smappy.plugins.drift_comet" in sys.modules


def test_only_one_section_is_open_at_a_time(window):
    tab = tab_named(window, "Analysis")
    first, second = tab.tab.instances[:2]
    tab.open_section(first.id)
    tab.open_section(second.id)
    open_now = [s.title for s in tab.sections if s.button.isChecked()]
    assert len(open_now) == 1


def test_a_pin_that_is_not_installed_says_so_instead_of_crashing(window):
    from smappy.workspace import Instance
    tab = tab_named(window, "Analysis")
    ghost = Instance(plugin="Gone/Away", label="Ghost")
    tab.tab.instances.append(ghost)
    tab.rebuild()
    tab.open_section(ghost.id)
    assert tab.slots[ghost.id].panel is None          # a message, not a traceback
    assert [s.title for s in tab.sections][-1] == "Ghost"


def test_values_survive_being_saved_and_reopened(window, tmp_path):
    tab = tab_named(window, "Analysis")
    instance = tab.tab.instances[0]
    tab.open_section(instance.id)
    tab.slots[instance.id].panel.form.restore({"segmentation_var": 11})
    instance.label = "COMET (coarse)"
    path = tmp_path / "saved.yaml"
    window.save_gui_to(path)

    reopened = load(path)
    back = next(t for t in reopened.tabs if t.name == "Analysis")
    assert back.instances[0].label == "COMET (coarse)"
    assert back.instances[0].values["segmentation_var"] == 11
    assert reopened.layout["open"]["Analysis"] == instance.id
    # the screen's business, not the GUI's; and a GUI starts on File
    assert "geometry" not in reopened.layout
    assert "active_tab" not in reopened.layout


def test_an_unopened_panel_keeps_the_values_it_was_given(window, tmp_path):
    """Saving must not blank the plugins you never looked at."""
    tab = tab_named(window, "Analysis")
    instance = tab.tab.instances[1]
    instance.values = {"pixelsize_nm": 42.0}
    assert tab.slots[instance.id].panel is None
    window.save_gui_to(tmp_path / "saved.yaml")
    back = load(tmp_path / "saved.yaml")
    kept = next(t for t in back.tabs if t.name == "Analysis").instances[1]
    assert kept.values == {"pixelsize_nm": 42.0}


def make_window():
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.session import Session
    session = Session()
    return ControlWindow(session, RenderWindow(session))


def comet_value(window):
    tab = tab_named(window, "Analysis")
    return next(i for i in tab.tab.instances if i.plugin == COMET).values.get(
        "segmentation_var")


def set_comet_value(window, value):
    tab = tab_named(window, "Analysis")
    instance = next(i for i in tab.tab.instances if i.plugin == COMET)
    tab.open_section(instance.id)
    tab.slots[instance.id].panel.form.restore({"segmentation_var": value})


def test_nothing_typed_survives_a_restart_unless_it_was_saved(window, tmp_path):
    """One installation, many users: a value one person typed must not become
    the next person's default by the program remembering it."""
    from PySide6.QtWidgets import QApplication
    set_comet_value(window, 17)
    window.tabs.setCurrentIndex(3)
    QApplication.instance().aboutToQuit.emit()
    written = {p.name for p in (tmp_path / "config").rglob("*") if p.is_file()}
    assert written <= {"window.yaml"}         # where the window was, nothing else

    again = make_window()
    assert comet_value(again) is None
    assert again.tabs.tabText(again.tabs.currentIndex()) == "File"
    assert again.workspace.path is None
    assert "default GUI" in again.windowTitle()


def test_a_saved_gui_is_what_the_next_start_opens(window, tmp_path):
    set_comet_value(window, 17)
    path = window.save_gui_to(tmp_path / "jonas.gui.yaml")
    assert window.windowTitle().endswith("jonas")

    again = make_window()
    assert again.workspace.path == path.resolve()
    assert comet_value(again) == 17
    assert again.windowTitle().endswith("jonas")
    assert again.tabs.tabText(again.tabs.currentIndex()) == "File"


def test_save_gui_overwrites_the_loaded_file_and_nothing_else(window, tmp_path):
    path = window.save_gui_to(tmp_path / "lab.gui.yaml")
    set_comet_value(window, 23)
    window.save_gui()
    assert next(i for t in load(path).tabs for i in t.instances
                if i.plugin == COMET).values["segmentation_var"] == 23


def test_save_gui_on_the_shipped_default_asks_where(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    target = tmp_path / "mine.gui.yaml"
    asked = []

    def ask(*args, **kwargs):
        asked.append(args)
        return str(target), ""
    monkeypatch.setattr(QFileDialog, "getSaveFileName", ask)
    assert window.workspace.path is None
    window.save_gui()
    assert asked and target.exists()
    assert window.workspace.path == target


def test_loading_a_gui_replaces_the_values_and_is_remembered(window, tmp_path):
    set_comet_value(window, 31)
    path = window.save_gui_to(tmp_path / "other.gui.yaml")
    window.load_default_gui()
    assert comet_value(window) is None
    assert make_window().workspace.path is None      # the default, next time too

    window.load_gui_from(path)
    assert comet_value(window) == 31
    assert window.tabs.tabText(window.tabs.currentIndex()) == "File"
    assert make_window().workspace.path == path.resolve()


def test_a_remembered_gui_that_has_gone_gives_the_default(window, tmp_path):
    path = window.save_gui_to(tmp_path / "gone.gui.yaml")
    path.unlink()
    assert make_window().workspace.path is None


def test_the_window_position_is_kept_apart_from_the_gui(window, tmp_path):
    from smappy import workspace
    window.save_window_state()
    assert workspace.window_state()["geometry"]
    make_window()                              # reads it back without error


def test_removing_and_reordering_from_the_tab_edits_the_workspace(window):
    tab = tab_named(window, "Analysis")
    first, second = tab.tab.instances[:2]
    tab.tab.move(first.id, 1)
    tab.rebuild()
    assert [s.title for s in tab.sections][:2] == ["RCC", "COMET"]
    tab.tab.instances.remove(second)
    tab.rebuild()
    assert "RCC" not in [s.title for s in tab.sections]
    assert second.id not in tab.slots            # its panel went with it


def test_a_tab_the_user_adds_starts_empty_and_builds(window):
    from smappy.workspace import Tab
    window.workspace.tabs.append(Tab(name="MyLab"))
    window.build_tabs()
    names = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert names[-1] == "MyLab"
    assert tab_named(window, "MyLab").sections == []


def test_a_plugin_dropped_in_a_folder_reaches_a_tab_and_runs(app, tmp_path, monkeypatch):
    """The whole story: copy a file into a folder, pin it, run it."""
    import numpy as np
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    from smappy import config
    config.load(reload=True)

    root = tmp_path / "myplugins"
    (root / "MyLab" / "Cluster").mkdir(parents=True)
    (root / "MyLab" / "Cluster" / "count_neighbours.py").write_text('''
"""Counts each localization's neighbours within a radius."""
from dataclasses import dataclass

import numpy as np

from smappy.locs import Localizations
from smappy.plugins import Plugin, Result, param


@dataclass
class Settings:
    radius_nm: float = param(50.0, label="radius", unit="nm", min=1)


class CountNeighbours(Plugin):
    """Counts each localization's neighbours within a radius."""
    Settings = Settings

    def run(self, ctx, settings):
        x = np.asarray(ctx.locs["x_nm"])
        counts = np.array([(np.abs(x - v) <= settings.radius_nm).sum() - 1 for v in x])
        ctx.report(f"median {np.median(counts):.0f} neighbours")
        columns = {k: np.asarray(ctx.locs[k]) for k in ctx.locs.keys()}
        columns["neighbours"] = counts.astype(float)
        return Result(locs=Localizations(columns), text="counted", settings=settings)
'''.lstrip())
    config.set_plugin_roots([root])
    plugins.discover(force=True)
    try:
        # the folder named it, and the docstring described it, with no decorator
        ref = plugins.refs()["MyLab/Cluster/Count Neighbours"]
        assert ref.name == "Count Neighbours"
        assert ref.description.startswith("Counts each localization")

        from smappy.gui.app import ControlWindow, RenderWindow
        from smappy.locs import Localizations
        from smappy.session import Session
        session = Session(Localizations({"x_nm": np.arange(20.0),
                                         "y_nm": np.zeros(20), "frame": np.arange(20)}))
        window = ControlWindow(session, RenderWindow(session))
        window.workspace.tabs.append(
            Tab(name="MyLab", instances=[Instance(plugin=ref.path)]))
        window.build_tabs()

        tab = tab_named(window, "MyLab")
        instance = tab.tab.instances[0]
        tab.open_section(instance.id)
        panel = tab.slots[instance.id].panel
        assert list(panel.form.fields) == ["radius_nm"]

        panel.form.restore({"radius_nm": 3.0})
        result = session.run(plugins.get(ref.path)(), panel.form.value())
        assert "neighbours" in result.locs.keys()
        assert result.locs["neighbours"].max() == 6      # +-3 nm on a unit grid
    finally:
        config.set_plugin_roots([])
        config.load(reload=True)
        plugins.discover(force=True)


# ------------------------------------------------------------ the menu

def test_the_plugins_menu_is_the_whole_tree_and_opens_a_window(window, monkeypatch):
    """A tab is a curated list and pinning is a decision; the menu is the
    other way in -- anything, once, in a window of its own."""
    window._fill_plugins_menu()

    def leaves(menu, prefix=""):
        found = {}
        for action in menu.actions():
            if action.menu():
                found.update(leaves(action.menu(), f"{prefix}{action.text()}/"))
            elif not action.isSeparator():
                found[f"{prefix}{action.text()}"] = action
        return found

    found = leaves(window.plugins_menu)
    # every scanned plugin, under its own group; the tree, not the pinned list.
    # The leaf is the plugin's *name*, as the chooser shows it, not the last
    # part of its path
    for path, ref in plugins.refs().items():
        assert f"{ref.group}/{ref.name}" in found, path
    assert "Find a plugin..." in found

    stats = f"{plugins.refs()[STATS].group}/{plugins.refs()[STATS].name}"
    found[stats].trigger()
    opened = window._plugin_windows[STATS]
    assert opened.isVisible() and opened.panel.plugin.path == STATS
    # again is the same window, so what was typed into it is still there
    found[stats].trigger()
    assert window._plugin_windows[STATS] is opened


def test_a_plugin_that_cannot_be_opened_says_so_rather_than_raising(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    said = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: said.append(a[-1]), raising=False)
    assert window.open_plugin_window("No/Such/Plugin") is None
    assert said and "No/Such/Plugin" in said[0]
