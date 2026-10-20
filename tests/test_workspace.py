"""Tabs as curated lists over one tree, and the GUI files that keep them."""
import pytest

from smappy import plugins
from smappy.workspace import Instance, Tab, Workspace, load

COMET = "Analysis/Drift/COMET"
RCC = "Analysis/Drift/RCC"
STATS = "Analysis/Measure/Localization Statistics"
SEGMENT = "Analysis/Regions/Segment regions"


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


def test_the_default_workspace_is_the_shipped_gui_file():
    """A list, not a rule: the tabs and pins of data/default.gui.yaml."""
    ws = Workspace.default()
    assert [t.name for t in ws.tabs] == ["File", "Localize", "Render", "Analysis", "Regions"]
    analysis = next(t for t in ws.tabs if t.name == "Analysis")
    assert [i.plugin for i in analysis.instances] == [
        "Analysis/Measure/Line Profile", STATS, RCC, COMET,
        "Analysis/Dual-colour/Assign colours"]
    assert next(t for t in ws.tabs if t.name == "Render").kind == "render"
    regions = next(t for t in ws.tabs if t.name == "Regions")
    assert regions.header == "roi"
    pinned = [i.plugin for i in regions.instances]
    assert pinned == [SEGMENT, "Analysis/Regions/Summarize/Export table"]
    # every instance has an id of its own, given as it is read
    ids = [i.id for t in ws.tabs for i in t.instances]
    assert all(ids) and len(set(ids)) == len(ids)
    # a tab the *user* adds starts empty
    assert Tab(name="Mine").instances == []


def test_every_plugin_the_shipped_gui_names_is_shipped():
    """The file is checked against the tree, so a renamed plugin is not
    silently missing from a fresh install."""
    import yaml
    from smappy.workspace import DEFAULT_GUI
    named = [i["plugin"] for t in yaml.safe_load(DEFAULT_GUI.read_text())["tabs"]
             for i in t.get("instances") or []]
    assert set(named) <= set(plugins.refs())
    assert sum(len(t.instances) for t in Workspace.default().tabs) == len(named)


def test_a_plugin_that_is_not_installed_is_left_out_of_the_default():
    refs = dict(plugins.refs())
    del refs[COMET]
    ws = Workspace.default(refs)
    analysis = next(t for t in ws.tabs if t.name == "Analysis")
    assert COMET not in {i.plugin for i in analysis.instances}
    assert RCC in {i.plugin for i in analysis.instances}


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
    titles = [i.title() for i in analysis.instances]
    assert titles[0] == titles[-1] == "COMET (coarse)"
    ws.layout = {"active_tab": "Analysis", "open": {"Analysis": analysis.instances[0].id}}
    path = ws.save(tmp_path / "w.yaml")

    back = load(path)
    tab = next(t for t in back.tabs if t.name == "Analysis")
    assert [i.title() for i in tab.instances] == titles
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
                                                 "Analysis", "Regions"]


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
    before = [i.plugin for i in tab.instances]
    tab.instances.append(Instance(plugin="Gone/Away"))
    gone = ws.prune(list(plugins.refs()))
    assert gone == ["Gone/Away"]
    assert [i.plugin for i in tab.instances] == before


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
        ["File", "Localize", "Render", "Analysis", "Regions"]
    localize = tab_named(window, "Localize")
    assert {"Bead calibration", "Gaussian 2D", "Spline 3D"} <= \
        {s.title for s in localize.sections}


def test_the_bead_calibration_section_opens_the_window_as_it_was_left(window):
    localize = tab_named(window, "Localize")
    instance = next(i for i in localize.tab.instances
                    if i.plugin == "Localize/Bead calibration")
    localize.open_section(instance.id)
    panel = localize.slots[instance.id].panel
    assert panel.run_button.text() == "Open"
    assert not panel.form.fields          # single or dual is the window's to ask
    panel.run_button.click()
    window.calibration_window.mode_box.setCurrentText("Dual colour")
    window.calibration_window.hide()
    panel.run_button.click()
    assert window.calibration_window.isVisible()
    assert window.calibration_window.is_dual


def test_the_menus_have_no_second_way_into_what_a_tab_opens(window):
    """The bead calibration is the Localize tab's, the per-region steps the
    Regions tab's.  The ROI manager is a window to look at, as the 3D view
    is, and is in View, once, with Ctrl+R.  The Plugins menu is the whole
    plugin tree, every plugin in it, and not counted."""
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QMenu

    def in_plugins(menu):
        while menu is not None:
            if menu is window.plugins_menu:
                return True
            menu = menu.parent() if isinstance(menu.parent(), QMenu) else None
        return False

    in_menus = [a.text() for m in window.findChildren(QMenu) if not in_plugins(m)
                for a in m.actions()]
    assert not [t for t in in_menus if "calibration" in t.lower()
                or "Per-region" in t]
    assert in_menus.count("ROI manager") == 1
    shortcuts = [a for a in window.actions() if a.shortcut().toString() == "Ctrl+R"]
    assert [a.text() for a in shortcuts] == ["ROI manager"]


def test_opening_a_section_is_what_imports_the_plugin(window):
    import sys
    tab = tab_named(window, "Analysis")
    instance = next(i for i in tab.tab.instances if i.plugin == COMET)
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
    at = next(k for k, i in enumerate(tab.tab.instances) if i.plugin == COMET)
    instance = tab.tab.instances[at]
    tab.open_section(instance.id)
    tab.slots[instance.id].panel.form.restore({"segmentation_var": 11})
    instance.label = "COMET (coarse)"
    path = tmp_path / "saved.yaml"
    window.save_gui_to(path)

    reopened = load(path)
    back = next(t for t in reopened.tabs if t.name == "Analysis")
    assert back.instances[at].label == "COMET (coarse)"
    assert back.instances[at].values["segmentation_var"] == 11
    assert reopened.layout["open"]["Analysis"] == instance.id
    # the screen's business, not the GUI's; and a GUI starts on File
    assert "geometry" not in reopened.layout
    assert "active_tab" not in reopened.layout


def test_a_saved_gui_keeps_the_filters_and_gives_them_to_the_next_file(window, tmp_path):
    session = window.session
    session.set_locs(_filtered_table(), undoable=False)
    session.layers[0].set_bound("photons", 500.0, None)
    session.layers[0].remove_bound("xy_err_nm")
    path = tmp_path / "filters.yaml"
    window.save_gui_to(path)
    saved = load(path).filters
    assert saved["photons"] == [500.0, None] and "xy_err_nm" not in saved
    # z is not in this table: the default for it is kept for a file that has it
    assert saved["z_nm"] == [-500.0, 500.0]

    session.layers[0].set_ranges({})
    window.load_gui_from(path)
    assert session.layers[0].ranges["photons"] == (500.0, None)
    assert "xy_err_nm" not in session.layers[0].ranges
    session.apply_default_filters(None)


def _filtered_table(n=500):
    import numpy as np
    from smappy.locs import Localizations
    rng = np.random.default_rng(1)
    return Localizations({"x_nm": rng.uniform(0, 5000, n), "y_nm": rng.uniform(0, 5000, n),
                          "frame": np.arange(n), "photons": rng.uniform(100, 2000, n),
                          "xy_err_nm": rng.uniform(5, 40, n)})


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
    assert "default workspace" in again.windowTitle()


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


def test_the_workspace_is_a_submenu_of_file_and_reset_gives_the_default(window, tmp_path):
    file_menu = next(a.menu() for a in window.menuBar().actions() if a.text() == "File")
    entries = [a.text() for a in file_menu.actions()]
    assert "Workspace" in entries and not any("GUI" in e for e in entries)
    actions = {a.text(): a for a in window.workspace_menu.actions() if a.text()}
    assert list(actions) == ["Save", "Save as...", "Open...", "Recent", "Reset",
                             "From data file..."]
    set_comet_value(window, 31)
    window.save_gui_to(tmp_path / "mine.gui.yaml")
    actions["Reset"].trigger()
    assert window.workspace.path is None and comet_value(window) is None


def test_recent_lists_the_workspaces_saved_or_loaded_and_opens_one_in_a_click(
        window, tmp_path):
    first = window.save_gui_to(tmp_path / "first.gui.yaml")
    set_comet_value(window, 41)
    second = window.save_gui_to(tmp_path / "second.gui.yaml")
    window.load_gui_from(first)                  # loading puts it back on top
    gone = window.save_gui_to(tmp_path / "gone.gui.yaml")
    gone.unlink()
    window.load_default_gui()                    # the default is not a file
    window._fill_recent_guis()
    entries = {a.text(): a for a in window.recent_gui_menu.actions()}
    assert list(entries) == ["first", "second"]  # newest first, the gone one left out
    entries["second"].trigger()
    assert window.workspace.path == second and comet_value(window) == 41
    # and it is what the next start opens; nothing was saved to get there
    assert make_window().workspace.path == second.resolve()


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
    assert [s.title for s in tab.sections][:2] == ["Localization Statistics", "Line Profile"]
    tab.tab.instances.remove(second)
    tab.rebuild()
    assert "Localization Statistics" not in [s.title for s in tab.sections]
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
    changes_locs = True
    Settings = Settings
    grouping = "ungrouped"        # it writes a column onto the localizations

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


def test_new_plugin_puts_a_working_one_under_user_in_the_menu(window, monkeypatch, tmp_path):
    """Plugins > New plugin...: a name asked, the template copied into the
    user's folder, and the plugin in the menu at the next opening."""
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QInputDialog, QMessageBox
    from smappy import config
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    config.load(reload=True)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("pair distances", True))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    shown = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: shown.append(url))
    try:
        window._fill_plugins_menu()
        new = [a for a in window.plugins_menu.actions() if a.text() == "New plugin..."]
        new[0].trigger()
        assert (config.user_plugin_dir() / "pair_distances.py").is_file()
        assert shown and shown[0].toLocalFile() == str(config.user_plugin_dir())
        window._fill_plugins_menu()
        user = [a.menu() for a in window.plugins_menu.actions()
                if a.menu() and a.text() == "User"]
        assert [a.text() for a in user[0].actions()] == ["Pair distances"]
        # changed on disk, it opens with the new code, not the window kept
        first = window.open_plugin_window("User/Pair Distances")
        file = config.user_plugin_dir() / "pair_distances.py"
        file.write_text(file.read_text().replace('"""The distance', '"""Changed: the distance'))
        import os, time
        os.utime(file, (time.time() + 5, time.time() + 5))
        second = window.open_plugin_window("User/Pair Distances")
        assert second is not first
        assert type(second.panel.plugin).__doc__.startswith("Changed")
    finally:
        monkeypatch.undo()
        config.load(reload=True)
        plugins.discover(force=True)


def test_a_plugin_that_cannot_be_opened_says_so_rather_than_raising(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    said = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: said.append(a[-1]), raising=False)
    assert window.open_plugin_window("No/Such/Plugin") is None
    assert said and "No/Such/Plugin" in said[0]
