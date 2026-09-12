"""Tabs as curated lists over one tree, and the state that survives a restart."""
import pytest

from smappy import plugins
from smappy.workspace import Instance, Tab, Workspace, load

COMET = "Analysis/Drift/COMET"
RCC = "Analysis/Drift/RCC"


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
    assert sorted(i.plugin for i in analysis.instances) == [COMET, RCC]
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
    assert [i.title() for i in tab.instances] == ["COMET (coarse)", "RCC", "COMET (coarse)"]
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
    assert [i.plugin for i in tab.instances] == [COMET, RCC]


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
        ["Gaussian 2D", "Spline 3D", "Spline 3D 2C"]


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
    window.save_workspace(path)

    reopened = load(path)
    back = next(t for t in reopened.tabs if t.name == "Analysis")
    assert back.instances[0].label == "COMET (coarse)"
    assert back.instances[0].values["segmentation_var"] == 11
    assert reopened.layout["open"]["Analysis"] == instance.id
    assert reopened.layout["geometry"]


def test_an_unopened_panel_keeps_the_values_it_was_given(window, tmp_path):
    """Saving must not blank the plugins you never looked at."""
    tab = tab_named(window, "Analysis")
    instance = tab.tab.instances[1]
    instance.values = {"pixelsize_nm": 42.0}
    assert tab.slots[instance.id].panel is None
    window.save_workspace(tmp_path / "saved.yaml")
    back = load(tmp_path / "saved.yaml")
    kept = next(t for t in back.tabs if t.name == "Analysis").instances[1]
    assert kept.values == {"pixelsize_nm": 42.0}


def test_removing_and_reordering_from_the_tab_edits_the_workspace(window):
    tab = tab_named(window, "Analysis")
    first, second = tab.tab.instances[:2]
    tab.tab.move(first.id, 1)
    tab.rebuild()
    assert [s.title for s in tab.sections] == ["RCC", "COMET"]
    tab.tab.instances.remove(second)
    tab.rebuild()
    assert [s.title for s in tab.sections] == ["COMET"]
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
