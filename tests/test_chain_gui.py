"""Chains in the GUI: built in their own panel, kept in the workspace until
saved, saved into the tree; the layers editor; the batch window's job and its
reading of the runner's lines."""
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from smappy import batch, chain, plugins
from smappy.chain import ChainSpec, Step
from smappy.session import Session
from smappy.workspace import Instance, Tab, Workspace

from test_batch_foundations import blinks


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def new_chain_tab(app, session):
    from smappy.gui.plugin_tab import PluginTab
    tab = PluginTab(Tab(name="A"), session)
    tab.add_chain()
    instance = tab.tab.instances[0]
    return tab, instance, tab.slots[instance.id].panel


def test_a_new_chain_is_built_in_its_own_panel_and_kept_inline(app):
    from smappy.gui.chain_panel import ChainPanel
    tab, instance, panel = new_chain_tab(app, Session(blinks()))
    assert isinstance(panel, ChainPanel) and panel.edit.isChecked()
    spec = panel.current_spec()
    spec.steps += [Step(plugin="Chain/Layers"),
                   Step(plugin="Analysis/Measure/Localization Statistics",
                        values={"bins": 33})]
    assert panel._apply(spec)
    assert [s["plugin"] for s in instance.chain["steps"]] == \
        ["Chain/Layers", "Analysis/Measure/Localization Statistics"]
    assert panel.form.value().localization_statistics.bins == 33
    # moving keeps what the form holds
    panel.form.set_values({"localization_statistics.bins": 44})
    panel.steps.setCurrentRow(1)
    panel.move_step(-1)
    assert instance.chain["steps"][0]["values"]["bins"] == 44
    # the workspace keeps the chain being built, and reads it back
    tab.save_values()
    back = Workspace.from_dict(Workspace(tabs=[tab.tab]).to_dict())
    assert back.tabs[0].instances[0].chain["steps"][0]["plugin"] == \
        "Analysis/Measure/Localization Statistics"
    assert back.prune(list(plugins.refs())) == []      # not pruned for being unsaved


def test_a_saved_chain_becomes_its_file_in_the_tree(app, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    tab, instance, panel = new_chain_tab(app, Session(blinks()))
    spec = panel.current_spec()
    spec.steps.append(Step(plugin="Analysis/Measure/Localization Statistics"))
    panel._apply(spec)
    target = plugins.chains_dir() / "gui_saved.chain.yaml"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    try:
        assert panel.save_chain() == target
        assert instance.plugin == "Analysis/Chains/Gui Saved" and instance.chain is None
        assert chain.read(target).steps[0].plugin == \
            "Analysis/Measure/Localization Statistics"
        assert tab.sections[0].title == "Gui Saved"
    finally:
        target.unlink(missing_ok=True)
        plugins.discover(force=True)


def test_the_layers_editor_reads_back_what_it_was_given_and_the_session(app):
    from smappy.gui.params import LayersField
    from smappy.plugins.chain_layers import ChainLayers
    spec = ChainLayers.specs()["layers"]
    field = LayersField(spec)
    given = [{"grouped": True, "start": "empty",
              "bounds": [{"field": "photons", "lo": 0.01, "quantile": True}]},
             {"grouped": False, "start": "defaults",
              "bounds": [{"field": "loc_precision_nm", "hi": 20.0, "required": True}]}]
    field.set(given)
    assert field.value() == given
    session = Session(blinks())
    session.layers[0].set_bound("photons", 100.0, None)
    field.session = session
    field.take_from_session()
    taken = field.value()
    assert taken[0]["grouped"] is True and taken[0]["start"] == "empty"
    assert {"field": "photons", "lo": 100.0} in taken[0]["bounds"]


def test_the_batch_window_writes_the_job_it_shows(app, tmp_path):
    from smappy.gui.batch_window import BatchWindow
    from smappy.batch import Input
    window = BatchWindow()
    window.set_chain(ChainSpec(steps=[Step(plugin="Analysis/Measure/Localization Statistics",
                                           label="stats")]))
    window.add_input(Input(folder=str(tmp_path), pattern=["*.h5"]))
    window.add_input(Input(file=str(tmp_path / "a.h5"), overrides={"stats.bins": 12}))
    window.figures.setCurrentIndex(window.figures.findData("svg"))
    job = window.job()
    assert [i.to_dict() for i in job.inputs] == [
        {"folder": str(tmp_path), "pattern": ["*.h5"]},
        {"file": str(tmp_path / "a.h5"), "overrides": {"stats.bins": 12}}]
    assert job.output.figures == "svg" and job.chain.steps[0].label == "stats"
    window.job_path = tmp_path / "j.batch.yaml"
    window.save_job(ask=False)
    again = BatchWindow()
    again.load_job(tmp_path / "j.batch.yaml")
    assert again.job().to_dict()["inputs"] == job.to_dict()["inputs"]


def test_the_batch_window_follows_the_runners_lines(app, tmp_path):
    from smappy.gui.batch_window import BatchWindow
    from smappy.batch import Input
    window = BatchWindow()
    (tmp_path / "d").mkdir()
    window.add_input(Input(folder=str(tmp_path / "d")))
    window.add_input(Input(file=str(tmp_path / "x.h5")))
    for line in ["SMAPPY_JOBS\t3",
                 f"SMAPPY_START\t1/3\t{tmp_path / 'd' / 'a.h5'}",
                 "SMAPPY_PROGRESS\t1/3\tloading",
                 f"SMAPPY_DONE\t1/3\t{tmp_path / 'out' / 'a'}",
                 f"SMAPPY_START\t2/3\t{tmp_path / 'd' / 'b.h5'}",
                 "SMAPPY_FAILED\t2/3\tOSError: broken",
                 f"SMAPPY_START\t3/3\t{tmp_path / 'x.h5'}",
                 f"SMAPPY_DONE\t3/3\t{tmp_path / 'out' / 'x'}",
                 f"SMAPPY_REPORT\t{tmp_path / 'out' / 'batch_report.json'}"]:
        window.handle_line(line)
    assert window.table.item(0, 4).text() == "1 done, 1 failed"
    assert window.table.item(1, 4).text() == "done"
    assert window.progress.value() == 3
    assert window.report_path.name == "batch_report.json"
