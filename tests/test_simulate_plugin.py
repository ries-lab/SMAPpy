"""*Simulate*: a scene chosen from a preset, its key parameters at the top,
the form following the components, and the three outputs
(`smappy.plugins.simulate`)."""
import numpy as np
import pytest

from smappy.plugins import Context, settings_from
from smappy.plugins.simulate import (DEFAULT_KEYS, Simulate, SimulateSettings,
                                     component_state, key_labels, modified, preset_keys,
                                     preset_values, presets)


def preset(name):
    return settings_from(SimulateSettings, preset_values(name))


def test_every_shipped_preset_is_a_scene_that_runs():
    for name in presets():
        settings = preset(name)
        settings.n_frames = 300
        result = Simulate().run(Context(), settings)
        locs = result.files[0][0]
        assert len(locs) > 0, name
        assert locs.metadata["simulation_settings"]["preset"] == name


def test_a_preset_fills_every_setting_and_says_when_one_moved():
    """What the preset leaves out is the default, not what was there before;
    any change is 'modified'."""
    settings = SimulateSettings(n_frames=123, background=99.0)
    filled = Simulate().react("preset", SimulateSettings(preset="Diffusion"))
    assert filled["n_frames"] == 1000 and filled["background"] == 10.0
    assert filled["components"][0]["motion.diffusion"] is True
    fresh = preset("Diffusion")
    assert not modified(fresh)
    fresh.components[0].motion.d_um2_s = 0.1
    assert modified(fresh)
    assert not modified(settings)                       # no preset, nothing to differ from


def test_a_presets_key_parameters_are_named_by_component_when_there_are_several():
    keys = key_labels(preset("Two-colour pores"), preset_keys("Two-colour pores"))
    assert keys["components.0.labelling.efficiency"] == "NPC: labelling efficiency"
    assert keys["components.1.labelling.efficiency"] == "second label: labelling efficiency"
    one = key_labels(preset("Diffusion"), preset_keys("Diffusion"))
    assert one["components.0.motion.d_um2_s"] == "D"
    assert preset_keys("") == DEFAULT_KEYS


def test_the_form_shows_each_component_what_it_reads():
    state = component_state(preset("Pores with beads"))
    shown, choices = state.shown, state.choices
    assert shown["components.0.placement"] and not shown["components.1.placement"]
    assert shown["components.1.structure.density_um2"]
    assert not shown["components.0.structure.density_um2"]
    assert not shown["components.0.structure.file"]
    assert state.active["components.1.blinking.bleaching"] is False     # always on
    assert ("beads", "beads") in choices["components.0.placement.same_as"]
    linked = component_state(preset("Two-colour pores")).shown
    assert not linked["components.1.blinking"]           # photophysics as NPC
    assert linked["components.1.placement.same_as"]
    assert not linked["components.1.placement.n"]


def test_choosing_a_built_in_structure_forgets_a_file():
    settings = SimulateSettings()
    settings.components[0].structure.file = "/somewhere/mine.yaml"
    assert Simulate().react("components.0.structure.preset", settings) == \
        {"components.0.structure.file": ""}


def test_a_split_camera_is_written_as_a_recipe_with_two_halves(tmp_path):
    from smappy.io.tiff import open_stack
    settings = preset("Ring and cross (two colours)")
    settings.output = "camera"
    settings.camera.split = True
    settings.n_frames = 20
    settings.camera.path = str(tmp_path / "two.sim.yaml")
    settings.camera.split_sigma_nm = 145.0
    settings.camera.load_truth = True
    result = Simulate().run(Context(), settings)
    source = open_stack(result.written)
    assert source.shape == (200, 100)
    first = next(iter(source.frames(chunk=5)))[1]
    assert first.shape[1:] == (200, 100)
    truth = result.files[0][0]
    assert set(np.unique(truth["channel"])) == {1, 2}
    assert sorted(np.unique(truth["ratio"])) == [0.25, 0.75]
    assert truth.metadata["sigma_nm"] == [130.0, 145.0]
    # the map between the halves is written beside the recipe, as a fitter's file
    from smappy.calibrate.transform import load_transform
    from smappy.simulate.camera import dual_transformation
    written = load_transform(tmp_path / "two_2ct.h5")
    np.testing.assert_allclose(written.transformation, dual_transformation(100))


def test_a_camera_recipe_without_a_file_is_named_after_its_preset(tmp_path, monkeypatch):
    from smappy import config
    monkeypatch.setattr(config, "last_folder", lambda: tmp_path)
    settings = preset("Nuclear pores (Nup96)")
    settings.output = "camera"
    settings.n_frames = 10
    state = Simulate().describe(None, settings)
    assert state.keys["camera.path"] == "simulation file"
    assert state.hints["camera.path"] == str(tmp_path / "Nuclear pores _Nup96_.sim.yaml")
    result = Simulate().run(Context(), settings)
    assert result.written == str(tmp_path / "Nuclear pores _Nup96_.sim.yaml")
    # Fit this is offered whatever is on disk: it writes these settings first
    assert Simulate().describe(None, settings).notice.window == "fit simulation"


def test_two_colours_as_camera_frames_put_the_split_camera_at_the_top_and_say_why():
    settings = preset("Two-colour pores")
    settings.output = "camera"
    state = Simulate().describe(None, settings)
    assert state.keys["camera.split"] == "split camera"
    assert "tick split camera" in state.notice.text
    settings.camera.split = True
    state = Simulate().describe(None, settings)
    assert "camera.split" in state.keys and "two-colour fitter" in state.notice.text
    one = preset("Nuclear pores (Nup96)")
    one.output = "camera"
    assert "camera.split" not in Simulate().describe(None, one).keys


@pytest.mark.slow
def test_two_colour_pores_as_a_split_camera_are_fitted_by_the_two_colour_fitter(tmp_path):
    """The two-colour workflow on simulated pores: the recipe is the fitter's
    file, the transformation written beside it is taken without measuring it,
    and the colours the fit assigns are the components'."""
    from scipy.spatial import cKDTree
    from smappy.plugins.fit import (CameraSettings, DualGaussianFit, OutputSettings)
    settings = preset("Two-colour pores")
    settings.output = "camera"
    settings.camera.split = True
    settings.camera.load_truth = True
    settings.n_frames = 400
    for c in settings.components:
        c.labelling.efficiency = 0.15                   # sparse enough to fit
    settings.camera.path = str(tmp_path / "pores.sim.yaml")
    made = Simulate().run(Context(), settings)
    truth = made.files[0][0]
    fitter = DualGaussianFit()
    fit = fitter.Settings()
    fit.source.path = made.written
    fit.output = OutputSettings(path=str(tmp_path / "pores.hdf5"))
    fit.camera = CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.1)
    updates = fitter.react("source.path", fit)
    assert updates["transform.path"] == str(tmp_path / "pores_2ct.h5")
    fit.transform.path = updates["transform.path"]
    locs = fitter.run(Context(), fit).locs
    got, want = [], []
    for f in np.unique(locs["frame"]):
        m, t = locs["frame"] == f, (truth["frame"] == f) & (truth["photons"] > 1500)
        if not t.any():
            continue
        d, i = cKDTree(np.column_stack([truth["x_nm"][t], truth["y_nm"][t]])).query(
            np.column_stack([locs["x_nm"][m], locs["y_nm"][m]]))
        near = (d < 60) & (locs["channel"][m] > 0)
        got += list(locs["channel"][m][near])
        want += list(truth["channel"][t][i[near]])
    assert len(got) > 100
    agree = np.mean(np.asarray(got) == np.asarray(want))
    assert max(agree, 1 - agree) > 0.9               # which colour is 1 is the ratios' order


def test_single_structure_settings_still_run_as_a_scene_of_one():
    from smappy.simulate import SimulationSettings, simulate
    plain = SimulationSettings(n_frames=200, seed=1)
    made = Simulate().run(Context(), plain).files[0][0]
    np.testing.assert_array_equal(made["x_nm"], simulate(plain)["x_nm"])


# ------------------------------------------------------------------ the GUI

@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_panel_shows_the_presets_key_parameters_which_edit_the_scene(app):
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.session import Session
    panel = PluginPanel(Simulate, Session())
    form = panel.form
    preset_field = form.fields["preset"]
    preset_field.widget.setCurrentIndex(preset_field.widget.findData("Two-colour pores"))
    assert form.keys_section.isVisibleTo(panel)
    assert [c.name for c in form.value().components] == ["NPC", "second label"]
    proxy = form.key_fields["components.1.labelling.efficiency"]
    assert proxy.value() == 0.5
    proxy.set(0.3)
    proxy.changed.emit()
    assert form.value().components[1].labelling.efficiency == 0.3
    assert preset_field.state.text() == "modified"
    # the component's own field shows the edit too
    lst = form.fields["components"]
    lst.list.setCurrentRow(1)
    assert lst.editor.fields["labelling"].fields["efficiency"].value() == 0.3
    # as camera frames, the split camera is ticked from the top
    out = form.fields["output"]
    out.widget.setCurrentIndex(out.widget.findData("camera"))
    panel._describe()
    split = form.key_fields["camera.split"]
    split.set(True)
    split.changed.emit()
    assert form.value().camera.split is True


def test_saved_components_can_be_added_to_a_scene(app):
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.session import Session
    panel = PluginPanel(Simulate, Session())
    lst = panel.form.fields["components"]
    lst._fill_add_menu()
    labels = [a.text() for a in lst.add_menu.actions()]
    assert "beads" in labels and "background" in labels
    next(a for a in lst.add_menu.actions() if a.text() == "beads").trigger()
    beads = panel.form.value().components[-1]
    assert beads.structure.kind == "beads" and beads.blinking.activation == "always"
