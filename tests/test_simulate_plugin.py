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
    settings.output = "split"
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
