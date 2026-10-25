"""Settings that hold a list of parts -- a simulated scene's components --
a field shown beside another, and presets kept as files: the settings
framework the redesigned simulator is built on (docs/simulation-redesign.md)."""
from dataclasses import dataclass
from typing import List

import pytest

from smappy import presets
from smappy.plugins import param, param_specs, settings_from, settings_values


@dataclass
class Photo:
    photons: float = param(5000.0, label="photons", min=0)
    photons_std: float = param(2500.0, label="±", beside=True, min=0)
    on_time: float = param(1.5, label="on time")


@dataclass
class Comp:
    name: str = param("component", label="name")
    on: bool = param(True, label="on")
    density: float = param(1.0, label="density", min=0)
    photo: Photo = param(default_factory=Photo, label="photophysics")


def default_components():
    return [Comp("NPC"), Comp("background", density=0.1)]


@dataclass
class Scene:
    frames: int = param(1000, label="frames")
    components: List[Comp] = param(default_factory=default_components,
                                   label="components")


def test_a_list_of_parts_is_one_value_of_flat_maps_and_reads_back():
    values = settings_values(Scene())
    assert values["components"][1] == {"name": "background", "on": True, "density": 0.1,
                                       "photo.photons": 5000.0, "photo.photons_std": 2500.0,
                                       "photo.on_time": 1.5}
    assert settings_from(Scene, values) == Scene()
    # nested maps too, as a log writes them; what an item leaves out is its default
    scene = settings_from(Scene, {"components": [{"name": "z", "photo": {"photons": 7}}]})
    assert scene.components == [Comp("z", photo=Photo(photons=7))]
    # an absent list is the default, a fresh one each time
    assert settings_from(Scene, {}).components == default_components()
    assert settings_from(Scene, {}).components is not settings_from(Scene, {}).components
    assert param_specs(Scene)["components"].item_type is Comp


def test_a_batch_checks_every_item_of_a_list():
    from smappy.batch import Problem, _check_values, _flat_specs
    problems: List[Problem] = []
    _check_values({"components": [{"density": -1}, {"photo.photons": "many"}, 3]},
                  _flat_specs(param_specs(Scene)), "step", problems)
    text = " | ".join(str(p) for p in problems)
    assert "components.0" in text and "below the minimum" in text
    assert "components.1" in text and "not a number" in text
    assert "components.2: not a mapping" in text


def test_a_page_explains_an_items_settings_once_for_every_item():
    from smappy.docs import leaves
    names = [dotted for dotted, _ in leaves(param_specs(Scene))]
    assert names == ["frames", "components", "components.name", "components.on",
                     "components.density", "components.photo.photons",
                     "components.photo.photons_std", "components.photo.on_time"]


def test_presets_are_files_the_users_first_and_only_the_users_deleted(tmp_path):
    mine, shipped = tmp_path / "mine", tmp_path / "shipped"
    shipped.mkdir()
    (shipped / "npc.yaml").write_text("frames: 10\ncomponents: []\n")
    (shipped / "ring.yaml").write_text("photo:\n  photons: 3\n")
    folders = [mine, shipped]
    assert presets.names(folders) == ["npc", "ring"]
    assert presets.read(folders, "ring") == {"photo.photons": 3}
    presets.write(folders, "npc", {"frames": 99})        # mine wins over the shipped
    assert presets.read(folders, "npc") == {"frames": 99}
    assert presets.is_users(folders, "npc") and not presets.is_users(folders, "ring")
    with pytest.raises(ValueError):
        presets.delete(folders, "ring")
    presets.delete(folders, "npc")
    assert presets.read(folders, "npc") == {"frames": 10, "components": []}
    with pytest.raises(ValueError):
        presets.write(folders, "a/b", {})


# ------------------------------------------------------------------ the form

@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_form_edits_the_selected_item_and_names_the_edit(app):
    from smappy.gui.params import SettingsForm
    form = SettingsForm(Scene)
    edits = []
    form.field_changed.connect(edits.append)
    lst = form.fields["components"]
    assert [lst.list.item(i).text() for i in range(lst.list.count())] == ["NPC", "background"]
    lst.list.setCurrentRow(1)
    density = lst.editor.fields["density"]
    assert density.value() == 0.1
    density.set(0.25)
    density.changed.emit()
    assert edits == ["components.1.density"]
    assert form.value().components[1].density == 0.25
    assert form.value().components[0].density == 1.0        # the other untouched


def test_the_list_adds_duplicates_moves_ticks_and_removes(app):
    from PySide6.QtCore import Qt
    from smappy.gui.params import SettingsForm
    form = SettingsForm(Scene)
    lst = form.fields["components"]
    lst.list.setCurrentRow(0)
    lst._duplicate()
    assert [c.name for c in form.value().components] == ["NPC", "NPC copy", "background"]
    lst._move(1)
    assert [c.name for c in form.value().components] == ["NPC", "background", "NPC copy"]
    lst.list.item(0).setCheckState(Qt.Unchecked)
    assert form.value().components[0].on is False
    lst.add({"name": "beads", "density": 0.01})
    assert form.value().components[-1] == Comp("beads", density=0.01)
    lst.list.setCurrentRow(0)
    lst._remove()
    assert [c.name for c in form.value().components] == ["background", "NPC copy", "beads"]


def test_the_plugin_reaches_an_item_by_its_index(app):
    """`FormState` and `react` name an item's field ``components.<i>.<name>``;
    what is said about an item not on show is shown when it is selected."""
    from smappy.gui.params import SettingsForm
    form = SettingsForm(Scene)
    lst = form.fields["components"]
    form.set_values({"components.1.photo.photons": 999.0})
    assert form.value().components[1].photo.photons == 999.0
    form.set_shown({"components.1.photo.photons_std": False})
    lst.list.setCurrentRow(1)
    label, widget = lst.editor.fields["photo"].beside["photons_std"]
    assert widget.isHidden() and label.isHidden()
    lst.list.setCurrentRow(0)
    assert not lst.editor.fields["photo"].beside["photons_std"][1].isHidden()


def test_a_form_with_a_list_saves_and_restores(app):
    from smappy.gui.params import SettingsForm
    form = SettingsForm(Scene)
    form.set_values({"components.0.name": "pores"})
    values = form.values()
    assert values["components"][0]["name"] == "pores"
    again = SettingsForm(Scene)
    assert again.restore(values) == []
    assert again.value() == form.value()


def test_a_field_beside_another_shares_its_row(app):
    from smappy.gui.params import SettingsForm
    form = SettingsForm(Photo)
    photons, std = form.fields["photons"], form.fields["photons_std"]
    form.resize(360, 200)
    form.show()
    app.processEvents()
    assert std.geometry().top() == photons.geometry().top() or \
        std.parent() is photons.parent()
    assert form.row_widgets["photons_std"] is form.row_widgets["photons"]
    form.close()


def test_a_preset_field_saves_the_whole_form_and_lists_it(app, tmp_path):
    from smappy.gui.params import SettingsForm

    @dataclass
    class WithPreset:
        preset: str = param("", label="preset", kind="preset",
                            store=lambda: [tmp_path / "presets"])
        frames: int = param(1000, label="frames")

    form = SettingsForm(WithPreset)
    form.set_values({"frames": 42})
    field = form.fields["preset"]
    form._save_preset("short", field)
    assert presets.read([tmp_path / "presets"], "short") == {"frames": 42}
    assert field.value() == "short" and field.delete_button.isEnabled()
