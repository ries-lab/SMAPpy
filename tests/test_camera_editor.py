"""Adding a camera to the database from a file it took.

A camera that is not in the database has its conversion typed into every
fit.  The editor writes the entry once, starting from the file just opened,
and the fitter says -- while the file is chosen, not when the run fails --
that the camera is unknown and where it is added.
"""
import json

import numpy as np
import pytest

from smappy import camera_db
from smappy.camera_db import (Camera, CameraDatabase, Parameter, State, draft,
                              problems, save_camera, with_mode)

PRIME = {"Core-Camera": "Prime95B", "Prime95B-SerialNumber": "XY123",
         "Prime95B-ReadoutRate": "200MHz 16bit", "Prime95B-Gain": "HDR",
         "Prime95B-Port": "Normal", "Prime95B-Exposure": "20",
         "ROI": "0-0-256-256", "PixelSizeUm": "0.11"}

EVOLVE = {"Core-Camera": "Evolve512", "Evolve512-SerialNumber": "A14G150010",
          "Evolve512-Port": "Multiplication Gain",
          "Evolve512-ReadoutRate": "10MHz 16bit", "Evolve512-Gain": "1",
          "Evolve512-MultiplierGain": "300", "ROI": "0-0-512-512"}

IXON = {"Core-Camera": "Andor", "Andor-Camera": "| iXon Ultra | DU897_BV | 8172 |",
        "Andor-Output_Amplifier": "Electron Multiplying",
        "Andor-Pre-Amp-Gain": "Gain 1", "Andor-ReadoutMode": "10.000 MHz",
        "Andor-Gain": "300", "ROI": "0-0-48-60"}


@pytest.fixture
def config(tmp_path, monkeypatch):
    """A user database of the test's own, and the shipped one afterwards."""
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path))
    camera_db.database(reload=True)
    yield tmp_path
    monkeypatch.undo()
    camera_db.database(reload=True)


# --------------------------------------------------------------- the draft

@pytest.mark.parametrize("tags, shipped", [(EVOLVE, "M2 Evolve"), (IXON, "M1 Andor")])
def test_a_draft_from_a_files_tags_is_the_entry_written_by_hand(tags, shipped):
    """The two shipped EMCCDs were entered by hand from SMAP's file; drafted
    from their tags alone, they come out the same but for the numbers."""
    entry = camera_db.database().get(shipped)
    made = draft(tags)
    assert made.prefix == entry.prefix
    assert made.identify.tag == entry.identify.tag
    assert made.identify.argument == entry.identify.argument
    for name in ("em_on", "emgain", "conversion", "offset", "roi"):
        assert made.parameters[name].to_dict() == entry.parameters[name].to_dict(), name
    assert set(made.states[0].match) == set(entry.states[0].match)
    # what the tags never say is left for the user, not guessed
    resolved = made.resolve(tags).values
    assert resolved["conversion"] is None and resolved["offset"] is None
    assert resolved["emgain"] == 300.0 and resolved["em_on"] is True


def test_a_draft_takes_a_calibrated_pixel_size_and_no_em_gain_from_an_scmos():
    made = draft(PRIME)
    values = made.resolve(PRIME).values
    assert values["pixelsize_um"] == 0.11
    assert values["em_on"] is False and values["emgain"] == 1.0
    assert values["exposure_ms"] == 20.0
    # the Photometrics Gain is a pre-amp gain: part of the mode, not the EM gain
    assert "{prefix}-Gain" in made.states[0].match


def test_a_file_with_no_device_drafts_an_entry_chosen_by_hand():
    made = draft({"PixelSizeUm": "0.1"})
    assert made.identify is None and made.prefix == ""
    assert made.parameters["conversion"].to_dict() == {"fixed": None}
    assert any("by hand" in p for p in problems(made, {"PixelSizeUm": "0.1"}))


# ---------------------------------------------------------------- the mode

def test_a_new_readout_mode_is_added_and_the_others_are_kept():
    evolve = camera_db.database().get("M2 Evolve")
    tags = dict(EVOLVE, **{"Evolve512-ReadoutRate": "100MHz 16bit"})
    assert evolve.state_for(tags) is None
    added = with_mode(evolve, tags, camera_db.mode_tags(evolve),
                      {"conversion": 3.1, "offset": 120.0})
    assert len(added.states) == len(evolve.states) + 1
    assert added.resolve(tags).values["conversion"] == 3.1
    assert added.resolve(EVOLVE).values["conversion"] == 6.7     # untouched
    assert evolve.state_for(tags) is None                        # a copy


def test_the_mode_a_file_already_matches_is_corrected_in_place():
    evolve = camera_db.database().get("M2 Evolve")
    fixed = with_mode(evolve, EVOLVE, camera_db.mode_tags(evolve),
                      {"conversion": 7.0, "offset": 400.0})
    assert len(fixed.states) == len(evolve.states)
    assert fixed.resolve(EVOLVE).values["conversion"] == 7.0


def test_a_mode_recognised_by_no_tag_is_not_added():
    camera = draft(PRIME)
    assert with_mode(camera, PRIME, [], {"conversion": 1.0}).states == camera.states


# -------------------------------------------------------------- saving

def test_a_saved_camera_is_recognised_by_the_next_file(config):
    camera = with_mode(draft(PRIME), PRIME, ["Prime95B-ReadoutRate"],
                       {"conversion": 1.2, "offset": 100.0})
    assert camera_db.database().identify(PRIME) is None
    path = save_camera(camera)
    assert path == config / "cameras.json"
    found = camera_db.database().resolve(PRIME)
    assert found.camera_name == "Prime95B" and found.missing == []
    assert found.values["conversion"] == 1.2


def test_saving_again_replaces_the_camera_by_name_and_keeps_the_others(config):
    save_camera(Camera(name="Other", parameters={"offset": Parameter(fixed=5.0)}))
    save_camera(Camera(name="Mine", parameters={"offset": Parameter(fixed=1.0)}))
    save_camera(Camera(name="Mine", parameters={"offset": Parameter(fixed=2.0)}))
    stored = json.loads((config / "cameras.json").read_text())["cameras"]
    assert [c["name"] for c in stored] == ["Other", "Mine"]
    assert stored[1]["parameters"]["offset"] == {"fixed": 2.0}


def test_a_shipped_camera_saved_by_the_user_takes_its_place(config):
    evolve = camera_db.database().get("M2 Evolve")
    tags = dict(EVOLVE, **{"Evolve512-ReadoutRate": "100MHz 16bit"})
    save_camera(with_mode(evolve, tags, camera_db.mode_tags(evolve),
                          {"conversion": 3.1, "offset": 120.0}))
    known = camera_db.database()
    assert known.names().count("M2 Evolve") == 1
    assert known.resolve(tags).values["conversion"] == 3.1
    assert known.resolve(EVOLVE).values["conversion"] == 6.7
    # the shipped file is not what was written
    shipped = CameraDatabase.load(camera_db.SHIPPED).get("M2 Evolve")
    assert len(shipped.states) == len(evolve.states)


def test_the_revision_moves_when_the_database_is_read_again(config):
    before = camera_db.revision()
    save_camera(Camera(name="Mine"))
    assert camera_db.revision() > before


def test_problems_say_what_the_fit_will_still_miss():
    camera = draft(PRIME)
    said = problems(camera, PRIME)
    assert any("conversion, offset" in p for p in said)
    named = Camera(name="", identify=Parameter(tag="X", argument="1"))
    assert "the camera needs a name" in problems(named, {"X": "2"})
    assert any("would not recognise" in p for p in problems(named, {"X": "2"}))


# ------------------------------------------------------------ the fitter

def a_tiff(path, tags):
    import tifffile
    tifffile.imwrite(path, np.zeros((2, 32, 32), np.uint16),
                     extratags=[(51123, "s", 0, json.dumps(tags), True)])
    return str(path)


def test_the_fitter_says_an_unknown_camera_is_unknown_and_where_it_is_added(
        config, tmp_path):
    from smappy.plugins.fit import GaussianFit
    plugin = GaussianFit()
    settings = plugin.Settings()
    assert plugin.describe(None, settings).notice is None                  # no file yet
    settings.source.path = a_tiff(tmp_path / "prime_MMStack.ome.tif", PRIME)
    notice = plugin.describe(None, settings).notice
    assert "Prime95B" in notice.text and "conversion" in notice.text
    assert notice.window == "camera editor" and notice.action == "Add camera..."

    save_camera(with_mode(draft(PRIME), PRIME, ["Prime95B-ReadoutRate"],
                          {"conversion": 1.2, "offset": 100.0}))
    assert plugin.describe(None, settings).notice is None                  # the cache noticed
    assert plugin.resolution(settings).values["conversion"] == 1.2
    assert ("Prime95B", "Prime95B") in plugin.describe(None, settings).choices["camera.camera"]


def test_the_fitter_offers_a_readout_mode_the_database_lacks(config, tmp_path):
    from smappy.plugins.fit import GaussianFit
    plugin = GaussianFit()
    settings = plugin.Settings()
    settings.source.path = a_tiff(tmp_path / "evolve_MMStack.ome.tif",
                                  dict(EVOLVE, **{"Evolve512-ReadoutRate": "1MHz"}))
    notice = plugin.describe(None, settings).notice
    assert notice.action == "Add readout mode..."
    assert "M2 Evolve" in notice.text and "conversion" in notice.text

    settings.source.path = a_tiff(tmp_path / "known_MMStack.ome.tif", EVOLVE)
    assert plugin.describe(None, settings).notice is None


# ------------------------------------------------------------------ the GUI

@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def rows_of(table):
    return {table.item(r, 0).text(): table.item(r, 1).text()
            for r in range(table.rowCount())}


def test_the_editor_starts_from_the_file_and_saves_what_is_typed(app, config):
    from smappy.gui.camera_editor import CameraEditor
    editor = CameraEditor(PRIME)
    assert editor.name.text() == "Prime95B"
    assert editor.identify_tag.currentText() == "Prime95B-SerialNumber"
    assert editor.identify_value.text() == "XY123"
    assert "Prime95B-ReadoutRate" in editor.ticked()
    assert "conversion, offset" in editor.problems.text()

    editor.rows["conversion"].value.setText("1.2")
    editor.rows["offset"].value.setText("100")
    editor.changed()
    assert rows_of(editor.table)["conversion"] == "1.2"
    assert "unknown" not in editor.problems.text()

    names = []
    editor.saved.connect(names.append)
    assert editor.save() == "Prime95B" and names == ["Prime95B"]
    found = camera_db.database().resolve(PRIME)
    assert found.values["conversion"] == 1.2 and found.values["offset"] == 100.0
    assert found.state is not None


def test_the_editor_gives_a_known_camera_the_mode_it_lacked(app, config):
    from smappy.gui.camera_editor import CameraEditor
    tags = dict(EVOLVE, **{"Evolve512-ReadoutRate": "1MHz"})
    evolve = camera_db.database().get("M2 Evolve")
    editor = CameraEditor(tags, evolve)
    assert editor.start_from.currentData() == "M2 Evolve"
    assert set(editor.ticked()) == {"Evolve512-Gain", "Evolve512-Port",
                                    "Evolve512-ReadoutRate"}
    assert "6 other readout modes" in editor.modes_note.text()
    assert "shipped camera" in editor.problems.text()
    editor.rows["conversion"].value.setText("12.5")
    editor.rows["offset"].value.setText("400")
    editor.save()
    known = camera_db.database().get("M2 Evolve")
    assert len(known.states) == len(evolve.states) + 1
    assert known.resolve(tags).values["conversion"] == 12.5


def test_a_tag_read_by_comparison_keeps_what_it_compares_with(app, config):
    from smappy.gui.camera_editor import CameraEditor
    editor = CameraEditor(PRIME)
    camera = editor.camera()
    assert camera.parameters["em_on"].to_dict() == {
        "tag": "{prefix}-Port", "read": "not_equals", "value": "Normal"}
    editor.rows["pixelsize_um"].value.setText("0.102, 0.103")
    assert editor.camera().parameters["pixelsize_um"].fixed == [0.102, 0.103]


def test_the_editor_opens_on_no_file(app, config):
    from smappy.gui.camera_editor import CameraEditor
    editor = CameraEditor()
    assert "no file" in editor.summary.text()
    assert editor.table.rowCount() == 0


def test_the_panel_shows_the_notice_and_its_button_opens_the_editor(
        app, config, tmp_path):
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.plugins.fit import GaussianFit
    from smappy.session import Session

    session = Session()
    opened = []
    session.window_openers["camera editor"] = opened.append
    panel = PluginPanel(GaussianFit, session)
    assert panel.notice.isHidden()
    panel.form.set_values({"source.path": a_tiff(tmp_path / "p_MMStack.ome.tif",
                                                 PRIME)})
    panel._react("source.path")
    assert not panel.notice.isHidden() and "Prime95B" in panel.notice.text()
    assert not panel.notice_button.isHidden()
    panel.notice_button.click()
    assert opened and opened[0].source.path.endswith("p_MMStack.ome.tif")

    save_camera(with_mode(draft(PRIME), PRIME, ["Prime95B-ReadoutRate"],
                          {"conversion": 1.2, "offset": 100.0}))
    panel.refresh_derived()
    assert panel.notice.isHidden()


def test_a_fit_in_a_chain_says_its_camera_is_unknown_and_opens_the_editor_on_its_own_settings(
        app, config, tmp_path):
    """A chain forwards its steps' form state with one hook, the notice
    included, and the button hands the window the fit's settings rather
    than the chain's, which a camera editor could not read."""
    from smappy.chain import ChainSpec, Step, chain_class
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.plugins.fit import GaussianFit
    from smappy.session import Session

    cls = chain_class(ChainSpec(steps=[Step(plugin="Localize/Gaussian 2D", label="fit")]))
    key = cls.steps[0][0]
    session = Session()
    opened = []
    session.window_openers["camera editor"] = opened.append
    panel = PluginPanel(cls, session)
    assert panel.notice.isHidden()
    path = a_tiff(tmp_path / "p_MMStack.ome.tif", PRIME)
    panel.form.set_values({f"{key}.source.path": path})
    panel._react(f"{key}.source.path")
    assert not panel.notice.isHidden()
    assert panel.notice.text().startswith("fit: ") and "Prime95B" in panel.notice.text()
    panel.notice_button.click()
    assert isinstance(opened[0], GaussianFit.Settings)
    assert opened[0].source.path == path


def test_without_an_editor_the_notice_has_no_button(app, config, tmp_path):
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.plugins.fit import GaussianFit
    from smappy.session import Session

    panel = PluginPanel(GaussianFit, Session())
    panel.form.set_values({"source.path": a_tiff(tmp_path / "p_MMStack.ome.tif",
                                                 PRIME)})
    panel._react("source.path")
    assert not panel.notice.isHidden() and panel.notice_button.isHidden()


def test_the_parameter_view_offers_the_editor_for_an_unknown_camera(
        app, config, tmp_path):
    from smappy.gui.camera_view import CameraParameters
    from smappy.plugins.fit import GaussianFit

    plugin = GaussianFit()
    settings = plugin.Settings()
    settings.source.path = a_tiff(tmp_path / "p_MMStack.ome.tif", PRIME)
    view = CameraParameters()
    asked = []
    view.open_editor = lambda metadata, camera: asked.append((metadata, camera))
    view.show_for(plugin, settings)
    assert not view.add_button.isHidden()
    view.add_button.click()
    assert asked[0][0]["Prime95B-SerialNumber"] == "XY123" and asked[0][1] is None

    settings.source.path = a_tiff(tmp_path / "e_MMStack.ome.tif", EVOLVE)
    view.show_for(plugin, settings)
    assert view.add_button.isHidden()                 # known, mode and all


def test_the_main_window_opens_the_editor_on_the_fitters_file(
        app, config, tmp_path):
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.session import Session
    from smappy.plugins.fit import GaussianFit

    session = Session()
    window = ControlWindow(session, RenderWindow(session))
    assert "camera editor" in session.window_openers
    settings = GaussianFit.Settings()
    settings.source.path = a_tiff(tmp_path / "p_MMStack.ome.tif", PRIME)
    session.window_openers["camera editor"](settings)
    editor = window._camera_editor
    assert editor.metadata["Prime95B-SerialNumber"] == "XY123"
    editor.rows["conversion"].value.setText("1.2")
    editor.rows["offset"].value.setText("100")
    editor.save()
    assert camera_db.database().identify(PRIME).name == "Prime95B"
    window.close()


def test_choosing_a_file_shows_its_camera_but_leaves_the_fields_on_auto(
        app, config, tmp_path):
    """The file's numbers are shown, greyed, and stay the file's: copied in,
    they overrode a later *camera* choice and outlived the file."""
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.plugins.fit import GaussianFit
    from smappy.session import Session

    panel = PluginPanel(GaussianFit, Session())
    panel.form.set_values({"source.path": a_tiff(tmp_path / "p_MMStack.ome.tif",
                                                 PRIME)})
    panel._react("source.path")
    panel.refresh_derived()
    assert panel.form.value().camera.pixelsize_um is None          # still auto
    shown = panel.form.fields["camera"].fields["pixelsize_um"]
    assert shown.widget.text() == "0.11"                           # the hint

    other = tmp_path / "bare.tif"
    import tifffile
    tifffile.imwrite(other, np.zeros((2, 32, 32), np.uint16))
    panel.form.set_values({"source.path": str(other)})
    panel._react("source.path")
    panel.refresh_derived()
    assert panel.form.value().camera.pixelsize_um is None          # nothing inherited
