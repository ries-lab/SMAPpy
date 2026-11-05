"""The microscope where files are opened, and the manager its entries are made in.

A file opener finds the microscope from the file's own tags and shows it in a line; a
button shows what was found and lets one be chosen by hand; the manager, from the
menu, edits the entries as tables.  The tags are the Ries lab's M2 and M5
(`tests/test_microscope_db.py`).
"""
import json

import numpy as np
import pytest

from smappy import camera_db, microscope_db
from smappy.gui.microscope_editor import (map_text, parse_follows, parse_map, parse_value,
                                          value_text)
from smappy.microscope_db import FROM_NAME, one_line

from test_emu import M2 as EMU_M2, write as write_emu
from test_microscope_db import M2_BEADS, M2_EVOLVE, M2_MOVIE, M5_BEADS

pytest.importorskip("PySide6")


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def config(tmp_path, monkeypatch):
    from smappy import config as settings
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    settings.load(reload=True)
    camera_db.database(reload=True)
    microscope_db.database(reload=True)
    yield tmp_path / "config"
    monkeypatch.undo()
    # a fit's file recognised here is the "last used" microscope (`config`, cached): read the session's
    # again, or a later test of an unrecognised file is given it as a stand-in
    settings.load(reload=True)
    camera_db.database(reload=True)
    microscope_db.database(reload=True)


def a_tiff(path, tags):
    """A stack whose Micro-Manager tags are `tags`."""
    import tifffile
    tifffile.imwrite(path, np.zeros((2, 32, 32), np.uint16),
                     extratags=[(51123, "s", 0, json.dumps(tags), True)])
    return str(path)


# ------------------------------------------------------------ text in a cell

def test_what_a_cell_shows_reads_back_as_it_was():
    assert parse_value(value_text(1.43)) == 1.43
    assert parse_value(value_text([127, 116])) == [127.0, 116.0]
    assert parse_value("astigmatic") == "astigmatic" and parse_value(" ") is None
    states = {"on": "28000", "off": "4000"}
    assert parse_map(map_text(states), values=False) == states
    assert parse_map("in: 127 116; out: 127") == {"in": [127.0, 116.0], "out": 127.0}
    assert parse_follows("filter", FROM_NAME) == {"device": "filter", "values": FROM_NAME}
    assert parse_follows("", "1.43") == 1.43


def test_one_line_says_the_microscope_its_state_and_its_pixel():
    from test_microscope_db import camera_of
    # the switches a parameter follows; the filters are left to the window
    assert one_line(camera_of(M2_BEADS).microscope) == \
        "M2 \u00b7 3DA on \u00b7 iXon 1 \u00b7 127 \u00d7 116 nm"
    assert one_line(camera_of(M2_MOVIE).microscope, short=True) == "M2, 2D"
    assert one_line(None) == "no microscope recognised"


# ------------------------------------------------------------------ the manager

def test_the_manager_shows_an_entry_as_tables_and_reads_it_back(app):
    from smappy.gui.microscope_editor import MicroscopeManager
    manager = MicroscopeManager()
    for name in ("M2", "M5"):
        shipped = microscope_db.database().get(name)
        manager.load(name)
        back = manager.entry().to_dict()
        for part in ("identify", "devices", "parameters", "channels", "camera", "versions",
                     "objective", "filters", "main_channel", "layout", "pupils"):
            assert back.get(part) == shipped.to_dict().get(part), (name, part)
    manager.close()


def test_the_manager_checks_the_entry_against_a_file_as_it_is_edited(app, tmp_path):
    from smappy.gui.microscope_editor import MicroscopeManager
    manager = MicroscopeManager(path=a_tiff(tmp_path / "beads.ome.tif", M2_BEADS))
    manager.load("M2")
    assert "recognised" in manager.file_label.text()
    shown = {manager.result.item(r, 0).text(): manager.result.item(r, 1).text()
             for r in range(manager.result.rowCount())}
    assert shown["device: 3DA"] == "on" and shown["pixel_nm"] == "127 × 116"
    manager.objective["NA"].setText("1.3")                   # checked again at once
    shown = {manager.result.item(r, 0).text(): manager.result.item(r, 1).text()
             for r in range(manager.result.rowCount())}
    assert shown["NA"] == "1.3"
    manager.close()


def test_an_emu_import_starts_an_entry_named_by_its_buttons(app, tmp_path):
    from smappy.gui.microscope_editor import MicroscopeManager
    manager = MicroscopeManager()
    warnings = manager.import_emu_file(write_emu(tmp_path, EMU_M2), three_d="3DA", update=False)
    entry = manager.entry()
    assert "3DA" in entry.devices and entry.devices["3DA"].states == {"on": "28000", "off": "4000"}
    assert entry.parameters["modality"]["device"] == "3DA"
    assert manager.until.text()                              # the configuration's date
    assert warnings and "empty" in manager.message.text()
    manager.close()


def test_saving_writes_the_users_file_and_keeps_the_old_entry_dated(app, config):
    from smappy.gui.microscope_editor import MicroscopeManager
    manager = MicroscopeManager()
    manager.load("M2")
    manager.objective["NA"].setText("1.45")
    manager.until.setText("2026-10-01")
    assert manager.save()
    stored = json.loads((config / "microscopes.json").read_text())["microscopes"]
    assert [m["name"] for m in stored] == ["M2"]
    assert stored[0]["objective"]["NA"] == 1.45
    # the arrangement and the mirror, read back from the channels tab
    assert stored[0]["main_channel"] == "transmitted"
    assert stored[0]["channels"]["reflected"]["mirror"] == "x"
    # the Evolve setup it shipped with, then what this save replaced
    assert stored[0]["versions"][0]["camera"] == "Evolve 1"
    assert stored[0]["versions"][-1] == {"until": "2026-10-01", "objective": {"NA": 1.43}}
    assert microscope_db.database().get("M2").objective["NA"] == 1.45
    assert "(yours)" in manager.entries.currentItem().text()
    manager.close()


def test_an_entry_without_a_name_is_not_saved(app, config):
    from smappy.gui.microscope_editor import MicroscopeManager
    manager = MicroscopeManager()
    manager.show_entry(microscope_db.Microscope(name=""))
    assert manager.save() is None and "name" in manager.message.text()
    manager.close()


# ------------------------------------------------- what a file opener shows

def test_the_found_microscope_shows_its_rows_and_offers_a_choice(app):
    from smappy.gui.microscope_view import FoundMicroscope
    found = microscope_db.database().resolve(M2_BEADS, channels=2)
    window = FoundMicroscope()
    window.show_resolution(found)
    assert window.summary.text().startswith("M2 ·")
    assert window.table.rowCount() == len(found.rows())
    chosen = []
    window.chosen.connect(chosen.append)
    window.choice.setCurrentIndex(window.choice.findData("M5"))
    window.choice.activated.emit(window.choice.currentIndex())
    assert chosen == ["M5"]
    window.show_resolution(None)
    assert "choose one below" in window.summary.text()
    window.close()


def test_psf_calibration_names_the_stacks_microscope_and_takes_one_chosen(app, tmp_path, config):
    from smappy.calibrate.psf_window import PSFCalibrationWindow
    window = PSFCalibrationWindow()
    tags = {k: v for k, v in M2_BEADS.items() if k != "Focus-lock-Serial ID"}
    window.add_paths([a_tiff(tmp_path / "beads_MMStack.ome.tif", tags)])
    form = window.optics_form
    assert form.title == "no microscope recognised" and form.status == "unrecognised"
    assert form.line.isVisibleTo(form)                        # said inside, in orange
    form.fields["microscope"].set("M2")                       # by hand, in the control
    form.fields["microscope"].changed.emit()
    assert form.title.startswith("M2 \u00b7") and window.sections["microscope"].title == form.title
    assert window.optics_camera.pixelsize_um_xy == (0.127, 0.116)
    assert form.fields["NA"]._hint == "1.43"
    window.close()


def test_the_fitter_names_the_microscope_and_takes_one_chosen(app, tmp_path, config):
    from smappy.plugins.fit import GaussianFit
    plugin = GaussianFit()
    settings = plugin.Settings()
    settings.source.path = a_tiff(tmp_path / "beads_MMStack.ome.tif", M2_BEADS)
    state = plugin.describe(None, settings)
    assert state.titles["camera"] == "M2 \u00b7 3DA on \u00b7 iXon 1 \u00b7 127 \u00d7 116 nm"
    assert state.hints["camera"]["status"] == "ok"
    assert state.shown["camera.read_noise_e"] is False          # EM gain on: the fit ignores it
    assert microscope_db.database() and \
        __import__("smappy.config", fromlist=["get"]).get("last_microscope") == "M2"
    # a file nothing recognises: the last microscope used stands in, and says so
    settings.source.path = a_tiff(tmp_path / "m5_MMStack.ome.tif",
                                  {k: v for k, v in M5_BEADS.items() if "iChrome" not in k})
    found = plugin.resolution(settings).microscope
    assert found.name == "M2" and found.fallback
    assert plugin.describe(None, settings).hints["camera"]["status"] == "fallback"
    settings.camera.microscope = "M5"
    assert plugin.resolution(settings).microscope.name == "M5"


def test_the_microscope_control_locks_its_values_and_colours_an_edit(app, tmp_path, config):
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.plugins.fit import GaussianFit
    from smappy.session import Session
    panel = PluginPanel(GaussianFit, Session())
    panel.form.set_values({"source.path": a_tiff(tmp_path / "beads_MMStack.ome.tif", M2_BEADS)})
    panel._describe()
    control = panel.form.fields["camera"]
    section = panel.form.sections["camera"]
    # folded, its title is the line; open, the rows are there, no window
    assert not section.button.isChecked() and section.title.startswith("M2 \u00b7 3DA on")
    conversion = control.fields["conversion"]
    assert conversion.widget.text() == "15.8" and not conversion.auto.isVisibleTo(conversion)
    assert control.fields["camera"].widget.itemText(0) == "auto (iXon 1)"
    assert "found:" in control.labels["conversion"].toolTip()            # where it comes from
    assert not conversion.widget.isEnabled() and control.fields["microscope"].isEnabled()
    assert not control.fields["camera"].widget.isEnabled()               # the camera is a value too
    # unlocked, typing over the found value makes it the user's
    control.lock.setChecked(False)
    conversion.widget.setText("12")
    conversion.widget.textEdited.emit("12")
    conversion.widget.editingFinished.emit()
    assert control.labels["conversion"].styleSheet().startswith("color: #2f7fe0")
    assert panel.form.values()["camera.conversion"] == 12.0              # stored as before
    assert conversion.reset_button.isVisibleTo(conversion)
    # and ↺ gives it back to the file
    conversion.reset_button.click()
    assert panel.form.values()["camera.conversion"] is None and conversion.widget.text() == "15.8"
    assert control.labels["conversion"].styleSheet() == ""
    # emptied, the same
    conversion.widget.setText("9")
    conversion.widget.textEdited.emit("9")
    conversion.widget.setText("")
    conversion.widget.editingFinished.emit()
    assert panel.form.values()["camera.conversion"] is None


def test_a_camera_chosen_in_the_control_is_the_one_the_fit_uses(app, tmp_path, config):
    from smappy.camera_db import database
    from smappy.plugins.fit import GaussianFit
    other = next(n for n in database().names() if n != "iXon 1")
    plugin = GaussianFit()
    settings = plugin.Settings()
    settings.source.path = a_tiff(tmp_path / "beads_MMStack.ome.tif", M2_BEADS)
    assert plugin.describe(None, settings).hints["camera.camera"] == "iXon 1"     # found from the file
    settings.camera.camera = other
    assert plugin.resolution(settings).camera_metadata().camera_name == other


def test_the_menu_opens_the_manager(app, config):
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.session import Session
    session = Session()
    window = ControlWindow(session, RenderWindow(session))
    from PySide6.QtGui import QAction
    assert "Microscope manager..." in [a.text() for a in window.findChildren(QAction)]
    session.window_openers["microscope manager"]()
    assert window._microscope_manager.isVisible()
    window.close()


# ------------------------------------------- a missing value blocks only what reads it

def test_a_gaussian_fit_runs_without_the_optics_a_psf_model_needs(app, tmp_path):
    """No NA, no emission, a lens in no known state: the fit only needs the camera."""
    from smappy.plugins.fit import GaussianFit
    tags = dict(M2_BEADS, **{"Servos-Position1": "abc"})          # 3DA reads nothing
    settings = GaussianFit.Settings()
    settings.source.path = a_tiff(tmp_path / "beads_MMStack.ome.tif", tags)
    resolved = GaussianFit().resolution(settings)
    assert resolved.microscope.state_names()["3DA"] is None
    camera = resolved.camera_metadata()
    camera.require()                                   # what a fit checks: complete
    assert camera.pixelsize_um == 0.127               # the file's, the lens unknown
    from smappy.calibrate.pupil import OpticsSettings
    assert OpticsSettings().missing(camera) == []      # NA and emission from M2 anyway


# ------------------------------------------- in situ's start astigmatism, from the entry

@pytest.mark.parametrize("tags, expected", [(M2_BEADS, 59.119), (M2_MOVIE, 0.0), (M2_EVOLVE, -73.0)])
def test_in_situ_starts_from_the_astigmatism_of_the_microscopes_3d_lens_state(tmp_path, config, tags, expected):
    # the iXon's from the pupil shipped with M2 (its beads of 260829); the Evolve's, mounted turned by 90
    # degrees, the other sign, from the entry -- the iXon's pupil is not the Evolve's; the lens out none
    from smappy.calibrate.insitu_run import InSituModelSettings, start_astigmatism
    from smappy.io.tiff import open_stack
    from smappy.plugins.fit import CameraSettings
    camera = CameraSettings().resolve(open_stack(a_tiff(tmp_path / "x_MMStack.ome.tif", tags)), require=False)
    assert start_astigmatism(InSituModelSettings(), camera)[0] == expected
    assert start_astigmatism(InSituModelSettings(astigmatism_nm=-40.0), camera) == (-40.0, "typed")


def test_the_astigmatism_field_shows_only_when_neither_the_entry_nor_a_start_model_gives_it(app, tmp_path,
                                                                                            config):
    from smappy.calibrate.psf_window import PSFCalibrationWindow
    window = PSFCalibrationWindow()
    window.method_box.setCurrentText("in situ (blinking molecules)")
    window._ask = lambda box: None
    form = window.insitu_form
    window.add_paths([a_tiff(tmp_path / "movie_MMStack.ome.tif", M2_BEADS)])
    field = form.fields["astigmatism_nm"]
    # the shipped pupil's, over the entry's 62
    assert field.widget.text() == "59.119" and form.labels["astigmatism_nm"].styleSheet() == ""
    assert not form.more_section.button.isChecked()                     # under more, closed
    # a start model: the field is not used, so not shown
    window.start_form.set_values({"start_from": str(tmp_path / "m.uipsf.h5")})
    window.start_form.changed.emit()
    assert not form.row_layouts["astigmatism_nm"].isRowVisible(field)
    # no start model and a microscope nobody knows: shown, orange, more opened
    window.start_form.set_values({"start_from": ""})
    window.paths.clear()
    unknown = {k: v for k, v in M5_BEADS.items() if "iChrome" not in k}
    window.optics_form.fields["microscope"].set("")
    window.add_paths([a_tiff(tmp_path / "other_MMStack.ome.tif", unknown)])
    window.optics_camera = type("Camera", (), {"microscope": {}})()   # nothing recognised, no fallback
    window._show_astigmatism()
    assert form.labels["astigmatism_nm"].styleSheet() == "color: #c07000"
    assert form.more_section.button.isChecked() and form.row_layouts["astigmatism_nm"].isRowVisible(field)
    window.close()


# ------------------------------------------- the objective's pupil, kept with the microscope

PUPIL = {"kind": "minimal", "phase_nm": {"5": -2.0, "6": 58.0, "11": 21.0}, "gamma": 0.36, "sigma_nm": [24.0, 24.0],
         "pixel_blur": True, "wavelength_nm": 676.0, "source": "beads_3dcal.h5", "date": "2026-10-23"}


def test_a_kept_pupil_is_found_for_files_taken_in_the_same_lens_state_and_survives_the_manager(
        app, tmp_path, config):
    from smappy.calibrate.insitu_run import InSituModelSettings, start_astigmatism
    from smappy.gui.microscope_editor import MicroscopeManager
    from smappy.io.tiff import open_stack
    from smappy.plugins.fit import CameraSettings

    def scope(tags, name):
        source = open_stack(a_tiff(tmp_path / f"{name}_MMStack.ome.tif", tags))
        return CameraSettings().resolve(source, require=False)
    beads = scope(M2_BEADS, "beads")
    key, path = microscope_db.keep_pupil(beads.microscope, PUPIL)
    assert key == "3DA: on \u00b7 far red" and path == microscope_db.user_file()   # by lens and colour
    assert microscope_db.pupil_of(beads.microscope)["phase_nm"]["11"] == 21.0
    assert microscope_db.pupil_of(scope(M2_MOVIE, "movie").microscope) is None      # the lens out
    # in situ's start astigmatism: the pupil's, over the entry's +62
    assert start_astigmatism(InSituModelSettings(), beads)[0] == 58.0
    # no version for it: the objective's pupil holds for the files taken before it was kept too
    entry = microscope_db.database().get("M2")
    assert all("pupils" not in v for v in entry.versions)
    assert microscope_db.pupil_of(scope(M2_BEADS, "again").microscope)["phase_nm"]["11"] == 21.0
    # a red file with the lens in does not take the far-red model; a far-red one does
    assert entry.pupil_for({"3DA": "on"}, 600.0) is None
    assert entry.pupil_for({"3DA": "on"}, 700.0)["gamma"] == 0.36
    manager = MicroscopeManager()
    manager.show_entry(entry)
    assert manager.pupil_table.item(0, 0).text() == "3DA: on \u00b7 far red"
    assert manager.entry().pupils == entry.pupils
    manager.close()


def test_gamma_is_the_microscope_s_whatever_its_3d_lens_and_from_its_nearest_colour(app, tmp_path, config):
    # the user's decision (November 2026): γ is the objective's apodisation, so a movie with the lens out takes
    # it from a pupil kept with the lens in, and of two colours the nearer; a typed ``gamma`` comes first
    from smappy.io.tiff import open_stack
    from smappy.plugins.fit import CameraSettings

    def scope(tags, name):
        return CameraSettings().resolve(open_stack(a_tiff(tmp_path / f"{name}_MMStack.ome.tif", tags)),
                                        require=False).microscope
    beads, movie = scope(M2_BEADS, "beads"), scope(M2_MOVIE, "movie")
    assert microscope_db.gamma_of(movie, 676.0) == (0.2017, None, "M2's kept pupil at 700 nm")    # shipped
    microscope_db.keep_pupil(beads, PUPIL)
    assert microscope_db.pupil_of(movie) is None                       # its pupil follows the lens, γ not
    assert microscope_db.gamma_of(movie, 680.0) == (0.36, None, "M2's kept pupil at 676 nm")
    entry = microscope_db.database().get("M2")
    entry.pupils["other"] = dict(PUPIL, gamma=0.2, na=1.41, wavelength_nm=580.0)    # its NA learnt: held with it
    assert microscope_db.gamma_of(movie, 600.0)[:2] == (0.2, 1.41) and microscope_db.gamma_of(movie, 670.0)[0] == 0.36
    assert microscope_db.gamma_of(dict(movie, values={"gamma": 0.1}), 670.0) == (0.1, None, "M2's entry")


def test_a_pupil_is_kept_only_for_a_recognised_microscope_in_its_current_setup(config):
    with pytest.raises(ValueError, match="no microscope recognised"):
        microscope_db.keep_pupil({}, PUPIL)
    with pytest.raises(ValueError, match="setup until 2026-07-15"):
        microscope_db.keep_pupil({"name": "M2", "version": "2026-07-15", "states": {}}, PUPIL)


def test_a_pupil_without_its_kind_is_neither_kept_nor_found(app, tmp_path, config):
    # the Zernike-magnitude dicts kept before pupils described themselves (November 2026): no legacy, the
    # user keeps the pupil again from a new bead model
    import copy

    from smappy.io.tiff import open_stack
    from smappy.plugins.fit import CameraSettings
    beads = CameraSettings().resolve(open_stack(a_tiff(tmp_path / "beads_MMStack.ome.tif", M2_BEADS)), require=False)
    old = {k: v for k, v in PUPIL.items() if k not in ("kind", "gamma")} | {"magnitude": {"11": 0.12}}
    with pytest.raises(ValueError, match="kind"):
        microscope_db.keep_pupil(beads.microscope, old)
    entry = copy.deepcopy(microscope_db.database().get("M2"))           # not the cached database's
    entry.pupils["3DA: on"] = old
    assert entry.pupil_for({"3DA": "on"}) is None


# ------------------------------------------------------------- the third round's tabs

def test_the_channels_are_drawn_on_the_chip_where_the_fitters_cut_them():
    from matplotlib.figure import Figure
    from smappy.gui.microscope_editor import draw_channels
    m2 = microscope_db.database().get("M2")
    ax = Figure().add_subplot()
    draw_channels(ax, m2, "iXon 1", (100, 0, 64, 32), np.zeros((32, 64)))
    (main, other) = ax.patches
    # the transmitted arm on the right half of the frame, the main one; its corner on the chip
    assert (main.get_x(), main.get_width()) == (131.5, 32) and other.get_x() == 99.5
    labels = [t.get_text() for t in ax.texts]
    assert labels == ["transmitted (main)", "reflected, mirrored x"]
    # a 4Pi's regions need no file: they are on the chip already
    ax = Figure().add_subplot()
    draw_channels(ax, microscope_db.database().get("4Pi-SMS example"))
    assert [p.get_x() + 0.5 for p in ax.patches] == [93, 469, 1409, 1769]
    # a split without a file says what it needs instead
    ax = Figure().add_subplot()
    draw_channels(ax, m2, "iXon 1")
    assert not ax.patches and "check with a file" in ax.texts[0].get_text()


def test_the_manager_draws_the_split_on_the_files_frame_as_it_is_edited(app, tmp_path):
    from smappy.gui.microscope_editor import MicroscopeManager
    manager = MicroscopeManager(path=a_tiff(tmp_path / "beads.ome.tif", M2_BEADS))
    manager.load("M2")
    assert len(manager.channels_figure.axes[0].patches) == 2
    assert not manager.layout_choice.isEnabled()             # the arms' sides say the split
    table = manager.tables["channels"]
    table.item(0, 4).setText("10 0 22 32")                   # the arms placed: drawn again there
    table.item(1, 4).setText("0 0 10 32")
    main = manager.channels_figure.axes[0].patches[0]
    assert (main.get_x(), main.get_width()) == (9.5, 22)
    assert manager.entry().channels[0].region == [10, 0, 22, 32]
    table.item(0, 2).setText("left")                         # a side its region contradicts: said
    assert "by its region" in manager.channels_figure.axes[0].texts[0].get_text()
    manager.close()


def test_a_versions_day_is_moved_and_a_version_removed_where_it_is_saved(app, config):
    from smappy.gui.microscope_editor import MicroscopeManager
    manager = MicroscopeManager()
    manager.load("M2")
    assert manager.version_table.item(0, 0).text() == "2026-07-15"
    assert "camera" in manager.version_table.item(0, 1).text()
    manager.version_table.item(0, 0).setText("2026-07-20")
    assert manager.save()
    stored = json.loads((config / "microscopes.json").read_text())["microscopes"][0]
    assert [v["until"] for v in stored["versions"]] == ["2026-07-20"]
    manager.version_table.selectRow(0)
    manager.remove_version()
    assert manager.save()
    assert microscope_db.database().get("M2").versions == []
    manager.version_table.setRowCount(0)
    manager.close()


def test_a_day_that_is_not_one_is_refused_before_saving(app, config):
    from smappy.gui.microscope_editor import MicroscopeManager
    manager = MicroscopeManager()
    manager.load("M2")
    manager.version_table.item(0, 0).setText("July")
    assert manager.save() is None and "not a day" in manager.message.text()
    manager.close()


def test_a_kept_pupil_is_drawn_and_can_be_dropped(app):
    from smappy.gui.microscope_editor import MicroscopeManager, pupil_phase
    pupil = {"kind": "minimal", "phase_nm": {"6": 50.0}, "gamma": 0.3}
    # Noll 6 is sqrt(6) rho^2 cos(2 phi): 50 nm of it is 122 nm at the rim along x, -122 along y
    phase = pupil_phase(pupil, size=65)
    assert phase[32, 64] == pytest.approx(50 * 6 ** 0.5) and phase[64, 32] == pytest.approx(-50 * 6 ** 0.5)
    assert np.isnan(phase[0, 0])
    entry = microscope_db.Microscope(name="test", pupils={"3DA: on · far red": pupil})
    manager = MicroscopeManager()
    manager.show_entry(entry)
    assert "Z6 +50" in manager.pupil_figure.axes[0].get_title()
    manager.pupil_table.selectRow(0)
    manager.remove_pupil()
    assert manager.entry().pupils == {}
    manager.close()


def test_the_filters_tab_types_a_band_and_names_what_no_name_gives(app):
    from PySide6.QtWidgets import QTableWidgetItem
    from smappy.gui.microscope_editor import MicroscopeManager, unnamed_filters
    entry = microscope_db.Microscope.from_dict({
        "name": "test",
        "devices": {"wheel": {"tag": "Wheel-Label", "states": {"676/37": "1", "665LP": "2"}}},
        "channels": {"transmitted": {"filter": "wheel"}}})
    assert unnamed_filters(entry) == ["665LP"]
    manager = MicroscopeManager()
    manager.show_entry(entry)
    assert "665LP" in manager.unnamed.text()
    table = manager.tables["filters"]
    table.insertRow(0)
    for column, text in enumerate(["665LP", "", "665", "720"]):
        table.setItem(0, column, QTableWidgetItem(text))
    assert manager.entry().filters == {"665LP": {"band_nm": [665.0, 720.0]}}
    assert manager.unnamed.text() == ""
    manager.close()
