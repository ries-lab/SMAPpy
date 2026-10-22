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
from test_microscope_db import M2_BEADS, M2_MOVIE, M5_BEADS

pytest.importorskip("PySide6")


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    camera_db.database(reload=True)
    microscope_db.database(reload=True)
    yield tmp_path / "config"
    monkeypatch.undo()
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
        for part in ("identify", "devices", "parameters", "channels", "camera", "versions"):
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
    table = manager.tables["parameters"]
    row = next(r for r in range(table.rowCount()) if table.item(r, 0).text() == "NA")
    table.item(row, 2).setText("1.3")                        # checked again at once
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
    table = manager.tables["parameters"]
    row = next(r for r in range(table.rowCount()) if table.item(r, 0).text() == "NA")
    table.item(row, 2).setText("1.45")
    manager.until.setText("2026-10-01")
    assert manager.save()
    stored = json.loads((config / "microscopes.json").read_text())["microscopes"]
    assert [m["name"] for m in stored] == ["M2"]
    assert stored[0]["parameters"]["NA"] == 1.45
    # the Evolve setup it shipped with, then what this save replaced
    assert stored[0]["versions"][0]["camera"] == "Evolve 1"
    assert stored[0]["versions"][-1] == {"until": "2026-10-01", "parameters": {"NA": 1.43}}
    assert microscope_db.database().get("M2").parameters["NA"] == 1.45
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
    assert form.line.text() == "no microscope recognised"
    form.fields["microscope"].set("M2")                       # by hand, in the control
    form.fields["microscope"].changed.emit()
    assert form.line.text().startswith("M2 \u00b7")
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
    assert control.line.text().startswith("M2 \u00b7 3DA on")
    assert "15.8 e-/ADU" in control.values_line.text()
    assert not control.fields["conversion"].isEnabled() and control.fields["microscope"].isEnabled()
    control.lock.setChecked(False)
    control.fields["conversion"].set(12.0)
    control.fields["conversion"].changed.emit()
    assert control.labels["conversion"].styleSheet().startswith("color: #1d5fb4")
    assert panel.form.values()["camera.conversion"] == 12.0      # stored as before
    control.open_window()
    assert control.window.isVisible()
    control.window.close()


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
