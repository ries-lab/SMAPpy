"""The calibrations made so far, and the spline fitters' *auto*: the one listed for the data's microscope."""
import json
import os

import numpy as np
import pytest

from smappy import calibration_list
from smappy.io.calibration import SplineCalibration, keep_microscope, save_spline_calibration

from test_microscope_db import M2_BEADS, M2_MOVIE, camera_of
from test_microscope_manager import a_tiff


@pytest.fixture
def listing(tmp_path, monkeypatch):
    # a configuration folder of the test's own: the list is in it
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    yield calibration_list.list_path()


def _calibration(path, tags=None, em_on=True):
    cal = SplineCalibration(np.zeros((64, 4, 4, 4), np.float32), 50.0, 2.0, em_on=em_on)
    if tags is not None:
        keep_microscope(cal, camera_of(tags))
    save_spline_calibration(path, cal)
    return str(path)


def test_a_calibration_is_found_for_data_taken_as_its_beads_were_and_only_those(tmp_path, listing):
    beads = camera_of(M2_BEADS).microscope
    old = calibration_list.add(_calibration(tmp_path / "a_3dcal.h5"), microscope=beads, made="2026-08-01")
    new = calibration_list.add(_calibration(tmp_path / "b_3dcal.h5"), microscope=beads, made="2026-09-20")
    calibration_list.add(_calibration(tmp_path / "c_3dcal.h5"), microscope=camera_of(M2_MOVIE).microscope)
    import datetime
    found = calibration_list.find(beads, True, False, day=datetime.date(2026, 8, 3))
    # the nearest in time to the data first; the lens out is offered, never chosen
    assert [(e.name, ok) for e, ok in found] == [("a_3dcal.h5", True), ("b_3dcal.h5", True),
                                                 ("c_3dcal.h5", False)]
    assert calibration_list.chosen(beads, True, False).path == new.path        # no day: the newest
    assert calibration_list.chosen(camera_of(M2_MOVIE).microscope, True, False).name == "c_3dcal.h5"
    assert calibration_list.chosen(beads, False, False) is None                # EM gain off: none
    assert calibration_list.chosen(beads, True, True) is None                  # two channels: none
    # a file gone is skipped; the list is links
    os.remove(old.path)
    assert [e.name for e, _ in calibration_list.find(beads, True, False)] == ["b_3dcal.h5", "c_3dcal.h5"]


def test_a_calibration_beside_the_data_is_found_without_being_listed(tmp_path, listing):
    folder = tmp_path / "260829_M2"
    folder.mkdir()
    _calibration(folder / "beads_3Dcal.h5", M2_BEADS)
    data = a_tiff(folder / "movie_MMStack.ome.tif", M2_BEADS)
    found = calibration_list.find(camera_of(M2_BEADS).microscope, True, False, data=data)
    assert [(e.name, ok) for e, ok in found] == [("beads_3Dcal.h5", True)]   # its file says its microscope


def test_the_fitter_on_auto_fits_with_the_listed_calibration_and_says_which(tmp_path, listing):
    from smappy.plugins.fit import SplineFit
    path = calibration_list.add(_calibration(tmp_path / "beads_3dcal.h5", M2_BEADS)).path
    plugin = SplineFit()
    settings = plugin.Settings()
    settings.source.path = a_tiff(tmp_path / "movie_MMStack.ome.tif", M2_BEADS)
    state = plugin.describe(None, settings)
    assert state.hints["model.calibration"].startswith("auto: beads_3dcal.h5")
    assert state.choices["model.calibration"][0][0] == path
    said = []
    resolved = plugin.with_calibration(settings, type("Ctx", (), {"report": said.append})())
    assert resolved.model.calibration == path and "beads_3dcal.h5" in said[0]
    # data with the lens out: nothing to fit with, said with what was found
    settings.source.path = a_tiff(tmp_path / "out_MMStack.ome.tif", M2_MOVIE)
    with pytest.raises(ValueError, match="1 other listed"):
        plugin.with_calibration(settings)


def test_a_fitted_file_stands_for_the_calibration_it_was_fitted_with(tmp_path, listing):
    import h5py
    from smappy.plugins.fit import SplineFit
    calibration = _calibration(tmp_path / "beads_3dcal.h5", M2_BEADS)
    fitted = tmp_path / "cells_locs.h5"
    with h5py.File(fitted, "w") as f:
        f.create_group("locs")
        f.attrs["metadata"] = json.dumps({"calibration": {"file": calibration}})
    assert calibration_list.recorded_calibration(fitted) == calibration
    assert calibration_list.recorded_calibration(calibration) is None          # a calibration is itself
    plugin = SplineFit()
    settings = plugin.Settings()
    settings.source.path = a_tiff(tmp_path / "movie_MMStack.ome.tif", M2_BEADS)
    settings.model.calibration = str(fitted)
    assert plugin.with_calibration(settings).model.calibration == calibration
    os.remove(calibration)
    with pytest.raises(ValueError, match="not there any more"):
        plugin.with_calibration(settings)


def test_the_calibration_field_offers_what_was_found_and_auto_in_a_menu(tmp_path, listing):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.plugins.fit import SplineFit
    from smappy.session import Session
    path = calibration_list.add(_calibration(tmp_path / "beads_3dcal.h5", M2_BEADS)).path
    panel = PluginPanel(SplineFit, Session())
    panel.form.set_values({"source.path": a_tiff(tmp_path / "movie_MMStack.ome.tif", M2_BEADS)})
    panel._describe()
    field = panel.form.fields["model"].fields["calibration"]
    assert field.widget.text() == "" and field.widget.placeholderText().startswith("auto: beads_3dcal.h5")
    assert field.found_button.isVisibleTo(field)
    menu = field._path_menu()
    actions = [a for a in menu.actions() if not a.isSeparator()]
    assert actions[0].text().startswith("auto") and actions[1].text().startswith("beads_3dcal.h5")
    actions[1].trigger()
    assert panel.form.values()["model.calibration"] == path
    field._path_menu().actions()[0].trigger()                                 # back to auto
    assert panel.form.values()["model.calibration"] == ""
