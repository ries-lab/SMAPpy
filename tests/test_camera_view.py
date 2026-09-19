"""Choosing the camera, and looking at what the fit will use.

A wrong conversion is invisible in the result, so the parameter view is the
only way to catch one -- and it is hidden until asked for, because the usual
answer is that the camera was recognised and the numbers are its own.
"""
import json

import numpy as np
import pytest

pytest.importorskip("PySide6")

from smappy.camera_db import CameraDatabase, Camera, Parameter, State   # noqa: E402
from smappy.io.tiff import ImageSource                                  # noqa: E402

EVOLVE = {
    "Core-Camera": "Evolve512",
    "Evolve512-SerialNumber": "A14G150010",
    "Evolve512-Port": "Multiplication Gain",
    "Evolve512-ReadoutRate": "10MHz 16bit",
    "Evolve512-Gain": "1",
    "Evolve512-MultiplierGain": "300",
    "ROI": "0-0-512-512",
}


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def a_source(**tags) -> ImageSource:
    return ImageSource(files=[], shape=(64, 64), dtype=np.uint16, n_frames=1,
                       n_frames_declared=1, mm_metadata={}, summary=dict(tags))


class FakeFitter:
    """Stands in for the fit plugin: the view only asks it to resolve."""

    def __init__(self, source, camera=""):
        self.source, self.camera = source, camera

    def resolution(self, settings):
        from smappy.io.tiff import resolve_camera
        return resolve_camera(self.source, camera=self.camera)


def test_the_view_says_which_camera_and_which_readout_mode(app):
    from smappy.gui.camera_view import CameraParameters

    window = CameraParameters()
    window.show_for(FakeFitter(a_source(**EVOLVE)), settings=object())
    assert "M2 Evolve" in window.summary.text()
    assert "A14G150010" in window.summary.text()        # identified by this
    assert "10MHz 16bit" in window.summary.text()       # and set up like this

    rows = {window.table.item(r, 0).text():
            (window.table.item(r, 1).text(), window.table.item(r, 2).text())
            for r in range(window.table.rowCount())}
    assert rows["conversion"][0] == "6.7"
    assert rows["conversion"][1].startswith("state:")    # not in the metadata
    assert rows["emgain"] == ("300", "metadata: Evolve512-MultiplierGain = 300")
    assert rows["pixelsize_um"][1].startswith("fixed:")
    assert "Every value the fit needs is known" in window.note.text()


def test_an_unrecognised_camera_says_so_rather_than_showing_numbers(app):
    from smappy.gui.camera_view import CameraParameters

    window = CameraParameters()
    window.show_for(FakeFitter(a_source(**{"Core-Camera": "Someone's camera"})),
                    settings=object())
    assert "not recognised" in window.summary.text()
    assert "conversion" in window.note.text()           # and what is missing


def test_an_unrecognised_readout_mode_is_called_out(app, tmp_path, monkeypatch):
    """The camera is right and its conversion is not known: the worst case."""
    from smappy.gui.camera_view import CameraParameters

    window = CameraParameters()
    window.show_for(FakeFitter(a_source(**dict(EVOLVE, **{
        "Evolve512-ReadoutRate": "100MHz 16bit"}))), settings=object())
    assert "readout mode not recognised" in window.summary.text()
    assert "conversion" in window.note.text()
    rows = {window.table.item(r, 0).text(): window.table.item(r, 1).text()
            for r in range(window.table.rowCount())}
    assert rows["conversion"] == "-"


def test_a_camera_can_be_chosen_when_the_file_identifies_none(app):
    from smappy.gui.camera_view import CameraParameters

    source = a_source(**{"Core-Camera": "unknown"})
    window = CameraParameters()
    window.show_for(FakeFitter(source, camera="M5 Fusion BT"), settings=object())
    assert "M5 Fusion BT" in window.summary.text()
    rows = {window.table.item(r, 0).text(): window.table.item(r, 1).text()
            for r in range(window.table.rowCount())}
    assert rows["conversion"] == "0.24"
    assert rows["pixelsize_um"] == "0.102, 0.103"       # x and y, both kept


def test_the_view_is_not_something_the_fitter_has_to_have(app):
    """Nothing chosen yet: it says so instead of failing."""
    from smappy.gui.camera_view import CameraParameters

    window = CameraParameters()
    window.refresh()
    assert "no acquisition" in window.summary.text()
    assert window.table.rowCount() == 0


def test_the_fit_plugin_offers_the_databases_cameras(app):
    from smappy.plugins.fit import CameraSettings, _camera_choices

    choices = _camera_choices()
    assert choices[0] == ("", "auto")
    assert ("M1 Andor", "M1 Andor") in choices
    specs = CameraSettings.__dataclass_fields__
    assert "camera" in specs and "pixelsize_y_um" in specs


def test_a_user_database_reaches_the_view(app, tmp_path, monkeypatch):
    """A lab's own cameras.json is what makes this worth anything."""
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path))
    from smappy import config
    from smappy.camera_db import database
    config.load(reload=True)

    CameraDatabase([Camera(
        name="Mine",
        prefix="Cam",
        identify=Parameter(tag="{prefix}-Serial", read="exact", argument="42"),
        parameters={"conversion": Parameter(state=True),
                    "offset": Parameter(fixed=100.0),
                    "pixelsize_um": Parameter(fixed=0.108)},
        states=[State(match={"{prefix}-Rate": "fast"}, values={"conversion": 2.5})],
    )]).save(tmp_path / "cameras.json")
    database(reload=True)

    from smappy.gui.camera_view import CameraParameters
    window = CameraParameters()
    window.show_for(FakeFitter(a_source(**{"Cam-Serial": "42", "Cam-Rate": "fast"})),
                    settings=object())
    assert "Mine" in window.summary.text()
    rows = {window.table.item(r, 0).text(): window.table.item(r, 1).text()
            for r in range(window.table.rowCount())}
    assert rows["conversion"] == "2.5"
    assert json.loads((tmp_path / "cameras.json").read_text())["version"] == 1
    database(reload=True)                    # leave the shipped one behind


def test_a_real_acquisition_reaches_the_view_through_the_fitter(app, tmp_path):
    """The whole path: a TIFF with an iXon's tags, opened by the fit plugin."""
    import json
    import tifffile
    from smappy.plugins.fit import GaussianFit

    tags = {"Core-Camera": "Andor",
            "Andor-Camera": "| iXon Ultra | DU897_BV | 8172 |",
            "Andor-Output_Amplifier": "Electron Multiplying",
            "Andor-Pre-Amp-Gain": "Gain 1", "Andor-ReadoutMode": "10.000 MHz",
            "Andor-Gain": "300", "ROI": "0-0-48-60"}
    path = tmp_path / "run_MMStack.ome.tif"
    tifffile.imwrite(path, np.zeros((2, 48, 60), np.uint16),
                     extratags=[(51123, "s", 0, json.dumps(tags), True)])

    plugin = GaussianFit()
    settings = plugin.Settings()
    settings.source.path = str(path)

    from smappy.gui.camera_view import CameraParameters
    window = CameraParameters()
    window.show_for(plugin, settings)
    assert "M1 Andor" in window.summary.text()
    rows = {window.table.item(r, 0).text(): window.table.item(r, 1).text()
            for r in range(window.table.rowCount())}
    # the two Micro-Manager never records, from the mode that it does
    assert rows["conversion"] == "15.8" and rows["offset"] == "196"
    assert "Every value the fit needs is known" in window.note.text()

    # and what the fit will actually use matches what is shown
    camera = plugin._camera(settings)
    assert camera.conversion == 15.8 and camera.offset == 196.0
    assert camera.emgain == 300.0 and camera.is_em

    # the user's own setting wins over all of it, and says so
    settings.camera.conversion = 4.2
    plugin._camera_key = None
    window.refresh()
    rows = {window.table.item(r, 0).text():
            (window.table.item(r, 1).text(), window.table.item(r, 2).text())
            for r in range(window.table.rowCount())}
    assert rows["conversion"] == ("4.2", "user: your setting")
