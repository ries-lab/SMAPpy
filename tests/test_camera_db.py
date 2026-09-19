"""The camera database: which camera took this, and what it was set to.

The two numbers a fit needs most are the two Micro-Manager does not record.
The conversion depends on the readout mode, which the metadata *does* carry;
an iXon reports no baseline at all.  So the database answers "which camera"
from one tag, "which readout mode" from a handful more, and every value it
supplies says where it came from.
"""
import json

import numpy as np
import pytest

from smappy.camera_db import (Camera, CameraDatabase, Parameter, State,
                              database, read_numbers)
from smappy.metadata import CameraMetadata, pixel_sizes

IXON = {
    "Andor-Camera": "| iXon Ultra | DU897_BV | 8172 |",
    "Andor-Output_Amplifier": "Electron Multiplying",
    "Andor-Pre-Amp-Gain": "Gain 1",
    "Andor-ReadoutMode": "10.000 MHz",
    "Andor-Gain": "300",
    "Andor-Exposure": "30",
    "ROI": "0-0-512-512",
}


def a_camera(**kwargs) -> Camera:
    defaults = dict(
        name="test",
        prefix="Cam",
        identify=Parameter(tag="{prefix}-SerialNumber", read="exact", argument="7"),
        parameters={
            "pixelsize_um": Parameter(fixed=0.1),
            "em_on": Parameter(tag="{prefix}-Port", read="not_equals",
                               argument="Normal"),
            "emgain": Parameter(tag="{prefix}-Gain", read="number"),
            "conversion": Parameter(state=True),
            "offset": Parameter(state=True),
        },
        states=[State(match={"{prefix}-Rate": "10MHz"},
                      values={"conversion": 6.7, "offset": 398.6}),
                State(match={"{prefix}-Rate": "20MHz"},
                      values={"conversion": 6.4, "offset": 421.0})],
    )
    defaults.update(kwargs)
    return Camera(**defaults)


def a_file(**extra):
    metadata = {"Cam-SerialNumber": "7", "Cam-Port": "Multiplication Gain",
                "Cam-Gain": "200", "Cam-Rate": "10MHz"}
    metadata.update(extra)
    return metadata


# ------------------------------------------------------------------- reading

def test_a_tag_is_read_the_way_the_camera_says():
    camera = a_camera()
    values = camera.resolve(a_file()).values
    assert values["em_on"] is True          # not "Normal": EM is on
    assert values["emgain"] == 200.0
    assert camera.resolve(a_file(**{"Cam-Port": "Normal"})).values["em_on"] is False


def test_an_roi_is_read_however_the_camera_writes_it():
    assert read_numbers("0-0-512-512") == [0, 0, 512, 512]
    assert read_numbers("[0 0 256 256]") == [0, 0, 256, 256]
    assert read_numbers("0,0,64,64") == [0, 0, 64, 64]
    assert read_numbers("nothing here") is None


def test_the_device_prefix_is_one_field_to_change():
    """A lab that renames its Micro-Manager device edits one line."""
    camera = a_camera(prefix="Andor")
    renamed = {"Andor-SerialNumber": "7", "Andor-Port": "Normal",
               "Andor-Gain": "1", "Andor-Rate": "20MHz"}
    assert camera.identifies(renamed)
    assert camera.resolve(renamed).values["conversion"] == 6.4


# --------------------------------------------------------------- identifying

def test_a_camera_is_identified_by_its_tag():
    db = CameraDatabase([a_camera()])
    assert db.identify(a_file()) is not None
    assert db.identify(a_file(**{"Cam-SerialNumber": "8"})) is None
    assert db.identify({}) is None


def test_a_file_from_an_unknown_camera_resolves_to_nothing_rather_than_a_guess():
    db = CameraDatabase([a_camera()])
    resolution = db.resolve({"Some-Other-Camera": "3"})
    assert resolution.camera is None and resolution.values == {}
    assert resolution.missing == ["conversion", "offset", "pixelsize_um"]


def test_a_camera_can_be_named_instead_of_identified():
    """What a file with no identifying tag needs: chosen, never guessed."""
    db = CameraDatabase([a_camera()])
    resolution = db.resolve({"Cam-Rate": "20MHz"}, camera="test")
    assert resolution.camera_name == "test"
    assert resolution.values["conversion"] == 6.4


# -------------------------------------------------------------- the state

def test_the_readout_mode_decides_the_conversion():
    camera = a_camera()
    for rate, conversion, offset in (("10MHz", 6.7, 398.6), ("20MHz", 6.4, 421.0)):
        resolution = camera.resolve(a_file(**{"Cam-Rate": rate}))
        assert resolution.values["conversion"] == conversion
        assert resolution.values["offset"] == offset
        assert rate in resolution.state_name


def test_an_unrecognised_readout_mode_says_so_rather_than_inventing_one():
    resolution = a_camera().resolve(a_file(**{"Cam-Rate": "33MHz"}))
    assert resolution.values["conversion"] is None
    assert "not recognised" in str(resolution.sources["conversion"])
    assert resolution.missing == ["conversion", "offset"]


def test_every_value_says_where_it_came_from():
    """A wrong conversion is invisible in the fit and obvious in the table."""
    resolution = a_camera().resolve(a_file())
    sources = {name: str(source) for name, source in resolution.sources.items()}
    assert sources["pixelsize_um"] == "fixed: in the database"
    assert sources["emgain"] == "metadata: Cam-Gain = 200"
    assert sources["conversion"].startswith("state: Cam-Rate = 10MHz")


def test_only_what_the_fit_reads_becomes_camera_metadata():
    camera = a_camera()
    camera.parameters["frame_interval_ms"] = Parameter(fixed=17.0)
    resolution = camera.resolve(a_file())
    assert resolution.values["frame_interval_ms"] == 17.0
    metadata = resolution.camera_metadata()
    assert isinstance(metadata, CameraMetadata)
    assert metadata.conversion == 6.7 and metadata.emgain == 200.0
    assert metadata.camera_name == "test"


# ------------------------------------------------------------- the database

def test_a_users_camera_replaces_the_shipped_one_of_the_same_name():
    shipped = CameraDatabase([a_camera(), a_camera(name="other")])
    mine = CameraDatabase([a_camera(name="other",
                                    parameters={"conversion": Parameter(fixed=1.0)})])
    joined = shipped.joined_with(mine)
    assert joined.names() == ["other", "test"]
    assert joined.get("other").parameters["conversion"].fixed == 1.0


def test_a_database_round_trips_through_json(tmp_path):
    before = CameraDatabase([a_camera()])
    path = before.save(tmp_path / "cameras.json")
    after = CameraDatabase.load(path)
    assert after.to_dict() == before.to_dict()
    assert after.resolve(a_file()).values["conversion"] == 6.7
    assert json.loads(path.read_text())["version"] == 1


# ------------------------------------------------------- the shipped cameras

def test_the_shipped_database_reads_and_identifies_an_ixon():
    db = database(reload=True)
    assert "M1 Andor" in db.names()
    resolution = db.resolve(IXON)
    assert resolution.camera_name == "M1 Andor"
    # the two the metadata does not carry, from the readout mode that it does
    assert resolution.values["conversion"] == 15.8
    assert resolution.values["offset"] == 196.0
    assert resolution.values["emgain"] == 300.0
    assert resolution.values["em_on"] is True
    assert resolution.missing == []


def test_the_shipped_ixon_changes_its_conversion_with_the_readout_mode():
    db = database(reload=True)
    faster = dict(IXON, **{"Andor-ReadoutMode": "17.000 MHz"})
    assert db.resolve(faster).values["conversion"] == 14.61
    gained = dict(IXON, **{"Andor-Pre-Amp-Gain": "Gain 2"})
    assert db.resolve(gained).values["conversion"] == 7.7


def test_a_users_file_joins_the_shipped_one(tmp_path, monkeypatch):
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path))
    from smappy import config
    config.load(reload=True)
    CameraDatabase([a_camera(name="mine")]).save(tmp_path / "cameras.json")
    db = database(reload=True)
    assert "mine" in db.names() and "M1 Andor" in db.names()
    assert db.resolve(a_file()).camera_name == "mine"


# -------------------------------------------------------------- pixel sizes

def test_one_pixel_size_means_both_directions():
    assert pixel_sizes(0.1) == (0.1, 0.1)
    assert pixel_sizes([0.1]) == (0.1, 0.1)
    assert pixel_sizes([0.102, 0.103]) == (0.102, 0.103)
    assert pixel_sizes(None) is None

    square = CameraMetadata(pixelsize_um=0.13)
    assert square.pixelsize_nm_xy == (130.0, 130.0) and square.square_pixels
    rectangular = CameraMetadata(pixelsize_um=[0.102, 0.103])
    assert rectangular.pixelsize_nm_xy == (102.0, 103.0)
    assert not rectangular.square_pixels
    assert "0.102 x 0.103 um" in str(rectangular)


def test_nm_coordinates_use_each_axis_pixel_size():
    from smappy.locs import Localizations, to_nm
    locs = Localizations({"x_pix": np.array([1.0, 2.0]),
                          "y_pix": np.array([1.0, 2.0]),
                          "loc_precision_pix": np.array([1.0, 1.0])})
    square = to_nm(locs, 100.0)
    assert square["x_nm"].tolist() == [100.0, 200.0]
    assert square["y_nm"].tolist() == [100.0, 200.0]
    assert square.metadata["pixelsize_nm"] == 100.0
    assert "pixelsize_nm_xy" not in square.metadata

    rectangular = to_nm(locs, (102.0, 103.0))
    assert rectangular["x_nm"].tolist() == [102.0, 204.0]
    assert rectangular["y_nm"].tolist() == [103.0, 206.0]
    # a width is neither x nor y; the mean is exact for square pixels and the
    # only sensible answer for these
    assert rectangular["loc_precision_nm"].tolist() == [102.5, 102.5]
    assert rectangular.metadata["pixelsize_nm_xy"] == [102.0, 103.0]


def test_the_shipped_camera_with_two_pixel_sizes_keeps_both():
    camera = database(reload=True).get("M5 Fusion BT")
    assert camera.parameters["pixelsize_um"].fixed == [0.102, 0.103]


# ------------------------------------------------------------ from SMAP's mat

def a_mat(tmp_path):
    """A ``*_cameras.mat`` shaped like SMAP's, with one camera in it."""
    import scipy.io

    def cell(rows):
        out = np.empty((len(rows), len(rows[0])), dtype=object)
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                out[i, j] = value
        return out

    par = cell([
        ["EMon", "metadata", "1", "Evolve-Port", "", "~strcmp(X,'Normal')", ""],
        ["cam_pixelsize_um", "fix", "0.106 0.107", "select", "", "", ""],
        ["conversion", "state dependent", "1", "select", "", "", ""],
        ["offset", "state dependent", "100", "select", "", "", ""],
        ["emgain", "metadata", "1", "Evolve-Gain", "", "str2double(X)", ""],
        ["numberOfFrames", "metadata", "0", "Frames", "", "str2double(X)", ""],
        ["roi", "metadata", "", "ROI direct", "", "str2num(X)", ""],
        ["comment", "fix", "settings not initialized", "select", "", "", ""],
    ])
    scipy.io.savemat(str(tmp_path / "cams.mat"), {
        "camtab": cell([["Lab cam", "Evolve-SerialNumber", "A1"]]),
        "cameras": {
            "par": par,
            "ID": {"name": "Lab cam", "tag": "Evolve-SerialNumber", "value": "A1"},
            "state": np.array([
                {"defpar": cell([["Evolve-Rate", "10MHz"], ["select", ""]]),
                 "par": cell([["conversion", "6.7"], ["offset", "398.6"],
                              ["emgain", "100"]])},
                {"defpar": cell([["Evolve-Rate", "20MHz"], ["select", ""]]),
                 "par": cell([["conversion", "6.4"], ["offset", "421"],
                              ["emgain", "100"]])},
            ], dtype=object),
        },
    })
    return tmp_path / "cams.mat"


def test_a_smap_settings_file_becomes_a_database(tmp_path):
    """A lab with a ``*_cameras.mat`` converts it once instead of retyping it."""
    pytest.importorskip("scipy")
    from smappy.io.cameras_mat import to_database

    db = to_database(a_mat(tmp_path))
    assert db.names() == ["Lab cam"]
    camera = db.get("Lab cam")
    assert camera.prefix == "Evolve"
    assert camera.identify.tag == "{prefix}-SerialNumber"
    # SMAP's placeholder comment is not a comment
    assert camera.comment == ""
    # x and y kept; SMAP's own bookkeeping rows dropped
    assert camera.parameters["pixelsize_um"].fixed == [0.106, 0.107]
    assert "numberOfFrames" not in camera.parameters
    # "ROI direct" is SMAP reading the image, not a tag: here it is the ROI key
    assert camera.parameters["roi"].tag == "ROI"

    metadata = {"Evolve-SerialNumber": "A1", "Evolve-Port": "Multiplication Gain",
                "Evolve-Gain": "300", "Evolve-Rate": "20MHz"}
    resolution = db.resolve(metadata)
    assert resolution.camera_name == "Lab cam"
    assert resolution.values["conversion"] == 6.4
    assert resolution.values["offset"] == 421.0
    assert resolution.values["em_on"] is True
    assert resolution.values["emgain"] == 300.0
