"""The microscope database: which microscope took this, and its optics.

Micro-Manager records every device's state and nothing about what it means.
An entry names the devices that matter and lets each parameter follow one of
them: the pixel the cylindrical lens, the wavelength the filter.  The tags
below are what the Ries lab's M2 and M5 files record (October 2026).
"""
import datetime
import json

import pytest

from smappy import camera_db, microscope_db
from smappy.camera_db import Parameter, Source
from smappy.io import emu
from smappy.microscope_db import (FROM_NAME, Channel, Device, Follows, Microscope,
                                  MicroscopeDatabase, acquisition_date, changes,
                                  save_microscope, wavelength_from_name)

# M2 with the iXon, 2026-08-29: the bead stacks, and the TIRF movie of the same
# afternoon, which differs from them in one device -- the cylindrical lens
M2_BEADS = {
    "Focus-lock-Serial ID": "iBEAM-SMART-785-S-A3-13644", "Luxx638-Serial Number": "1311-08",
    "Core-Camera": "Andor", "Andor-Camera": "| iXon Ultra | DU897_BV | 8172 |",
    "Andor-Output_Amplifier": "Electron Multiplying", "Andor-Pre-Amp-Gain": "Gain 1",
    "Andor-ReadoutMode": "10.000 MHz", "Andor-Gain": "25",
    "Servos-Position0": "10000", "Servos-Position1": "28000",
    "Servos-Position2": "0", "Servos-Position3": "32000", "TTL-State0": "0",
    "Thorlabs ELL9-Label": "Position 0", "Thorlabs ELL9-1-Label": "Position 0",
    "Luxx638-Laser Operation Select": "On", "PixelSizeUm": "0.127",
    "StartTime": "2026-08-29 17:34:14.224 +0200",
}
M2_MOVIE = dict(M2_BEADS, **{"Servos-Position1": "4000",
                             "StartTime": "2026-08-29 19:02:00.000 +0200"})

# M2 with the Evolve, 2026-01-05
M2_EVOLVE = dict(M2_BEADS, **{
    "Core-Camera": "Evolve512", "Evolve512-SerialNumber": "A14G150010",
    "Servos-Position0": "20000", "Thorlabs ELL9-Label": "Position 3",
    "StartTime": "2026-01-05 10:19:59.185 +0100"})
del M2_EVOLVE["Andor-Camera"]

# M5, 2026-09-01: the 3D lens on a linear stage, two filter wheels
M5_BEADS = {
    "iChrome-MLE-TCP-System: Serial Number": "iCHROME-MLE-L-CD_30087",
    "Core-Camera": "HamamatsuHam_DCAM", "HamamatsuHam_DCAM-CameraID": "S/N: 500975",
    "Thorlabs ELL20-Position (um)": "0", "Thorlabs ELL17/ELL20-Position (um)": "18005",
    "Thorlabs ELL6-State": "1", "Thorlabs Filter Wheel-State": "3",
    "Thorlabs Filter Wheel-1-State": "0", "StartTime": "2026-09-01 12:26:03.521 +0200",
}


def lens(**kwargs) -> Device:
    return Device(**dict(dict(name="3D lens", tags=["Servos-Position1"], read="nearest",
                              states={"in": 28000, "out": 4000}), **kwargs))


def a_microscope(**kwargs) -> Microscope:
    defaults = dict(
        name="scope",
        identify={"Focus-lock-Serial ID": "iBEAM-SMART-785-S-A3-13644"},
        devices={"3D lens": lens(),
                 "filter": Device(name="filter", tags=["Servos-Position3"], read="nearest",
                                  states={"600/60": 23500, "700/100": 32000, "None": 57000})},
        parameters={"NA": 1.43, "wavelength_nm": {"device": "filter", "values": FROM_NAME},
                    "pixel_nm": {"device": "3D lens",
                                 "values": {"in": [127, 116], "out": [127, 127]}}},
    )
    defaults.update(kwargs)
    return Microscope(**defaults)


# ------------------------------------------------------------------ devices

def test_a_servo_is_read_by_its_nearest_state_split_at_the_midpoint():
    device = lens()
    assert device.state({"Servos-Position1": "28000"})[0] == "in"
    assert device.state({"Servos-Position1": "27900"})[0] == "in"     # re-trimmed
    assert device.state({"Servos-Position1": "16001"})[0] == "in"
    assert device.state({"Servos-Position1": "15999"})[0] == "out"
    assert device.state({"Servos-Position1": "4000"})[0] == "out"


def test_a_tolerance_rejects_a_reading_far_from_every_state():
    device = lens(tolerance=2000)
    state, raw, source = device.state({"Servos-Position1": "16000"})
    assert state is None and raw == "16000"
    assert source.kind == "missing" and "no state within 2000" in source.detail


def test_a_label_is_matched_exactly_and_a_stranger_says_so():
    device = Device(name="reflected", tags=["Thorlabs ELL9-Label"],
                    states={"600/60": "Position 3", "676/37": "Position 1"})
    assert device.state({"Thorlabs ELL9-Label": "Position 1"})[0] == "676/37"
    state, _, source = device.state({"Thorlabs ELL9-Label": "Position 2"})
    assert state is None and "no state reads that" in source.detail


def test_a_renamed_device_is_found_by_its_older_tag():
    device = Device(name="BFP", tags=["ELL9 BFP-Label", "Thorlabs ELL9-1-Label"],
                    states={"on": "Position 1", "off": "Position 0"})
    state, _, source = device.state({"Thorlabs ELL9-1-Label": "Position 0"})
    assert state == "off" and source.detail == "Thorlabs ELL9-1-Label = Position 0"
    state, _, source = device.state({})
    assert state is None and "ELL9 BFP-Label or Thorlabs ELL9-1-Label" in source.detail


def test_two_states_at_the_same_value_read_as_the_first():
    device = Device(name="slider", tags=["Servos-Position0"], read="nearest",
                    states={"empty": 10000, "685/70": 10000, "676/37": 20000})
    assert device.state({"Servos-Position0": "10000"})[0] == "empty"


def test_a_continuous_device_keeps_its_position_and_no_state():
    device = Device(name="TIRF", tags=["Thorlabs ELL17/ELL20-Position (um)"], read="value")
    assert device.state(M5_BEADS)[:2] == (None, 18005.0)


def test_a_device_round_trips_through_its_dict():
    device = lens(tags=["a", "b"], tolerance=500.0, emu="Two-state device 2")
    assert Device.from_dict("3D lens", device.to_dict()) == device


# --------------------------------------------------------------- parameters

@pytest.mark.parametrize("name, wavelength", [("676/37", 676.0), ("700/100", 700.0),
                                              (" 525/45", 525.0), ("665LP", None),
                                              ("None", None), ("Empty", None)])
def test_a_band_pass_filter_gives_its_centre_and_nothing_else_does(name, wavelength):
    assert wavelength_from_name(name) == wavelength


def test_each_parameter_follows_its_device_and_says_so():
    found = a_microscope().resolve(M2_BEADS)
    assert found.values["NA"] == 1.43
    assert found.values["wavelength_nm"] == 700.0
    assert found.pixel_nm == (127.0, 116.0)
    assert str(found.sources["pixel_nm"]) == "device: 3D lens is in"
    assert str(found.sources["NA"]) == "fixed: in the microscope's entry"


def test_the_movie_without_the_lens_gets_the_square_pixel():
    found = a_microscope().resolve(M2_MOVIE)
    assert found.state_names()["3D lens"] == "out"
    assert found.pixel_nm == (127.0, 127.0)


def test_a_state_the_table_lacks_is_missing_with_the_reason():
    scope = a_microscope(parameters={"pixel_nm": {"device": "3D lens",
                                                  "values": {"in": [127, 116]}}})
    found = scope.resolve(M2_MOVIE)
    assert found.values["pixel_nm"] is None
    assert found.sources["pixel_nm"].detail == "3D lens is out: no value for it"


def test_a_filter_without_a_band_in_its_name_is_missing():
    found = a_microscope().resolve(dict(M2_BEADS, **{"Servos-Position3": "57000"}))
    assert found.values["wavelength_nm"] is None
    assert "None: no wavelength in the name" in found.sources["wavelength_nm"].detail


# ------------------------------------------------------- cameras, channels

def setups(**kwargs) -> Microscope:
    """Set up with the iXon now, with the Evolve until 2026-07-15 -- M2's history."""
    return a_microscope(camera="iXon 1", versions=[{
        "until": "2026-07-15", "camera": "Evolve 1",
        "parameters": {"pixel_nm": {"device": "3D lens",
                                    "values": {"in": [116, 127], "out": [127, 127]}}}}],
        **kwargs)


def test_a_file_is_read_with_the_setup_of_its_day():
    scope = setups()
    now = scope.resolve(M2_BEADS, camera="iXon 1")
    assert now.pixel_nm == (127.0, 116.0) and now.camera == "iXon 1" and now.warnings() == []
    january = scope.resolve(M2_EVOLVE, camera="Evolve 1")
    assert january.pixel_nm == (116.0, 127.0) and january.version == "2026-07-15"
    assert january.warnings() == []


def test_a_camera_swapped_before_the_entry_was_finds_its_setup():
    scope = setups()
    # the iXon already in on 2026-07-10, the entry saying Evolve until the 15th
    early = scope.resolve(dict(M2_BEADS, StartTime="2026-07-10 10:00:00 +0200"), camera="iXon 1")
    assert early.pixel_nm == (127.0, 116.0) and early.camera == "iXon 1"
    assert early.warnings() == ["taken with iXon 1, not Evolve 1: read as scope was set up "
                                "with iXon 1"]


def test_a_camera_the_microscope_never_had_is_a_warning():
    found = setups().resolve(M2_BEADS, camera="Fusion BT 1")
    assert found.pixel_nm == (127.0, 116.0)               # the setup of its day, still
    assert "never set up with" in found.warnings()[0]


def test_a_camera_is_the_same_under_the_name_it_had_before():
    scope = a_microscope(camera="M1 Andor")
    assert scope.resolve(M2_BEADS, camera="iXon 1").warnings() == []


def test_a_file_whose_camera_names_itself_nowhere_gets_the_microscopes():
    from smappy.io.tiff import resolve_camera
    tags = {k: v for k, v in M2_BEADS.items() if k != "Andor-Camera"}
    found = resolve_camera(a_file(tags))
    assert found.camera_name == "iXon 1"
    assert found.values["conversion"] == 15.8              # the iXon's, for its readout mode


def two_arms(**kwargs) -> Microscope:
    devices = dict(a_microscope().devices,
                   transmitted=Device(name="transmitted", tags=["Servos-Position0"],
                                      read="nearest", states={"empty": 10000, "676/37": 20000}),
                   reflected=Device(name="reflected", tags=["Thorlabs ELL9-Label"],
                                    states={"Empty": "Position 0", "676/37": "Position 1",
                                            "600/60": "Position 3"}))
    return a_microscope(devices=devices, channels=[
        Channel("transmitted", "transmitted", "right"),
        Channel("reflected", "reflected", "left")], **kwargs)


def test_dual_channel_data_gets_a_wavelength_per_side():
    found = two_arms().resolve(M2_EVOLVE, camera="iXon 1", channels=2)
    assert [(c.name, c.side, c.wavelength_nm) for c in found.channels] == [
        ("transmitted", "right", 676.0), ("reflected", "left", 600.0)]


def test_single_channel_data_is_the_transmitted_arm_over_the_whole_chip():
    found = two_arms().resolve(M2_EVOLVE, camera="iXon 1", channels=1)
    assert [(c.name, c.side, c.wavelength_nm) for c in found.channels] == [
        ("transmitted", "", 676.0)]


def test_an_empty_arm_takes_the_wavelength_of_the_common_filter():
    found = two_arms().resolve(M2_BEADS, camera="iXon 1", channels=2)
    transmitted, reflected = found.channels
    assert transmitted.wavelength_nm == 700.0          # the filter wheel's 700/100
    assert "transmitted is empty" in transmitted.source.detail
    assert "filter is 700/100" in transmitted.source.detail
    assert reflected.wavelength_nm == 700.0


# ------------------------------------------------------------- recognising

def test_a_microscope_is_recognised_by_a_device_that_stays_with_it():
    db = MicroscopeDatabase([a_microscope()])
    assert db.identify(M2_BEADS).name == "scope"
    assert db.identify(M5_BEADS) is None
    assert db.resolve(M5_BEADS).microscope is None
    assert db.resolve(M5_BEADS, microscope="scope").name == "scope"   # chosen


def test_the_acquisition_date_is_read_from_start_time():
    assert acquisition_date(M2_BEADS) == datetime.date(2026, 8, 29)
    assert acquisition_date({}) is None


# ---------------------------------------------------------------- versions

def test_an_older_file_is_read_with_the_entry_as_it_stood_then():
    scope = a_microscope(versions=[{"until": "2026-03-01", "devices": {
        "3D lens": lens(states={"in": 37000, "out": 4000}).to_dict()}}])
    # 20 000: nearer 4000 than 37 000, the old states; nearer 28 000 than 4000
    early = dict(M2_MOVIE, **{"Servos-Position1": "20000",
                              "StartTime": "2026-02-10 10:00:00 +0100"})
    late = dict(early, StartTime="2026-04-10 10:00:00 +0200")
    assert scope.resolve(early).state_names()["3D lens"] == "out"
    assert scope.resolve(early).version == "2026-03-01"
    assert scope.resolve(late).state_names()["3D lens"] == "in"
    assert scope.resolve(late).version is None


def test_versions_are_laid_over_newest_first():
    scope = a_microscope(parameters={"NA": 1.5}, versions=[
        {"until": "2026-02-01", "parameters": {"NA": 1.3}},
        {"until": "2026-06-01", "parameters": {"NA": 1.4}}])
    assert scope.as_of(datetime.date(2026, 1, 1)).parameters["NA"] == 1.3
    assert scope.as_of(datetime.date(2026, 3, 1)).parameters["NA"] == 1.4
    assert scope.as_of(datetime.date(2026, 7, 1)).parameters["NA"] == 1.5


def test_a_device_added_later_did_not_exist_before():
    old = a_microscope()
    new = a_microscope(devices=dict(old.devices, BFP=Device(name="BFP", tags=["x"])))
    diff = changes(old, new)
    assert diff == {"devices": {"BFP": None}}
    dated = a_microscope(devices=new.devices, versions=[dict(until="2026-03-01", **diff)])
    assert "BFP" not in dated.as_of(datetime.date(2026, 1, 1)).devices
    assert "BFP" in dated.as_of(datetime.date(2026, 5, 1)).devices


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path))
    camera_db.database(reload=True)
    microscope_db.database(reload=True)
    yield tmp_path
    monkeypatch.undo()
    camera_db.database(reload=True)
    microscope_db.database(reload=True)


def test_saving_a_changed_entry_keeps_the_old_one_dated(config):
    save_microscope(a_microscope())
    changed = a_microscope(devices=dict(a_microscope().devices,
                                        **{"3D lens": lens(states={"in": 30000, "out": 4000})}))
    path = save_microscope(changed, until="2026-10-01")
    stored = json.loads(path.read_text())["microscopes"][0]
    assert stored["devices"]["3D lens"]["states"]["in"] == 30000
    assert stored["versions"] == [{"until": "2026-10-01", "devices": {
        "3D lens": lens().to_dict()}}]
    assert microscope_db.database().get("scope").versions == stored["versions"]


def test_saving_an_unchanged_entry_adds_no_version(config):
    save_microscope(a_microscope())
    path = save_microscope(a_microscope())
    assert "versions" not in json.loads(path.read_text())["microscopes"][0]


def test_microscopes_and_cameras_share_the_users_file(config):
    camera_db.save_camera(camera_db.Camera(name="mine"))
    save_microscope(a_microscope())
    camera_db.save_camera(camera_db.Camera(name="other"))
    stored = json.loads((config / "microscopes.json").read_text())
    assert [c["name"] for c in stored["cameras"]] == ["mine", "other"]
    assert [m["name"] for m in stored["microscopes"]] == ["scope"]


# ------------------------------------------------------- the shipped ones

def test_the_shipped_m2_tells_the_beads_from_the_movie_without_the_lens():
    db = microscope_db.database(reload=True)
    beads, movie = db.resolve(M2_BEADS), db.resolve(M2_MOVIE)
    assert beads.name == movie.name == "M2"
    assert beads.camera == "iXon 1"
    assert beads.state_names()["3DA"] == "on" and movie.state_names()["3DA"] == "off"
    assert beads.values["modality"] == "astigmatic" and movie.values["modality"] == "2D"
    assert beads.pixel_nm == (127.0, 116.0) and movie.pixel_nm == (127.0, 127.0)
    assert beads.values["NA"] == 1.43 and beads.values["n_immersion"] == 1.518


def test_the_shipped_m2_reads_the_evolve_pixel_the_other_way_round():
    found = microscope_db.database(reload=True).resolve(M2_EVOLVE)
    assert found.camera == "Evolve 1" and found.pixel_nm == (116.0, 127.0)
    # the ELL9 slider under the name it had before September 2026
    assert found.state_names()["Channel 2 (reflected)"] == "600/60"


def test_the_shipped_m2_puts_the_transmitted_arm_on_the_ixons_right():
    found = microscope_db.database(reload=True).resolve(M2_BEADS, channels=2)
    assert [(c.name, c.side) for c in found.channels] == [("transmitted", "right"),
                                                          ("reflected", "left")]
    # the Evolve is turned by 90 degrees against the iXon: the split runs upper and lower
    evolve = microscope_db.database().resolve(M2_EVOLVE, channels=2)
    assert [(c.name, c.side) for c in evolve.channels] == [("transmitted", "upper"),
                                                           ("reflected", "lower")]


def test_the_shipped_m5_has_its_3d_lens_on_a_stage():
    found = microscope_db.database(reload=True).resolve(M5_BEADS)
    assert found.name == "M5" and found.camera == "Fusion BT 1"
    assert found.state_names()["3D"] == "on"
    assert found.values["modality"] == "astigmatic"
    assert found.values["NA"] == 1.5
    assert [c.wavelength_nm for c in found.channels] == [685.0]
    out = microscope_db.database().resolve(
        dict(M5_BEADS, **{"Thorlabs ELL20-Position (um)": "50000"}))
    assert out.values["modality"] == "2D"
    assert found.pixel_nm is None          # not the microscope's: the camera's
    assert found.warnings() == []


# ------------------------------------------------- what the fit and calibration read

def a_file(tags):
    """What `io.tiff.resolve_camera` reads of a stack: its tags, and no simulated camera."""
    from types import SimpleNamespace
    return SimpleNamespace(summary=dict(tags), mm_metadata={}, camera=None)


def test_the_lens_state_sets_the_pixel_over_the_files_square_one():
    from smappy.io.tiff import resolve_camera
    beads = resolve_camera(a_file(M2_BEADS))
    assert beads.values["pixelsize_um"] == [0.127, 0.116]       # the file says 0.127 square
    assert str(beads.sources["pixelsize_um"]) == "microscope: M2, 3DA is on"
    movie = resolve_camera(a_file(M2_MOVIE)).camera_metadata()
    assert movie.pixelsize_um_xy == (0.127, 0.127)
    assert movie.microscope["states"]["3DA"] == "off"


def test_the_users_pixel_still_wins_over_the_microscope():
    from smappy.io.tiff import resolve_camera
    found = resolve_camera(a_file(M2_BEADS), overrides={"pixelsize_um": 0.1})
    assert found.values["pixelsize_um"] == 0.1


def test_a_file_no_microscope_recognises_keeps_the_files_pixel():
    from smappy.io.tiff import resolve_camera
    tags = {k: v for k, v in M2_BEADS.items() if k != "Luxx638-Serial Number"}
    found = resolve_camera(a_file(tags))
    assert found.microscope is None
    assert found.values["pixelsize_um"] == 0.127
    assert found.camera_metadata().microscope is None


def test_the_microscope_travels_with_the_camera_metadata():
    from smappy.io.tiff import resolve_camera
    scope = resolve_camera(a_file(M2_BEADS)).camera_metadata().microscope
    assert scope["name"] == "M2" and scope["camera"] == "iXon 1"
    assert scope["values"]["NA"] == 1.43 and scope["values"]["modality"] == "astigmatic"
    assert "3DA" in scope["followed"] and "laser 638" not in scope["followed"]
    assert [(c["name"], c["side"]) for c in scope["channels"]] == [("transmitted", "right"),
                                                                   ("reflected", "left")]
    json.dumps(scope)                                           # it is saved as JSON


def camera_of(tags):
    from smappy.io.tiff import resolve_camera
    return resolve_camera(a_file(tags)).camera_metadata()


def test_psf_calibration_takes_the_optics_from_the_microscope_on_auto():
    from smappy.calibrate.pupil import OpticsSettings
    camera = camera_of(M2_BEADS)
    found = OpticsSettings().resolved(camera)
    # one channel: the transmitted arm's slider is empty, so the filter wheel's 700/100
    assert found["NA"] == 1.43 and found["n_immersion"] == 1.518
    assert found["wavelengths"] == [700.0]
    assert found["pixel"] == (127.0, 116.0)
    typed = OpticsSettings(NA=1.4, wavelength_nm=650.0, n_immersion=1.515).resolved(camera)
    assert (typed["NA"], typed["wavelengths"], typed["n_immersion"]) == (1.4, [650.0], 1.515)


def test_an_objectives_coverslip_is_saved_with_the_microscope_and_reaches_the_calibration(config):
    from smappy.calibrate.pupil import OpticsSettings
    # a water objective, its turret a device: each objective with its own coverslip glass and design
    turret = Device(name="objective", tags=["Turret-Label"], states={"60x W": "Position-1", "100x oil": "Position-2"})
    water = a_microscope(
        name="water scope", identify={"Laser-Serial": "W-1"}, devices={"objective": turret},
        parameters={"NA": {"device": "objective", "values": {"60x W": 1.2, "100x oil": 1.45}},
                    "n_immersion": {"device": "objective", "values": {"60x W": 1.333, "100x oil": 1.518}},
                    "n_coverslip": {"device": "objective", "values": {"60x W": 1.52}},
                    "coverslip_design_um": {"device": "objective", "values": {"60x W": 150}},
                    "wavelength_nm": 670})
    save_microscope(water)
    stored = microscope_db.database(reload=True).get("water scope")
    assert stored.parameters["n_coverslip"] == {"device": "objective", "values": {"60x W": 1.52}}
    assert stored.parameters["coverslip_design_um"]["values"] == {"60x W": 150}
    tags = {"Laser-Serial": "W-1", "Turret-Label": "Position-1", "PixelSizeUm": "0.11"}
    found = OpticsSettings().resolved(camera_of(tags))
    assert (found["n_immersion"], found["n_coverslip"]) == (1.333, 1.52)
    assert found["coverslip_design_um"] == found["coverslip_um"] == 150.0      # the thickness: the design's
    assert OpticsSettings().coverslip_matters(camera_of(tags))
    typed = OpticsSettings(coverslip_um=172.0).resolved(camera_of(tags))
    assert (typed["coverslip_um"], typed["coverslip_design_um"]) == (172.0, 150.0)
    # the oil objective of the same microscope says nothing: the coverslip is the immersion, 170 µm
    oil = OpticsSettings().resolved(camera_of(dict(tags, **{"Turret-Label": "Position-2"})))
    assert (oil["n_coverslip"], oil["coverslip_design_um"]) == (1.518, 170.0)
    assert not OpticsSettings().coverslip_matters(camera_of(dict(tags, **{"Turret-Label": "Position-2"})))


def test_dual_colour_puts_the_arm_on_the_main_half_first():
    from smappy.calibrate.pupil import OpticsSettings
    tags = dict(M2_BEADS, **{"Servos-Position0": "20000", "Thorlabs ELL9-Label": "Position 3"})
    camera = camera_of(tags)                     # transmitted 676/37 on the right, reflected 600/60 left
    # auto: the microscope's main channel, its first arm, the transmitted one -- as the dual
    # calibration takes it (`dual.resolve_arrangement`); it was the left half's, the reflected
    auto = OpticsSettings().resolved(camera, 2)
    assert auto["wavelengths"] == [676.0, 600.0]
    assert OpticsSettings().resolved(camera, 2, main="left")["wavelengths"] == [600.0, 676.0]
    assert OpticsSettings().hints(camera, 2)["wavelength2_nm"] == 600.0


def test_without_a_microscope_the_na_and_emission_must_be_typed():
    from smappy.calibrate.pupil import OpticsSettings
    tags = {k: v for k, v in M2_BEADS.items() if k != "Luxx638-Serial Number"}
    camera = camera_of(tags)
    assert OpticsSettings().missing(camera) == ["NA", "emission"]
    assert OpticsSettings(NA=1.43, wavelength_nm=670.0).missing(camera) == []
    assert OpticsSettings().resolved(camera)["n_immersion"] == 1.518
    assert OpticsSettings().missing(None) == ["NA", "emission"]      # no stacks yet


# ------------------------------------------------- calibration against data

def test_beads_with_the_lens_and_data_without_it_are_told_apart():
    from smappy.io.calibration import microscope_differences
    beads, movie = camera_of(M2_BEADS).microscope, camera_of(M2_MOVIE).microscope
    assert microscope_differences(beads, movie) == ["3DA: on for the beads, off for the data"]
    assert microscope_differences(beads, camera_of(M2_BEADS).microscope) == []
    lasers = dict(M2_BEADS, **{"Luxx638-Laser Operation Select": "Off"})
    assert microscope_differences(beads, camera_of(lasers).microscope) == []
    assert microscope_differences(beads, camera_of(M5_BEADS).microscope) == [
        "the beads were taken on M2, the data on M5"]
    assert microscope_differences(None, movie) == []


def test_a_saved_calibration_keeps_its_beads_microscope(tmp_path):
    import warnings

    import numpy as np

    from smappy.io.calibration import (SplineCalibration, keep_microscope, load_spline_calibration,
                                       save_spline_calibration, warn_on_microscope_mismatch)
    cal = SplineCalibration(np.zeros((64, 4, 4, 4), np.float32), 50.0, 2.0)
    keep_microscope(cal, camera_of(M2_BEADS))
    path = tmp_path / "beads_3dcal.h5"
    save_spline_calibration(path, cal)
    loaded = load_spline_calibration(path)
    assert loaded.parameters["microscope"]["states"]["3DA"] == "on"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_on_microscope_mismatch(loaded, camera_of(M2_MOVIE).microscope)
    assert any("3DA: on for the beads, off for the data" in str(w.message) for w in caught)


def test_an_arm_without_a_wavelength_is_missing_not_the_others():
    from smappy.calibrate.pupil import OpticsSettings
    # 260821: the transmitted slider empty, the wheel at None -- the band is the splitting
    # dichroic's, which nothing records
    tags = dict(M2_BEADS, **{"Servos-Position3": "57000", "Thorlabs ELL9-Label": "Position 1"})
    camera = camera_of(tags)
    # the main channel the transmitted arm, whose slider is the empty one
    assert OpticsSettings().resolved(camera, 2)["wavelengths"] == [None, 676.0]
    assert OpticsSettings().missing(camera, 2) == ["emission"]
    assert OpticsSettings(wavelength_nm=700.0).missing(camera, 2) == []
    # with the reflected arm chosen as the main channel, the 2nd is the missing one
    assert OpticsSettings().missing(camera, 2, main="left") == ["emission, 2nd"]
    # a microscope with no second arm: the same light on both halves
    one_arm = camera_of(M5_BEADS)
    one_arm.microscope["channels"] = one_arm.microscope["channels"][:1]
    assert OpticsSettings().resolved(one_arm, 2)["wavelengths"] == [685.0, 685.0]


# ------------------------------------------- the objective, the filters, the arrangement (2026-11)

def test_the_objective_section_gives_the_na_and_wins_over_an_older_entrys_parameters():
    entry = Microscope(name="scope", parameters={"NA": 1.3, "n_immersion": 1.33},
                       objective={"name": "Olympus 100x NA1.50", "NA": 1.5, "immersion": "oil",
                                  "n_immersion": 1.518})
    found = entry.resolve({})
    assert found.values["NA"] == 1.5 and found.values["n_immersion"] == 1.518
    assert found.values["objective"] == "Olympus 100x NA1.50" and found.values["immersion"] == "oil"
    assert found.sources["NA"].kind == "fixed"
    # an entry from before the section still reads its parameters
    assert Microscope(name="old", parameters={"NA": 1.3}).resolve({}).values["NA"] == 1.3
    assert Microscope.from_dict(entry.to_dict()).objective == entry.objective


def test_a_filter_whose_name_is_not_its_band_takes_the_band_typed_for_it():
    wheel = Device(name="wheel", tags=["Wheel-State"], states={"676/37": "0", "665LP": "1", "Empty": "2"})
    entry = Microscope(name="scope", devices={"wheel": wheel},
                       parameters={"wavelength_nm": {"device": "wheel", "values": FROM_NAME}},
                       filters={"665LP": {"band_nm": [665, 715]}})
    assert entry.resolve({"Wheel-State": "0"}).values["wavelength_nm"] == 676       # the name
    assert entry.resolve({"Wheel-State": "1"}).values["wavelength_nm"] == 690       # the band's centre
    empty = entry.resolve({"Wheel-State": "2"})
    assert empty.values["wavelength_nm"] is None and "no band typed" in empty.sources["wavelength_nm"].detail


def test_a_kept_pupil_is_told_apart_by_the_colour_not_the_filter():
    assert [microscope_db.colour_of(w) for w in (452, 525, 600, 676, 700, None)] == \
        ["blue", "green", "red", "far red", "far red", ""]
    entry = Microscope(name="scope", devices={"3D lens": lens()},
                       parameters={"modality": {"device": "3D lens",
                                                "values": {"in": "astigmatic", "out": "2D"}}})
    assert entry.pupil_key({"3D lens": "in"}, 676) == "3D lens: in · far red"
    entry.pupils = {"3D lens: in · red": {"kind": "minimal", "gamma": 0.2},
                    "3D lens: in": {"kind": "minimal", "gamma": 0.3}}
    assert entry.pupil_for({"3D lens": "in"}, 600)["gamma"] == 0.2
    assert entry.pupil_for({"3D lens": "in"}, 700)["gamma"] == 0.3      # kept before colours were
    assert entry.pupil_for({"3D lens": "out"}, 600) is None


def test_m2s_split_is_its_arms_the_transmitted_one_main_on_the_right_the_reflected_mirrored():
    m2 = microscope_db.database(reload=True).get("M2")
    assert m2.arrangement() == {"layout": "right-left mirrored", "mirrored": True,
                                "main_channel": "right", "mirrors": {"right": "", "left": "x"}}
    layout = m2.split_layout((0, 0, 512, 512))
    assert layout.regions == [(256, 0, 256, 512), (0, 0, 256, 512)]
    assert layout.flips == [(False, False), (True, False)]
    # a camera ROI: the frame's middle, on the chip
    assert m2.split_layout((100, 50, 300, 200)).regions == [(250, 50, 150, 200), (100, 50, 150, 200)]
    found = microscope_db.database().resolve(M2_BEADS, channels=2)
    assert [(c.name, c.side, c.mirror) for c in found.channels] == [
        ("transmitted", "right", ""), ("reflected", "left", "x")]
    assert found.values["objective"] == "Leica 160x NA1.43"


def test_the_arms_regions_are_where_the_split_lies_whatever_the_frame_and_say_their_sides():
    m2 = microscope_db.database(reload=True).get("M2")
    transmitted, reflected = m2.channels
    transmitted.region, reflected.region = {"iXon 1": [240, 0, 272, 512]}, [0, 0, 236, 512]
    for roi in ((0, 0, 512, 512), (100, 0, 300, 512)):         # the frame does not move them
        assert m2.split_layout(roi).regions == [(240, 0, 272, 512), (0, 0, 236, 512)]
    # without sides, the regions say them
    transmitted.side = reflected.side = ""
    assert m2.arrangement()["main_channel"] == "right"
    assert m2.arrangement()["regions"] == {"right": [240, 0, 272, 512], "left": [0, 0, 236, 512]}
    # a side that contradicts its region is refused, not resolved one way or the other
    transmitted.side = "left"
    with pytest.raises(ValueError, match="by its region"):
        m2.arrangement()
    # a version holds them: the arrangement as it stood
    transmitted.side = ""
    m2.versions = [{"until": "2026-10-01", "main_channel": "reflected", "channels": {
        "transmitted": {"filter": "Channel 1 (transmitted)", "region": [256, 0, 256, 512]},
        "reflected": {"filter": "Channel 2 (reflected)", "region": [0, 0, 256, 512],
                      "mirror": "x"}}}]
    before = m2.as_of(datetime.date(2026, 9, 1))
    assert before.arrangement()["main_channel"] == "left"
    assert before.split_layout((0, 0, 512, 512)).regions == [(0, 0, 256, 512), (256, 0, 256, 512)]


def test_the_fitters_cut_the_channels_at_the_arms_regions_and_read_a_typed_split_only_without():
    from smappy.calibrate.dual import ArrangementSettings, chip_channels
    from smappy.plugins.fit import ChannelsSettings
    m2 = microscope_db.database(reload=True).get("M2")
    m2.channels[0].region, m2.channels[1].region = [240, 0, 272, 512], [0, 0, 236, 512]
    for typed in (None, 300):                                 # the entry's regions win
        told, chip = chip_channels(ArrangementSettings(split_position=typed), (0, 0, 512, 512), m2)
        assert chip.regions == [(240, 0, 272, 512), (0, 0, 236, 512)]
        assert chip.flips == [(False, False), (True, False)] and told.main_channel == "right"
    # a layout chosen by hand is not the entry's: its split is the typed one, else the middle
    chosen = ChannelsSettings(split="two channels", layout="right-left", main_channel="left",
                              split_position=300)
    _, chip = chip_channels(chosen, (0, 0, 512, 512), m2)
    assert chip.regions == [(0, 0, 300, 512), (300, 0, 212, 512)]


def test_the_main_channel_is_swapped_on_the_microscopes_layout_keeping_its_regions():
    """The dye registered onto is chosen per experiment: the other half as main, at the entry's
    regions and mirror, without choosing the layout by hand (2026-11)."""
    from smappy.calibrate.dual import ArrangementSettings, chip_channels, resolve_arrangement
    m2 = microscope_db.database(reload=True).get("M2")
    m2.channels[0].region, m2.channels[1].region = [240, 0, 272, 512], [0, 0, 236, 512]
    told, chip = chip_channels(ArrangementSettings(main_channel="left"), (0, 0, 512, 512), m2)
    assert told.layout == "right-left mirrored" and told.main_channel == "left"
    assert chip.regions == [(0, 0, 236, 512), (240, 0, 272, 512)]
    assert chip.flips == [(False, False), (True, False)]       # still mirrored against each other
    # the microscope's is its transmitted arm; a half off the layout's axis is refused by name
    assert resolve_arrangement(ArrangementSettings(), m2).main_channel == "right"
    with pytest.raises(ValueError, match="microscope's entry splits the chip right-left"):
        resolve_arrangement(ArrangementSettings(main_channel="upper"), m2)


def test_an_entry_from_before_the_arms_said_the_split_reads_its_mirror_and_warns_of_its_split():
    raw = microscope_db.database(reload=True).get("M5").to_dict()
    raw["layout"] = "right-left mirrored"
    raw["channels"]["reflected"].pop("mirror")
    raw["split_position"] = 256
    with pytest.warns(UserWarning, match="give the arms their regions"):
        entry = Microscope.from_dict(raw)
    assert entry.layout == "right-left" and entry.channels[1].mirror == "x"
    assert "split_position" not in entry.to_dict()


def test_m5_and_m6_know_their_layout_before_their_arms_sides():
    db = microscope_db.database(reload=True)
    assert db.get("M5").objective == {"name": "Olympus 100x NA1.50", "NA": 1.5, "immersion": "oil",
                                      "n_immersion": 1.518}
    assert db.get("M6").objective["name"] == "Olympus 100x NA1.49" and db.get("M6").objective["NA"] == 1.49
    for name in ("M5", "M6"):
        entry = db.get(name)
        assert entry.layout == "right-left"                        # the axis; the mirror the arm's
        assert entry.arrangement()["layout"] == "right-left mirrored"
        assert entry.arrangement()["main_channel"] == "left"      # as a dual calibration's auto
        assert entry.split_layout((0, 0, 1000, 300)).flips == [(False, False), (True, False)]


M6_MOVIE = {
    "Core-Camera": "HamamatsuHam_DCAM", "iChrome-MLE-Laser 1: 1. Enable": "1",
    "Thorlabs ELL20-Position (um)": "44317", "Thorlabs ELL17-Position (um)": "19040",
    "Thorlabs ELL6-State": "1", "transmitted_ Filter Wheel-2-State": "2",
    "reflected_ Filter Wheel-State": "3", "StartTime": "2026-11-04 10:00:00.000 +0100",
}


def test_m6_is_chosen_by_hand_and_reads_its_own_devices():
    db = microscope_db.database(reload=True)
    assert db.resolve(M6_MOVIE).microscope is None                  # nothing identifies it yet
    found = db.resolve(M6_MOVIE, microscope="M6", camera="Fusion BT 2", channels=2)
    states = found.state_names()
    assert states["3D"] == "on" and states["TIRF"] == "on" and states["laser 640"] == "on"
    assert [(c.name, c.wavelength_nm) for c in found.channels] == [("transmitted", 676.0),
                                                                    ("reflected", 700.0)]
    camera = camera_db.database(reload=True).get("Fusion BT 2")
    assert camera is not None and camera.parameters["pixelsize_um"].fixed == [0.102, 0.108]


def test_the_shipped_m2_is_recognised_before_and_after_its_focus_lock_laser_was_exchanged():
    """M2 is recognised by its Luxx 638's serial, which every file of 2026 records; the focus-lock
    laser it was recognised by was exchanged in November 2026 and the files stopped matching."""
    from smappy.microscope_db import database
    before = dict(M2_BEADS)
    after = dict(M2_BEADS, **{"Focus-lock-Serial ID": "iBEAM-SMART-785-S-A3-13738",
                              "StartTime": "2026-11-06 17:28:02.385 +0100"})
    for tags in (before, after, M2_EVOLVE):
        assert database().identify(tags).name == "M2"
