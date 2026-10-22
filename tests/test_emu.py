"""Reading an EMU configuration into microscope devices.

The configuration says which Micro-Manager property is which control and
what its states are called -- the chain that turns ``Servos-Position1 =
28000`` into *3DA on*.  The configuration below is cut down from M2's.
"""
import json

import pytest

from smappy.io import emu
from smappy.microscope_db import FROM_NAME

M2 = {
    "properties": {
        "Filter wheel position": "Servos-Position3",
        "Filter wheel position state 0": "5000",
        "Filter wheel position state 1": "32000",
        "Filter wheel position state 2": "57000",
        "Slider 1 position": "Servos-Position0",
        "Slider 1 position state 0": "20000",
        "Slider 1 position state 1": "10000",
        "Slider 1 position state 2": "10000",
        "Slider 2 position": "ELL9 - Reflected Slider-Label",
        "Slider 2 position state 0": "Position 3",
        "Slider 2 position state 1": "Position 1",
        "Two-state device 1": "ELL9 BFP-Label",
        "Two-state device 1 - On value": "Position 1",
        "Two-state device 1 - Off value": "Position 0",
        "Two-state device 2": "Servos-Position1",
        "Two-state device 2 - On value": "28000",
        "Two-state device 2 - Off value": "4000",
        "Two-state device 4": "Unallocated",
        "Two-state device 4 - On value": "Enter value",
        "Laser 3 enable": "Luxx638-Laser Operation Select",
        "Laser 3 enable - On value": "On",
        "Laser 3 enable - Off value": "Off",
        "QPD X": "Analog Input-AnalogInput0",
    },
    "parameters": {
        "Filters - Filter names": "440/40,700/100,None",
        "2C - Slider 1 title": "Channel 1 (transmitted)",
        "2C - Slider 1 names": "676/37,685/70,empty",
        "2C - Slider 2 title": "Channel 2 (reflected)",
        "2C - Slider 2 names": "600/60,676/37",
        "Controls - Two-state device 1 name": "BFP",
        "Controls - Two-state device 2 name": "3DA",
        "Controls - Enable two-state device 4": "false",
        "Laser 3 - Name": "638",
    },
    "settings": {"Additional FW tab title": "2C"},
}


def write(tmp_path, plugin, name="M2-config.uicfg"):
    path = tmp_path / name
    path.write_text(json.dumps({
        "defaultConfigurationName": "ht-SMLM", "globalSettings": {},
        "pluginConfigurations": [dict(configurationName="ht-SMLM", pluginName="ht-SMLM",
                                      **plugin)]}))
    return path


def test_each_allocated_control_becomes_a_device_with_named_states(tmp_path):
    found, warnings = emu.devices(emu.read(write(tmp_path, M2)))
    assert list(found) == ["filter wheel", "Channel 1 (transmitted)",
                           "Channel 2 (reflected)", "BFP", "3DA", "laser 638"]
    assert found["filter wheel"].states == {"440/40": "5000", "700/100": "32000",
                                            "None": "57000"}
    assert found["3DA"].states == {"on": "28000", "off": "4000"}
    assert found["3DA"].tags == ["Servos-Position1"] and found["3DA"].read == "nearest"
    assert found["BFP"].read == "exact"                 # a label, compared as text
    assert found["3DA"].emu == "Two-state device 2"


def test_a_value_given_to_two_states_is_said(tmp_path):
    _, warnings = emu.devices(emu.read(write(tmp_path, M2)))
    assert warnings == ["Channel 1 (transmitted): empty and 685/70 are both at 10000; "
                        "a file reading 10000 is taken as 685/70"]


def test_a_device_of_the_mojo_fpga_is_also_looked_up_without_its_prefix(tmp_path):
    old = json.loads(json.dumps(M2))
    old["properties"]["Two-state device 2"] = "Mojo-Servos-Position1"
    found, _ = emu.devices(emu.read(write(tmp_path, old)))
    assert found["3DA"].tags == ["Mojo-Servos-Position1", "Servos-Position1"]


def test_a_draft_follows_the_filter_and_the_lens_it_is_told_of(tmp_path):
    entry, _ = emu.draft(emu.read(write(tmp_path, M2)), "M2", three_d="3DA")
    assert entry.parameters["wavelength_nm"] == {"device": "filter wheel",
                                                 "values": FROM_NAME}
    assert entry.parameters["modality"] == {"device": "3DA", "values": {
        "on": "astigmatic", "off": "2D"}}
    assert [(c.name, c.filter) for c in entry.channels] == [
        ("transmitted", "Channel 1 (transmitted)"), ("reflected", "Channel 2 (reflected)")]
    found = entry.resolve({"Servos-Position1": "27500", "Servos-Position3": "32000"})
    assert found.values["modality"] == "astigmatic"
    assert found.values["wavelength_nm"] == 700.0


def test_a_lens_the_configuration_does_not_have_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no device '3D'"):
        emu.draft(emu.read(write(tmp_path, M2)), three_d="3D")


def test_importing_again_keeps_the_names_and_the_old_tag_behind_the_new(tmp_path):
    entry, _ = emu.draft(emu.read(write(tmp_path, M2)), "M2", three_d="3DA")
    entry.devices["3D lens"] = entry.devices.pop("3DA")       # renamed by hand
    entry.devices["3D lens"].name = "3D lens"
    entry.parameters["modality"]["device"] = "3D lens"
    newer = json.loads(json.dumps(M2))
    newer["properties"]["Two-state device 2"] = "Servos-Position5"
    newer["properties"]["Two-state device 2 - On value"] = "30000"
    updated, _ = emu.update(entry, emu.read(write(tmp_path, newer, "newer.uicfg")))
    device = updated.devices["3D lens"]
    assert "3DA" not in updated.devices
    assert device.tags == ["Servos-Position5", "Servos-Position1"]
    assert device.states == {"on": "30000", "off": "4000"}
    assert updated.parameters["modality"]["device"] == "3D lens"


def test_the_configuration_file_carries_its_date(tmp_path):
    assert emu.read(write(tmp_path, M2)).modified is not None
