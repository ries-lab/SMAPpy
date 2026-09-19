"""Metadata merging and Micro-Manager parsing rules."""
import pytest

from smappy.metadata import CameraMetadata
from smappy.io.tiff import _parse_roi
from smappy.io.cameras_mat import reader_for


def test_roi_separator_is_not_a_minus_sign():
    assert _parse_roi("208-273-200-200") == (208, 273, 200, 200)
    assert _parse_roi("nonsense") is None
    assert _parse_roi(None) is None


def test_matlab_expressions_become_named_readers():
    """SMAP stores how to read a tag as a MATLAB expression; the database
    stores it as a name, and the converter is what translates."""
    assert reader_for("str2double(X)") == {"read": "number"}
    assert reader_for("str2num(X)") == {"read": "numbers"}
    assert reader_for("str2double(X)>0") == {"read": "positive"}
    assert reader_for("~strcmp(X,'Normal')") == {"read": "not_equals",
                                                 "argument": "Normal"}
    assert reader_for("strcmp(X,'On')") == {"read": "equals", "argument": "On"}
    assert reader_for("") == {"read": "text"}


def test_overrides_do_not_clear_unspecified_fields():
    """A config that only sets the pixel size must not switch EM off."""
    from_file = CameraMetadata(conversion=6.7, offset=400.0, em_on=True, emgain=100)
    merged = from_file.merged_with(CameraMetadata(pixelsize_um=0.127))
    assert merged.em_on is True and merged.emgain == 100
    assert merged.pixelsize_um == 0.127
    assert merged.adu_to_photons == pytest.approx(0.067)
    assert merged.excess_noise == 2.0


def test_overrides_win_when_set():
    from_file = CameraMetadata(conversion=6.7, offset=400.0, em_on=True, emgain=100)
    merged = from_file.merged_with(CameraMetadata(offset=398.6, em_on=False))
    assert merged.offset == 398.6
    assert merged.is_em is False
    assert merged.adu_to_photons == 6.7
    assert merged.excess_noise == 1.0


def test_missing_values_are_reported():
    with pytest.raises(ValueError, match="pixelsize_um"):
        CameraMetadata(conversion=6.7, offset=400.0).require()


def test_shipped_camera_presets_are_complete():
    """Every camera YAML in ``examples`` is a camera you can fit with.

    A preset that left out the conversion or the offset would only fail once
    the metadata failed to fill it in -- for an iXon, which reports no
    baseline, that is at the fit.
    """
    from pathlib import Path

    presets = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.yaml"))
    assert [p.stem for p in presets] == ["camera_andor_ixon897", "camera_evolve512"]
    for path in presets:
        CameraMetadata.from_yaml(path).require()


def test_andor_preset_matches_the_smap_settings_file():
    """The iXon preset holds one readout state of SMAP's RiesLab_cameras.mat.

    The values are the mode the camera is normally run in (pre-amp Gain 1,
    10 MHz, electron multiplying); EM is left to the image metadata, which
    records it per acquisition.
    """
    from pathlib import Path

    cam = CameraMetadata.from_yaml(
        Path(__file__).resolve().parents[1] / "examples" / "camera_andor_ixon897.yaml")
    assert (cam.conversion, cam.offset) == (15.8, 196)
    assert cam.pixelsize_um == 0.127
    assert cam.em_on is None and cam.emgain is None
