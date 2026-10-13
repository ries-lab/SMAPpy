"""Metadata merging and Micro-Manager parsing rules."""
import pytest

from smappy.metadata import CameraMetadata
from smappy.io.tiff import _parse_roi


def test_roi_separator_is_not_a_minus_sign():
    assert _parse_roi("208-273-200-200") == (208, 273, 200, 200)
    assert _parse_roi("nonsense") is None
    assert _parse_roi(None) is None


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


def test_missing_values_are_reported_but_a_pixel_size_is_not_required():
    """Without a pixel size a fit still runs, on an assumed one
    (`DEFAULT_PIXELSIZE_NM`); without a conversion it cannot."""
    CameraMetadata(conversion=6.7, offset=400.0).require()
    with pytest.raises(ValueError, match="conversion"):
        CameraMetadata(offset=400.0, pixelsize_um=0.1).require()
