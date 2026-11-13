"""The sample's refractive index as a choice with presets (`smappy.calibrate.medium`): the same field in the PSF
calibration's optics, Spline 3D's focus and the sample part, with the old numbers still read."""
import pytest

from smappy.calibrate.depth import SampleSettings
from smappy.calibrate.medium import BUFFER, WATER, cleared, for_drawing, resolve, settle
from smappy.calibrate.pupil import OpticsSettings
from smappy.plugins import settings_from, settings_values
from smappy.plugins.fit import SplineModelSettings


def test_the_presets_stand_for_water_a_blinking_buffer_and_the_immersion():
    assert resolve("water", None, None) == WATER == 1.33
    assert resolve("buffer", None, None) == BUFFER == 1.345
    assert resolve("matched", None, None, 1.518) == 1.518
    assert resolve("matched", None, None) == "matched"            # for the model's immersion
    assert resolve("typed", 1.36, None) == 1.36
    assert resolve("", None, 1.33) == 1.33 and resolve("typed", None, 1.4) == 1.4
    with pytest.raises(ValueError):
        settle("glycerol", None)


def test_the_calibration_keeps_water_and_4pi_the_immersion_on_auto():
    assert OpticsSettings().sample_index() == 1.33
    assert OpticsSettings().sample_index(True, 1.406) == 1.406
    assert OpticsSettings(medium="buffer").sample_index(True, 1.406) == 1.345
    assert OpticsSettings(medium="matched").sample_index(False, 1.406) == 1.406


def test_drawing_in_the_sample_keeps_the_calibrations_index_on_auto():
    assert SampleSettings().n_med() is None and SplineModelSettings().n_med() is None
    assert SampleSettings(medium="buffer").n_med() == 1.345
    assert SplineModelSettings(medium="matched").n_med() == "matched"


@pytest.mark.parametrize("cls, name", [(OpticsSettings, "n_medium"), (SampleSettings, "sample_index"),
                                       (SplineModelSettings, "sample_index")])
def test_an_old_typed_number_still_reads_as_typed(cls, name):
    old = settings_from(cls, {name: 1.36})
    assert settle(old.medium, getattr(old, name)) == "typed"
    assert resolve(old.medium, getattr(old, name), 1.33) == 1.36
    again = settings_from(cls, settings_values(old))
    assert getattr(again, name) == 1.36 and resolve(again.medium, getattr(again, name), 1.33) == 1.36
    assert settings_from(cls, {}).medium == ""


def test_leaving_typed_drops_the_number_so_auto_is_auto_again():
    assert cleared("model.medium", "", 1.36, "model.medium", "model.sample_index") == {"model.sample_index": None}
    assert cleared("model.medium", "typed", 1.36, "model.medium", "model.sample_index") == {}
    assert cleared("model.focus", "", 1.36, "model.medium", "model.sample_index") == {}
    assert for_drawing("water", 1.36) == 1.33


def test_a_number_typed_over_a_preset_makes_it_typed():
    assert cleared("model.sample_index", "buffer", 1.36, "model.medium", "model.sample_index") == \
        {"model.medium": "typed"}
    assert cleared("model.sample_index", "typed", 1.36, "model.medium", "model.sample_index") == {}
    assert cleared("model.sample_index", "buffer", None, "model.medium", "model.sample_index") == {}
