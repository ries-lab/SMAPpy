"""The DECODE fitter's form and checks, without torch (`smappy.plugins.fit_decode`).

The model file is written here with numpy alone -- weights that no network is built from, a
header that says what a trained model's says -- because the form, and so these, must never need
torch.  What the network does is tests/decode_tests/test_fit_plugin.py.
"""
import subprocess
import sys

import numpy as np
import pytest
import tifffile

from smappy.io.calibration import SplineCalibration, save_spline_calibration
from smappy.locs import Localizations
from smappy.plugins.fit_decode import model_summary, range_warnings, read_model
from smappy.training import model_file
from smappy.training.job import coefficients_sha256


def _calibration(width):
    g = np.exp(-0.5 * ((np.arange(13) - 6.0) / width) ** 2)
    coeff = np.zeros((64, 11, 13, 13), np.float32)
    coeff[0] = (np.outer(g, g) / np.outer(g, g).sum())[None]
    return SplineCalibration(coeff=coeff, dz=100.0, z0=5.0, x0=7.0, em_on=False)


@pytest.fixture
def files(tmp_path):
    cal = _calibration(1.5)
    save_spline_calibration(tmp_path / "beads.h5", cal)
    save_spline_calibration(tmp_path / "other.h5", _calibration(2.5))
    meta = {"method": "decode-v0.10.2", "architecture": "SigmaMUNet", "channels_in": 3,
            "calibration_sha256": coefficients_sha256(cal), "calibration_em_on": False,
            "calibration_source": str(tmp_path / "beads.h5"), "epoch": 311, "epochs": 1000,
            "calibration_microscope": {"name": "M2", "version": 1, "followed": ["3DA"], "states": {"3DA": "on"}},
            "trained_on": {"photons_mu_sig": [3000.0, 600.0], "background": [20.0, 60.0], "z_nm": [-400.0, 400.0]}}
    model_file.write(tmp_path / "model.safetensors", {"w": np.zeros(3, np.float32)}, meta)
    tifffile.imwrite(tmp_path / "frames.tif", np.full((4, 16, 16), 110, np.uint16), photometric="minisblack")
    return tmp_path


def _settings(files, **more):
    from smappy.gui.live_session import fit_settings
    values = {"source.path": str(files / "frames.tif"), "camera.conversion": 0.5, "camera.offset": 100.0,
              "camera.pixelsize_um": 0.1, "model.path": str(files / "model.safetensors")}
    values.update(more)
    return fit_settings("Localize/DECODE", values)


def test_the_model_info_says_what_the_model_was_trained_on_and_how_the_data_differ(files):
    meta = read_model(files / "model.safetensors")
    rows = dict(model_summary(meta))
    assert rows["PSF calibration"] == "beads.h5" and rows["trained"] == "epoch 311 of at most 1000"
    assert rows["photons a frame"] == "3,000 \u00b1 600" and "differs from the data" not in rows
    data = {"name": "M2", "version": 1, "followed": ["3DA"], "states": {"3DA": "off"}}
    first = model_summary(meta, data)[0]
    assert first == ("differs from the data", "3DA on for the beads, off for the data")


def test_a_file_that_is_not_a_decode_model_is_refused(files):
    model_file.write(files / "x.safetensors", {"w": np.zeros(1, np.float32)}, {"method": "other"})
    with pytest.raises(ValueError, match="not a DECODE model"):
        read_model(files / "x.safetensors")
    with pytest.raises(ValueError, match="no such model"):
        read_model(files / "missing.safetensors")


def test_the_form_warns_of_other_conditions_and_else_shows_the_model_in_a_line(files, monkeypatch):
    from smappy import plugins
    from smappy.plugins import fit_decode
    plugin = plugins.get("Localize/DECODE")()
    monkeypatch.setattr(fit_decode.DecodeFit, "data_conditions",
                        lambda self, s: {"name": "M1", "version": 1, "followed": [], "states": {}})
    assert "another microscope: M2 for the model's beads, M1" in plugin.describe(None, _settings(files)).notice.text
    monkeypatch.setattr(fit_decode.DecodeFit, "data_conditions", lambda self, s: None)      # not known: no word
    assert plugin.describe(None, _settings(files)).notice.text == \
        "model: trained on beads.h5, microscope M2, epoch 311"


def test_the_form_loads_no_torch(files):
    code = ("import sys; from smappy import plugins; from smappy.gui.live_session import fit_settings; "
            f"s = fit_settings('Localize/DECODE', {{'source.path': {str(files / 'frames.tif')!r}, "
            f"'model.path': {str(files / 'model.safetensors')!r}, "
            f"'camera.conversion': 0.5, 'camera.offset': 100.0}}); "
            "state = plugins.get('Localize/DECODE')().describe(None, s); "
            "print('torch' in sys.modules, state.notice is not None)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "True"]


def test_a_fit_brighter_than_the_training_drew_is_warned_about(files):
    meta = read_model(files / "model.safetensors")
    n = 100
    calm = Localizations({"photons": np.full(n, 3000.0, np.float32), "background": np.full(n, 30.0, np.float32)})
    assert range_warnings(calm, meta) == []
    bright = Localizations({"photons": np.r_[np.full(80, 3000.0), np.full(20, 9000.0)].astype(np.float32),
                            "background": np.r_[np.full(90, 30.0), np.full(10, 200.0)].astype(np.float32)})
    said = range_warnings(bright, meta)
    assert len(said) == 2 and "20%" in said[0] and "10%" in said[1]


def test_without_torch_the_form_and_a_run_here_say_what_to_install(files, monkeypatch):
    from smappy import plugins
    from smappy.plugins import Context, fit_decode
    monkeypatch.setattr(fit_decode, "torch_here", lambda: False)
    plugin = plugins.get("Localize/DECODE")()
    assert "smappy-smlm[decode]" in plugin.describe(None, _settings(files)).notice.text
    with pytest.raises(ValueError, match=r"smappy-smlm\[decode\]"):
        plugin.run(Context(), _settings(files))
    # on a training host the network runs there: nothing to install here
    assert "smappy-smlm" not in (plugin.describe(None, _settings(files, **{"model.run_on": "box"})).notice or
                                 type("n", (), {"text": ""})).text
