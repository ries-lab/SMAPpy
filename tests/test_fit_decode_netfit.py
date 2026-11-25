"""SMAPpy's own network in the DECODE fitter (*DECODE, optimised*; docs/decode/gui-plan.md, step 1).

A one-channel netfit model is fitted by ``Localize/DECODE`` exactly as the port's is: loaded by
`training.predictors.load`, driven by the `WindowEngine`, its columns SMAPpy's.  The network here is
untrained (8 features), so the tests are of the plumbing; that netfit's numbers are good is the
study's (docs/decode/poc-plan.md).
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

torch = pytest.importorskip("torch")

import smappy
from smappy import plugins
from smappy.locs import Localizations
from smappy.netfit import network
from smappy.netfit.network import Scaling
from smappy.netfit.settings import SimulationSettings, TrainingSettings
from smappy.netfit.train import METHOD, trained_on
from smappy.plugins import Context
from smappy.plugins.fit_decode import model_summary, range_warnings, read_model
from smappy.training import model_file

SCALING = Scaling(input_offset=100.0, input_scale=100.0, phot_max=10000.0, z_max=1000.0, bg_max=200.0)


def _write(path, channels=None, psf=None):
    s = TrainingSettings()
    s.parts.backbone.settings["features"] = 8
    torch.manual_seed(0)
    model = network.build(s.parts, 3).eval()
    sim = SimulationSettings(photons_range=(500.0, 8000.0), background=(10.0, 80.0))
    meta = {"method": METHOD, "settings": s.to_dict(), "scaling": SCALING.__dict__, "window": 3,
            "calibration_em_on": False, "calibration_source": "/beads/bead_3dcal.h5", "epoch": 12, "epochs": 100,
            "trained_on": trained_on(sim)}
    if channels:
        meta["channels"] = channels
    if psf:
        meta["psf"] = psf
    model_file.write(path, {k: v.numpy() for k, v in model.state_dict().items()}, meta)
    return path


@pytest.fixture
def files(tmp_path):
    rng = np.random.default_rng(1)
    tifffile.imwrite(tmp_path / "frames.tif", rng.poisson(120, (12, 24, 28)).astype(np.uint16), photometric="minisblack")
    return {"model": _write(tmp_path / "model.safetensors"), "tif": tmp_path / "frames.tif", "root": tmp_path}


def _settings(files, **more):
    from smappy.gui.live_session import fit_settings
    values = {"source.path": str(files["tif"]), "camera.conversion": 1.0, "camera.offset": 0.0,
              "camera.pixelsize_um": 0.1, "camera.em_on": False, "model.path": str(files["model"]),
              "model.device": "cpu", "model.threshold": 0.0, "model.batch": 4, "output.path": "",
              "output.raw_frames": 0}
    values.update(more)
    return fit_settings("Localize/DECODE", values)


def test_a_netfit_model_is_fitted_by_the_decode_fitter_as_its_own_predictor_localizes_the_movie(files):
    from smappy.netfit.predict import Predictor
    locs = plugins.get("Localize/DECODE")().run(Context(), _settings(files)).locs
    for name in ("frame", "x_nm", "y_nm", "z_nm", "photons", "background", "prob", "x_err_nm", "z_err_nm"):
        assert name in locs, name
    assert set(np.unique(locs["frame"])) <= set(range(12)) and len(np.unique(locs["frame"])) == 12
    # the same molecules as the predictor gives on the whole movie, its ends repeated for context
    frames = tifffile.imread(files["tif"]).astype(np.float32)
    stack = np.concatenate([frames[:1], frames, frames[-1:]])
    cols = Predictor(files["model"], "cpu", threshold=0.0, batch=4)(stack, 0)
    assert len(cols["frame"]) == len(locs)
    np.testing.assert_allclose(np.sort(np.asarray(locs["x_nm"])), np.sort(cols["x_pix"] * 100.0), rtol=1e-5)


def test_the_form_says_what_a_netfit_model_is_without_loading_torch(files):
    code = ("import sys; from smappy import plugins; from smappy.gui.live_session import fit_settings; "
            f"s = fit_settings('Localize/DECODE', {{'source.path': {str(files['tif'])!r}, "
            f"'model.path': {str(files['model'])!r}, 'camera.conversion': 1.0, 'camera.offset': 0.0, "
            "'camera.pixelsize_um': 0.1}); "
            "state = plugins.get('Localize/DECODE')().describe(None, s); "
            "print('torch' in sys.modules); print(state.notice.text)")
    # this tree's smappy, not the one installed (`conftest.py` puts it first only in this process)
    env = {**os.environ, "PYTHONPATH": str(Path(smappy.__file__).parents[1])}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env=env).stdout.splitlines()
    assert out[0] == "False"
    assert out[1] == "model (DECODE, optimised): trained on bead_3dcal.h5, epoch 12"
    rows = dict(model_summary(read_model(files["model"])))
    assert rows["model"] == "DECODE, optimised (netfit-v1)" and rows["photons a frame"] == "500 to 8,000"
    assert rows["frame window"] == "3" and rows["background"] == "10 to 80 photons a pixel"


def test_a_fit_brighter_than_a_netfit_training_drew_is_warned_about(files):
    meta = read_model(files["model"])
    calm = Localizations({"photons": np.full(100, 3000.0, np.float32), "background": np.full(100, 30.0, np.float32)})
    assert range_warnings(calm, meta) == []
    bright = Localizations({"photons": np.r_[np.full(90, 3000.0), np.full(10, 9000.0)].astype(np.float32),
                            "background": np.full(100, 30.0, np.float32)})
    said = range_warnings(bright, meta)
    assert len(said) == 1 and "10%" in said[0] and "the top of its range" in said[0]


def test_the_one_channel_loader_refuses_a_two_channel_model_which_the_fitter_cuts_itself(files):
    """`predictors.load` gives the engine a predictor of whole frames: a two-channel model needs its
    channels cut first (`netfit.fit_two`, the fitter's ``two_channel_model``); the remote server, which
    loads through it, runs one-channel models only."""
    from smappy.training.predictors import load
    path = _write(files["root"] / "two.safetensors", channels={"shift": [0, 0], "box": [0, 8, 0, 8], "main": 0})
    assert read_model(path)["channels"]
    with pytest.raises(ValueError, match="reads two channels"):
        load(path, "cpu")


def test_the_automatic_batch_fills_the_memory_it_is_given_and_is_fixed_on_a_cpu(monkeypatch):
    from smappy.netfit import predict
    assert predict.auto_batch("cpu", 256, 256) == predict.CPU_BATCH
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (8e9, 16e9))
    # half of 8 GB over 5 000 bytes a pixel of 256 x 256 frames
    assert predict.auto_batch("cuda:0", 256, 256) == int(4e9 // (predict.BYTES_PER_PIXEL * 256 * 256))
    assert predict.auto_batch("cuda:0", 4096, 4096) == 1


def test_a_model_trained_on_gaussians_reports_their_width_where_another_reports_z(files):
    from smappy.plugins.fit_decode import summary_line
    psf = {"sigma_nm": [110.0, 170.0], "sigma_ref_nm": 140.0, "step_nm": 5.0, "pixel_nm": [100.0, 100.0]}
    model = _write(files["root"] / "gauss.safetensors", psf=psf)
    plain = plugins.get("Localize/DECODE")().run(Context(), _settings(files)).locs
    locs = plugins.get("Localize/DECODE")().run(Context(), _settings(files, **{"model.path": str(model)})).locs
    assert "z_nm" not in locs and "z_err_nm" not in locs
    # the same network: its z is the width less the range's middle
    np.testing.assert_allclose(np.sort(np.asarray(locs["sigma_nm"])), np.sort(np.asarray(plain["z_nm"]) + 140.0),
                               rtol=1e-5)
    np.testing.assert_allclose(np.sort(np.asarray(locs["sigma_err_nm"])), np.sort(np.asarray(plain["z_err_nm"])),
                               rtol=1e-5)
    meta = read_model(model)
    assert "Gaussians 110 -- 170 nm wide" in summary_line(meta)
    rows = dict(model_summary(meta))
    assert rows["width (sigma)"] == "110 to 170 nm" and "z" not in rows and rows["PSF calibration"] == "none: Gaussians"
