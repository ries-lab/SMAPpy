"""A two-channel model in the DECODE fitter (*aligned channels*; docs/decode/gui-plan.md, step 3).

The fitter cuts the camera's frames into the network's two channels by the registration -- moved by
how far the splitter moved since the beads, typed or measured -- and puts the localizations back on
the calibration's main half, each channel's photons beside them.  The network here is untrained (8
features), so these test the plumbing against `netfit.fit_two` and `channels.TwoChannels` directly;
Fig. 2a through the fitter gave the study script's localizations (plex-plan §16).
"""
import hashlib

import numpy as np
import pytest
import tifffile

torch = pytest.importorskip("torch")

from test_netfit_channels import SHAPE, dual

from smappy import plugins
from smappy.calibrate.dual import save_dual_color_calibration
from smappy.netfit import channels as CH
from smappy.netfit import network
from smappy.netfit.settings import ChannelSettings, SimulationSettings, TrainingSettings
from smappy.netfit.train import METHOD, trained_on
from smappy.plugins import Context
from smappy.training import model_file


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def files(tmp_path):
    return make_files(tmp_path)


def make_files(tmp_path):
    """An untrained two-channel model, its calibration and a movie in ``tmp_path``."""
    cal = dual()
    save_dual_color_calibration(tmp_path / "beads_2ct.h5", cal)
    s = TrainingSettings()
    s.channels = ChannelSettings(kind="colour")
    s.parts.backbone.settings.update(features=8, two_fuse="early")
    sim = SimulationSettings(photons_range=(1000.0, 8000.0), background=(10.0, 80.0))
    torch.manual_seed(0)
    model = network.build(s.parts, 3, s.channels).eval()
    two = CH.TwoChannels.from_calibration(cal)
    meta = {"method": METHOD, "settings": s.to_dict(), "window": 3,
            "scaling": network.Scaling.for_simulation(sim, s.channels).__dict__,
            "calibration_em_on": False, "calibration_source": str(tmp_path / "beads_2ct.h5"),
            "calibration_sha256": _sha(tmp_path / "beads_2ct.h5"), "trained_on": trained_on(sim, s.channels),
            "channels": two.to_dict()}
    model_file.write(tmp_path / "two.safetensors", {k: v.numpy() for k, v in model.state_dict().items()}, meta)
    rng = np.random.default_rng(3)
    tifffile.imwrite(tmp_path / "frames.tif", rng.poisson(60, (8, *SHAPE)).astype(np.uint16), photometric="minisblack")
    return {"root": tmp_path, "model": tmp_path / "two.safetensors", "tif": tmp_path / "frames.tif", "cal": cal}


def _settings(files, **more):
    from smappy.gui.live_session import fit_settings
    values = {"source.path": str(files["tif"]), "camera.conversion": 1.0, "camera.offset": 0.0,
              "camera.pixelsize_um": 0.1, "camera.em_on": False, "model.path": str(files["model"]),
              "model.device": "cpu", "model.threshold": 0.0, "model.batch": 4, "output.path": "",
              "output.raw_frames": 0, "finish.assign_colors": False}
    values.update(more)
    return fit_settings("Localize/DECODE", values)


def test_a_two_channel_model_is_fitted_on_the_main_half_with_each_channel_s_photons(files):
    from smappy.netfit.fit_two import TwoChannelPredictor
    from smappy.netfit.predict import Predictor
    locs = plugins.get("Localize/DECODE")().run(Context(), _settings(files, **{"two_channels.moved_rows": 1.5})).locs
    for name in ("photons_ch0", "photons_ch1", "background_ch0", "background_ch1", "x_nm", "z_nm", "prob"):
        assert name in locs, name
    assert len(np.unique(locs["frame"])) == 8
    # the localizations the wrapper gives on the whole movie, its ends repeated for context
    frames = tifffile.imread(files["tif"]).astype(np.float32)
    stack = np.concatenate([frames[:1], frames, frames[-1:]])
    direct = TwoChannelPredictor(Predictor(files["model"], "cpu", threshold=0.0, batch=4), files["cal"],
                                 moved=(0.0, 1.5))(stack, 0)
    assert len(direct["frame"]) == len(locs)
    np.testing.assert_allclose(np.sort(np.asarray(locs["y_nm"])), np.sort(direct["y_pix"] * 100.0), rtol=1e-5)
    # on the main half (rows above the split), as the two-colour spline fitters report them
    assert np.asarray(locs["y_nm"]).max() < 80 * 100.0


def test_the_move_since_the_beads_moves_where_channel_one_is_read():
    from smappy.netfit.fit_two import moved_calibration
    cal = dual()
    p = np.array([[30.0, 20.0], [60.0, 50.0]])
    moved = moved_calibration(cal, (0.5, 2.0))
    before = CH.TwoChannels.from_calibration(cal)
    after = CH.TwoChannels.from_calibration(moved)
    # channel 1's partner of each grid position, in channel 1's own pixels, 2 rows and half a column on
    d0 = before.layout.to_channel(1, p)
    d1 = after.layout.to_channel(1, p)
    np.testing.assert_allclose(np.abs(d1 - d0), [[0.5, 2.0], [0.5, 2.0]], atol=1e-9)


def test_the_move_measured_is_where_the_network_finds_the_fewest_molecules(monkeypatch):
    from smappy.netfit import fit_two
    truth = (-1.25, 3.5)
    monkeypatch.setattr(fit_two, "count_at", lambda p, cal, corner, photons, m:
                        500 + 40 * (m[0] - truth[0]) ** 2 + 60 * (m[1] - truth[1]) ** 2)
    assert fit_two.measure_move(None, None, (0, 0), None) == pytest.approx(truth)


def test_the_form_shows_the_two_channels_part_for_a_two_channel_model_alone(files):
    from test_fit_decode_netfit import _write
    plugin = plugins.get("Localize/DECODE")()
    s = _settings(files)
    assert plugin.describe(None, s).shown.get("two_channels") is True
    one = _write(files["root"] / "one.safetensors")
    assert plugin.describe(None, _settings(files, **{"model.path": str(one)})).shown.get("two_channels") is False


def test_another_calibration_than_the_model_s_is_refused(files):
    from smappy.calibrate.dual import DualColorCalibration
    cal = files["cal"]
    other = DualColorCalibration(cal.main, cal.secondary, np.asarray(cal.transformation) * 1.0001, cal.geometry,
                                 cal.parameters)
    save_dual_color_calibration(files["root"] / "other_2ct.h5", other)
    with pytest.raises(ValueError, match="not the calibration the model was trained on"):
        plugins.get("Localize/DECODE")().run(Context(), _settings(
            files, **{"two_channels.calibration": str(files["root"] / "other_2ct.h5")}))


def test_a_move_measured_on_the_first_frames_is_what_the_run_s_record_says(files, monkeypatch):
    import json

    import h5py

    from smappy.netfit import fit_two
    monkeypatch.setattr(fit_two, "measure_move", lambda predictor, cal, corner, photons, start: (0.5, -1.25))
    out = files["root"] / "fit_locs.hdf5"
    r = plugins.get("Localize/DECODE")().run(Context(), _settings(files, **{"two_channels.measure_move": True,
                                                                             "output.path": str(out)}))
    moved = json.loads(h5py.File(out, "r").attrs["metadata"])["decode"]["moved_since_beads"]
    assert moved == {"rows": -1.25, "columns": 0.5, "measured": True}
    assert any("-1.25 rows and 0.50 columns" in w for w in r.locs.metadata.get("warnings", []))


def test_the_goodness_draws_each_molecule_over_the_network_s_own_background_map(files):
    """Over a level a frame, an uneven background's light under a spot counts against it (plex-plan §16)."""
    from smappy.netfit.fit_two import TwoChannelPredictor
    from smappy.netfit.goodness import fit_goodness
    from smappy.netfit.predict import Predictor
    frames = tifffile.imread(files["tif"]).astype(np.float32)
    stack = np.concatenate([frames[:1], frames, frames[-1:]])
    predictor = Predictor(files["model"], "cpu", threshold=0.0, batch=4)
    wrapped = TwoChannelPredictor(predictor, files["cal"], goodness=True)
    assert predictor.keep_background
    out = wrapped(stack, 0)
    # the map the goodness used: the predictor's, on the channels as cut, one per frame read
    two = wrapped.channels(stack.shape[1:])
    raw = predictor(two.cut(stack), 0, two.offsets())
    bg = raw["background_map"]
    assert bg.shape == (8, 2, *two.cut(stack).shape[-2:]) and np.isfinite(bg).all()
    expected = fit_goodness(two, wrapped.cal, wrapped.kind, two.cut(stack)[1:-1], np.asarray(raw["frame"], int),
                            np.c_[raw["x_pix"], raw["y_pix"]], raw["z_nm"], raw["photons_c"], None, False,
                            background_map=bg)
    np.testing.assert_allclose(out["goodness"], expected, rtol=1e-5)
    # without the goodness nothing is kept
    plain = Predictor(files["model"], "cpu", threshold=0.0, batch=4)
    assert "background_map" not in TwoChannelPredictor(plain, files["cal"])(stack, 0)
