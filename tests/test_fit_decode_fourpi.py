"""A 4Pi netfit model in the DECODE fitter (docs/4pi.md §30): the camera's frames cut into the channels as
Spline 4Pi cuts them, the phase out, *4Pi z* after the fit, and the form showing the 4Pi part for such a
model only.  The network is untrained (8 features): these are tests of the plumbing; that a trained one's
numbers are good is §30's."""
import numpy as np
import pytest
import tifffile

torch = pytest.importorskip("torch")

from smappy import fourpi, plugins
from smappy.channels import from_corners
from smappy.netfit import network
from smappy.netfit import fourpi as NF
from smappy.netfit.settings import ChannelSettings, SimulationSettings, TrainingSettings
from smappy.netfit.train import METHOD, trained_on
from smappy.plugins import Context
from smappy.training import model_file
from smappy.training.predictors import fourpi as is_fourpi
from smappy.training.predictors import two_channel


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    root = tmp_path_factory.mktemp("decode4pi")
    cal = fourpi.model_calibration()
    layout = from_corners([0, 70, 140, 210], [0] * 4, 64, 64, flip_x=[0, 0, 1, 1])
    maps = [np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]]) for dx, dy in [(0, 0), (0.6, -0.4), (-1.3, 1.2), (2.4, 0.7)]]
    layout = layout.with_maps(maps)
    cal.layout, cal.transforms = layout.to_dict(), np.stack(maps)
    cal_path = root / "model_4pi.h5"
    fourpi.save_fourpi_calibration(cal_path, cal)
    # an untrained 4Pi network, its header as a training writes it
    s = TrainingSettings(channels=ChannelSettings(kind="4pi"))
    s.parts.backbone.settings.update(features=8, two_fuse="early")
    torch.manual_seed(0)
    net = network.build(s.parts, 3, s.channels).eval()
    sim = SimulationSettings(photons_range=(500.0, 8000.0), background=(5.0, 20.0))
    scaling = network.Scaling.for_simulation(sim)
    four = NF.FourChannels.from_layout(layout)
    meta = {"method": METHOD, "settings": s.to_dict(), "scaling": scaling.__dict__, "window": 3,
            "calibration_em_on": False, "calibration_source": str(cal_path), "trained_on": trained_on(sim),
            "channels": four.to_dict()}
    model = root / "model.safetensors"
    model_file.write(model, {k: v.numpy() for k, v in net.state_dict().items()}, meta)
    # a movie: molecules in channel 0's pixels, every channel drawn where the maps put them
    rng = np.random.default_rng(1)
    n_frames, per = 10, 6
    frame = np.repeat(np.arange(n_frames), per)
    x, y = rng.uniform(10, 54, (2, len(frame)))
    frames = fourpi.render_frames(cal, layout, frame, x, y, rng.uniform(-400, 400, len(frame)), 3000.0,
                                  rng.uniform(-np.pi, np.pi, len(frame)), n_frames=n_frames, background=10.0)
    tifffile.imwrite(root / "movie.tif", frames.astype(np.uint16), photometric="minisblack")
    return {"model": model, "cal": cal_path, "tif": root / "movie.tif", "layout": layout, "frames": frames}


def _settings(files, **more):
    from smappy.gui.live_session import fit_settings
    values = {"source.path": str(files["tif"]), "camera.conversion": 1.0, "camera.offset": 0.0,
              "camera.pixelsize_um": 0.1, "camera.em_on": False, "model.path": str(files["model"]),
              "model.device": "cpu", "model.threshold": 0.0, "model.batch": 4, "output.path": "",
              "output.raw_frames": 0}
    values.update(more)
    return fit_settings("Localize/DECODE", values)


def test_a_4pi_model_is_told_from_a_two_channel_one_by_its_header(files):
    meta = model_file.read_metadata(files["model"])
    assert is_fourpi(meta) and not two_channel(meta)


def test_the_decode_fitter_cuts_the_frames_as_spline_4pi_and_puts_out_the_phase_and_4pi_z(files):
    """The fitter's localizations are the 4Pi predictor's on the channels the layout cuts (its ends repeated
    for the window), on the frame where channel 0 saw them; the phase is there, the table keeps the model's
    slope, and *after the fit* has run *4Pi z* on it."""
    from smappy.netfit.predict import Predictor
    locs = plugins.get("Localize/DECODE")().run(Context(), _settings(files)).locs
    for name in ("phase_rad", "phase_err_rad", "z_env_nm", "fringe_probability", "cavity_phase_rad",
                 "background_ch3"):
        assert name in locs, name
    assert locs.metadata["fourpi"]["phase_per_nm"] == pytest.approx(fourpi.model_calibration().phase_per_nm)
    four = NF.FourChannels.from_layout(files["layout"])
    pred = Predictor(files["model"], "cpu", threshold=0.0, batch=4)
    direct = NF.localize(pred, four, files["layout"].cut(files["frames"].astype(np.uint16).astype(np.float32)))
    assert len(direct["frame"]) == len(locs) > 0
    np.testing.assert_allclose(np.sort(np.asarray(locs["x_nm"])), np.sort(direct["x_pix"] * 100.0), rtol=1e-5)
    np.testing.assert_allclose(np.sort(np.asarray(locs["phase_rad"])), np.sort(direct["phase_rad"]), atol=1e-5)


def test_the_form_shows_the_4pi_part_and_4pi_z_for_a_4pi_model_only(files, tmp_path):
    from smappy.plugins.fit_decode import DecodeFit
    state = DecodeFit().describe(None, _settings(files))
    assert state.shown["fourpi"] and not state.shown["two_channels"]
    assert state.shown["finish.fourpi_z"]
    # a one-channel model: no 4Pi part, no 4Pi z
    s = TrainingSettings()
    s.parts.backbone.settings["features"] = 8
    net = network.build(s.parts, 3)
    one = tmp_path / "one.safetensors"
    model_file.write(one, {k: v.detach().numpy() for k, v in net.state_dict().items()},
                     {"method": METHOD, "settings": s.to_dict(), "window": 3,
                      "scaling": network.Scaling.for_simulation(SimulationSettings()).__dict__})
    state = DecodeFit().describe(None, _settings(files, **{"model.path": str(one)}))
    assert not state.shown["fourpi"] and not state.shown.get("finish.fourpi_z", False)
