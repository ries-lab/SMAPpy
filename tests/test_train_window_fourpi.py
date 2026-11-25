"""A 4Pi job in the *Train DECODE* window, without the window (docs/4pi.md §30): Spline 4Pi prefits the movie, the
job is written with the movie's layout in its calibration, the training reads it and draws in that layout, the
form offers the 4Pi model alone, and a model is tried on the movie as the fitter would run it."""
import json

import numpy as np
import pytest
import tifffile

pytest.importorskip("decode_smlm")

from smappy import fourpi
from smappy.channels import from_corners
from smappy.plugins.fit import CameraSettings


@pytest.fixture(scope="module")
def movie(tmp_path_factory):
    """A 4Pi movie in a strip of four channels (two mirrored, each moved a little against the first), 400 frames
    of six molecules blinking under a drifting cavity, and its model with that layout kept."""
    root = tmp_path_factory.mktemp("train4pi")
    cal = fourpi.model_calibration()
    layout = from_corners([0, 70, 140, 210], [0] * 4, 64, 64, flip_x=[0, 0, 1, 1])
    maps = [np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]]) for dx, dy in [(0, 0), (0.6, -0.4), (-1.3, 1.2), (2.4, 0.7)]]
    layout = layout.with_maps(maps)
    cal.layout, cal.transforms = layout.to_dict(), np.stack(maps)
    cal_path = root / "model_4pi.h5"
    fourpi.save_fourpi_calibration(cal_path, cal)
    rng = np.random.default_rng(2)
    n_frames, per = 400, 6
    frame = np.repeat(np.arange(n_frames), per)
    x, y = rng.uniform(10, 54, (2, len(frame)))
    cavity = fourpi.cavity_drift(n_frames, rate_rad=0.003, start_rad=0.4, seed=3)
    frames = fourpi.render_frames(cal, layout, frame, x, y, rng.uniform(-400, 400, len(frame)),
                                  rng.uniform(1500, 6000, len(frame)), cavity[frame], n_frames=n_frames, background=15.0)
    tifffile.imwrite(root / "movie.tif", (frames / 0.5 + 100).astype(np.uint16), photometric="minisblack")
    return root / "movie.tif", cal_path


def _settings(movie, **more):
    from smappy.gui.train_window import PrepareSettings
    path, cal_path = movie
    return PrepareSettings(movie=str(path), calibration=str(cal_path), density="HD", prefit_frames=400,
                           camera=CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.1, em_on=False), **more)


@pytest.fixture(scope="module")
def prepared(movie):
    from smappy.gui.train_window import run_prefit
    return run_prefit(_settings(movie))


def test_a_4pi_calibration_trains_the_4pi_model_alone(movie):
    from smappy.gui.train_window import model_choices, psf_kind
    channels, line = psf_kind(str(movie[1]))
    assert channels == 4 and "4Pi" in line
    assert [k for k, _ in model_choices(4)] == ["optimised"]


@pytest.mark.slow
def test_the_prefit_fits_with_spline_4pi_and_finds_each_channels_background_and_the_cavitys_drift(prepared):
    """The prefit's Spline 4Pi and 4Pi z on the movie: every channel's background range holds the 15 photons
    drawn, the photons' range holds the 1500 -- 6000 drawn, and the cavity's drift is the one drawn."""
    from smappy.gui.train_window import channels_of
    f = prepared.fourpi
    assert channels_of(prepared) == 4 and len(f["background"]) == 4
    for lo, hi in f["background"]:
        assert lo < 15 < hi
    lo, hi = f["photons_range"]
    assert lo < 1500 and hi > 6000
    assert f["rate_measured"] == pytest.approx(0.003, abs=0.0015)
    assert len(f["layout"]["regions"]) == 4


@pytest.mark.slow
def test_a_4pi_job_trains_in_the_movies_layout(prepared, tmp_path):
    """The job: the 4Pi recipe's settings with the prefit's ranges, the 4Pi model with the movie's layout as its
    calibration; the training reads it and draws batches of four channels with the phase."""
    import torch

    from smappy.gui.train_window import with_settings, write_job
    from smappy.netfit import job as netjob
    from smappy.netfit.train import Drawer
    folder = write_job(prepared, with_settings(prepared.estimates, _form(prepared)), tmp_path / "job", on_time=(1.0, 3.0))
    s = netjob.read_settings(folder)
    assert s.channels.kind == "4pi" and s.channels.count == 4
    assert s.parts.backbone.settings["two_fuse"] == "early" and s.simulation.window == 3
    assert [tuple(b) for b in s.channels.background] == [tuple(b) for b in prepared.fourpi["background"]]
    assert fourpi.load_fourpi_calibration(folder / "calibration.h5").layout["regions"] == prepared.fourpi["layout"]["regions"]
    assert json.loads((folder / "job.json").read_text())["data"]["fourpi_layout"] == prepared.fourpi["layout"]
    b = Drawer(s, "cpu")(2, torch.Generator().manual_seed(0))
    assert b.frames.shape[1] == 4 and b.phase is not None


@pytest.mark.slow
def test_typed_estimates_write_a_4pi_job_without_a_prefit(movie, tmp_path):
    from smappy.gui.train_window import EstimateSettings, prepared_by_hand, with_settings, write_job
    from smappy.netfit import job as netjob
    typed = EstimateSettings(photons_min=800.0, photons_max=9000.0, background_min=5.0, background_max=40.0,
                             z_min_nm=-500.0, z_max_nm=500.0)
    p = prepared_by_hand(_settings(movie), typed)
    folder = write_job(p, with_settings(p.estimates, typed), tmp_path / "job")
    s = netjob.read_settings(folder)
    assert s.channels.kind == "4pi" and s.simulation.photons_range == (800.0, 9000.0)
    assert all(tuple(b) == (5.0, 40.0) for b in s.channels.background)


@pytest.mark.slow
def test_a_4pi_model_is_tried_on_its_movie_cut_as_the_fitter_cuts_it(prepared, tmp_path):
    """*Try*: the job's model (untrained here) on the movie's first frames, through the 4Pi predictor in the
    movie's layout -- positions on the frame, a density per µm² of one channel."""
    import torch

    from smappy.gui.train_window import try_on_data, with_settings, write_job
    from smappy.netfit import job as netjob
    from smappy.netfit import network
    from smappy.netfit.train import METHOD
    from smappy.training import model_file
    folder = write_job(prepared, with_settings(prepared.estimates, _form(prepared)), tmp_path / "job")
    s = netjob.read_settings(folder)
    s.parts.backbone.settings["features"] = 8
    torch.manual_seed(0)
    net = network.build(s.parts, s.simulation.window, s.channels)
    meta = {"method": METHOD, "settings": s.to_dict(), "window": s.simulation.window,
            "scaling": network.Scaling.for_simulation(s.simulation, s.channels).__dict__}
    model_file.write(folder / "model.safetensors", {k: v.detach().numpy() for k, v in net.state_dict().items()}, meta)
    photons, locs, found, trained = try_on_data(folder, frames=6)
    assert photons.ndim == 2 and locs.ndim == 2 and np.isfinite(trained)


def _form(prepared):
    from smappy.gui.train_window import estimate_settings
    return estimate_settings(prepared.estimates, 4)
