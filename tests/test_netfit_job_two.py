"""A two-channel training from the window (docs/decode/gui-plan.md, step 4): the calibration a two-channel
one, the job's channels as the prefit found them and the form chose, trained by ``smappy-decode train``.

The prefit itself (Spline 3D 2C on the movie) was checked on the C = 2 suite's simulated sets: the dyes
found at 0.775 and 0.925 where they were drawn at 0.78 and 0.92, the kind and the brighter half right
(2026-11-22); here its pieces are tested on numbers, and a job written without a prefit is trained.
"""
import json

import numpy as np
import pytest
import tifffile
from test_netfit_channels import SHAPE, dual

from smappy.calibrate.dual import save_dual_color_calibration
from smappy.netfit import prefit_two
from smappy.netfit.job import two_channel_settings
from smappy.training import job as jobfile
from smappy.training import progress


def test_the_dyes_are_the_peaks_of_the_shares_and_a_lone_outlier_is_none():
    rng = np.random.default_rng(0)
    share = np.r_[rng.normal(0.78, 0.03, 4000), rng.normal(0.92, 0.02, 800), [0.1, 0.4]]
    assert prefit_two.dye_shares(share) == pytest.approx([0.78, 0.92], abs=0.015)
    assert prefit_two.dye_shares(rng.normal(0.5, 0.05, 3000)) == pytest.approx([0.5], abs=0.015)


def test_the_kind_is_a_biplane_where_the_planes_focus_200_nm_apart():
    from dataclasses import replace
    cal = dual()
    assert prefit_two.kind_of(cal) == "colour"
    shifted = np.roll(cal.secondary.psf, 5, axis=0)                 # 5 slices of 50 nm deeper
    assert prefit_two.kind_of(replace(cal, secondary=replace(cal.secondary, psf=shifted))) == "biplane"


def test_the_channels_are_in_the_grid_s_order_with_the_dyes_turned_where_the_grid_is_the_other_half():
    two = prefit_two.TwoChannels("colour", [(10.0, 50.0), (20.0, 300.0)], brighter=1, dyes=[0.78, 0.92])
    ch, fuse = two_channel_settings(two, main="auto", dyes="prefit", spread=0.06)
    assert ch.main == 1 and ch.background == [[20.0, 300.0], [10.0, 50.0]]
    assert ch.dyes == pytest.approx([0.08, 0.22]) and ch.share_sd == 0.06 and fuse == "early"
    free, _ = two_channel_settings(two, main="0", dyes="independent")
    assert free.main == 0 and free.dyes is None
    with pytest.raises(ValueError, match="no dyes"):
        two_channel_settings(prefit_two.TwoChannels("colour", [(0, 1), (0, 1)], 0), dyes="prefit")


@pytest.fixture
def movie(tmp_path):
    save_dual_color_calibration(tmp_path / "beads_2ct.h5", dual())
    rng = np.random.default_rng(1)
    path = tmp_path / "data" / "cells.tif"
    path.parent.mkdir()
    tifffile.imwrite(path, rng.poisson(120, (20, *SHAPE)).astype(np.uint16), photometric="minisblack")
    return path, tmp_path / "beads_2ct.h5"


def _settings(movie, cal, **two):
    from smappy.gui.train_window import PrepareSettings, TwoChannelPrepare
    from smappy.plugins.fit import CameraSettings
    return PrepareSettings(movie=str(movie), calibration=str(cal), density="LD", blocks=1, block_frames=20,
                           model=two.pop("model", "optimised"), two_channels=TwoChannelPrepare(**two),
                           camera=CameraSettings(conversion=1.0, offset=0.0, pixelsize_um=0.1, em_on=False))


def test_a_two_channel_calibration_trains_the_optimised_model_or_the_plex_like_one_alone(movie):
    from smappy.gui.train_window import two_channel_problem
    path, cal = movie
    assert two_channel_problem(_settings(path, cal)) == ""
    assert two_channel_problem(_settings(path, cal, model="plexlike")) == ""
    assert "two-channel calibration trains" in two_channel_problem(_settings(path, cal, model="decode"))


def test_a_two_channel_job_without_a_prefit_trains_through_the_decode_command(movie, tmp_path):
    torch = pytest.importorskip("torch")
    from decode_smlm.cli import main
    from smappy.gui.train_window import EstimateSettings, prepared_by_hand, write_job
    path, cal = movie
    prepared = prepared_by_hand(_settings(path, cal, main="1", dyes="typed", typed="0.8"),
                                EstimateSettings(photons=3000.0, photons_sd=500.0, background_min=10.0,
                                                 background_max=60.0, z_min_nm=-300.0, z_max_nm=300.0))
    folder = write_job(prepared, prepared.estimates, tmp_path / "job")
    job, params = jobfile.read(folder)
    ch = params["channels"]
    assert ch["kind"] == "colour" and ch["main"] == 1 and ch["dyes"] == pytest.approx([0.2])
    assert params["parts"]["backbone"]["settings"]["two_fuse"] == "early"
    # the C = 2 study's winners: 64 features, the ConvGRU over 3 frames, on-time 1, the schedule in samples
    sim = params["simulation"]
    assert params["parts"]["backbone"]["settings"]["features"] == 64 and sim["window"] == 3
    assert (sim["lifetime_frames"], sim["lifetime_max_frames"], sim["target_photons_min"]) == (1.0, 0.0, 100.0)
    assert params["parts"]["schedule"]["settings"]["decay_samples"] == 327680 and sim["movie_windows"] == 1
    assert jobfile.check(folder) == [] and job.calibration["sha256"] == jobfile.file_sha256(cal)
    s = json.loads((folder / "settings.json").read_text())
    s["simulation"].update(frame_px=16, density_per_40px=15.0, movie_windows=2)
    s["parts"]["backbone"]["settings"]["features"] = 8
    s["parts"]["schedule"]["settings"].update(samples_per_epoch=32, batch=8, test_samples=8, epochs=2,
                                              precision="fp32", compile=False)
    (folder / "settings.json").write_text(json.dumps(s, indent=1))
    assert main(["train", str(folder), "--device", "cpu", "--all-epochs"]) == 0
    kinds = [line["kind"] for line in progress.read(folder)]
    assert kinds[0] == "start" and kinds.count("epoch") == 2 and kinds[-1] == "finished"
    # the model knows its channels and its calibration: what the DECODE fitter reads
    from smappy.training import model_file
    meta = model_file.read_metadata(folder / "model.safetensors")
    assert meta["channels"]["main"] == 1 and meta["calibration_sha256"] == jobfile.file_sha256(cal)
    del torch


def test_a_smap_pair_s_photons_are_each_half_s_own_light_and_a_biplane_s_total_is_both():
    """SMAP's two-channel calibrations hold the beads' split in their PSFs (Fig. 1c's 0.99 and 0.77 at
    focus): each half's light is its share, so a linked N is the total (Jonas, 2026-11-22)."""
    from dataclasses import replace

    from smappy.netfit.fit_two import biplane_total
    from smappy.psf import SMAP_PAIR, light_at_focus, signal_integral
    cal = dual()
    second = replace(cal.secondary, psf=cal.secondary.psf * 0.77, parameters={"smappy_conversion": SMAP_PAIR + " ..."})
    assert signal_integral(second) == pytest.approx(light_at_focus(second))
    assert light_at_focus(second) == pytest.approx(0.77 * light_at_focus(cal.secondary))
    assert signal_integral(replace(second, parameters={})) == 1.0        # not SMAP's pair: as before
    pair = replace(cal, secondary=second)
    assert biplane_total(pair) == pytest.approx(light_at_focus(cal.main) + light_at_focus(second))
