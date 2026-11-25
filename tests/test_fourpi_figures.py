"""The 4Pi figures: the channels' split against the phase, the wrong-fringe risk along z, the fringe
boundaries and z by fringe of *4Pi z*, and the Preview's one molecule of *Spline 4Pi*."""
from types import SimpleNamespace

import numpy as np
import pytest

from smappy import fourpi
from smappy.psf import Global4PiPSF


@pytest.fixture(scope="module")
def cal():
    return fourpi.model_calibration()


@pytest.fixture(scope="module")
def engine_table(cal, tmp_path_factory):
    """Simulated 4Pi frames fitted by the engine, as Spline 4Pi fits them, with the model on disk where
    the table says it was fitted with it."""
    from smappy.channels import from_corners
    from smappy.metadata import CameraMetadata
    from smappy.multichannel import MultiChannelEngine
    from smappy.pipeline import FitSettings
    from smappy.plugins.fit import DetectionSettings
    layout = from_corners([0, 70, 140, 210], [0] * 4, 64, 64, flip_x=[0, 0, 1, 1])
    rng = np.random.default_rng(1)
    n_frames = 400
    frame = np.repeat(np.arange(n_frames), 3)
    x, y = rng.uniform(8, 56, (2, len(frame)))
    z = rng.uniform(-300, 300, len(frame))
    frames = fourpi.render_frames(cal, layout, frame, x, y, z, 2500, phase0=0.4,
                                  n_frames=n_frames, background=10)
    camera = CameraMetadata(conversion=1.0, offset=0.0, pixelsize_um=0.1)
    engine = MultiChannelEngine(camera, DetectionSettings().finder(), Global4PiPSF(cal), layout,
                                FitSettings(roisize=13), keep_rois=True)
    engine.push(layout.cut(frames), 0)
    locs = engine.flush()
    path = tmp_path_factory.mktemp("model") / "model_4pi.h5"
    fourpi.save_fourpi_calibration(path, cal)
    locs.metadata["fourpi"]["calibration"] = str(path)
    return locs, engine, camera


def test_the_models_split_over_a_roi_is_its_spots_summed(cal):
    """I + A cos phi + B sin phi summed over the ROI, from three phases, is the spot summed at any
    other phase."""
    z = np.array([-300.0, 0.0, 250.0])
    _, (I, A, B) = fourpi.model_channel_sums(cal, 13, z)
    phase = np.array([0.4, -2.0, 2.9])
    centre = np.full(3, 6.0)
    direct = fourpi.spots(cal, centre, centre, cal.z_nm_to_index(z), phase, 13).sum(axis=(-2, -1))
    assert np.allclose(I.T + A.T * np.cos(phase)[:, None] + B.T * np.sin(phase)[:, None], direct,
                       rtol=1e-4)
    shares = fourpi.model_shares(cal, z, phase, 13)
    assert np.allclose(shares.sum(axis=1), 1)
    assert np.allclose(shares, direct / direct.sum(axis=1, keepdims=True), rtol=1e-3)


def test_the_fit_writes_each_channels_photons_and_they_split_as_the_model_says(cal, engine_table):
    locs, _, _ = engine_table
    names = [f"roi_photons_ch{c}" for c in range(4)]
    assert all(n in locs for n in names)
    assert locs.metadata["fourpi"]["roisize"] == 13
    sums = np.stack([np.asarray(locs[n], float) for n in names], 1)
    # the ROIs hold most of a molecule's 2500 photons, the background taken off
    assert 1500 < np.median(sums.sum(axis=1)) < 2700
    model = fourpi.model_shares(cal, locs["z_env_nm"], locs["phase_rad"], 13)
    binned = fourpi.channel_shares(sums, locs["phase_rad"], bins=12, model=model)
    full = binned["counts"] >= 10
    assert full.sum() >= 10
    # the data's medians on the model's line, which swings by tenths
    assert np.nanmax(np.abs(binned["median"][full] - binned["model"][full])) < 0.03
    assert np.nanmax(binned["model"]) - np.nanmin(binned["model"]) > 0.2


def test_the_wrong_fringe_risk_is_where_the_probabilities_are_low():
    rng = np.random.default_rng(0)
    z = rng.uniform(-500, 500, 20000)
    p = np.where(np.abs(z) > 300, 0.8, 0.999)
    centres, expected, unsure, counts = fourpi.fringe_risk(z, p, bins=20)
    inner, outer = np.abs(centres) < 250, np.abs(centres) > 350
    assert np.allclose(expected[inner], 0.001)
    assert np.allclose(unsure[inner], 0)
    assert np.allclose(expected[outer], 0.2)
    assert np.allclose(unsure[outer], 1)
    assert counts.sum() > 0.97 * len(z)


def test_the_model_drawn_for_a_fitted_molecule_leaves_only_noise(cal):
    """`fourpi.molecule_model` evaluates the channels where the fitter does: the residual of a fit is
    Poisson noise, a mean squared Pearson residual of one."""
    from smappy.metadata import CameraMetadata
    from smappy.plugins.fit_4pi import brightest_molecule
    rois, _ = fourpi.simulate_rois(cal, 200, photons=3000, background=10, seed=4)
    model = Global4PiPSF(cal)
    link = fourpi.link(200, 4)
    result = model.fit(rois, link)
    fitted = fourpi.molecule_model(cal, model, result.theta, link, 13)
    chi = np.mean((rois - fitted) ** 2 / fitted, axis=(1, 2, 3))
    assert np.median(chi) == pytest.approx(1.0, abs=0.1)
    camera = CameraMetadata(conversion=1.0, offset=0.0, pixelsize_um=0.1)
    data, drawn, numbers = brightest_molecule(model, SimpleNamespace(images=rois, link=link, roisize=13),
                                              camera)
    assert data.shape == drawn.shape == (4, 13, 13)
    assert np.mean((data - drawn) ** 2 / drawn) < 1.3
    assert 2000 < numbers["photons"] < 4000


def test_the_preview_picks_a_molecule_the_fit_describes_not_two_in_one_box(cal, engine_table):
    from smappy.plugins.fit_4pi import brightest_molecule
    _, engine, camera = engine_table
    data, drawn, _ = brightest_molecule(engine.model, engine.rois, camera)
    assert np.mean((data - drawn) ** 2 / drawn) < 1.3


def test_4pi_z_draws_its_panels_and_the_channels_split(cal, engine_table):
    from matplotlib.figure import Figure

    from smappy.plugins import Context
    from smappy.plugins.fourpi_z import FourPiZ, FourPiZSettings
    locs, _, _ = engine_table
    result = FourPiZ().run(Context(locs=locs), FourPiZSettings(window_frames=100))
    figures = {f.name: f for f in result.figures()}
    assert set(figures) == {"", "channels"}
    assert figures[""].panels == 4
    for figure in figures.values():
        target = Figure()
        figure.draw_into(target)
        assert len(target.axes) == max(figure.panels, 1)
    shares = result.data["channel_shares"]
    assert shares["model"] is not None


def test_a_table_fitted_without_the_channels_photons_gets_no_split_figure(cal, engine_table):
    from smappy.plugins import Context
    from smappy.plugins.fourpi_z import FourPiZ, FourPiZSettings
    locs, _, _ = engine_table
    old = locs[np.arange(len(locs))]
    for c in range(4):
        del old.columns[f"roi_photons_ch{c}"]
    result = FourPiZ().run(Context(locs=old), FourPiZSettings(window_frames=100))
    assert [f.name for f in result.figures()] == [""]


def test_noise_free_data_equal_to_the_model_lie_on_its_line_though_the_shares_are_skewed_within_a_bin(cal):
    """The model's line is the median of its shares, as the dots are the data's: a bin holds molecules at
    every z, whose shares swing far near the focus and little away from it.  The mean against the median drew
    data equal to the model swinging 1.2 times its line on the 2017 tubulin's model (docs/4pi.md §34)."""
    rng = np.random.default_rng(0)
    z, phase = rng.uniform(-500, 500, 20000), rng.uniform(-np.pi, np.pi, 20000)
    shares = fourpi.model_shares(cal, z, phase, 13)
    binned = fourpi.channel_shares(shares * 5000.0, phase, model=shares)
    assert np.nanmax(np.abs(binned["median"] - binned["model"])) < 1e-9
    means = np.array([shares[np.digitize(phase, np.linspace(-np.pi, np.pi, 25)) - 1 == b].mean(0)
                      for b in range(24)])
    assert np.nanmax(np.abs(binned["median"] - means)) > 5e-4          # what the mean would have drawn
