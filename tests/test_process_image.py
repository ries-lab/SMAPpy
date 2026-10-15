"""Process image: a background is removed, what sits on it is kept, and the
result is a new image layer exactly over the original.

The planes are built with known parts -- a constant, a slope, a Gaussian spot
of known height -- and each method has to take away the smooth part and
leave the spot.
"""
import dataclasses

import numpy as np
import pytest

from smappy import plugins
from smappy.image_processing import (gaussian_background, local_contrast,
                                     rolling_ball_background, subtract_background,
                                     wavelet_background)
from smappy.images import ImageData
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.process_image import (BackgroundSettings, ContrastSettings,
                                          ProcessImageSettings, process_image)
from smappy.session import Session

PATH = "Analysis/Process/Process image"
METHODS = [("gaussian", {}), ("ball", {}), ("wavelet", {})]


def grid(ny=80, nx=100):
    return np.mgrid[0:ny, 0:nx].astype(float)


def slope():
    yy, xx = grid()
    return 50.0 + 0.3 * xx - 0.2 * yy


def spot(height=100.0, at=(30, 40), sigma=1.5):
    yy, xx = grid()
    return height * np.exp(-((xx - at[1]) ** 2 + (yy - at[0]) ** 2) / (2 * sigma ** 2))


@pytest.mark.parametrize("method", ["gaussian", "ball", "wavelet"])
def test_a_constant_plus_a_slope_is_removed_to_the_edges(method):
    """Point reflection continues the slope beyond the edges, so nothing is
    left of it, the rim included."""
    out, background = subtract_background(slope(), method, sigma_px=10,
                                          radius_px=8, level=3)
    assert np.abs(out).max() < 1e-9
    np.testing.assert_allclose(background, slope(), atol=1e-9)


@pytest.mark.parametrize("method", ["gaussian", "ball", "wavelet"])
def test_a_spot_on_a_background_keeps_its_height(method):
    """The spot comes out at its height over zero, and everything away from
    it at zero: the rounds keep the spot out of the Gaussian's and the
    wavelet's estimate, and the ball is too big to rise into it -- but for
    a few counts: its height is in the plane's units, as in ImageJ, and a
    ball of 8 rises about 2.5 into the foot of a spot 3 pixels wide."""
    out, _ = subtract_background(slope() + spot(), method, sigma_px=10,
                                 radius_px=8, level=3)
    yy, xx = grid()
    assert out[30, 40] == pytest.approx(100.0, rel=0.03 if method == "ball" else 0.01)
    assert np.abs(out[np.hypot(xx - 40, yy - 30) > 8]).max() < 1.0


def test_without_rounds_a_gaussian_high_pass_takes_a_share_of_the_spot():
    """What the rounds are for: a plain high-pass digs a moat around a
    spot and lowers it; refined, it does neither."""
    plain = slope() + spot() - gaussian_background(slope() + spot(), 4, rounds=0)
    refined = slope() + spot() - gaussian_background(slope() + spot(), 4, rounds=3)
    assert plain.min() < -5 and plain[30, 40] < 90
    assert refined.min() > -0.5 and refined[30, 40] > 99


def test_on_noise_the_spot_is_kept_and_the_background_is_at_zero():
    rng = np.random.default_rng(3)
    truth = slope() + spot(height=300.0)
    rows = []
    for _ in range(20):
        noisy = rng.poisson(truth).astype(float)
        out = noisy - gaussian_background(noisy, 10)
        rows.append(out)
    mean = np.mean(rows, axis=0)
    assert mean[30, 40] == pytest.approx(300.0, rel=0.05)
    yy, xx = grid()
    away = np.hypot(xx - 40, yy - 30) > 8
    assert abs(np.median(mean[away])) < 0.5
    w = np.mean([n - wavelet_background(n, 3) for n in
                 (rng.poisson(truth).astype(float) for _ in range(20))], axis=0)
    assert w[30, 40] == pytest.approx(300.0, rel=0.05)


def test_a_large_ball_is_rolled_on_a_shrunk_plane_and_still_finds_a_flat_background():
    yy, xx = grid(200, 200)
    plane = 20.0 + 80.0 * np.exp(-((xx - 100) ** 2 + (yy - 100) ** 2) / (2 * 3.0 ** 2))
    background = rolling_ball_background(plane, 40)        # shrunk by 4
    assert background.min() == pytest.approx(20.0, abs=1e-9)
    assert background.max() < 20.5                          # of the spot's 100


def test_local_contrast_makes_a_dim_and_a_bright_half_alike():
    """The same texture at a tenth of the amplitude on the left comes out
    with the same local SD on both sides."""
    rng = np.random.default_rng(0)
    texture = rng.normal(size=(80, 160))
    plane = texture * np.where(np.arange(160) < 80, 1.0, 10.0)[None, :]
    out = local_contrast(plane, 5.0, floor=0.0)
    left, right = out[10:-10, 15:65].std(), out[10:-10, 95:145].std()
    assert right / left == pytest.approx(1.0, abs=0.1)
    assert plane[:, 95:145].std() / plane[:, 15:65].std() > 8


def test_the_floor_keeps_a_flat_area_flat():
    rng = np.random.default_rng(1)
    plane = np.zeros((80, 160))
    plane[:, 80:] = rng.normal(size=(80, 80)) * 10
    plane[:, :80] = rng.normal(size=(80, 80)) * 0.01       # nearly empty
    out = local_contrast(plane, 5.0, floor=0.5)
    assert out[10:-10, 10:60].std() < 0.01


# ------------------------------------------------------------- the plugin
def session_with(image):
    locs = Localizations({"x_nm": np.array([500.0]), "y_nm": np.array([500.0]),
                          "frame": np.array([0])}, {})
    session = Session(locs)
    session.add_image(image)
    return session


def test_the_new_layer_has_the_original_s_placement():
    plane = (slope() + spot()).astype(np.float32)
    image = ImageData(plane, 100.0, x0=300.0, y0=-200.0, name="wf",
                      pixelsize_y=110.0)
    image.place(np.array([[0.9, -0.2, 40.0], [0.2, 0.9, -15.0], [0.0, 0.0, 1.0]]))
    session = session_with(image)
    n = len(session.layers)
    s = ProcessImageSettings(bg=BackgroundSettings(sigma_nm=1000.0))
    session.run(plugins.get(PATH)(), s)
    assert len(session.layers) == n + 1
    new = session.layers[-1].image
    assert new is not image and new.data.dtype == np.float32
    np.testing.assert_allclose(new.placement, image.placement)
    np.testing.assert_allclose(new.grid, image.grid)
    assert new.bounds == pytest.approx(image.bounds)
    assert new.name == "wf (background subtracted)"
    assert any(im is new for im in session.images)
    # the original untouched, and the new one is the background subtracted
    np.testing.assert_array_equal(image.data, plane)
    assert new.data[30, 40] == pytest.approx(100.0, rel=0.02)


def test_an_unplaced_image_stays_unplaced_and_a_preview_adds_nothing():
    session = session_with(ImageData(slope().astype(np.float32), 50.0, name="wf"))
    plugin = plugins.get(PATH)()
    n = len(session.layers)
    s = ProcessImageSettings(contrast=True)
    result = plugin.preview(session.context(0), s)
    assert len(session.layers) == n and not result.image_layers
    session.run(plugin, s)
    assert session.layers[-1].image.placement is None
    assert "contrast normalised" in session.layers[-1].name


def test_a_stack_gives_the_plane_shown_or_every_plane():
    planes = np.stack([slope() + spot(height=h) for h in (50.0, 100.0, 150.0)])
    image = ImageData(planes.astype(np.float32), 100.0, name="stack",
                      frames=np.array([10, 20, 30]))
    session = session_with(image)
    session.layers[-1].frame = 2
    plugin = plugins.get(PATH)()
    session.run(plugin, ProcessImageSettings())
    one = session.layers[-1].image
    assert one.data.ndim == 2 and one.data[30, 40] == pytest.approx(150.0, rel=0.02)
    assert list(one.frames) == [30]
    session.run(plugin, ProcessImageSettings(planes="all", layer=2))
    every = session.layers[-1].image
    assert every.data.shape == planes.shape
    assert every.data[:, 30, 40] == pytest.approx([50.0, 100.0, 150.0], rel=0.02)


def test_the_form_lists_image_layers_and_shows_only_the_chosen_method():
    session = session_with(ImageData(np.ones((8, 8), np.float32), 100.0, name="wf"))
    plugin = plugins.get(PATH)()
    s = ProcessImageSettings(bg=BackgroundSettings(method="ball"))
    state = plugin.describe(session.context(0), s)
    assert (2, "2: wf") in state.choices["layer"]
    assert state.shown["bg.radius_nm"] and not state.shown["bg.sigma_nm"]
    assert not state.shown["lcn"]
    assert "10.0 pixels" in state.notice.text


def test_without_an_image_layer_or_an_operation_it_says_so():
    plugin = plugins.get(PATH)()
    session = Session(Localizations({"x_nm": np.zeros(1), "y_nm": np.zeros(1),
                                     "frame": np.zeros(1, int)}, {}))
    with pytest.raises(ValueError, match="no image layer"):
        session.run(plugin, ProcessImageSettings())
    session.add_image(ImageData(np.ones((8, 8), np.float32), 100.0, name="wf"))
    with pytest.raises(ValueError, match="tick an operation"):
        session.run(plugin, ProcessImageSettings(background=False))


def test_the_new_layer_is_saved_with_the_file(tmp_path):
    session = session_with(ImageData((slope() + spot()).astype(np.float32), 100.0,
                                     x0=50.0, name="wf"))
    session.run(plugins.get(PATH)(), ProcessImageSettings())
    path = tmp_path / "t.h5"
    session.save(path)
    from smappy.io.hdf5 import load_images
    names = [im.name for im in load_images(path)]
    assert "wf (background subtracted)" in names
