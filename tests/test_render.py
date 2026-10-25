"""Geometry, accumulation and the display path, on positions with known pixels."""
import numpy as np
import pytest

from smappy import lut as luts
from smappy.render import (FieldOfView, SigmaSettings, normalize, render,
                            render_histogram, to_rgb)


def test_pixel_grid_and_edges():
    fov = FieldOfView.from_range((0, 100), (0, 50), 10.0)
    assert fov.shape == (5, 10)
    assert fov.extent == (0.0, 100.0, 50.0, 0.0)
    # a position sits in the pixel whose half-open interval contains it
    px, py = fov.to_pixels([0.0, 9.999, 10.0, 99.9], [0.0, 0.0, 0.0, 49.9])
    assert np.floor(px).tolist() == [0, 0, 1, 9]

    img = render_histogram([5.0, 15.0, 15.0], [5.0, 45.0, 45.0], fov)
    assert img.n_locs == 3
    assert img.weight[0, 0] == 1 and img.weight[4, 1] == 2
    assert img.weight.sum() == 3


def test_localizations_outside_are_dropped():
    fov = FieldOfView.from_range((0, 10), (0, 10), 1.0)
    img = render_histogram([-0.1, 5.0, 10.0], [5.0, 5.0, 5.0], fov)
    assert img.n_locs == 1 and img.weight.sum() == 1


def test_weights_are_summed():
    fov = FieldOfView.from_range((0, 4), (0, 4), 1.0)
    img = render_histogram([0.5, 0.5, 2.5], [0.5, 0.5, 0.5], fov, weights=[2.0, 3.0, 1.0])
    assert img.weight[0, 0] == 5.0 and img.weight[0, 2] == 1.0


def test_fit_keeps_pixels_square_and_centred():
    fov = FieldOfView.fit((0, 1000), (0, 500), 100, 100)
    assert fov.pixelsize == 10.0                      # the larger range decides
    assert fov.x0 == 0.0 and fov.y0 == -250.0         # the shorter one is centred
    assert 0.5 * (fov.y0 + fov.y1) == 250.0


def test_zoom_keeps_the_cursor_fixed():
    fov = FieldOfView.from_range((0, 100), (0, 100), 1.0)
    zoomed = fov.zoomed(0.5, cx=25.0, cy=75.0)
    assert zoomed.pixelsize == 0.5 and zoomed.shape == fov.shape
    # the anchor keeps its position on the canvas; the view shrinks around it
    assert np.allclose(zoomed.to_pixels(25.0, 75.0), fov.to_pixels(25.0, 75.0))
    assert (zoomed.x1 - zoomed.x0) == 0.5 * (fov.x1 - fov.x0)


def test_intensity_lut_is_applied_after_normalization():
    fov = FieldOfView.from_range((0, 2), (0, 1), 1.0)
    img = render_histogram([0.5], [0.5], fov)
    rgb = to_rgb(img, lut="gray", imax=1.0)
    assert np.allclose(rgb[0, 0], 1.0) and np.allclose(rgb[0, 1], 0.0)


def test_normalize_falls_back_to_the_maximum():
    """A very sparse image has a zero quantile; SMAP then uses the maximum."""
    image = np.zeros((100, 100), np.float32)
    image[0, 0] = 7.0
    norm, imax = normalize(image, contrast=0.3)  # a quantile that lands on zero
    assert imax == 7.0 and norm.max() == 1.0


def test_luts_are_well_formed():
    for name in luts.names():
        table = luts.get(name)
        assert table.shape == (256, 3) and table.min() >= 0 and table.max() <= 1


def test_invert_is_smaps_lutinvert_on_intensity_and_a_turn_round_on_a_column():
    """SMAP's two inversions.  An intensity ramp: ``sum(c) - c`` (`lutinvert`),
    which on `hot` runs black through cyan to white.  A LUT coding a column:
    the table turned round, since the sum of a rainbow's channels is no
    colour of it."""
    one_hue = np.array([[0.4, 0.0, 0.0]], np.float32)
    assert np.allclose(luts.invert_sum(one_hue), [[0.0, 0.4, 0.4]])
    yellow = np.array([[1.0, 1.0, 0.0]], np.float32)
    assert np.allclose(luts.invert_sum(yellow), [[1.0, 1.0, 1.0]])
    assert np.allclose(luts.get("hot", invert=True), luts.invert_sum(luts.get("hot")))
    for name in luts.names():
        table = luts.get(name, invert=True)
        assert table.min() >= 0 and table.max() <= 1, name

    turbo = luts.get("turbo")
    values = np.array([0.0, 1.0])
    plain = luts.colors(values, "turbo", 0.0, 1.0)
    turned = luts.colors(values, "turbo", 0.0, 1.0, invert=True)
    assert np.allclose(plain, turbo[[0, -1]]) and np.allclose(turned, turbo[[-1, 0]])

def test_a_table_with_no_columns_yet_renders_as_an_empty_picture():
    """A live fit's table, before its first block: the view asks for a picture
    of it as soon as the fit starts, and used to get a KeyError."""
    from smappy.locs import Localizations
    from smappy.render import render_locs
    fov = FieldOfView.fit((0, 1000), (0, 500), 40, 20)
    image = render_locs(Localizations({}, {}), fov)
    assert image.weight.shape == (fov.ny, fov.nx) and not image.weight.any()
    assert image.n_locs == 0


def test_dl_draws_each_spot_as_wide_as_its_psf_on_camera_pixels():
    from smappy.locs import Localizations
    from smappy.render import RenderSettings, render_locs
    n = 20000
    rng = np.random.default_rng(0)
    locs = Localizations({"x_nm": np.full(n, 1030.0, np.float32),
                          "y_nm": np.full(n, 1070.0, np.float32),
                          "sigma_nm": np.full(n, 150.0, np.float32)},
                         {"units": "nm", "pixelsize_nm": 110.0})
    fov = FieldOfView.from_range((0, 2200), (0, 2200), 11.0)
    image = render_locs(locs, fov, RenderSettings(mode="dl")).weight
    blocks = image.reshape(20, 10, 20, 10)       # 10 view pixels per camera pixel
    assert np.allclose(blocks, blocks[:, :1, :, :1])   # flat within a camera pixel
    camera = blocks[:, 0, :, 0]
    # the PSF's width comes back from the camera image's second moment
    centres = (np.arange(20) + 0.5) * 110.0
    profile = camera.sum(axis=0)
    mean = (profile * centres).sum() / profile.sum()
    var = (profile * (centres - mean) ** 2).sum() / profile.sum()
    assert mean == pytest.approx(1030.0, abs=5)
    assert np.sqrt(var - 110.0 ** 2 / 12) == pytest.approx(150.0, rel=0.05)


def test_dl_draws_a_failed_or_missing_psf_one_camera_pixel_wide():
    from smappy.locs import Localizations
    from smappy.render import psf_sigmas
    locs = Localizations({"x_nm": np.zeros(3, np.float32), "y_nm": np.zeros(3, np.float32),
                          "sigma_nm": np.array([120.0, -5.0, 4000.0], np.float32)},
                         {"units": "nm", "camera": {"pixelsize_um": [0.127, 0.127]}})
    sx, sy = psf_sigmas(locs)
    np.testing.assert_allclose(sx, [120.0, 127.0, 127.0])
    assert sy is None


def test_an_absolute_maximum_keeps_a_structure_as_bright_at_every_zoom():
    # SMAP's absolute Imax is per pixel, so a zoom out saturates the picture;
    # given per um^2 it is scaled by the pixel's area instead
    from smappy.render import DisplaySettings, normalize
    rng = np.random.default_rng(3)
    x, y = rng.uniform(0, 2000, 20000), rng.uniform(0, 2000, 20000)   # 5000 / um^2
    display = DisplaySettings(absolute=True, maximum=10000.0)
    for pixelsize in (5.0, 10.0, 20.0, 40.0):
        n = int(2000 / pixelsize)
        rendered = render(x, y, FieldOfView(0, 0, pixelsize, n, n), sigma=30.0)
        shown, _ = normalize(rendered.weight, display.pixel_max(rendered))
        inner = shown[n // 4: 3 * n // 4, n // 4: 3 * n // 4]
        assert inner.mean() == pytest.approx(0.5, rel=0.05)


def test_switching_to_the_absolute_scale_starts_where_the_quantile_was():
    from smappy.render import DisplaySettings, normalize
    rng = np.random.default_rng(4)
    fov = FieldOfView(0, 0, 10.0, 200, 200)
    rendered = render(rng.normal(1000, 300, 5000), rng.normal(1000, 300, 5000), fov,
                      sigma=20.0)
    quantile = DisplaySettings(contrast=2.5)
    absolute = quantile.as_absolute(rendered)
    assert absolute.absolute and absolute.contrast == 2.5
    assert absolute.pixel_max(rendered) == pytest.approx(
        normalize(rendered.weight, None, 2.5)[1])


@pytest.mark.parametrize("mode", ["gauss", "hist", "tracks", "voronoi"])
def test_a_field_coloured_picture_with_nothing_in_view_is_black_not_the_luts_lowest_colour(mode):
    from smappy.locs import Localizations
    from smappy.render import DisplaySettings, RenderSettings, render_locs
    n = 50
    locs = Localizations({"x_nm": np.linspace(0, 500, n, dtype=np.float32),
                          "y_nm": np.linspace(0, 500, n, dtype=np.float32),
                          "z_nm": np.linspace(-300, 300, n, dtype=np.float32),
                          "frame": np.arange(n), "track_id": np.arange(n) // 10})
    fov = FieldOfView(0, 0, 10, 50, 50)
    display = DisplaySettings(lut="jet")
    nothing = render_locs(locs, fov, RenderSettings(mode=mode, color_field="z_nm"), display,
                          select=np.zeros(0, np.int64))
    assert nothing.color is not None
    assert display.apply(nothing).max() == 0.0
