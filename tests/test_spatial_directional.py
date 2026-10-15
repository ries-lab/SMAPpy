"""Spatial statistics' directional approaches: the correlation image and the
correlation along a line (`smappy.correlation`)."""
import numpy as np
import pytest

from smappy.chain import Block, ChainSpec, Step, chain_class
from smappy.correlation import (Frame, directional, lateral_mask, period_of, select_peak,
                                spectrum, spectrum_period, z_range)
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.spatial_statistics import (DirectionalSettings, SpatialStatistics,
                                               SpatialStatisticsSettings)
from smappy.regions import Region
from smappy.session import Session

PERIOD = 190.0
SPATIAL = "Analysis/Measure/Spatial statistics"


def rings(rng, n=6000, length=4000.0, width=400.0, background=2000):
    """Rings across a strip every 190 nm: along x the positions bunch at
    multiples of the period (15 nm spread), across it they are uniform; and
    a uniform background."""
    k = rng.integers(0, int(length // PERIOD) + 1, n)
    x = np.r_[PERIOD * k + rng.normal(0, 15, n), rng.uniform(0, length, background)]
    y = np.r_[rng.uniform(0, width, n), rng.uniform(0, width, background)]
    return x, y


def turned(x, y, angle, centre=(0.0, 0.0)):
    c, s = np.cos(angle), np.sin(angle)
    return centre[0] + x * c - y * s, centre[1] + x * s + y * c


def settings(**directional):
    return SpatialStatisticsSettings(
        pair_correlation=False, ripley_l=False, image=True, line=True,
        directional=DirectionalSettings(**{"bin_nm": 10.0, "max_shift_nm": 800.0,
                                           **directional}))


def run(locs, region, s, region_id="1"):
    return SpatialStatistics().run(Context(locs=locs, region=region, region_id=region_id), s)


def table(x, y, **more):
    return Localizations({"x_nm": np.asarray(x, float), "y_nm": np.asarray(y, float),
                          **more}, {})


# --------------------------------------------------------------- the period

def test_rings_every_190_nm_give_their_period_along_a_line():
    rng = np.random.default_rng(1)
    x, y = rings(rng)
    line = Region.line((0, 200), (4000, 200), 400)
    result = run(table(x, y), line, settings())
    assert result.values["period_nm"] == pytest.approx(PERIOD, abs=3)
    assert result.values["period_g"] > 1.5
    assert result.values["period_fft_nm"] == pytest.approx(PERIOD, abs=5)
    assert "lag_nm" in result.arrays and result.arrays["line_g"].shape == \
        result.arrays["lag_nm"].shape


def test_turned_the_period_is_found_along_a_tilted_line_roi():
    rng = np.random.default_rng(2)
    x, y = rings(rng)
    angle = np.radians(35)
    tx, ty = turned(x, y - 200, angle, (1000, 3000))
    p0 = (1000.0, 3000.0)
    p1 = (1000 + 4000 * np.cos(angle), 3000 + 4000 * np.sin(angle))
    result = run(table(tx, ty), Region.line(p0, p1, 400), settings())
    assert result.values["period_nm"] == pytest.approx(PERIOD, abs=3)
    # along x of the table the same rings are smeared: no strong period
    flat = run(table(tx, ty), Region.rect(*np.min([p0, p1], 0) - 300,
                                          *np.max([p0, p1], 0) + 300), settings())
    assert not flat.values["period_g"] > 1.3


def test_an_rois_named_line_turns_the_axes():
    rng = np.random.default_rng(3)
    x, y = rings(rng)
    angle = np.radians(-50)
    tx, ty = turned(x - 2000, y - 200, angle, (5000, 5000))
    session = Session(table(tx, ty, frame=np.arange(len(tx), dtype=np.int64)))
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(4200, "circle")
    file_id = next(iter(project.sources))
    roi = project.add_roi(file_id, [5000.0, 5000.0])
    roi_id = roi if isinstance(roi, str) else getattr(roi, "id", roi)
    name = project.line_names[0]
    ends = [turned(np.array([-2000.0, 2000.0]), np.zeros(2), angle, (5000, 5000))]
    (ex, ey), = ends
    project.set_line(roi_id, name, [(ex[0], ey[0]), (ex[1], ey[1])])
    step = Step(plugin=SPATIAL, values={"pair_correlation": False, "ripley_l": False,
                                         "line": True, "directional.bin_nm": 10.0,
                                         "directional.max_shift_nm": 800.0})
    cls = chain_class(ChainSpec(steps=[Block(steps=[step])]))
    session.run(cls(), cls.Settings())
    (row,) = project.results()
    assert row["period_nm"] == pytest.approx(PERIOD, abs=4)


def test_a_structure_periodic_in_z_gives_its_period_in_z():
    rng = np.random.default_rng(4)
    n = 8000
    z = PERIOD * rng.integers(-2, 3, n) + rng.normal(0, 15, n)
    z[:20] = 5000.0                              # far-out mislocalizations
    x, y = rng.uniform(0, 2000, n), rng.uniform(0, 2000, n)
    result = run(table(x, y, z_nm=z), Region.rect(0, 0, 2000, 2000),
                 settings(axis="z", plane="xz", max_shift_nm=600.0))
    assert result.values["period_nm"] == pytest.approx(PERIOD, abs=4)
    found = result.data["directional"]
    assert found.z[1] <= 500.0 and found.image_axes == ("x", "z")


def test_the_z_range_is_robust_and_clipped():
    z = np.r_[np.linspace(-300, 300, 1000), [4000.0, -9000.0]]
    lo, hi = z_range(z)
    assert -305 < lo < -290 and 290 < hi < 305
    assert z_range(np.linspace(-2000, 2000, 1000)) == (-500.0, 500.0)
    assert z_range([]) is None


# ---------------------------------------------------- cross-correlation

def test_two_layers_shifted_by_d_cross_correlate_with_a_peak_at_d():
    rng = np.random.default_rng(5)
    n = 4000
    x, y = rng.uniform(0, 4000, n), rng.uniform(0, 400, n)
    d = 60.0
    session = Session(table(np.r_[x, x + d], np.r_[y, y],
                            channel=np.repeat([1, 2], n).astype(np.int32),
                            frame=np.arange(2 * n, dtype=np.int64)))
    session.show_grouped(0, False)
    session.layers[0].set_bound("channel", 1, 1)
    second = session.add_layer(like=0)
    second.set_bound("channel", 2, 2)
    result = session.run(SpatialStatistics(), settings(bin_nm=5.0, max_shift_nm=300.0))
    assert result.values["shift_nm_L1xL2"] == pytest.approx(d, abs=2)
    found = result.data["directional"]
    image = found.images[(0, 1)]
    i, j = np.unravel_index(np.nanargmax(image), image.shape)
    assert found.lags[i] == pytest.approx(d, abs=5) and abs(found.lags[j]) <= 5
    # the auto-correlations of uniform layers are 1 away from 0
    auto = found.images[(0, 0)]
    centre = len(found.lags) // 2
    assert np.nanmean(auto[centre + 5:, :]) == pytest.approx(1, abs=0.05)


def test_uniform_points_correlate_to_1_in_any_window_shape():
    rng = np.random.default_rng(6)
    disc = Region.circle((0, 0), 1500)
    p = rng.uniform(-1500, 1500, (20000, 2))
    p = p[disc.mask(p[:, 0], p[:, 1])]
    mask, u0, v0 = lateral_mask(disc, Frame(), 10)
    found = directional([[{"u": p[:, 0], "v": p[:, 1]}]], [(mask, u0, v0)], 10, 600)
    g = found.line[(0, 0)]
    far = np.abs(found.lags) > 50
    assert np.nanmean(g[far]) == pytest.approx(1, abs=0.03)
    assert np.nanmean(found.images[(0, 0)]) == pytest.approx(1, abs=0.05)


# ----------------------------------------------------------- the helpers

def test_the_peak_nearest_the_guess_is_taken_else_the_first_strong_one():
    lag = np.arange(-800, 801, 10.0)
    g = 1 + 0.6 * np.cos(2 * np.pi * lag / PERIOD) * np.exp(-np.abs(lag) / 2000)
    assert period_of(lag, g)[0] == pytest.approx(PERIOD, abs=2)
    assert period_of(lag, g, guess=400)[0] == pytest.approx(2 * PERIOD, abs=3)
    assert select_peak([100, 200], [1, 5]) == (200.0, 5.0)
    assert select_peak([100, 200], [1, 5], guess=120) == (100.0, 1.0)


def test_the_spectrum_finds_a_sine_profiles_period():
    x = np.arange(400) * 5.0
    profile = 10 + 4 * np.sin(2 * np.pi * x / PERIOD)
    f, mag = spectrum(profile, 5.0)
    period, height = spectrum_period(f, mag, bin_nm=5.0)
    # 190 nm is not a whole number of cycles in 2000 nm: the peak leaks into
    # the next bin and its height is below the amplitude
    assert period == pytest.approx(PERIOD, rel=0.03) and 2 < height <= 4


def test_the_form_shows_each_directional_part_only_while_on():
    plugin = SpatialStatistics()
    off = plugin.describe(None, SpatialStatisticsSettings())
    assert off.shown["directional"] is False
    on = plugin.describe(None, SpatialStatisticsSettings(line=True))
    assert on.shown["directional"] is True and on.shown["directional.axis"] is True
    assert on.shown["directional.plane"] is False
