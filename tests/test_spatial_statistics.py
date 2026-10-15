"""Spatial statistics: pair counts, the window's edge correction, the
envelopes, and the values a region gets; Neighbourhood's *express as*."""
import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist

from smappy.locs import Localizations
from smappy.plugins import Context, PreflightQuestion
from smappy.plugins.neighbourhood import (NeighbourhoodSettings, default_name, express,
                                          neighbourhood)
from smappy.plugins.spatial_statistics import (
    Points, RadialSettings, SpatialStatistics, SpatialStatisticsSettings, first_maximum,
    keep_blinks, mad_test, pair_counts, parse_distances, spatial_statistics,
    window_coverage, window_of_points, window_of_region)
from smappy.regions import Region

SQUARE = Region.rect(0, 0, 5000, 5000)
DISC = Region.circle((0, 0), 3000)
ELL = Region.polygon([[0, 0], [6000, 0], [6000, 2000], [2000, 2000], [2000, 6000],
                      [0, 6000]])


def uniform_in(region, n, rng):
    """``n`` points uniform in the shape itself, by rejection."""
    x0, y0, x1, y1 = region.bounds
    out = np.zeros((0, 2))
    while len(out) < n:
        p = np.column_stack([rng.uniform(x0, x1, 2 * n), rng.uniform(y0, y1, 2 * n)])
        out = np.vstack([out, p[region.mask(p[:, 0], p[:, 1])]])
    return out[:n]


def disc_clusters(radius, rng, side=10000.0, n_clusters=150, per_cluster=30):
    """Clusters: discs of ``radius`` filled uniformly, centres at random."""
    centre = rng.uniform(0, side, (n_clusters, 2))
    r = radius * np.sqrt(rng.random((n_clusters, per_cluster)))
    t = rng.uniform(0, 2 * np.pi, (n_clusters, per_cluster))
    p = np.stack([centre[:, None, 0] + r * np.cos(t),
                  centre[:, None, 1] + r * np.sin(t)], -1).reshape(-1, 2)
    return p[((p >= 0) & (p <= side)).all(axis=1)]


def bins(edges):
    return 0.5 * (edges[1:] + edges[:-1])


# ------------------------------------------------------------- pair counts

def test_the_cpp_pair_histogram_equals_a_brute_force_one():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 3000, (2000, 2))
    b = rng.uniform(500, 3500, (1500, 2))
    edges = np.arange(0, 301, 10)
    assert np.array_equal(pair_counts(a, None, 300, 10),
                          2 * np.histogram(pdist(a), bins=edges)[0])
    assert np.array_equal(pair_counts(a, b, 300, 10),
                          np.histogram(cdist(a, b).ravel(), bins=edges)[0])
    # two layers over one table: a row in both is not its own pair
    same = np.full(len(a), -1)
    both = np.vstack([b, a[:100]])
    same[:100] = len(b) + np.arange(100)
    d = cdist(a, both)
    d[np.arange(100), len(b) + np.arange(100)] = np.inf
    assert np.array_equal(pair_counts(a, both, 300, 10, same),
                          np.histogram(d.ravel(), bins=edges)[0])


def test_the_pair_histogram_takes_every_pair_whatever_the_cells():
    rng = np.random.default_rng(2)
    a = np.vstack([rng.normal(0, 30, (300, 2)), rng.uniform(-5e4, 5e4, (300, 2))])
    edges = np.arange(0, 51, 1.0)
    assert np.array_equal(pair_counts(a, None, 50, 1),
                          2 * np.histogram(pdist(a), bins=edges)[0])


# ------------------------------------------------------------- the window

def test_a_window_raster_has_the_shapes_area():
    for region, area in ((SQUARE, 25e6), (DISC, np.pi * 9e6), (ELL, 20e6)):
        assert window_of_region(region).area == pytest.approx(area, rel=0.005)


def test_the_set_covariance_of_a_rectangle_is_the_known_one():
    w = window_of_region(Region.rect(0, 0, 4000, 2000))
    r = np.array([0.0, 300.0, 1000.0])
    theta = (np.arange(10000) + 0.5) * np.pi / 10000
    exact = [np.mean((4000 - x * np.abs(np.cos(theta))) * (2000 - x * np.abs(np.sin(theta))))
             for x in r]
    assert w.covariance(r) == pytest.approx(exact, rel=0.002)


def test_the_occupied_area_of_a_uniform_rectangle_is_the_rectangle():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 10000, (20000, 2))
    w = window_of_points(p[:, 0], p[:, 1], "occupied", 200)
    assert w.area == pytest.approx(1e8, rel=0.01)
    assert w.contains(p[:, 0], p[:, 1]).all()
    # a cell in a larger field: the empty outside is not window
    q = p[np.hypot(p[:, 0] - 5000, p[:, 1] - 5000) < 3000]
    assert window_of_points(q[:, 0], q[:, 1]).area == pytest.approx(np.pi * 9e6, rel=0.05)


# ---------------------------------------------------- CSR in three windows

@pytest.mark.parametrize("region", [SQUARE, DISC, ELL], ids=["square", "disc", "L"])
def test_csr_gives_g_1_and_l_minus_r_0_inside_the_envelope(region):
    rng = np.random.default_rng(4)
    p = uniform_in(region, 5000, rng)
    edges, (curve,) = spatial_statistics([[Points(p)]], [window_of_region(region)],
                                         600, 10, n_sims=39, seed=1)
    assert np.nanmean(curve.g) == pytest.approx(1.0, abs=0.02)
    assert np.nanmax(np.abs(curve.g[10:] - 1)) < 0.12
    assert np.nanmax(np.abs(curve.L_r)) < 5
    assert curve.test["p"] > 0.05
    assert curve.test["T"] <= curve.test["T_crit"]


def test_without_edge_correction_the_same_points_would_look_dispersed():
    """The point of the window: dividing by the pairs a plane would give
    makes g fall with r, the pairs that would cross the edge being missing."""
    rng = np.random.default_rng(5)
    p = uniform_in(DISC, 5000, rng)
    edges = np.arange(0, 601, 10.0)
    counts = pair_counts(p, None, 600, 10)
    plane = 5000 * 4999 / (np.pi * 9e6) ** 2 * np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) \
        * np.pi * 9e6
    assert (counts / plane)[-10:].mean() < 0.9


# --------------------------------------------------------------- clusters

@pytest.mark.parametrize("radius", [50.0, 100.0, 200.0])
def test_clusters_of_known_radius_are_found_at_that_scale(radius):
    """The maximum of L - r of discs of radius R lies between R and 2R
    (Kiskowski et al. 2009), and the data leave the envelope."""
    rng = np.random.default_rng(6)
    p = disc_clusters(radius, rng)
    w = window_of_points(p[:, 0], p[:, 1], "box")
    edges, (curve,) = spatial_statistics([[Points(p)]], [w], 6 * radius, radius / 20,
                                         n_sims=19)
    at, height = first_maximum(bins(edges), curve.L_r, 0.0)
    assert radius <= at <= 2 * radius
    assert height > radius
    assert curve.test["p"] == pytest.approx(1 / 20)
    assert curve.test["T"] > curve.test["T_crit"]


# ------------------------------------------------------- cross-correlation

def test_two_independent_layers_cross_correlate_to_1():
    rng = np.random.default_rng(7)
    a, b = rng.uniform(0, 10000, (5000, 2)), rng.uniform(0, 10000, (5000, 2))
    w = window_of_region(Region.rect(0, 0, 10000, 10000))
    edges, curves = spatial_statistics([[Points(a)], [Points(b)]], [w], 300, 5,
                                       n_sims=19)
    cross = curves[2]
    assert cross.is_cross and cross.name == "layer 1 x layer 2"
    assert np.nanmean(cross.g) == pytest.approx(1.0, abs=0.03)
    assert cross.test["p"] > 0.05


def test_a_layer_against_a_copy_shifted_by_d_peaks_at_d():
    rng = np.random.default_rng(8)
    a = rng.uniform(0, 10000, (5000, 2))
    for d in (40.0, 120.0):
        b = a + [d * np.cos(0.7), d * np.sin(0.7)]
        w = window_of_points(*np.vstack([a, b]).T, "box")
        edges, curves = spatial_statistics([[Points(a)], [Points(b)]], [w], 300, 4,
                                           null="none")
        assert bins(edges)[np.nanargmax(curves[2].g)] == pytest.approx(d, abs=4)


# ------------------------------------------------------------ keep blinks

def blinking(rng, n_molecules=1500, side=8000.0, precision=15.0):
    """Molecules at random, each seen a few times with the precision."""
    m = rng.uniform(0, side, (n_molecules, 2))
    k = rng.geometric(0.3, n_molecules)
    owner = np.repeat(np.arange(n_molecules), k)
    return m[owner] + rng.normal(0, precision, (len(owner), 2)), owner


def test_keep_blinks_moves_each_blink_whole():
    rng = np.random.default_rng(9)
    xy, owner = blinking(rng, 200)
    w = window_of_points(xy[:, 0], xy[:, 1], "box")
    moved = keep_blinks(w, xy, owner, rng)
    for g in range(5):
        here = owner == g
        assert np.allclose(moved[here] - moved[here].mean(0), xy[here] - xy[here].mean(0))
    assert not np.allclose(moved, xy)


def test_blinking_is_clustered_for_csr_and_random_for_keep_blinks():
    rng = np.random.default_rng(10)
    xy, owner = blinking(rng)
    w = window_of_points(xy[:, 0], xy[:, 1], "box")
    layers = [[Points(xy, owner)]]
    _, (csr_curve,) = spatial_statistics(layers, [w], 300, 5, null="csr", n_sims=19)
    _, (kept,) = spatial_statistics(layers, [w], 300, 5, null="keep_blinks", n_sims=39,
                                    seed=2)
    assert csr_curve.test["p"] == pytest.approx(1 / 20)
    assert kept.test["p"] > 0.05


def test_keep_blinks_without_group_id_says_what_to_do():
    rng = np.random.default_rng(11)
    xy = rng.uniform(0, 1000, (100, 2))
    w = window_of_points(xy[:, 0], xy[:, 1], "box")
    with pytest.raises(ValueError, match="group_id"):
        spatial_statistics([[Points(xy)]], [w], 100, 10, null="keep_blinks", n_sims=3)


# ------------------------------------------------------------ the helpers

def test_the_mad_test_ranks_the_data_among_the_simulations():
    rng = np.random.default_rng(12)
    sims = rng.normal(0, 1, (99, 50))
    assert mad_test(np.zeros(50), sims)["p"] > 0.5
    assert mad_test(np.full(50, 10.0), sims)["p"] == pytest.approx(0.01)


def test_the_first_maximum_is_the_first_of_two_peaks():
    r = np.arange(0, 600, 5.0)
    y = 1 + 0.8 * np.exp(-(r - 190) ** 2 / 800) + np.exp(-(r - 380) ** 2 / 800)
    at, height = first_maximum(r, y, 1.0)
    assert at == pytest.approx(190, abs=5)
    assert height == pytest.approx(1.8, abs=0.1)
    # a bump below half the highest is not the first maximum
    y2 = 1 + 0.3 * np.exp(-(r - 100) ** 2 / 200) + np.exp(-(r - 380) ** 2 / 800)
    assert first_maximum(r, y2, 1.0)[0] == pytest.approx(380, abs=5)
    assert np.isnan(first_maximum(r, np.ones_like(r) * 0.5, 1.0)[0])


def test_distances_are_read_from_a_list():
    assert parse_distances("95, 190;380 ") == [95.0, 190.0, 380.0]
    assert parse_distances("") == []
    with pytest.raises(ValueError, match="not a number"):
        parse_distances("95, ten")


# --------------------------------------------------------------- the plugin

def table(xy, **extra):
    return Localizations({"x_nm": xy[:, 0], "y_nm": xy[:, 1], **extra}, {})


def settings(**radial):
    return SpatialStatisticsSettings(ripley_k=True,
                                     radial=RadialSettings(**{"simulations": 9, **radial}))


def test_the_plugin_hands_back_values_and_arrays_for_a_region():
    rng = np.random.default_rng(13)
    p = disc_clusters(80.0, rng, side=4000, n_clusters=60)
    locs = table(p)
    region = Region.rect(0, 0, 4000, 4000)
    ctx = Context(locs=locs, region=region, region_id="1")
    result = SpatialStatistics().run(ctx, settings(r_max_nm=400, dr_nm=5, at_nm="100"))
    v, a = result.values, result.arrays
    assert 80 <= v["lr_max_r_nm"] <= 160
    assert v["g_max"] > 2 and np.isfinite(v["g_at_100nm"]) and np.isfinite(v["lr_at_100nm"])
    assert v["leaves_envelope"] == 1.0 and v["p_csr"] == pytest.approx(0.1)
    n = len(a["r_nm"])
    assert n == 80
    for key in ("pairs", "expected", "g", "K_nm2", "lr_nm", "g_lo", "g_hi",
                "lr_nm_lo", "lr_nm_hi"):
        assert a[key].shape == (n,)
    assert np.allclose(a["g"], a["pairs"] / a["expected"], equal_nan=True)
    assert set(result.plots) == {"K(r)", "L(r) - r", "window"}
    assert "rect ROI" in result.text


def test_the_plugin_keeps_files_apart():
    rng = np.random.default_rng(14)
    a, b = rng.uniform(0, 3000, (2000, 2)), rng.uniform(0, 3000, (2000, 2))
    locs = table(np.vstack([a, b]), filenumber=np.repeat([1, 2], 2000))
    result = SpatialStatistics().run(Context(locs=locs),
                                     settings(window="box", r_max_nm=200, null="none"))
    one = pair_counts(a, None, 200, 5) + pair_counts(b, None, 200, 5)
    assert np.array_equal(result.arrays["pairs"], one)
    assert np.nanmean(result.arrays["g"]) == pytest.approx(1, abs=0.05)
    assert result.data["files"] == [1, 2]


def test_a_large_run_asks_first_and_offers_one_without_the_envelope():
    rng = np.random.default_rng(15)
    locs = table(rng.uniform(0, 40000, (1_000_000, 2)))
    question = SpatialStatistics().preflight(Context(locs=locs), settings(simulations=99))
    assert isinstance(question, PreflightQuestion)
    assert question.choices[0].settings.radial.null == "none"
    small = table(rng.uniform(0, 4000, (5000, 2)))
    assert SpatialStatistics().preflight(Context(locs=small), settings()) is None


def test_the_form_shows_the_radial_part_only_with_a_radial_approach():
    plugin = SpatialStatistics()
    off = SpatialStatisticsSettings(pair_correlation=False, ripley_l=False)
    assert plugin.describe(None, off).shown["radial"] is False
    on = plugin.describe(None, SpatialStatisticsSettings())
    assert on.shown["radial"] is True
    assert on.shown["radial.occupied_bin_nm"] is True
    nothing = SpatialStatisticsSettings(radial=RadialSettings(null="none"))
    assert plugin.describe(None, nothing).shown["radial.simulations"] is False


# ------------------------------------------------- Neighbourhood's express as

def test_a_circle_at_a_straight_edge_covers_half_its_area():
    w = window_of_region(Region.rect(0, 0, 10000, 10000))
    cover = window_coverage(w, [5000, 0.0, 0.0], [5000, 5000, 0.0], 100.0)
    assert cover == pytest.approx(np.pi * 1e4 * np.array([1, 0.5, 0.25]), rel=0.02)


def test_local_l_is_r_and_the_relative_density_1_in_a_uniform_field_to_the_edge():
    rng = np.random.default_rng(16)
    p = rng.uniform(0, 20000, (200000, 2))
    t = table(p)
    keep = np.ones(len(p), dtype=bool)
    edge = ((p < 300) | (p > 19700)).any(axis=1)
    for how, expected in (("relative", 1.0), ("local_l", 80.0)):
        s = NeighbourhoodSettings(radius_nm=80, express_as=how, window="box")
        v = neighbourhood((t, keep), (t, keep), s)
        # local L is the square root: Jensen puts its mean a little below r
        tol = 0.01 if how == "relative" else 0.02
        assert np.mean(v[~edge]) == pytest.approx(expected, rel=tol)
        assert np.mean(v[edge]) == pytest.approx(expected, rel=tol)


def test_express_is_getis_and_franklins_local_l_away_from_the_edge():
    counts = np.array([0.0, 10, 40])
    area, others, r = 1e8, 9999.0, 50.0
    coverage = np.full(3, np.pi * r ** 2)
    expected = np.sqrt(area * counts / (np.pi * others))
    assert express(counts, coverage, others, area, r, "local_l") == pytest.approx(expected)
    assert express(counts, coverage, others, area, r, "count") == pytest.approx(counts)


def test_clusters_stand_out_in_local_l_and_the_name_says_what_it_is():
    rng = np.random.default_rng(17)
    p = np.vstack([disc_clusters(60.0, rng, n_clusters=50), rng.uniform(0, 10000, (3000, 2))])
    t = table(p)
    keep = np.ones(len(p), dtype=bool)
    s = NeighbourhoodSettings(radius_nm=60, express_as="local_l")
    v = neighbourhood((t, keep), (t, keep), s)
    assert np.median(v[:-3000]) > 3 * np.median(v[-3000:])
    assert default_name(s) == "local_l_r60"
    assert default_name(NeighbourhoodSettings(express_as="relative")) == "density_rel_r50"


# ------------------------------------------------------------ in a session

def two_colour(rng, side=6000.0):
    """Two channels, the second a copy of the first 60 nm away."""
    a = rng.uniform(0, side, (3000, 2))
    b = a + [60.0, 0.0]
    xy = np.vstack([a, b])
    return table(xy, channel=np.repeat([1, 2], len(a)).astype(np.int32),
                 frame=np.arange(len(xy), dtype=np.int64))


def test_each_visible_layer_and_each_pair_in_a_session():
    from smappy.session import Session
    rng = np.random.default_rng(18)
    session = Session(two_colour(rng))
    session.show_grouped(0, False)
    session.layers[0].set_bound("channel", 1, 1)
    second = session.add_layer(like=0)
    second.set_bound("channel", 2, 2)
    result = SpatialStatistics().run(session.context(),
                                     settings(r_max_nm=200, dr_nm=4, null="none"))
    names = [c.name for c in result.data["curves"]]
    assert len(names) == 3 and " x " in names[2]
    mid = result.arrays["r_nm"]
    assert mid[np.nanargmax(result.arrays["g_L1xL2"])] == pytest.approx(60, abs=4)
    assert np.nanmean(result.arrays["g_L2"]) == pytest.approx(1, abs=0.05)
    assert "g_max_r_nm_L1xL2" in result.values


def test_the_drawn_roi_is_the_window():
    from smappy.session import Session
    rng = np.random.default_rng(19)
    session = Session(table(rng.uniform(0, 10000, (20000, 2)),
                            frame=np.arange(20000, dtype=np.int64)))
    session.show_grouped(0, False)
    session.set_roi(Region.circle((5000, 5000), 2000))
    result = SpatialStatistics().run(session.context(), settings(r_max_nm=300, null="none"))
    assert result.data["area_nm2"] == pytest.approx(np.pi * 4e6, rel=0.01)
    assert np.nanmean(result.arrays["g"]) == pytest.approx(1, abs=0.04)
    assert "circle ROI" in result.text


def test_it_runs_per_cluster():
    from smappy.roi_manager.pipeline import Step
    from smappy.session import Session
    from smappy import plugins
    rng = np.random.default_rng(20)
    p = rng.uniform(0, 4000, (6000, 2))
    cluster = np.where(p[:, 0] < 2000, 1, 2).astype(np.int32)
    session = Session(table(p, cluster_id=cluster, frame=np.arange(6000, dtype=np.int64)))
    session.show_grouped(0, False)
    path = "Analysis/Measure/Spatial statistics"
    step = plugins.get(path)
    results = session.evaluate_regions(
        "cluster_id", [Step(label="pcf", path=path, plugin=step(),
                            settings=settings(window="box", r_max_nm=200,
                                              null="none"))])["pcf"]
    columns = results.columns()
    assert len(columns["lr_max_r_nm"]) == 2
