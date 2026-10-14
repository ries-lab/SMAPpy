"""Analysis/Measure/Cluster shape: known shapes give their numbers."""
import math

import numpy as np
import pytest

from smappy import plugins
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.cluster_shape import cluster_shape
from smappy.session import Session

PLUGIN = "Analysis/Measure/Cluster shape"


def disc(n, radius, rng, centre=(0.0, 0.0)):
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    return centre[0] + r * np.cos(phi), centre[1] + r * np.sin(phi)


def ellipse(n, a, b, angle_deg, rng, centre=(0.0, 0.0)):
    u, v = disc(n, 1.0, rng)
    u, v = a * u, b * v
    t = np.radians(angle_deg)
    return (centre[0] + u * np.cos(t) - v * np.sin(t),
            centre[1] + u * np.sin(t) + v * np.cos(t))


def run(locs_or_ctx, **settings):
    plugin = plugins.get(PLUGIN)
    ctx = locs_or_ctx if isinstance(locs_or_ctx, Context) else Context(locs=locs_or_ctx)
    return plugin().run(ctx, plugin.Settings(**settings))


def test_a_disc_gives_its_area_radius_of_gyration_and_no_elongation():
    rng = np.random.default_rng(0)
    x, y = disc(20000, 500.0, rng, centre=(3000.0, -200.0))
    v = cluster_shape(x, y)
    assert v["n_locs"] == 20000
    assert v["centre_x_nm"] == pytest.approx(3000, abs=5)
    assert v["centre_y_nm"] == pytest.approx(-200, abs=5)
    assert v["area_nm2"] == pytest.approx(math.pi * 500 ** 2, rel=0.01)
    assert v["area_nm2"] < math.pi * 500 ** 2              # the hull lies inside
    assert v["density_per_um2"] == pytest.approx(20000 / (math.pi * 0.25), rel=0.01)
    assert v["rg_nm"] == pytest.approx(500 / math.sqrt(2), rel=0.01)
    assert v["std_long_nm"] == pytest.approx(250, rel=0.02)
    assert v["std_short_nm"] == pytest.approx(250, rel=0.02)
    assert v["elongation"] == pytest.approx(1.0, abs=0.04)
    assert v["extent_long_nm"] == pytest.approx(1000, rel=0.01)
    assert "volume_nm3" not in v


@pytest.mark.parametrize("angle", [0.0, 30.0, 120.0])
def test_an_ellipse_gives_its_axes_angle_and_elongation(angle):
    rng = np.random.default_rng(1)
    x, y = ellipse(20000, 400.0, 100.0, angle, rng)
    v = cluster_shape(x, y)
    # a uniform ellipse's standard deviation along an axis is half the semi-axis
    assert v["std_long_nm"] == pytest.approx(200, rel=0.02)
    assert v["std_short_nm"] == pytest.approx(50, rel=0.02)
    assert v["elongation"] == pytest.approx(4.0, rel=0.03)
    off = (v["angle_deg"] - angle + 90) % 180 - 90       # 179.9 is 0
    assert off == pytest.approx(0, abs=1.0)
    assert 0 <= v["angle_deg"] < 180
    assert v["area_nm2"] == pytest.approx(math.pi * 400 * 100, rel=0.01)
    assert v["extent_long_nm"] == pytest.approx(800, rel=0.01)
    assert v["extent_short_nm"] == pytest.approx(200, rel=0.01)
    assert v["rg_nm"] == pytest.approx(math.sqrt((400 ** 2 + 100 ** 2) / 4), rel=0.01)


def test_a_line_has_its_length_and_angle_and_no_area():
    rng = np.random.default_rng(2)
    s = rng.uniform(0, 1000, 5000)
    t = np.radians(60)
    v = cluster_shape(100 + s * np.cos(t), 50 + s * np.sin(t))
    assert v["area_nm2"] == 0
    assert math.isnan(v["density_per_um2"])              # nothing to divide by
    assert v["std_long_nm"] == pytest.approx(1000 / math.sqrt(12), rel=0.02)
    assert v["std_short_nm"] == 0
    assert math.isnan(v["elongation"])
    assert v["extent_long_nm"] == pytest.approx(1000, rel=0.002)
    assert v["angle_deg"] == pytest.approx(60, abs=1e-6)
    # a line of some width is only very elongated
    w = cluster_shape(s, rng.normal(0, 10, 5000))
    assert w["elongation"] == pytest.approx(1000 / math.sqrt(12) / 10, rel=0.05)
    assert w["angle_deg"] == pytest.approx(0, abs=0.5) or w["angle_deg"] > 179.5


def test_a_cylinder_gives_its_volume_and_z_extent():
    rng = np.random.default_rng(3)
    x, y = disc(20000, 300.0, rng)
    z = rng.uniform(-100, 100, 20000)
    v = cluster_shape(x, y, z)
    assert v["volume_nm3"] == pytest.approx(math.pi * 300 ** 2 * 200, rel=0.02)
    assert v["z_extent_nm"] == pytest.approx(200, rel=0.01)
    assert v["centre_z_nm"] == pytest.approx(0, abs=3)


def test_too_few_points_give_what_they_can_and_nan_for_the_rest():
    empty = cluster_shape([], [], [])
    assert empty["n_locs"] == 0 and math.isnan(empty["area_nm2"])
    assert math.isnan(empty["volume_nm3"])
    one = cluster_shape([5.0], [7.0])
    assert one["centre_x_nm"] == 5 and one["rg_nm"] == 0 and one["area_nm2"] == 0
    assert math.isnan(one["std_long_nm"])
    two = cluster_shape([0.0, 30.0], [0.0, 40.0])
    assert two["extent_long_nm"] == pytest.approx(50) and two["area_nm2"] == 0
    assert two["angle_deg"] == pytest.approx(math.degrees(math.atan2(40, 30)))


def shapes_table():
    """Three clusters: a disc, an ellipse at 45 degrees and a wide line."""
    rng = np.random.default_rng(4)
    parts = [disc(3000, 200.0, rng, centre=(1000.0, 1000.0)),
             ellipse(3000, 300.0, 75.0, 45.0, rng, centre=(3000.0, 1000.0)),
             (rng.uniform(4500, 5500, 3000), rng.normal(1000, 5, 3000))]
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    n = len(x)
    return Localizations({"x_nm": x, "y_nm": y, "frame": np.arange(n) % 500,
                          "photons": np.full(n, 1000.0), "xy_err_nm": np.full(n, 10.0),
                          "cluster_id": np.repeat([1, 2, 3], 3000).astype(np.int32)})


def test_it_runs_per_cluster():
    session = Session(shapes_table())
    session.show_grouped(0, False)
    step = plugins.get(PLUGIN)
    from smappy.roi_manager.pipeline import Step
    table = session.evaluate_regions(
        "cluster_id", [Step(label="shape", path=PLUGIN, plugin=step(),
                            settings=step.Settings())])["shape"]
    columns = table.columns()
    assert list(columns["n_locs"]) == [3000, 3000, 3000]
    assert columns["rg_nm"][0] == pytest.approx(200 / math.sqrt(2), rel=0.03)
    assert columns["angle_deg"][1] == pytest.approx(45, abs=2)
    assert columns["elongation"][1] == pytest.approx(4, rel=0.06)
    assert columns["extent_long_nm"][2] == pytest.approx(1000, rel=0.01)
    assert columns["elongation"][2] > 50


def test_on_the_view_it_measures_each_layer_and_names_the_second_l2():
    session = Session(shapes_table())
    session.show_grouped(0, False)
    second = session.add_layer(like=0)
    second.set_bound("cluster_id", 1, 1)                 # the disc only
    result = run(session.context())
    assert result.values["n_locs"] == 9000
    assert result.values["n_locs_L2"] == 3000
    assert result.values["rg_nm_L2"] == pytest.approx(200 / math.sqrt(2), rel=0.03)
    assert "area" in result.text


def test_z_is_used_when_the_table_has_it_and_asked_for():
    rng = np.random.default_rng(5)
    x, y = disc(2000, 100.0, rng)
    locs = Localizations({"x_nm": x, "y_nm": y, "z_nm": rng.uniform(0, 50, 2000)})
    assert run(locs).values["z_extent_nm"] == pytest.approx(50, rel=0.02)
    assert "z_extent_nm" not in run(locs, use_z=False).values
    with pytest.raises(ValueError, match="no localizations"):
        run(Localizations({"x_nm": np.zeros(0), "y_nm": np.zeros(0)}))
