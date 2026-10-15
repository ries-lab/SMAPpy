"""Curves: the mean of many regions' curves with its spread, per keyword,
pooled where the evaluator says how."""
import numpy as np
import pytest

from smappy.chain import Block, ChainSpec, Step, chain_class
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.curves import (NO_GROUP, Curves, CurvesSettings, Region, spread_of,
                                   summarise)
from smappy.plugins.spatial_statistics import (Points, RadialSettings, SpatialStatistics,
                                               SpatialStatisticsSettings,
                                               spatial_statistics, window_of_region)
from smappy.regions import Region as Shape
from smappy.session import Session

SPATIAL = "Analysis/Measure/Spatial statistics"
SIDE = 2000.0           # each ROI's square
GAP = 5000.0            # between ROI centres: no pair reaches across


# ------------------------------------------------------------- the numbers

def test_regions_with_known_curves_give_their_mean_and_sd():
    x = np.linspace(0, 10, 11)
    curves = [x, 2 * x, 3 * x, 6 * x]
    regions = [Region(str(i), {"x_nm": x, "y": c}, {"v": float(i)})
               for i, c in enumerate(curves)]
    x_name, grid, names, (group,) = summarise(regions)
    assert x_name == "x_nm" and names == ["y"] and group.name == NO_GROUP
    assert np.allclose(group.mean["y"], 3 * x)
    assert np.allclose(group.spread["y"], np.std([1, 2, 3, 6], ddof=1) * x)
    assert group.values == {"v": 1.5} and group.n == 4
    _, _, _, (sem,) = summarise(regions, spread="sem")
    assert np.allclose(sem.spread["y"], group.spread["y"] / 2)


def test_a_missing_value_is_left_out_of_its_point_only():
    stack = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, np.nan], [5.0, 4.0, 7.0]])
    mean, sd, count = spread_of(stack)
    assert np.allclose(mean, [3.0, 3.0, 7.0])
    assert count.tolist() == [3, 2, 1]
    assert sd[0] == pytest.approx(2.0) and np.isnan(sd[2])


def test_a_region_on_another_grid_is_refused_by_name():
    a = Region("1", {"r_nm": np.arange(5.0), "g": np.ones(5)})
    b = Region("7", {"r_nm": np.arange(5.0) * 2, "g": np.ones(5)})
    with pytest.raises(ValueError, match="region 7 .* another grid"):
        summarise([a, b])
    c = Region("8", {"r_nm": np.arange(6.0), "g": np.ones(6)})
    with pytest.raises(ValueError, match="region 8"):
        summarise([a, c])


def test_the_envelope_edges_are_not_curves_to_average():
    x = np.arange(4.0)
    r = Region("1", {"r_nm": x, "g": x, "g_lo": x, "lr_nm_hi_L2": x, "lr_nm_L2": x})
    assert summarise([r])[2] == ["g", "lr_nm_L2"]


def test_groups_are_kept_apart_in_order_with_the_ungrouped_last():
    x = np.arange(3.0)
    regions = [Region("1", {"x_nm": x, "y": x}, group=NO_GROUP),
               Region("2", {"x_nm": x, "y": x + 10}, group="treated"),
               Region("3", {"x_nm": x, "y": x + 20}, group="control"),
               Region("4", {"x_nm": x, "y": x + 30}, group="treated")]
    _, _, _, groups = summarise(regions)
    assert [g.name for g in groups] == ["treated", "control", NO_GROUP]
    assert np.allclose(groups[0].mean["y"], x + 20)
    assert groups[0].ids == ["2", "4"]


# ------------------------------------------------------------ the pooling

def squares(rng, k=4, n=(300, 800, 1500, 3000)):
    """``k`` squares of equal size far apart, uniform points in each."""
    regions = []
    for i in range(k):
        x0 = i * GAP
        shape = Shape.rect(x0, 0, x0 + SIDE, SIDE)
        p = np.column_stack([rng.uniform(x0, x0 + SIDE, n[i]), rng.uniform(0, SIDE, n[i])])
        regions.append((shape, p))
    return regions


def measured(regions, **radial):
    plugin = SpatialStatistics()
    s = SpatialStatisticsSettings(ripley_k=True, radial=RadialSettings(
        **{"r_max_nm": 300, "dr_nm": 10, "null": "none", **radial}))
    out = []
    for i, (shape, p) in enumerate(regions):
        locs = Localizations({"x_nm": p[:, 0], "y_nm": p[:, 1]}, {})
        result = plugin.run(Context(locs=locs, region=shape, region_id=str(i)), s)
        out.append(Region(str(i), dict(result.arrays), dict(result.values)))
    return out


def test_the_pooled_estimator_is_g_of_the_union_of_equal_windows():
    rng = np.random.default_rng(1)
    regions = squares(rng)
    _, x, names, (group,) = summarise(measured(regions), pool=SpatialStatistics().pool)
    assert names == ["g", "K_nm2", "lr_nm"] and group.pooled == names
    # each square a file of one measurement: the counts and expectations add
    windows = [window_of_region(shape) for shape, _ in regions]
    _, (union,) = spatial_statistics([[Points(p) for _, p in regions]], windows, 300, 10)
    assert np.allclose(group.mean["g"], union.g)
    assert np.allclose(group.mean["lr_nm"], union.L_r)
    # and, for squares of equal counts, the one window that is all four, far
    # apart: the same to k n (n - 1) over k n (k n - 1), a fraction of a
    # percent (with unequal counts one density for all of them is wrong, and
    # pooling is not that)
    from smappy.plugins.spatial_statistics import Window
    regions = squares(rng, n=(1500,) * 4)
    windows = [window_of_region(shape) for shape, _ in regions]
    _, _, _, (group,) = summarise(measured(regions), pool=SpatialStatistics().pool)
    big = window_of_region(Shape.rect(0, 0, 3 * GAP + SIDE, SIDE), pixels=1200)
    inside = np.zeros_like(big.mask)
    for w in windows:
        x0, _, x1, _ = w.extent
        cols = (np.arange(big.mask.shape[1]) + 0.5) * big.px + big.x0
        inside[:, (cols > x0) & (cols < x1)] = True
    whole = Window(inside, big.x0, big.y0, big.px, big.py)
    every = np.vstack([p for _, p in regions])
    _, (one,) = spatial_statistics([[Points(every)]], [whole], 300, 10)
    assert np.nanmean(group.mean["g"] / one.g) == pytest.approx(1.0, abs=0.01)


def test_pooling_weighs_regions_by_their_pairs_and_the_mean_does_not():
    rng = np.random.default_rng(2)
    regions = squares(rng, k=2, n=(40, 4000))
    found = measured(regions)
    _, _, _, (pooled,) = summarise(found, pool=SpatialStatistics().pool)
    _, _, _, (plain,) = summarise(found, names=["g"])
    # the sparse square's g is noise; the mean takes half of it, pooling ~1 %
    assert np.nanstd(pooled.mean["g"] - 1) < 0.5 * np.nanstd(plain.mean["g"] - 1)


# ------------------------------------------------------- in a session

def keyword_session(rng):
    """Six ROIs: three of clustered points ("clustered"), three uniform
    ("random")."""
    parts = []
    for i in range(6):
        x0 = i * GAP
        if i < 3:
            centres = np.column_stack([rng.uniform(x0 + 200, x0 + SIDE - 200, 40),
                                       rng.uniform(200, SIDE - 200, 40)])
            p = (centres[:, None, :] + rng.normal(0, 25, (40, 25, 2))).reshape(-1, 2)
        else:
            p = np.column_stack([rng.uniform(x0, x0 + SIDE, 1000),
                                 rng.uniform(0, SIDE, 1000)])
        parts.append(p)
    p = np.vstack(parts)
    session = Session(Localizations({"x_nm": p[:, 0], "y_nm": p[:, 1],
                                     "frame": np.arange(len(p), dtype=np.int64)}, {}))
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(SIDE, "square")
    file_id = next(iter(project.sources))
    project.set_annotation_lists({"condition": ["clustered", "random"]})
    for i in range(6):
        roi = project.add_roi(file_id, [i * GAP + SIDE / 2, SIDE / 2])
        roi_id = roi if isinstance(roi, str) else getattr(roi, "id", roi)
        project.annotate(roi_id, "condition", "clustered" if i < 3 else "random")
    step = Step(plugin=SPATIAL, values={"radial.r_max_nm": 300, "radial.dr_nm": 10,
                                         "radial.null": "none"})
    cls = chain_class(ChainSpec(steps=[Block(steps=[step])]))
    session.run(cls(), cls.Settings())
    return session


def test_keyword_groups_are_kept_apart_in_a_session():
    session = keyword_session(np.random.default_rng(3))
    result = session.run(Curves(), CurvesSettings(group_by="condition"))
    groups = result.data["groups"]
    assert set(groups) == {"clustered", "random"}
    assert groups["clustered"]["n"] == 3 and groups["random"]["n"] == 3
    assert groups["clustered"]["pooled"] == ["g", "K_nm2", "lr_nm"]
    clustered = np.nanmax(groups["clustered"]["mean"]["lr_nm"])
    random = np.nanmax(np.abs(groups["random"]["mean"]["lr_nm"]))
    assert clustered > 30 and random < 10
    assert "clustered: 3 ROIs" in result.text and "lr_max_r_nm" in result.text
    # without a keyword list: one group of six
    one = session.run(Curves(), CurvesSettings())
    assert list(one.data["groups"]) == [NO_GROUP]
    assert one.data["groups"][NO_GROUP]["n"] == 6


def test_the_form_offers_the_keyword_lists_and_the_arrays():
    session = keyword_session(np.random.default_rng(4))
    state = Curves().describe(session.context(), CurvesSettings())
    assert ("condition", "condition") in state.choices["group_by"]
    assert ("r_nm", "r_nm") in state.choices["x"]
    assert "g_lo" not in state.hints["curves"]
