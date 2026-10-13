"""One region type for the drawn ROI and the ROI manager's ROIs."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.regions import Region, inside_polygon
from smappy.roi_manager import ROIProject


def test_an_roi_manager_roi_is_a_region_of_the_project_shape():
    circle = Region.site([10.0, 20.0], "circle", 100.0)
    assert circle.kind == "circle" and circle.radius == 50.0
    assert circle.bounds == (-40.0, -30.0, 60.0, 70.0)
    square = Region.site([10.0, 20.0], "square", 100.0)
    assert square.kind == "rect" and square.bounds == (-40.0, -30.0, 60.0, 70.0)
    own = Region.site([0.0, 0.0], "circle", 100.0, polygon=[[0, 0], [5, 0], [0, 5]])
    assert own.kind == "polygon"                 # its own outline beats the project's


def test_the_outline_counts_as_inside_for_every_kind():
    x = np.array([0.0, 10.0, 5.0, 10.0001])
    y = np.array([0.0, 5.0, 10.0, 5.0])
    for region in (Region.rect(0, 0, 10, 10),
                   Region.polygon([[0, 0], [10, 0], [10, 10], [0, 10]])):
        assert region.mask(x, y).tolist() == [True, True, True, False]
    circle = Region.circle([0.0, 0.0], 5.0)
    assert circle.mask([5.0, 0.0, 5.0001], [0.0, -5.0, 0.0]).tolist() == [True, True, False]


def test_the_project_selects_through_the_region():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-200, 200, (2, 5000))
    project = ROIProject()
    source = project.add_source(Localizations({"x_nm": x, "y_nm": y}, {}), name="a")
    roi = project.add_roi(source.id, [0.0, 0.0])
    for shape in ("circle", "square"):
        project.set_geometry(150, shape)
        expected = np.flatnonzero(project.region(roi).mask(x, y))
        assert np.array_equal(project.indices(roi), expected)
    inside = (x ** 2 + y ** 2) <= 75 ** 2
    project.set_geometry(150, "circle")
    assert np.array_equal(project.indices(roi), np.flatnonzero(inside))


def test_rim_distance_is_to_the_outline_from_either_side():
    square = Region.square([0.0, 0.0], 10.0)
    assert square.rim_distance(0.0, 0.0) == pytest.approx(5.0)
    assert square.rim_distance(8.0, 0.0) == pytest.approx(3.0)
    assert square.rim_distance(8.0, 9.0) == pytest.approx(5.0)       # to the corner
    circle = Region.circle([0.0, 0.0], 5.0)
    assert circle.rim_distance(0.0, 1.0) == pytest.approx(4.0)
    assert circle.rim_distance(0.0, 7.0) == pytest.approx(2.0)
    triangle = Region.polygon([[0, 0], [10, 0], [0, 10]])
    assert triangle.rim_distance(1.0, 1.0) == pytest.approx(1.0)


def test_vertices_and_outline_close_the_shape():
    rect = Region.rect(0, 0, 4, 2)
    assert rect.vertices().tolist() == [[0, 0], [4, 0], [4, 2], [0, 2]]
    xs, ys = rect.outline()
    assert (xs[0], ys[0]) == (xs[-1], ys[-1]) and len(xs) == 5
    xs, ys = Region.circle([1.0, 1.0], 2.0).outline()
    assert np.allclose(np.hypot(xs - 1, ys - 1), 2.0)
    assert xs[0] == pytest.approx(xs[-1]) and ys[0] == pytest.approx(ys[-1])


def test_a_circle_survives_its_dict_and_the_others_keep_theirs():
    circle = Region.circle([3.0, 4.0], 7.0)
    back = Region.from_dict(circle.to_dict())
    assert back.kind == "circle" and back.radius == 7.0
    assert back.points.tolist() == [[3.0, 4.0]]
    assert "radius" not in Region.rect(0, 0, 1, 1).to_dict()


def test_inside_polygon_only_tests_the_bounding_box_but_answers_everywhere():
    x = np.array([-100.0, 1.0, 100.0])
    y = np.array([0.0, 1.0, 0.0])
    assert inside_polygon(x, y, [[0, 0], [2, 0], [2, 2], [0, 2]]).tolist() == \
        [False, True, False]


# --------------------------------------------------- measuring in a region

def _blobs(seed=0, n=3000, shift=0.0):
    rng = np.random.default_rng(seed)
    xy = np.vstack([rng.normal(c, 40, (n // 2, 2)) for c in ([1000, 1000], [4000, 4000])])
    xy += shift
    m = len(xy)
    return Localizations({"x_nm": xy[:, 0].astype(np.float32),
                          "y_nm": xy[:, 1].astype(np.float32),
                          "frame": np.arange(m, dtype=np.int64),
                          "photons": rng.gamma(4, 400, m).astype(np.float32),
                          "xy_err_nm": rng.uniform(5, 15, m).astype(np.float32)}, {})


def test_a_region_context_holds_the_roi_unfiltered_and_the_filter_as_its_selection():
    locs = _blobs()
    project = ROIProject()
    source = project.add_source(locs, name="a")
    project.set_filters({"photons": (1500, None)})
    roi = project.add_roi(source.id, [1000.0, 1000.0])
    ctx = project.region_context(roi)
    x, y = np.asarray(locs["x_nm"], float), np.asarray(locs["y_nm"], float)
    inside = project.region(roi).mask(x, y)
    assert len(ctx.locs) == inside.sum()               # membership: the shape alone
    bright = np.asarray(locs["photons"])[inside] >= 1500
    assert np.array_equal(ctx.selection.mask, bright)  # the filter, inside it
    assert len(ctx.selection.apply(ctx.locs)) == len(project.extract(roi))
    assert ctx.region.kind == "circle" and ctx.region.center.tolist() == [1000.0, 1000.0]
    assert ctx.session is None and ctx.rois is project


def test_a_region_context_offers_the_grouped_table_when_the_project_is_grouped():
    project = ROIProject()
    source = project.add_source(_blobs(), name="a")
    project.grouped = True
    roi = project.add_roi(source.id, [4000.0, 4000.0])
    ctx = project.region_context(roi)
    grouped, selection = ctx.table()
    assert ctx.served_grouped and len(grouped) == len(selection.mask)
    assert ctx.locs is grouped                         # what the project shows
    whole, _ = ctx.ungrouped()
    assert 0 < len(grouped) <= len(whole)              # a row per blink, no more
    ungrouped, _ = project.region_context(roi, grouping="ungrouped").table()
    assert len(ungrouped) == len(whole)


def test_in_the_session_every_layer_is_cut_to_the_roi_and_the_drawn_roi_is_not(tmp_path):
    from smappy.io.hdf5 import save_localizations
    from smappy.session import Session
    save_localizations(tmp_path / "a.hdf5", _blobs(seed=1))
    save_localizations(tmp_path / "b.hdf5", _blobs(seed=2))     # the same places
    s = Session()
    s.load(tmp_path / "a.hdf5")
    s.load(tmp_path / "b.hdf5", append=True)
    s.add_layer()
    s.layers[1].set_bound("photons", 2000.0, None)
    s.roi = Region.rect(3000, 3000, 5000, 5000)       # drawn round the other blob
    project = s.rois
    project.sync()
    first = next(iter(project.sources))
    roi = project.add_roi(first, [1000.0, 1000.0])
    ctx = project.region_context(roi)
    whole, _ = ctx.ungrouped()
    assert len(ctx.locs) < len(whole)                           # the layer's blinks
    files = np.asarray(whole["filenumber"])
    assert len(whole) > 0 and (files == files[0]).all()       # its own file only
    x = np.asarray(whole["x_nm"])
    assert (x < 2000).all()                                     # not the drawn ROI's
    one, two = ctx.layer_selection(0), ctx.layer_selection(1)
    assert len(two) < len(one) <= len(whole)
    assert (np.asarray(whole["photons"])[two.mask] >= 2000).all()


def test_npc_corners_counts_round_the_place_the_roi_was_put():
    from smappy.plugins.npc import NPCCorners
    rng = np.random.default_rng(3)
    theta = np.repeat(np.arange(8) * np.pi / 4, 30) + rng.normal(0, 0.05, 240)
    r = 55 + rng.normal(0, 4, 240)
    locs = Localizations({"x_nm": (2000 + r * np.cos(theta)).astype(np.float32),
                          "y_nm": (2000 + r * np.sin(theta)).astype(np.float32),
                          "photons": np.full(240, 2000, np.float32),
                          "xy_err_nm": np.full(240, 5, np.float32)}, {})
    project = ROIProject()
    source = project.add_source(locs, name="pore")
    # a polygon lopsided round the pore: its box's middle is not the pore's
    polygon = [[1900, 1900], [2300, 1900], [2300, 2100], [1900, 2100]]
    roi = project.add_roi(source.id, [2000.0, 2000.0], polygon=polygon)
    ctx = project.region_context(roi)
    assert ctx.region.center.tolist() == [2000.0, 2000.0]
    result = NPCCorners().run(ctx, NPCCorners.Settings())
    assert result.data["n_corners"] == 8
