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
