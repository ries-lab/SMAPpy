"""Analysis ROIs backed by the session: the layer's view, saved with the file."""
import numpy as np
import pytest

from smappy.io.hdf5 import save_localizations
from smappy.locs import Localizations
from smappy.regions import Region
from smappy.session import Session


def _table(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    # two blobs and a background, so the finder has something to find
    blob = np.concatenate([rng.normal(c, 40, (n // 4, 2)) for c in ([2000, 2000], [6000, 6000])])
    back = rng.uniform(0, 8000, (n // 2, 2))
    xy = np.vstack([blob, back])
    m = len(xy)
    return Localizations({"x_nm": xy[:, 0].astype(np.float32), "y_nm": xy[:, 1].astype(np.float32),
                          "frame": np.arange(m, dtype=np.int64),
                          "photons": rng.gamma(4, 400, m).astype(np.float32),
                          "loc_precision_nm": rng.uniform(5, 15, m).astype(np.float32)}, {})


def test_rois_follow_the_layer_and_survive_a_save(tmp_path):
    path = tmp_path / "locs.hdf5"
    save_localizations(path, _table())
    s = Session()
    s.load(path)
    project = s.rois
    assert [x.name for x in project.sources.values()] == ["locs.hdf5"]
    file_id = next(iter(project.sources))

    # the ROI sees what the layer shows: tighten the filter, fewer localizations
    roi = project.add_roi(file_id, [2000, 2000])
    wide = len(project.extract(roi))
    s.layers[0].set_bound("loc_precision_nm", None, 8.0)
    project.sync()
    assert 0 < len(project.extract(roi)) < wide

    s.layers[0].set_bound("loc_precision_nm", None, 25.0)   # back to the default
    project.sync()
    found = project.find(file_id, parameters={"min_count": 5})
    assert found                                   # the blobs are found
    for r in project.rois.values():
        r.reviewed = True
    project.evaluate()
    before = len(project.results())
    assert before == len(project.rois)

    out = s.save(tmp_path / "with_rois.hdf5")
    back = Session()
    back.load(out)
    restored = back.rois
    assert len(restored.rois) == len(project.rois)
    assert len(restored.results()) == before      # same filter, so still current
    assert restored.size_nm == project.size_nm and restored.shape == project.shape
    assert len(restored.extract(next(iter(restored.rois.values())))) > 0


def test_geometry_and_review_round_trip(tmp_path):
    path = tmp_path / "locs.hdf5"
    save_localizations(path, _table(1000))
    s = Session()
    s.load(path)
    project = s.rois
    file_id = next(iter(project.sources))
    project.set_geometry(450, "square")
    a = project.add_roi(file_id, [2000, 2000])
    b = project.add_roi(file_id, [6000, 6000])
    b.use = False
    project.navigation["roi"] = a.id
    out = s.save(tmp_path / "saved.hdf5")

    back = Session()
    back.load(out)
    restored = back.rois
    assert restored.shape == "square" and restored.size_nm == 450
    assert restored.navigation["roi"] == a.id
    assert {r.id: r.use for r in restored.rois.values()} == {a.id: True, b.id: False}


def test_a_changed_table_makes_results_outdated(tmp_path):
    """A drift correction or a refit must not leave stale numbers looking current."""
    path = tmp_path / "locs.hdf5"
    save_localizations(path, _table(1000))
    s = Session()
    s.load(path)
    project = s.rois
    file_id = next(iter(project.sources))
    roi = project.add_roi(file_id, [2000, 2000])
    project.evaluate()
    assert len(project.results()) == 1

    moved = dict(s.locs.columns)
    moved["x_nm"] = moved["x_nm"] + np.float32(7.0)      # as a drift correction would
    s.set_locs(Localizations(moved, dict(s.locs.metadata)))
    project.sync()
    assert project.results() == []                        # outdated, not silently reused
    assert project.stale_steps(roi.id)


def test_new_rois_count_without_a_review_step(tmp_path):
    """Review is not exposed in the GUI, so an ROI counts as soon as it exists."""
    path = tmp_path / "locs.hdf5"
    save_localizations(path, _table(1000))
    s = Session()
    s.load(path)
    project = s.rois
    file_id = next(iter(project.sources))
    found = project.find(file_id, parameters={"min_count": 5})
    manual = project.add_roi(file_id, [2000, 2000])
    assert found and all(r.reviewed for r in project.rois.values())
    project.evaluate()
    assert len(project.results()) == len(found) + 1
    manual.use = False
    assert len(project.results()) == len(found)      # `use` is what excludes


def test_numbering_is_stable_and_names_the_file(tmp_path):
    path = tmp_path / "locs.hdf5"
    save_localizations(path, _table(500))
    s = Session()
    s.load(path)
    project = s.rois
    file_id = next(iter(project.sources))
    a = project.add_roi(file_id, [1000, 1000])
    b = project.add_roi(file_id, [2000, 2000])
    assert project.numbers() == {a.id: 1, b.id: 2}
    assert project.file_number(file_id) == 1
    project.rois.pop(a.id)
    assert project.numbers() == {b.id: 1}


def _manager(tmp_path, qtbot=None):
    """A manager window over a small file, for the drawing tests."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    path = tmp_path / "locs.hdf5"
    save_localizations(path, _table(2000))
    s = Session()
    s.load(path)
    from smappy.gui.roi_window import ROIManagerWindow
    window = ROIManagerWindow(s)
    window.resize(900, 700)
    return s, window


def test_a_drawn_polygon_becomes_the_roi_and_selects_by_it(tmp_path):
    session, window = _manager(tmp_path)
    file_id = next(iter(session.rois.sources))
    roi = session.rois.add_roi(file_id, [2000, 2000])
    window.refresh()
    window.select(roi.id)

    window._start_drawing("polygon")
    square = [(1800, 1800), (2200, 1800), (2200, 2200), (1800, 2200)]
    for x, y in square:
        window._roi_clicked(float(x), float(y))
    window._roi_clicked(*square[0])                 # closing on the first vertex
    assert window.drawing is None
    assert roi.polygon is not None and len(roi.polygon) == 4
    assert np.allclose(roi.center, [2000, 2000])    # the centroid

    # the model now selects by the polygon, not by the global circle
    inside = session.rois.extract(roi)
    assert len(inside) > 0
    assert inside["x_nm"].min() >= 1800 and inside["x_nm"].max() <= 2200
    assert "polygon" in session.rois.geometry(roi)

    window._clear_shape()
    assert roi.polygon is None


def test_direction_takes_two_clicks_and_is_saved_with_the_file(tmp_path):
    session, window = _manager(tmp_path)
    file_id = next(iter(session.rois.sources))
    roi = session.rois.add_roi(file_id, [2000, 2000])
    window.refresh()
    window.select(roi.id)

    window._start_drawing("direction")
    window._roi_clicked(1900.0, 2000.0)
    assert roi.direction is None                    # one click is not a direction
    window._roi_clicked(2100.0, 2050.0)
    assert roi.direction == [[1900.0, 2000.0], [2100.0, 2050.0]]
    assert window.drawing is None

    out = session.save(tmp_path / "with_direction.hdf5")
    back = Session()
    back.load(out)
    restored = next(iter(back.rois.rois.values()))
    assert restored.direction == roi.direction

    window._clear_line()
    assert roi.direction is None


def test_escape_abandons_a_half_drawn_polygon(tmp_path):
    session, window = _manager(tmp_path)
    file_id = next(iter(session.rois.sources))
    roi = session.rois.add_roi(file_id, [2000, 2000])
    window.refresh()
    window.select(roi.id)
    window._start_drawing("polygon")
    window._roi_clicked(1900.0, 1900.0)
    window._roi_clicked(2100.0, 1900.0)
    window._cancel_drawing()
    assert window.drawing is None and window._points == [] and roi.polygon is None


def test_a_drafted_roi_keeps_the_shape_it_was_drawn_with(tmp_path):
    session, window = _manager(tmp_path)
    window.refresh()
    window._zoom_clicked(3000.0, 3000.0)            # a draft
    assert window.draft is not None
    window._start_drawing("polygon")
    for x, y in ((2900, 2900), (3100, 2900), (3000, 3150)):
        window._roi_clicked(float(x), float(y))
    window._roi_clicked(2900.0, 2900.0)
    assert window.draft_polygon is not None and window.draft is not None
    before = len(session.rois.rois)
    window._add()
    assert len(session.rois.rois) == before + 1
    stored = list(session.rois.rois.values())[-1]
    assert stored.polygon is not None and len(stored.polygon) == 3
    assert window.draft is None and window.draft_polygon is None


# --------------------------------------------------------------------- tiles
def test_tiles_cover_the_file_and_are_walked_in_reading_order(tmp_path):
    """A grid is the point of the feature: every part of the data comes up
    once, in a fixed order, rather than whichever bright spot the eye found."""
    session, window = _manager(tmp_path)
    project = session.rois
    file_id = next(iter(project.sources))
    x0, y0, x1, y1 = project.state(file_id).index.bounds

    project.set_tiles(2000.0)
    tiles = project.tiles(file_id)
    nx = int(np.ceil((x1 - x0) / 2000.0))
    ny = int(np.ceil((y1 - y0) / 2000.0))
    assert len(tiles) == nx * ny
    assert tiles[0][:2] == (x0, y0)                      # from the corner
    assert tiles[1][0] > tiles[0][0] and tiles[1][1] == tiles[0][1]   # along a row
    assert tiles[nx][1] > tiles[0][1]                    # then the next row
    assert tiles[-1][2] >= x1 and tiles[-1][3] >= y1     # and they cover it all

    project.set_tiles(0.0)
    assert project.tiles(file_id) == []


def test_the_arrow_keys_walk_the_tiles_and_move_the_zoom(tmp_path):
    session, window = _manager(tmp_path)
    window.tile_size.setValue(2000.0)
    window.tiles_button.setChecked(True)
    window._on_tiles(True)

    tiles = session.rois.tiles(window.current_file())
    assert window.tile == 0
    assert window.zoom_pane.width_nm == pytest.approx(2000.0)
    x0, y0, x1, y1 = tiles[0]
    assert window.zoom_center == pytest.approx([(x0 + x1) / 2, (y0 + y1) / 2])

    window._step_tile(1)
    assert window.tile == 1
    window._step_tile(-1)
    assert window.tile == 0
    window._step_tile(-1)                       # wraps at the start of the file
    assert window.tile == len(tiles) - 1


def test_a_click_on_a_tile_edge_goes_there_and_one_inside_still_centres(tmp_path):
    session, window = _manager(tmp_path)
    window.tile_size.setValue(2000.0)
    window._on_tiles(True)
    window.redraw()                             # the file pane needs its fov
    tiles = session.rois.tiles(window.current_file())
    x0, y0, x1, y1 = tiles[2]

    window._file_clicked(float(x0), float((y0 + y1) / 2))     # on the left edge
    assert window.tile == 2

    inside = ((x0 + x1) / 2, (y0 + y1) / 2 + (y1 - y0) / 4)
    window._file_clicked(*map(float, inside))
    assert window.tile == 2                     # unchanged: a free look, not a tile
    assert window.zoom_center == pytest.approx(list(inside))


def test_the_tile_grid_travels_with_the_file(tmp_path):
    path = tmp_path / "locs.hdf5"
    save_localizations(path, _table())
    session = Session()
    session.load(path)
    session.rois.set_tiles(1500.0)
    session.save(path)

    reopened = Session()
    reopened.load(path)
    assert reopened.rois.tile_nm == 1500.0


def test_dragging_a_line_roi_wider_takes_the_slab_with_it():
    """`from ROI` is for taking an ROI the slab is *not* following.  While it
    is following, the width the line is dragged to is the box's second axis."""
    session = Session(_table())
    session.set_roi(Region.line((1000, 1000), (3000, 2000), 250.0))
    assert session.slab_follows_roi
    assert session.slab.size[1] == pytest.approx(250.0)

    session.roi_edited(Region.line((1000, 1000), (3000, 2000), 700.0))
    assert session.slab.size[1] == pytest.approx(700.0)

    session.set_slab(session.slab, follow_roi=False)     # a dragged face stops it
    session.roi_edited(Region.line((1000, 1000), (3000, 2000), 100.0))
    assert session.slab.size[1] == pytest.approx(700.0)
