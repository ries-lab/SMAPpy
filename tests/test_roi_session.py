"""Analysis ROIs backed by the session: the layer's view, saved with the file."""
import json
from dataclasses import asdict

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
                          "xy_err_nm": rng.uniform(5, 15, m).astype(np.float32)}, {})


def rois_of(*tables, filters=None, grouped=False):
    """``(project, source)``: the ROI project of a headless session over
    ``tables``, a file each, and the first file's source -- how a script
    makes ROIs.  ``filters`` are the layer's bounds (None: the ones a layer
    opens with); ``grouped`` whether it merges blinks.  Other tests import
    this."""
    from smappy.io.formats import FileInfo
    session = Session()
    for k, locs in enumerate(tables):
        session.add_file(locs, FileInfo(name=f"file{k + 1}", path="", format="test"),
                         append=k > 0)
    session.show_grouped(0, grouped)
    if filters is not None:
        session.layers[0].set_ranges(filters)
    project = session.rois
    return project, project.source_of(0)


def set_bounds(project, ranges=None, grouped=None):
    """The layer the project follows given new bounds, or blinks merged or
    not, and the project told."""
    session = project.session
    if ranges is not None:
        session.layers[0].set_ranges(ranges)
    if grouped is not None:
        session.show_grouped(0, grouped)
    project.sync()


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
    s.layers[0].set_bound("xy_err_nm", None, 8.0)
    project.sync()
    assert 0 < len(project.extract(roi)) < wide

    s.layers[0].set_bound("xy_err_nm", None, 25.0)   # back to the default
    project.sync()
    found = project.find(file_id, parameters={"min_count": 5})
    assert found                                   # the blobs are found
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


def test_results_of_a_table_that_never_came_from_a_file_survive_a_save(tmp_path):
    # a simulated or freshly fitted table has no `filenumber` until it is
    # reopened, and grouping writes its columns onto the table in place:
    # neither is a change to the data, so neither may make a result stale
    s = Session(_table())
    s.show_grouped(0, True)
    project = s.rois
    file_id = next(iter(project.sources))
    assert project.find(file_id, parameters={"min_count": 5})
    project.evaluate()
    grouped = len(project.results())
    assert grouped == len(project.rois)

    out = s.save(tmp_path / "grouped.hdf5", gui_state=False)
    back = Session()
    back.load(out)
    assert "filenumber" in back.locs and "filenumber" not in s.locs
    back.show_grouped(0, True)
    assert len(back.rois.results()) == grouped

    # evaluated ungrouped, then grouped and ungrouped again before saving
    # (a new session shows its layer grouped: switch it off first)
    s = Session(_table())
    s.show_grouped(0, False)
    project = s.rois
    project.find(next(iter(project.sources)), parameters={"keep": "n_locs >= 5"})
    project.evaluate()
    ungrouped = len(project.results())
    assert ungrouped == len(project.rois)
    s.show_grouped(0, True)
    s.layers[0].locs                                  # drawn: linked
    s.show_grouped(0, False)
    assert "group_id" in s.locs
    assert len(project.results()) == ungrouped
    out = s.save(tmp_path / "ungrouped.hdf5", gui_state=False)
    back = Session()
    back.load(out)
    back.show_grouped(0, False)
    assert len(back.rois.results()) == ungrouped


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
    assert found
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
    assert project.numbers() == {b.id: 2}            # not renumbered
    c = project.add_roi(file_id, [3000, 3000])
    assert c.number == 3                              # and 1 is not given again


def test_removing_roi_3_of_5_leaves_roi_5_as_5_in_the_list_the_column_and_the_results(
        tmp_path):
    from smappy.roi_manager.pipeline import Step
    from smappy.plugins import Plugin, Result
    from smappy.plugins.region_column import RegionsToColumn

    class Count(Plugin):
        changes_locs = False
        def run(self, ctx, settings=None):
            return Result(values={"n": float(len(ctx.selection.apply(ctx.locs)))})

    session, window = _manager(tmp_path)
    project = session.rois
    file_id = next(iter(project.sources))
    project.set_geometry(300)
    centres = [[1000, 1000], [2500, 1000], [4000, 1000], [5500, 1000], [7000, 1000]]
    rois = [project.add_roi(file_id, c) for c in centres]
    fifth = rois[4]

    def listed():
        model = window.rows
        return {model.ids[i]: model.data(model.index(i, 0)) for i in range(model.rowCount())}

    window.refresh()
    window.select(rois[2].id)
    window._remove()                                  # ROI 3, undoably
    assert rois[2].id not in project.rois
    assert listed()[fifth.id] == "5"
    assert fifth.number == 5

    project.evaluate(steps=[Step(label="Count", path="Test/Count", plugin=Count(),
                                 settings=None)])
    assert sorted(r["roi_id"] for r in project.results()) == [1, 2, 4, 5]
    session.run(RegionsToColumn(), RegionsToColumn.Settings())
    inside = project.region(fifth).mask(np.asarray(session.locs["x_nm"]),
                                        np.asarray(session.locs["y_nm"]))
    assert inside.any()
    assert (np.asarray(session.locs["roi_id"])[inside] == 5).all()
    assert 3 not in set(np.asarray(session.locs["roi_id"]).tolist())

    # undo brings ROI 3 back as 3, redo takes it away again, and the next
    # ROI made is 6 -- a number is never given twice
    window.undo()
    assert project.numbers()[rois[2].id] == 3
    window._fill_table()
    assert listed()[rois[2].id] == "3"
    window.redo()
    assert rois[2].id not in project.rois
    assert project.add_roi(file_id, [1000, 4000]).number == 6


def test_roi_numbers_travel_with_the_file_and_old_files_get_them_in_order(tmp_path):
    path = tmp_path / "locs.hdf5"
    save_localizations(path, _table(500))
    s = Session()
    s.load(path)
    project = s.rois
    file_id = next(iter(project.sources))
    a, b, c = (project.add_roi(file_id, [x, 1000]) for x in (1000, 2000, 3000))
    project.rois.pop(a.id)
    out = s.save(tmp_path / "numbered.hdf5")
    back = Session()
    back.load(out)
    assert sorted(back.rois.numbers().values()) == [2, 3]
    assert back.rois.add_roi(next(iter(back.rois.sources)), [4000, 1000]).number == 4

    # a project saved before ROIs had numbers: numbered in the order made
    doc = project.to_dict()
    for saved in doc["rois"]:
        saved.pop("number")
    doc.pop("next_number")
    project.from_dict(doc)
    assert [r.number for r in project.rois.values()] == [1, 2]
    assert project.next_number == 3


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


def test_a_double_click_closes_a_polygon_drawn_with_the_mouse(tmp_path):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    session, window = _manager(tmp_path)
    file_id = next(iter(session.rois.sources))
    roi = session.rois.add_roi(file_id, [2000, 2000])
    window.refresh()
    window.select(roi.id)
    window.show()
    QTest.qWaitForWindowExposed(window)
    window._start_drawing("polygon")
    viewport = window.roi_pane.graphics.viewport()
    centre = viewport.rect().center()
    for dx, dy in ((-40, -40), (40, -40)):
        QTest.mouseClick(viewport, Qt.LeftButton, Qt.NoModifier, centre + QPoint(dx, dy))
    # a double-click is a click and then the double-click event, as the
    # system sends it (QTest's mouseDClick sends the second alone)
    last = centre + QPoint(40, 40)
    QTest.mouseClick(viewport, Qt.LeftButton, Qt.NoModifier, last)
    QTest.mouseDClick(viewport, Qt.LeftButton, Qt.NoModifier, last)
    assert window.drawing is None
    assert roi.polygon is not None and len(roi.polygon) == 3     # no vertex twice
    window.close()


def test_a_polygons_corner_is_dragged_in_one_undo_step(tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    session, window = _manager(tmp_path)
    file_id = next(iter(session.rois.sources))
    roi = session.rois.add_roi(file_id, [2000, 2000])
    window.refresh()
    window.select(roi.id)
    window.show()
    QTest.qWaitForWindowExposed(window)
    window._start_drawing("polygon")
    square = [(1900, 1900), (2100, 1900), (2100, 2100), (1900, 2100)]
    for x, y in square + square[:1]:
        window._roi_clicked(float(x), float(y))
    pane = window.roi_pane
    assert len(pane.shapes["corners"].getData()[0]) == 4         # marked to grab

    def at(x, y):          # the widget's pixel over a data point
        from PySide6.QtCore import QPointF
        scene = pane.content.mapToScene(QPointF(x, y))
        return pane.graphics.mapFromScene(scene)

    viewport = pane.graphics.viewport()
    start, end = at(2100, 2100), at(2200, 2150)
    QTest.mousePress(viewport, Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(viewport, end)
    assert roi.polygon[2] == [2100.0, 2100.0]                    # only at the release
    QTest.mouseRelease(viewport, Qt.LeftButton, Qt.NoModifier, end)
    p = pane.fov.pixelsize
    assert roi.polygon[2] == [pytest.approx(2200, abs=p), pytest.approx(2150, abs=p)]
    assert roi.polygon[0] == [1900.0, 1900.0]
    assert np.allclose(roi.center, np.mean(roi.polygon, axis=0))
    window.undo()
    assert roi_now(session, roi).polygon[2] == [2100.0, 2100.0]
    window.close()


def test_a_lines_end_is_dragged_and_a_click_on_it_only_chooses_it(tmp_path):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest
    session, window = _manager(tmp_path)
    project = session.rois
    project.set_line_names(["line", "axis"])
    roi = project.add_roi(next(iter(project.sources)), [2000, 2000])
    project.set_line(roi.id, "axis", [[1900.0, 2000.0], [2100.0, 2000.0]])
    window.refresh()
    window.select(roi.id)
    window.show()
    QTest.qWaitForWindowExposed(window)
    pane = window.roi_pane
    viewport = pane.graphics.viewport()

    def at(x, y):
        return pane.graphics.mapFromScene(pane.content.mapToScene(QPointF(x, y)))

    steps = len(project.history._undo)
    QTest.mouseClick(viewport, Qt.LeftButton, Qt.NoModifier, at(2100, 2000))
    assert window.line_name == "axis"                            # chosen, not moved
    assert roi.lines["axis"] == [[1900.0, 2000.0], [2100.0, 2000.0]]
    assert len(project.history._undo) == steps                          # and no undo step
    QTest.mousePress(viewport, Qt.LeftButton, Qt.NoModifier, at(2100, 2000))
    QTest.mouseMove(viewport, at(2100, 2150))
    QTest.mouseRelease(viewport, Qt.LeftButton, Qt.NoModifier, at(2100, 2150))
    p = pane.fov.pixelsize
    end = project.rois[roi.id].lines["axis"]
    assert end[0] == [1900.0, 2000.0]
    assert end[1] == [pytest.approx(2100, abs=p), pytest.approx(2150, abs=p)]
    window.close()


def test_a_click_centres_the_stored_roi_and_its_line_stays_on_the_data(tmp_path):
    session, window = _manager(tmp_path)
    project = session.rois
    roi = project.add_roi(next(iter(project.sources)), [2000, 2000])
    project.set_line(roi.id, "line", [[1950.0, 2000.0], [2050.0, 2000.0]])
    window.refresh()
    window.select(roi.id)
    window._roi_clicked(2030.0, 2040.0)
    moved = project.rois[roi.id]
    assert moved.center == [2030.0, 2040.0]
    assert moved.lines["line"] == [[1950.0, 2000.0], [2050.0, 2000.0]]   # as a draft's
    window.undo()
    assert project.rois[roi.id].center == [2000.0, 2000.0]
    window.close()


def roi_now(session, roi):
    return session.rois.rois[roi.id]


def test_a_line_takes_two_clicks_and_is_saved_with_the_file(tmp_path):
    session, window = _manager(tmp_path)
    file_id = next(iter(session.rois.sources))
    roi = session.rois.add_roi(file_id, [2000, 2000])
    window.refresh()
    window.select(roi.id)

    window._start_drawing("line")
    window._roi_clicked(1900.0, 2000.0)
    assert roi.lines == {}                          # one click is not a line
    window._roi_clicked(2100.0, 2050.0)
    assert roi.lines == {"line": [[1900.0, 2000.0], [2100.0, 2050.0]]}
    assert window.drawing is None

    out = session.save(tmp_path / "with_direction.hdf5")
    back = Session()
    back.load(out)
    restored = next(iter(back.rois.rois.values()))
    assert restored.lines == roi.lines

    window._clear_line()
    assert roi.lines == {}


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


def test_roi_results_are_columns_beside_the_table_and_can_be_left_out(tmp_path):
    import h5py
    import json
    s = Session(_table())
    s.show_grouped(0, False)
    project = s.rois
    project.find(next(iter(project.sources)), parameters={"keep": "n_locs >= 5"})
    project.evaluate()
    n = len(project.results())
    out = s.save(tmp_path / "with.hdf5", gui_state=False)
    with h5py.File(out, "r") as f:
        attribute = json.loads(f.attrs["metadata"])["roi_project"]
        assert "tables" not in attribute and attribute["rois"]
        assert "regions/0/values/n_locs" in f
    back = Session()
    back.load(out)
    back.show_grouped(0, False)
    assert len(back.rois.results()) == n

    out = s.save(tmp_path / "without.hdf5", gui_state=False, region_results=False)
    with h5py.File(out, "r") as f:
        assert "regions" not in f
    back = Session()
    back.load(out)
    back.show_grouped(0, False)
    # what the segmenter measured goes with the ROIs; the evaluations do not
    assert len(back.rois.rois) == len(project.rois)
    assert not any("photons_mean" in row for row in back.rois.results())


def _drag(window, path):
    """A press, moves along ``path`` and a release, in data nm, on the ROI image."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest
    pane = window.roi_pane
    viewport = pane.graphics.viewport()

    def at(x, y):
        return pane.graphics.mapFromScene(pane.content.mapToScene(QPointF(x, y)))

    QTest.mousePress(viewport, Qt.LeftButton, Qt.NoModifier, at(*path[0]))
    for p in path[1:]:
        QTest.mouseMove(viewport, at(*p))
    QTest.mouseRelease(viewport, Qt.LeftButton, Qt.NoModifier, at(*path[-1]))


def _shown(tmp_path):
    from PySide6.QtTest import QTest
    session, window = _manager(tmp_path)
    roi = session.rois.add_roi(next(iter(session.rois.sources)), [2000, 2000])
    window.refresh()
    window.select(roi.id)
    window.show()
    QTest.qWaitForWindowExposed(window)
    return session, window, roi


def test_a_freehand_outline_is_thinned_to_its_corners_and_becomes_the_polygon(tmp_path):
    session, window, roi = _shown(tmp_path)
    window.set_shape_tool("freehand")
    assert window.polygon_button.text() == "Freehand"
    window.polygon_button.click()
    assert window.drawing == "freehand"
    # a square traced with a point every 10 nm
    side = np.linspace(1900, 2100, 21)
    path = ([(x, 1900) for x in side] + [(2100, y) for y in side[1:]]
            + [(x, 2100) for x in side[::-1][1:]] + [(1900, y) for y in side[::-1][1:]])
    _drag(window, path)
    assert window.drawing is None
    p = window.roi_pane.fov.pixelsize
    corners = np.asarray(roi_now(session, roi).polygon)
    assert 4 <= len(corners) <= 6                       # a square's corners, not 80 points
    assert corners.min(axis=0) == pytest.approx([1900, 1900], abs=2 * p)
    assert corners.max(axis=0) == pytest.approx([2100, 2100], abs=2 * p)
    window.close()


def test_an_ellipse_is_dragged_as_its_box_and_its_handle_stretches_and_turns_it(tmp_path):
    session, window, roi = _shown(tmp_path)
    window.set_shape_tool("ellipse")
    window.polygon_button.click()
    _drag(window, [(1900, 1950), (2000, 2000), (2100, 2050)])
    p = window.roi_pane.fov.pixelsize
    drawn = roi_now(session, roi)
    assert drawn.polygon is None
    assert drawn.ellipse["center"] == [pytest.approx(2000, abs=p), pytest.approx(2000, abs=p)]
    assert drawn.ellipse["axes"] == [pytest.approx(100, abs=p), pytest.approx(50, abs=p)]
    assert drawn.ellipse["tilt"] == 0
    assert np.allclose(drawn.center, drawn.ellipse["center"])
    held = session.rois.extract(drawn)
    u = (held["x_nm"] - 2000) / 100
    v = (held["y_nm"] - 2000) / 50
    assert len(held) and (u ** 2 + v ** 2 <= 1.01).all()     # exactly the ellipse
    # the end of the long axis dragged up and further out: longer, turned
    c = np.asarray(drawn.ellipse["center"])
    _drag(window, [tuple(c + [drawn.ellipse["axes"][0], 0]), (2000 + 150, 2000 + 150)])
    turned = roi_now(session, roi).ellipse
    assert turned["tilt"] == pytest.approx(45, abs=1)
    assert turned["axes"][0] == pytest.approx(np.hypot(150, 150), abs=2 * p)
    assert turned["axes"][1] == drawn.ellipse["axes"][1]
    window.undo()
    assert roi_now(session, roi).ellipse["tilt"] == 0
    window._clear_shape()
    assert roi_now(session, roi).ellipse is None
    window.close()


def test_an_ellipse_is_saved_with_the_rois_and_read_back(tmp_path):
    from smappy.roi_manager.core import ROI
    session, _ = _manager(tmp_path)
    project = session.rois
    roi = project.add_roi(next(iter(project.sources)), [0, 0],
                          ellipse={"center": [0, 0], "axes": [80, 40], "tilt": 30})
    back = ROI.from_dict(json.loads(json.dumps(asdict(roi))))
    assert back.ellipse == {"center": [0.0, 0.0], "axes": [80.0, 40.0], "tilt": 30.0}
    assert str(project.region(back)) == "ellipse ROI 160 x 80"
    assert "ellipse" in project.geometry(back)


def test_the_shape_menu_chooses_and_clears_and_a_right_click_removes_a_corner(tmp_path):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest
    session, window, roi = _shown(tmp_path)
    window._fill_shape_menu()
    assert [a.text() for a in window.shape_menu.actions()] == [
        "Polygon", "Freehand", "Ellipse", "", "Clear shape"]
    window._start_drawing("polygon")
    pentagon = [(2000 + 100 * np.cos(a), 2000 + 100 * np.sin(a))
                for a in np.linspace(0, 2 * np.pi, 6)[:-1]]
    for x, y in pentagon + pentagon[:1]:
        window._roi_clicked(float(x), float(y))
    assert len(roi_now(session, roi).polygon) == 5
    pane = window.roi_pane

    def at(x, y):
        return pane.graphics.mapFromScene(pane.content.mapToScene(QPointF(x, y)))

    viewport = pane.graphics.viewport()
    QTest.mouseClick(viewport, Qt.RightButton, Qt.NoModifier, at(*pentagon[2]))
    left = roi_now(session, roi).polygon
    assert len(left) == 4
    assert not any(np.allclose(p, pentagon[2]) for p in left)
    QTest.mouseClick(viewport, Qt.RightButton, Qt.NoModifier, at(2000, 2000))
    assert len(roi_now(session, roi).polygon) == 4               # not on a corner
    window._fill_shape_menu()
    next(a for a in window.shape_menu.actions() if a.text() == "Ellipse").trigger()
    assert window.polygon_button.text() == "Ellipse"
    next(a for a in window.shape_menu.actions() if a.text() == "Clear shape").trigger()
    assert roi_now(session, roi).polygon is None
    window.close()


def test_the_geometry_and_the_drawn_region_are_the_managers(tmp_path):
    session, window, roi = _shown(tmp_path)
    project = session.rois
    window.shape_box.setCurrentText("square")
    window.size_box.setValue(250)
    window.view_box.setValue(900)
    assert (project.shape, project.size_nm, project.preview_nm) == ("square", 250, 900)
    window.from_region.click()                          # nothing drawn: said so
    assert "draw a region" in window.status.currentMessage()
    session.set_roi(Region.rect(1500, 1500, 1700, 1600))
    window.from_region.click()
    added = project.rois[window.active]
    assert added.id != roi.id and len(added.polygon) == 4
    assert np.allclose(added.center, [1600, 1550])
    window.close()


def test_the_roi_image_draws_what_a_step_drew_over_the_roi(tmp_path):
    from smappy.plugins.circle_fit import CircleFit
    from smappy.roi_manager.pipeline import Step
    session, window, roi = _shown(tmp_path)
    project = session.rois
    window.redraw()
    assert window.roi_pane.shapes["overlay"].getData()[0] is None or \
        len(window.roi_pane.shapes["overlay"].getData()[0]) == 0
    steps = [Step(label="Circle fit", path="Analysis/Measure/Circle fit",
                  plugin=CircleFit(), settings=CircleFit.Settings())]
    project.evaluate(steps=steps)
    window.redraw()
    xs, ys = window.roi_pane.shapes["overlay"].getData()
    assert xs is not None and np.isfinite(xs).sum() > 10        # a circle, drawn
    entry = project.tables["Circle fit"].entry(roi.id)
    (circle,) = entry["overlays"]
    centre, r = np.asarray(circle["points"][0]), circle["radius"]
    drawn = np.column_stack((xs, ys))[np.isfinite(xs)]
    assert np.allclose(np.hypot(*(drawn - centre).T), r)
    window.close()


def test_dragging_inside_a_polygon_moves_it_and_a_click_inside_only_centres(tmp_path):
    session, window, roi = _shown(tmp_path)
    project = session.rois
    square = [[1900.0, 1900.0], [2100.0, 1900.0], [2100.0, 2100.0], [1900.0, 2100.0]]
    window._start_drawing("polygon")
    for x, y in square + square[:1]:
        window._roi_clicked(x, y)
    project.set_line(roi.id, "line", [[1950.0, 2000.0], [2050.0, 2000.0]])
    window.redraw()
    _drag(window, [(2000, 2000), (2030, 2010), (2060, 2020)])
    moved = roi_now(session, roi)
    p = window.roi_pane.fov.pixelsize
    assert np.allclose(moved.polygon, np.asarray(square) + [60, 20], atol=p)
    assert np.allclose(moved.center, np.mean(moved.polygon, axis=0))
    assert moved.lines["line"] == [[1950.0, 2000.0], [2050.0, 2000.0]]   # on the structure
    window.undo()
    assert roi_now(session, roi).polygon == square
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest
    pane = window.roi_pane
    # 40 nm off the line: within RIM_PIXELS of it the click chooses the line, and a pixel is 2.4 nm
    # when an earlier test in the worker left the window smaller
    at = pane.graphics.mapFromScene(pane.content.mapToScene(QPointF(2010.0, 1960.0)))
    QTest.mouseClick(pane.graphics.viewport(), Qt.LeftButton, Qt.NoModifier, at)
    clicked = roi_now(session, roi)
    assert clicked.polygon == square                        # the outline stays
    assert np.allclose(clicked.center, [2010, 1960], atol=p)
    window.close()
