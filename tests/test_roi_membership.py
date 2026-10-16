"""An ROI as a filter: a shape, an id, a frame range and a z range."""

import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.roi_manager.core import IdLookup, ROI, value_range
from test_roi_session import rois_of, set_bounds              # noqa: E402


def crossing_tracks():
    """Two tracks through the origin, one along x and one along y, ten
    frames each, so a circle round the origin holds both."""
    t = np.arange(10)
    x = np.r_[(t - 4.5) * 20, np.zeros(10)]
    y = np.r_[np.zeros(10), (t - 4.5) * 20]
    return Localizations({"x_nm": x, "y_nm": y,
                          "z_nm": np.r_[t * 10.0, -t * 10.0],
                          "frame": np.r_[t, t].astype(np.int64),
                          "photons": np.full(20, 1000.0),
                          "xy_err_nm": np.full(20, 5.0),
                          "track_id": np.r_[np.full(10, 1), np.full(10, 2)]},
                         {"units": "nm"})


def test_a_track_roi_holds_its_own_track_and_not_the_one_crossing_it():
    p, s = rois_of(crossing_tracks(), filters={})
    circle = p.add_roi(s.id, [0, 0])
    assert len(p.indices(circle)) == 20          # the shape takes in both
    track = p.add_roi(s.id, [0, 0], ids={"track_id": 2}, spatial=False)
    assert p.indices(track).tolist() == list(range(10, 20))
    assert p.region(track) is None
    both = p.add_roi(s.id, [0, 0], ids={"track_id": 1})   # shape and id
    p.set_geometry(50)                           # a circle of radius 25
    assert p.indices(both).tolist() == [4, 5]


def test_frame_and_z_ranges_narrow_an_roi_with_open_ends():
    p, s = rois_of(crossing_tracks(), filters={})
    track = p.add_roi(s.id, [0, 0], ids={"track_id": 1}, spatial=False,
                      frames=[2, 5])
    assert p.indices(track).tolist() == [2, 3, 4, 5]
    p.set_ranges([track.id], frames=[7, None])
    assert p.indices(track).tolist() == [7, 8, 9]
    p.set_ranges([track.id], frames=None, z=[None, 15])
    assert p.indices(track).tolist() == [0, 1]
    p.undo()
    assert p.rois[track.id].frames == [7, None] and p.rois[track.id].z is None


def test_a_z_range_on_a_table_without_z_cuts_nothing():
    locs = crossing_tracks()
    del locs.columns["z_nm"]
    p, s = rois_of(locs, filters={})
    roi = p.add_roi(s.id, [0, 0], z=[0, 10])
    assert len(p.indices(roi)) == 20


def test_a_layer_filter_applies_inside_an_id_roi():
    locs = crossing_tracks()
    locs.columns["photons"] = np.arange(20) * 100.0
    p, s = rois_of(locs, filters={})
    track = p.add_roi(s.id, [0, 0], ids={"track_id": 2}, spatial=False)
    set_bounds(p, {"photons": (1500, None)})
    assert p.indices(track).tolist() == list(range(15, 20))


def test_a_measurement_in_an_id_roi_sees_its_rows_and_no_shape():
    p, s = rois_of(crossing_tracks(), filters={})
    track = p.add_roi(s.id, [0, 0], ids={"track_id": 1}, spatial=False, frames=[0, 3])
    ctx = p.region_context(track)
    assert ctx.region is None
    assert np.asarray(ctx.locs["track_id"]).tolist() == [1, 1, 1, 1]
    assert np.asarray(ctx.locs["frame"]).tolist() == [0, 1, 2, 3]
    run = p.evaluate()
    assert run["records"][track.id]


def test_blinks_of_a_ranged_roi_are_those_its_localizations_belong_to():
    # one emitter on for frames 0-5, another for 10-12: two blinks
    n = 9
    frame = np.r_[np.arange(6), np.arange(10, 13)].astype(np.int64)
    locs = Localizations({"x_nm": np.r_[np.zeros(6), np.full(3, 30.0)],
                          "y_nm": np.zeros(n), "frame": frame,
                          "photons": np.full(n, 1000.0), "xy_err_nm": np.full(n, 5.0)},
                         {"units": "nm"})
    p, s = rois_of(locs, filters={}, grouped=True)
    roi = p.add_roi(s.id, [0, 0], frames=[4, 11])
    ctx = p.region_context(roi)
    assert np.asarray(ctx.ungrouped()[0]["frame"]).tolist() == [4, 5, 10, 11]
    blinks = ctx.locs
    assert len(blinks) == 2                     # both blinks, each half in range
    assert sorted(np.asarray(blinks["n_in_group"]).tolist()) == [3, 6]


def test_ranges_and_ids_are_part_of_the_signature_and_the_saved_file():
    p, s = rois_of(crossing_tracks(), filters={})
    plain = p.add_roi(s.id, [0, 0])
    assert set(p.geometry(plain)) == {"center", "shape", "size_nm", "direction"}
    assert p.to_dict()["format_version"] == 1  # readable by an older smappy
    p.evaluate()
    assert not p.stale_steps(plain.id)
    p.set_ranges([plain.id], frames=[0, 4])
    assert p.stale_steps(plain.id)
    track = p.add_roi(s.id, [0, 0], ids={"track_id": 2}, spatial=False, z=[-50, 0])
    doc = p.to_dict()
    assert doc["format_version"] == 2
    p.from_dict(doc)
    again = p.rois[track.id]
    assert (again.ids, again.spatial, again.z) == ({"track_id": 2}, False, [-50.0, 0.0])
    assert p.indices(again).tolist() == list(range(10, 16))
    doc["format_version"] = 3
    with pytest.raises(ValueError, match="newer"):
        p.from_dict(doc)


def test_an_roi_of_an_id_cannot_be_moved_or_made_without_an_id():
    p, s = rois_of(crossing_tracks(), filters={})
    track = p.add_roi(s.id, [0, 0], ids={"track_id": 2}, spatial=False)
    with pytest.raises(ValueError, match="cannot be moved"):
        p.move_roi(track.id, [10, 10])
    with pytest.raises(ValueError, match="needs an id"):
        p.add_roi(s.id, [0, 0], spatial=False)
    with pytest.raises(ValueError, match="whole number"):
        p.add_roi(s.id, [0, 0], ids={"track_id": 1.5})
    with pytest.raises(ValueError, match="low to high"):
        p.add_roi(s.id, [0, 0], frames=[5, 1])
    with pytest.raises(ValueError, match="needs an id"):
        ROI.from_dict({"file_id": s.id, "center": [0, 0], "spatial": False})


def test_value_range_and_the_id_lookup():
    assert value_range(None) is None
    assert value_range([None, None]) is None
    assert value_range([1, None]) == [1.0, None]
    lookup = IdLookup(np.array([3, 1, 3, 2, 1, 3]))
    assert lookup.rows(3).tolist() == [0, 2, 5]
    assert lookup.rows([1, 2]).tolist() == [1, 3, 4]
    assert lookup.rows([7]).tolist() == []
