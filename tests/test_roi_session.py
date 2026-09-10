"""Analysis ROIs backed by the session: the layer's view, saved with the file."""
import numpy as np
import pytest

from smappy.io.hdf5 import save_localizations
from smappy.locs import Localizations
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
    _, stale = project.latest(roi.id)
    assert stale
