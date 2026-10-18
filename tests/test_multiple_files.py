"""Several files in one table: each file's own columns, its own frames, and
*Combine files* making them one."""
import numpy as np
import pytest

from smappy.io.formats import FileInfo
from smappy.locs import Localizations, frames_apart
from smappy.plugins import Context, Selection
from smappy.session import Session


def _table(n=200, frames=50, seed=0, **extra):
    rng = np.random.default_rng(seed)
    columns = {"x_nm": rng.random(n).astype(np.float32) * 1e3,
               "y_nm": rng.random(n).astype(np.float32) * 1e3,
               "frame": np.sort(rng.integers(0, frames, n)).astype(np.int32),
               "photons": np.full(n, 500.0, np.float32)}
    columns.update(extra)
    return Localizations(columns, {})


def _session(*tables, **info):
    s = Session()
    for k, t in enumerate(tables):
        s.add_file(t, FileInfo(f"f{k}", f"f{k}.h5", "smappy", **info), append=k > 0)
    return s


# ------------------------------------------------------- each file's columns

def test_each_file_remembers_the_columns_it_was_read_with():
    a = _table()
    b = _table(z_nm=np.zeros(200, np.float32))
    s = _session(a, b)
    assert "z_nm" not in s.files[0].columns and "z_nm" in s.files[1].columns
    assert "filenumber" not in s.files[1].columns
    assert s.files_without("z_nm") == [0]
    assert s.files_without("x_nm") == []
    # the padding is in the table, and the list says it is padding
    assert np.isnan(np.asarray(s.locs["z_nm"])[np.asarray(s.locs["filenumber"]) == 0]).all()


def test_the_columns_survive_the_file_entry_and_an_older_entry_has_none():
    info = FileInfo("a", "a.h5", "smappy", columns=["x_nm", "frame"])
    assert FileInfo.from_dict(info.to_dict()).columns == ["x_nm", "frame"]
    old = {"name": "a", "path": "a.h5", "format": "smappy"}
    assert FileInfo.from_dict(old).columns is None
    s = Session()
    s.files = [FileInfo.from_dict(old)]
    assert s.files_without("z_nm") == []          # unknown is never "missing"


# ------------------------------------------------------- frames of one file

def test_frames_apart_keeps_each_files_gaps_and_never_lets_two_files_meet():
    s = _session(_table(frames=50), _table(frames=30, seed=1))
    apart = frames_apart(s.locs)
    files = np.asarray(s.locs["filenumber"])
    frame = np.asarray(s.locs["frame"])
    for k in (0, 1):
        assert np.array_equal(np.diff(apart[files == k]), np.diff(frame[files == k]))
    assert apart[files == 1].min() - apart[files == 0].max() > 50
    one = _table()
    assert np.array_equal(frames_apart(one), np.asarray(one["frame"]))


def test_nena_never_pairs_two_files_that_share_frame_numbers():
    from smappy.plugins.precision import displacement_pairs
    # one localization in frame 0 of the first file, one 1 nm away in frame 1
    # of the second: consecutive by their numbers, two molecules in truth
    t = Localizations({"x_nm": np.array([0.0, 1.0]), "y_nm": np.array([0.0, 1.0]),
                       "frame": np.array([0, 1]), "filenumber": np.array([0, 1])}, {})
    x, y = np.asarray(t["x_nm"]), np.asarray(t["y_nm"])
    across = displacement_pairs(x, y, np.asarray(t["frame"]), 10.0, gaps=(1,))
    assert across[1].n_found == 1                   # the numbers alone pair them
    apart = displacement_pairs(x, y, frames_apart(t), 10.0, gaps=(1,))
    assert 1 not in apart or apart[1].n_found == 0


# ------------------------------------------------------------- drift per file

def _drifting_file(sign, seed):
    from test_drift import simulate
    locs, truth, true_xyz = simulate(n_points=300, n_frames=200, per_frame=150,
                                     seed=seed)
    if sign < 0:                       # the other file drifts the other way
        columns = dict(locs.columns)
        for axis, name in enumerate(("x_nm", "y_nm", "z_nm")):
            columns[name] = (true_xyz[:, axis] - truth[np.asarray(locs["frame"]), axis]
                             ).astype(np.float32)
        locs = Localizations(columns, dict(locs.metadata))
        truth = -truth
    return locs, truth, true_xyz


def _residual(corrected, true_xyz, rows):
    """How far the corrected positions are from the truth, up to the
    constant offset a drift is defined to."""
    xy = np.column_stack([np.asarray(corrected["x_nm"])[rows],
                          np.asarray(corrected["y_nm"])[rows]]) - true_xyz[:, :2]
    return np.abs(xy - xy.mean(0)).mean()


def test_rcc_corrects_each_file_by_its_own_curve_and_leaves_the_unselected_alone():
    from smappy.plugins.drift_rcc import RCCDrift
    from smappy.rcc import RCCSettings
    a, truth_a, xyz_a = _drifting_file(+1, seed=1)
    b, truth_b, xyz_b = _drifting_file(-1, seed=2)
    s = _session(a, b)
    files = np.asarray(s.locs["filenumber"])
    settings = RCCSettings(n_timepoints=10, pixelsize_nm=10.0, max_drift_nm=300.0,
                           z_pixelsize_nm=10.0, tile_nm=250.0,
                           axial_max_drift_nm=300.0, group_dx_nm=0.0, group_dt=0)
    before = _residual(s.locs, xyz_b, files == 1)
    result = RCCDrift().run(Context(session=s, locs=s.locs,
                                    selection=Selection(np.ones(len(files), bool))),
                            settings)
    assert set(result.data["drifts"]) == {"f0", "f1"}
    assert _residual(result.locs, xyz_a, files == 0) < 15.0
    assert _residual(result.locs, xyz_b, files == 1) < 15.0 < before
    # the two curves are of their own files: they drift opposite ways
    da, db = (d.drift[:, 0] - d.drift[:, 0].mean() for d in result.data["drifts"].values())
    assert np.corrcoef(da, db)[0, 1] < -0.9
    # showing one file corrects that one, and the other keeps its positions
    only = RCCDrift().run(Context(session=s, locs=s.locs,
                                  selection=Selection(files == 0)), settings)
    assert "drift" in only.data and "drifts" not in only.data
    assert np.array_equal(np.asarray(only.locs["x_nm"])[files == 1],
                          np.asarray(s.locs["x_nm"])[files == 1])
    assert _residual(only.locs, xyz_a, files == 0) < 15.0


def test_several_files_drift_curves_are_kept_and_drawn_again():
    from smappy.drift import Drift
    from smappy.plugins.drift_rcc import RCCDrift
    from smappy.plugins.drift_result import drift_result
    curves = [("f0", Drift(np.zeros((10, 3)), method="rcc")),
              ("f1", Drift(np.ones((12, 3)), method="rcc"))]
    result = drift_result(None, curves, None, lambda d: d.plot)
    plugin = RCCDrift()
    restored = plugin.restore(plugin.keep(result))
    assert list(restored.data["drifts"]) == ["f0", "f1"]
    assert len(restored.data["drifts"]["f1"]) == 12
    assert restored.plot.panels == 2


# -------------------------------------------------------------- combine files

def test_combining_continues_frames_and_time_and_keeps_ids_apart():
    from smappy.plugins.combine_files import combine
    a = _table(frames=50, track_id=np.r_[np.zeros(100), np.arange(1, 101)].astype(np.int32),
               time_s=np.linspace(0, 0.49, 200))
    b = _table(frames=30, seed=1, track_id=np.r_[np.zeros(50), np.full(150, 3)].astype(np.int32),
               time_s=np.linspace(0, 0.29, 200))
    s = _session(a, b, frame_time_ms=10.0)
    files = np.asarray(s.locs["filenumber"])
    combined, warnings = combine(s.locs, gap=5)
    assert not warnings
    frame = np.asarray(combined["frame"])
    last_a = int(np.asarray(a["frame"]).max())
    assert np.array_equal(frame[files == 0], np.asarray(a["frame"]))
    assert np.array_equal(frame[files == 1], np.asarray(b["frame"]) + last_a + 1 + 5)
    # time moves with the frames: the first file's frames (+ gap) at 10 ms
    t = np.asarray(combined["time_s"])
    assert np.allclose(t[files == 1], np.asarray(b["time_s"]) + (last_a + 1 + 5) * 0.01)
    ids = np.asarray(combined["track_id"])
    assert (ids[files == 1][:50] == 0).all()                 # "in none" stays
    assert (ids[files == 1][50:] == 100 + 3).all()            # numbered on from 100
    assert (np.asarray(combined["filenumber"]) == 0).all()
    assert len(combined.metadata["files"]) == 1
    assert combined.metadata["files"][0]["name"] == "f0 + f1"


def test_combining_drops_the_linking_and_says_what_differs():
    from smappy.plugins.combine_files import combine
    a = _table(group_id=np.arange(200, dtype=np.int64))
    b = _table(seed=1, z_nm=np.zeros(200, np.float32))
    s = Session()
    s.add_file(a, FileInfo("a", "a", "smappy", pixelsize_nm=100.0))
    s.add_file(b, FileInfo("b", "b", "smappy", pixelsize_nm=117.0), append=True)
    combined, warnings = combine(s.locs)
    assert "group_id" not in combined
    assert any("pixel size" in w and "117" in w for w in warnings)
    assert any(w.startswith("a was read without z_nm") for w in warnings)


def test_combine_files_makes_the_session_one_file_and_undo_brings_them_back():
    from smappy.plugins.combine_files import CombineFiles, CombineSettings
    s = _session(_table(frames=50), _table(frames=30, seed=1))
    s.layers[0].set_files([1])
    plugin = CombineFiles()
    result = plugin.run(Context(session=s, locs=s.locs), CombineSettings())
    s.apply(plugin, result)
    assert s.file_names() == ["f0 + f1"]
    assert s.layers[0].files is None              # shows every file again
    assert len(s.layers[0].state.tables.locs) == 400
    s.undo()
    assert s.file_names() == ["f0", "f1"]
    assert sorted(np.unique(np.asarray(s.locs["filenumber"]))) == [0, 1]


def test_one_file_is_refused_with_nothing_to_combine():
    from smappy.plugins.combine_files import combine
    s = _session(_table())
    with pytest.raises(ValueError, match="nothing to combine"):
        combine(s.locs)


# ------------------------------------------- a layer's filter over its files

def test_a_filter_restricted_to_spans_keeps_what_a_mask_over_every_row_keeps():
    from smappy.filter import LocFilter
    t = _table(n=5000, photons=np.random.default_rng(3).random(5000).astype(np.float32))
    spans = [(100, 900), (2000, 2600)]
    inside = np.zeros(5000, bool)
    for a, b in spans:
        inside[a:b] = True
    full = LocFilter(t, photons=(0.2, 0.7)).set_mask("rows", inside)
    restricted = LocFilter(t, spans=spans, photons=(0.2, 0.7))
    assert np.array_equal(restricted.mask, full.mask)
    assert np.array_equal(restricted.indices, np.flatnonzero(full.mask))
    assert len(restricted) == len(full)
    restricted.set("x_nm", 100, 600)
    full.set("x_nm", 100, 600)
    assert np.array_equal(restricted.mask, full.mask)
    assert restricted.counts()["x_nm"] == int((full._masks["x_nm"] & inside).sum())
    assert np.array_equal(restricted.mask_without("photons"), full.mask_without("photons"))
    restricted.restrict(None)                      # lifted: every row again
    assert np.array_equal(restricted.mask, full.mask_without("rows"))


def test_a_layer_showing_one_file_tests_its_bounds_on_that_files_rows_only():
    s = _session(_table(n=300), _table(n=500, seed=1), _table(n=200, seed=2))
    state = s.layers[0].state
    s.layers[0].show_grouped(False)
    s.layers[0].set_files([1])
    f = state.filter_for(state.current)
    assert f.spans == [(300, 800)]
    files = np.asarray(s.locs["filenumber"])
    s.layers[0].set_bound("photons", 0, None)
    assert np.array_equal(f.mask, files == 1)
    s.layers[0].set_files([0, 1])                  # touching files are one span
    assert state.filter_for(state.current).spans == [(0, 800)]
    s.layers[0].set_files(None)
    assert state.filter_for(state.current).spans is None


def test_files_whose_rows_are_not_blocks_are_chosen_by_a_mask():
    from smappy.layer_state import ViewState
    t = _table(n=400)
    t.columns["filenumber"] = np.tile([0, 1], 200).astype(np.int32)
    state = ViewState(t)
    state.set_files([1])
    f = state.filter_for(state.current)
    assert f.spans is None
    assert np.array_equal(f.mask, np.tile([False, True], 200))


# ------------------------------------------------- linking file by file

def _blinking(n_files=3, seed=0):
    """Files of molecules seen in a few consecutive frames each."""
    tables = []
    for k in range(n_files):
        rng = np.random.default_rng(seed + k)
        m = 400
        x, y = rng.random(m) * 5e3, rng.random(m) * 5e3
        start = rng.integers(0, 500, m)
        length = rng.integers(1, 4, m)
        mol = np.repeat(np.arange(m), length)
        frame = start[mol] + np.concatenate([np.arange(l) for l in length])
        order = np.argsort(frame, kind="stable")
        mol, frame = mol[order], frame[order]
        tables.append(Localizations({
            "x_nm": (x[mol] + rng.normal(0, 5, len(mol))).astype(np.float32),
            "y_nm": (y[mol] + rng.normal(0, 5, len(mol))).astype(np.float32),
            "frame": frame.astype(np.int32),
            "photons": np.full(len(mol), 1000.0, np.float32),
            "xy_err_nm": np.full(len(mol), 5.0, np.float32)}, {}))
    return _session(*tables)


def test_a_layer_showing_one_file_links_that_file_alone():
    s = _blinking()
    s.layers[0].set_files([1])
    s.layers[0].show_grouped(True)
    blinks = s.layers[0].locs
    tables = s.tables
    assert tables._linked == {1}
    assert set(np.unique(np.asarray(blinks["filenumber"]))) == {1}
    files = np.asarray(s.locs["filenumber"])
    gid = np.asarray(s.locs["group_id"])
    assert (gid[files != 1] == 0).all() and (gid[files == 1] > 0).all()
    # switching back and forth links nothing again
    s.layers[0].set_files([1])
    assert s.layers[0].locs is blinks


def test_files_linked_one_by_one_give_the_blinks_of_the_whole_table():
    from smappy.group import group
    s = _blinking()
    whole, _ = group(Localizations(dict(s.locs.columns), {}), s.group_settings,
                     attach_columns=False)
    s.layers[0].set_files([2])
    s.layers[0].show_grouped(True)
    s.layers[0].locs
    s.layers[0].set_files(None)                    # now every file
    blinks = s.layers[0].locs
    assert s.tables._linked == {0, 1, 2}
    assert len(blinks) == len(whole)
    for k in range(3):
        mine = np.asarray(blinks["n_in_group"])[np.asarray(blinks["filenumber"]) == k]
        theirs = np.asarray(whole["n_in_group"])[np.asarray(whole["filenumber"]) == k]
        assert np.array_equal(np.sort(mine), np.sort(theirs))
    # the ids on the localizations are the grouped table's own, all distinct
    ids = np.asarray(blinks["group_id"])
    assert len(np.unique(ids)) == len(ids)
    row = {int(g): i for i, g in enumerate(ids)}
    gid = np.asarray(s.locs["group_id"])
    at = np.array([row[int(g)] for g in gid])
    assert np.array_equal(np.asarray(blinks["filenumber"])[at], np.asarray(s.locs["filenumber"]))
    assert np.array_equal(np.asarray(blinks["n_in_group"])[at], np.asarray(s.locs["n_in_group"]))


def test_new_merge_settings_forget_the_files_linked_before():
    from dataclasses import replace
    s = _blinking()
    s.layers[0].set_files([0])
    s.layers[0].show_grouped(True)
    s.layers[0].locs
    s.set_group_settings(replace(s.group_settings, dt=3))
    assert not s.tables.is_current
    s.layers[0].locs
    assert s.tables._linked == {0}
    gid = np.asarray(s.locs["group_id"])
    assert (gid[np.asarray(s.locs["filenumber"]) != 0] == 0).all()


def test_a_worker_links_the_wanted_files_one_job_at_a_time():
    from smappy.group import group
    s = _blinking()
    s.layers[0].set_files([0, 2])
    s.layers[0].show_grouped(True)
    done = []
    while True:
        job = s.tables.link_job()
        if job is None:
            break
        locs, settings, key = job
        done.append(key[2])
        assert set(np.unique(np.asarray(locs["filenumber"]))) == {key[2]}
        grouped, index = group(locs, settings, attach_columns=False)
        assert s.tables.take_grouped(grouped, index, key)
    assert done == [0, 2]
    assert s.tables.is_current and s.tables._linked == {0, 2}
