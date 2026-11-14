"""A SMAP `_sml.mat` of several files: the positions of one experiment
saved together, and SMAP's averaged sites as a file of their own.

MATLAB writes one file's ``saveloc.file`` as a struct and several as a
struct array -- each field a column of references, one per file -- which
`write_sml` lays out as ``save -v7.3`` does, and `write_sml_v5` as scipy
writes a v5 file.  A real file is read when ``SMAPPY_TEST_SML_FILES`` names
one of several files.
"""
import os

import h5py
import numpy as np
import pytest

from smappy.io.formats import load
from smappy.session import Session

CAMERA = {"cam_pixelsize_um": [0.127, 0.127], "conversion": 5.0, "offset": 398.6,
          "emgain": 200.0, "EMon": 1, "roi": [137.0, 286.0, 223.0, 208.0],
          "exposure": 25.0, "timediff": 35.31}
AVERAGE = {"cam_pixelsize_um": [0.1, 0.1], "conversion": 1.0, "offset": 0.0,
           "roi": [0.0, 0.0, 512.0, 512.0]}


def _chars(text):
    return np.frombuffer(text.encode(), np.uint8).astype(np.uint16)[:, None]


def _rows(numbers, seed=0):
    """Localizations of SMAP's files ``numbers`` (from 1), in SMAP's fields."""
    rng = np.random.default_rng(seed)
    numbers = np.asarray(numbers)
    n = len(numbers)
    return {"xnm": rng.uniform(0, 5000, n), "ynm": rng.uniform(0, 5000, n),
            "frame": np.arange(1, n + 1, dtype=float), "phot": rng.uniform(100, 1000, n),
            "filenumber": numbers.astype(np.uint16)}


def write_sml(path, files, rows, sites=()):
    """``files`` a list of (name, info): one a struct, several a struct array;
    ``sites`` SMAP's ROI manager sites (`test_sml_sites._site`)."""
    with h5py.File(path, "w") as f:
        if sites:
            from test_sml_sites import _Writer
            f.create_group("saveloc/siteexplorer")
            writer = _Writer(f)
            writer.objects("saveloc/siteexplorer/sites", "SEsites", list(sites))
            refs = writer.refs
        else:
            refs = f.create_group("#refs#")
        loc = f.create_group("saveloc/loc")
        for k, v in rows.items():
            loc[k] = np.asarray(v)[None]
        file = f.create_group("saveloc/file")
        if len(files) == 1:
            (name, info), = files
            file["name"] = _chars(name)
            for k, v in info.items():
                file[f"info/{k}"] = np.atleast_2d(v)
            return path
        names = file.create_dataset("name", (len(files), 1), dtype=h5py.ref_dtype)
        infos = file.create_dataset("info", (len(files), 1), dtype=h5py.ref_dtype)
        for i, (name, info) in enumerate(files):
            names[i, 0] = refs.create_dataset(f"name{i}", data=_chars(name)).ref
            group = refs.create_group(f"info{i}")
            for k, v in info.items():
                group[k] = np.atleast_2d(v)
            infos[i, 0] = group.ref
    return path


def write_sml_v5(path, files, rows):
    import scipy.io as sio
    entries = np.empty(len(files), dtype=[("name", object), ("info", object)])
    for i, (name, info) in enumerate(files):
        entries[i] = (name, info)
    sio.savemat(path, {"saveloc": {"loc": {k: np.asarray(v, float) for k, v in rows.items()},
                                   "file": entries}})
    return path


@pytest.mark.parametrize("write", [write_sml, write_sml_v5])
def test_a_smap_file_of_several_files_has_its_camera_read(tmp_path, write):
    """With several files, each field of ``saveloc.file`` is a column of
    references, and the camera used to be looked for as values: the table
    came without its pixel size."""
    files = [("C:\\data\\Pos0_sml.mat", CAMERA), ("C:\\data\\Pos1_sml.mat", CAMERA),
             ("all_average_sites_F_3_sml.mat", AVERAGE)]
    path = write(tmp_path / "several_sml.mat", files, _rows([1, 1, 2, 2, 3]))
    locs, info = load(path)
    assert info.pixelsize_nm == pytest.approx(127.0)
    camera = locs.metadata["camera"]
    assert camera["conversion"] == 5.0 and camera["roi"] == [137, 286, 223, 208]
    assert camera["frame_time_ms"] == pytest.approx(35.31)


@pytest.mark.parametrize("write", [write_sml, write_sml_v5])
def test_rows_of_no_file_are_left_out_not_read_as_localizations_at_the_origin(tmp_path, write):
    """SMAP shows a row whose ``filenumber`` names none of its files nowhere;
    those found were all zeros, and came in at (0, 0) in frame -1."""
    files = [("C:\\data\\Pos0_sml.mat", CAMERA), ("C:\\data\\Pos1_sml.mat", CAMERA)]
    rows = _rows([0, 1, 1, 2, 0, 2, 2])
    for name in ("xnm", "ynm", "frame", "phot"):
        rows[name][rows["filenumber"] == 0] = 0
    path = write(tmp_path / "leftovers_sml.mat", files, rows)
    locs, _ = load(path)
    assert len(locs) == 5
    assert np.all(locs["frame"] >= 0) and np.all(locs["x_nm"] > 0)
    assert np.allclose(np.sort(locs["photons"]), np.sort(rows["phot"][rows["filenumber"] > 0]))


def _open(path):
    session = Session()
    session.add_file(*load(path))
    return session


@pytest.mark.parametrize("write", [write_sml, write_sml_v5])
def test_the_files_of_a_smap_file_open_as_files_each_with_its_own_camera(tmp_path, write):
    """As one file, the positions of an experiment, which share the camera's
    coordinates, lay on top of each other."""
    files = [("C:\\data\\Pos0_sml.mat", CAMERA), ("C:\\data\\Pos1_sml.mat", CAMERA),
             ("all_average_sites_F_3_sml.mat", AVERAGE)]
    rows = _rows([0, 1, 1, 2, 2, 2, 3, 3])
    session = _open(write(tmp_path / "several_sml.mat", files, rows))
    assert session.file_names() == ["Pos0_sml.mat", "Pos1_sml.mat",
                                    "all_average_sites_F_3_sml.mat"]
    assert session.files[1].path == "C:\\data\\Pos1_sml.mat"
    assert [f.pixelsize_nm for f in session.files] == pytest.approx([127, 127, 100])
    assert [f.frame_time_ms for f in session.files][:2] == pytest.approx([35.31, 35.31])
    assert session.files[2].metadata["camera"]["roi"] == [0, 0, 512, 512]
    # each row in its file: SMAP counts from 1, and the row of none is gone
    kept = rows["filenumber"] > 0
    assert np.array_equal(session.locs["filenumber"], rows["filenumber"][kept] - 1)
    assert np.allclose(session.locs["photons"], rows["phot"][kept])


def test_a_smap_file_of_several_saved_as_hdf5_opens_as_the_same_files(tmp_path):
    files = [("C:\\data\\Pos0_sml.mat", CAMERA), ("C:\\data\\Pos1_sml.mat", CAMERA),
             ("all_average_sites_F_3_sml.mat", AVERAGE)]
    rows = _rows([1, 1, 2, 2, 2, 3, 3])
    smap = _open(write_sml(tmp_path / "several_sml.mat", files, rows))
    smap.save(tmp_path / "several.hdf5")
    again = _open(tmp_path / "several.hdf5")
    assert again.file_names() == smap.file_names()
    assert [f.pixelsize_nm for f in again.files] == pytest.approx([127, 127, 100])
    for name in smap.locs.columns:
        assert np.array_equal(again.locs[name], smap.locs[name]), name


def test_a_smap_file_of_one_file_stays_one_file(tmp_path):
    path = write_sml(tmp_path / "one_sml.mat", [("C:\\data\\Pos0_sml.mat", CAMERA)],
                     _rows([1, 1, 1]))
    locs, info = load(path)
    assert "files" not in locs.metadata and "filenumber" not in locs
    session = _open(path)
    assert session.file_names() == ["one_sml.mat"] and len(session.locs) == 3


def test_each_smap_site_is_an_roi_on_its_own_file_holding_its_files_localizations(tmp_path):
    """As one file, a site held the localizations of every position at its
    place: here Pos1's site and Pos0's localizations are at the same spot."""
    from test_sml_sites import _site
    files = [("C:\\data\\Pos0_sml.mat", CAMERA), ("C:\\data\\Pos1_sml.mat", CAMERA)]
    rows = _rows([1] * 30 + [2] * 20)
    rows["xnm"][:] = 1000 + np.random.default_rng(1).normal(0, 20, 50)
    rows["ynm"][:] = 2000 + np.random.default_rng(2).normal(0, 20, 50)
    rows["frame"] *= 100                 # apart in time: no two merged into a blink
    on_pos1 = _site(4, (1000.0, 2000.0))
    on_pos1["info"]["filenumber"] = 2.0
    on_pos0 = _site(9, (1000.0, 2000.0))
    path = write_sml(tmp_path / "sites_sml.mat", files, rows, sites=[on_pos1, on_pos0])
    project = _open(path).rois
    rois = {r.number: r for r in project.rois.values()}
    assert project.sources[rois[4].file_id].number == 1
    assert project.sources[rois[9].file_id].number == 0
    assert len(project.indices(rois[4])) == 20 and len(project.indices(rois[9])) == 30
