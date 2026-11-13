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


def write_sml(path, files, rows):
    """``files`` a list of (name, info): one a struct, several a struct array."""
    with h5py.File(path, "w") as f:
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
