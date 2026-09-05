"""Readers: each format lands in smappy's columns; tables of unlike columns join."""
import json
import zipfile

import h5py
import numpy as np
import pytest

from smappy.images import ImageData, load_image
from smappy.io.formats import guess_csv_mapping, load, reader_for
from smappy.locs import Localizations, concat
from smappy.render import FieldOfView
from smappy.session import Session


def test_smap_sml_v73(tmp_path):
    path = tmp_path / "run_sml.mat"
    n = 50
    with h5py.File(path, "w") as f:            # the shape MATLAB's -v7.3 writes
        loc = f.create_group("saveloc/loc")
        loc["xnm"] = np.arange(n, dtype=np.float32)[None]
        loc["ynm"] = np.ones((1, n), np.float32)
        loc["frame"] = np.arange(1, n + 1, dtype=np.float64)[None]
        loc["phot"] = np.full((1, n), 500, np.float32)
        loc["locprecnm"] = np.full((1, n), 8, np.float32)
        loc["LLrel"] = np.zeros((1, n), np.float32)
        loc["filenumber"] = np.ones((1, n), np.float32)
        info = f.create_group("saveloc/file/info")
        info["cam_pixelsize_um"] = np.array([[0.1]])
    locs, meta = load(path)
    assert reader_for(path).name == "SMAP"
    assert set(locs) == {"x_nm", "y_nm", "frame", "photons", "loc_precision_nm", "logl_rel"}
    assert locs["frame"][0] == 0 and locs["frame"].dtype == np.int64   # 1-based -> 0-based
    assert meta.pixelsize_nm == 100.0 and meta.n == n


def test_minflux_npy_and_zip(tmp_path):
    n, k = 30, 3
    itr = np.zeros((n, k), dtype=[("itr", "i4"), ("loc", "f8", (3,)), ("eco", "f8"),
                                  ("efo", "f8"), ("cfr", "f8"), ("dcr", "f8")])
    itr["loc"][:, -1, :2] = np.random.default_rng(0).random((n, 2)) * 1e-6   # metres
    itr["eco"][:, -1] = 400
    rec = np.zeros(n, dtype=[("itr", itr.dtype, (k,)), ("tim", "f8"), ("tid", "i8"), ("vld", "?")])
    rec["itr"] = itr
    rec["tim"] = np.arange(n)[::-1] * 1e-3                 # reverse time: gets sorted
    rec["tid"] = np.arange(n) // 10
    rec["vld"] = True
    rec["vld"][0] = False
    np.save(tmp_path / "m.npy", rec)
    locs, info = load(tmp_path / "m.npy")
    assert len(locs) == n - 1 and "z_nm" not in locs and info.format == "MINFLUX"
    assert locs["x_nm"].max() < 1000 and np.all(np.diff(locs["time_s"]) >= 0)
    assert np.allclose(locs["loc_precision_nm"], 150 / np.sqrt(400))
    with zipfile.ZipFile(tmp_path / "m.zip", "w") as z:
        z.write(tmp_path / "m.npy", "export.npy")
    assert len(load(tmp_path / "m.zip")[0]) == n - 1


def test_csv_thunderstorm_and_mapping(tmp_path):
    path = tmp_path / "ts.csv"
    path.write_text('"id","frame","x [nm]","y [nm]","sigma [nm]","intensity [photon]","uncertainty [nm]"\n'
                    '1,1,100.5,200.5,150,800,9\n2,3,110.5,210.5,160,900,8\n')
    assert guess_csv_mapping(['x [nm]', 'y [nm]']) == {'x [nm]': 'x_nm', 'y [nm]': 'y_nm'}
    locs, info = load(path)
    assert len(locs) == 2 and locs["photons"][1] == 900 and locs["frame"][1] == 3
    bare = tmp_path / "bare.csv"
    bare.write_text("1.5,2.5,7\n3.5,4.5,8\n")
    with pytest.raises(ValueError):
        load(bare)
    locs, _ = load(bare, mapping={"column 1": "x_nm", "column 2": "y_nm", "column 3": "frame"},
                   units="px", pixelsize_nm=100)
    assert locs["x_nm"][1] == 350 and locs["frame"][0] == 7


def test_concat_fills_missing_columns():
    a = Localizations({"x_nm": np.ones(3, np.float32), "frame": np.arange(3)}, {})
    b = Localizations({"x_nm": np.zeros(2, np.float32), "z_nm": np.ones(2, np.float32)}, {})
    c = concat([a, b])
    assert len(c) == 5 and np.isnan(c["z_nm"][:3]).all() and c["frame"].dtype.kind == "i"
    assert (c["frame"][3:] == 0).all()


def test_session_joins_files_and_layers_pick_them():
    a = Localizations({"x_nm": np.arange(10, dtype=np.float32), "y_nm": np.zeros(10, np.float32),
                       "frame": np.arange(10)}, {})
    b = Localizations({"x_nm": np.arange(5, dtype=np.float32), "y_nm": np.ones(5, np.float32),
                       "frame": np.arange(5)}, {})
    from smappy.io.formats import FileInfo
    s = Session()
    s.add_file(a, FileInfo("a", "/a", "test"))
    s.add_file(b, FileInfo("b", "/b", "test"), append=True)
    assert s.file_names() == ["a", "b"] and len(s.locs) == 15
    assert set(np.unique(s.locs["filenumber"])) == {0, 1}
    layer = s.layers[0]
    layer.set_files([1])
    assert len(s.selection(0)) == 5
    layer.set_files(None)
    assert len(s.selection(0)) == 15
    s.undo()
    assert len(s.locs) == 10 and s.file_names() == ["a"]


def test_image_layer_resamples_and_reads_pixel_size(tmp_path):
    import tifffile
    img = np.arange(16, dtype=np.uint16).reshape(4, 4)
    tifffile.imwrite(tmp_path / "wf.tif", img, imagej=True, resolution=(10, 10),
                     metadata={"unit": "um"})                       # 100 nm pixels
    data = load_image(tmp_path / "wf.tif")
    assert data.pixelsize == 100.0 and data.bounds == (0, 0, 400, 400)
    rendered = data.resample(FieldOfView(0, 0, 50.0, 8, 8))     # 2 view pixels per image pixel
    assert rendered.weight[0, 0] == 0 and rendered.weight[1, 2] == 1 and rendered.weight[7, 7] == 15
    outside = data.resample(FieldOfView(1000, 1000, 50.0, 4, 4))
    assert outside.weight.sum() == 0
    s = Session()
    layer = s.add_image(data)
    assert layer.is_image and s.full_view()[0][1] >= 400
    rgb, _ = layer.render(FieldOfView(0, 0, 100.0, 4, 4))
    assert rgb.shape == (4, 4, 3)
