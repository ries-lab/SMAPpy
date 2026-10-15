"""Other programs' files, read and written: text tables (with saved
conversions), Picasso's HDF5 and DECODE's, and the Export plugin."""
import json

import h5py
import numpy as np
import pytest

from smappy import plugins
from smappy.io import formats
from smappy.io.conversions import Conversion, conversion_for, save_conversion
from smappy.locs import Localizations
from smappy.plugins import Context, Selection


def table(n=6, pixelsize=100.0):
    rng = np.random.default_rng(3)
    return Localizations({
        "x_nm": rng.uniform(200, 3000, n).astype(np.float32),
        "y_nm": rng.uniform(200, 3000, n).astype(np.float32),
        "z_nm": rng.normal(0, 100, n).astype(np.float32),
        "frame": np.arange(n, dtype=np.int64),
        "photons": np.full(n, 1500, np.float32),
        "background": np.full(n, 12, np.float32),
        "x_err_nm": np.full(n, 8, np.float32), "y_err_nm": np.full(n, 10, np.float32),
        "xy_err_nm": np.full(n, 9, np.float32), "z_err_nm": np.full(n, 25, np.float32),
        "sigma_nm": np.full(n, 140, np.float32),
        "cluster_id": (np.arange(n) % 3).astype(np.int32),
        "chi2": rng.uniform(0, 1, n).astype(np.float32),
    }, {"pixelsize_nm": pixelsize})


@pytest.fixture(autouse=True)
def own_config(tmp_path, monkeypatch):
    """Saved conversions go to a directory of the test's own."""
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))


# ------------------------------------------------------------------- text

@pytest.mark.parametrize("suffix", [".csv", ".tsv"])
def test_an_exported_csv_reads_back_with_every_column(tmp_path, suffix):
    locs = table()
    path = formats.writer_named("csv").save(locs, tmp_path / f"t{suffix}")
    back, info = formats.load(path)
    assert set(back.keys()) >= set(locs.keys())        # chi2 and cluster_id too
    for name in locs.keys():
        assert np.allclose(back[name], locs[name], atol=1e-3), name
    assert back["frame"].dtype.kind == "i" and back["cluster_id"].dtype.kind == "i"


def test_thunderstorms_names_write_frames_from_one_and_read_back_from_zero(tmp_path):
    locs = table()
    path = formats.writer_named("csv").save(locs, tmp_path / "ts.csv", names="ThunderSTORM")
    header = path.read_text().splitlines()[0].split(",")
    assert "x [nm]" in header and "intensity [photon]" in header and "chi2" in header
    first = path.read_text().splitlines()[1].split(",")
    assert int(first[header.index("frame")]) == 1
    back, info = formats.load(path)
    assert info.metadata["conversion"] == "ThunderSTORM"
    assert list(back["frame"]) == list(locs["frame"])
    assert np.allclose(back["photons"], locs["photons"])


def test_only_the_chosen_columns_are_written(tmp_path):
    path = formats.writer_named("csv").save(table(), tmp_path / "t.csv",
                                            columns=["x_nm", "y_nm", "frame"])
    assert path.read_text().splitlines()[0] == "x_nm,y_nm,frame"


def test_a_table_in_pixels_or_microns_is_read_in_nanometres(tmp_path):
    path = tmp_path / "px.csv"
    path.write_text("X,Y,Z,T,keep\n1.5,2,30,1,7\n3,4,40,2,8\n")
    mapping = {"X": "x_nm", "Y": "y_nm", "Z": "z_nm", "T": "frame"}
    locs, _ = formats.load(path, mapping=mapping, units="px", pixelsize_nm=110.0,
                           first_frame=1)
    assert np.allclose(locs["x_nm"], [165, 330]) and np.allclose(locs["y_nm"], [220, 440])
    assert np.allclose(locs["z_nm"], [30, 40])          # a pixel has no depth
    assert list(locs["frame"]) == [0, 1] and list(locs["keep"]) == [7, 8]
    microns, _ = formats.load(path, mapping=mapping, units="um")
    assert np.allclose(microns["x_nm"], [1500, 3000]) and np.allclose(microns["z_nm"], [3e4, 4e4])


def test_a_saved_conversion_is_found_again_by_the_headers(tmp_path):
    path = tmp_path / "vutara.csv"
    path.write_text("probe,x_um,y_um,frm\n1,1.5,2.0,1\n2,3.0,4.0,2\n")
    with pytest.raises(ValueError, match="no column for x_nm"):
        formats.load(path)
    save_conversion(Conversion(name="Vutara", headers=["x_um", "y_um", "frm"],
                               mapping={"x_um": "x_nm", "y_um": "y_nm", "frm": "frame"},
                               units="um", first_frame=1))
    assert conversion_for(["probe", "x_um", "y_um", "frm"]).name == "Vutara"
    assert conversion_for(["x_um", "y_um"]) is None       # it wants all three
    locs, info = formats.load(path)
    assert np.allclose(locs["x_nm"], [1500, 3000]) and list(locs["frame"]) == [0, 1]
    assert list(locs["probe"]) == [1, 2] and info.metadata["conversion"] == "Vutara"


# ---------------------------------------------------------------- Picasso

def test_the_picasso_file_has_picassos_fields_in_pixels_and_its_metadata(tmp_path):
    locs = table(pixelsize=100.0)
    path = formats.writer_named("Picasso").save(locs, tmp_path / "p.hdf5", group="cluster_id")
    with h5py.File(path, "r") as f:
        rows = f["locs"][...]
        info = json.loads(f["metadata"][()])
    assert isinstance(rows, np.ndarray) and rows.dtype.names[:9] == (
        "frame", "x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy")
    assert {"ellipticity", "net_gradient", "z", "lpz", "group"} <= set(rows.dtype.names)
    assert rows.dtype["frame"] == np.uint32
    assert np.allclose(rows["x"], locs["x_nm"] / 100) and np.allclose(rows["lpy"], 0.1)
    assert np.allclose(rows["z"], locs["z_nm"])           # nm, as Picasso has it
    assert list(rows["group"]) == list(locs["cluster_id"])
    assert info[0]["Pixelsize"] == 100.0 and {"Width", "Height", "Frames"} <= set(info[0])
    assert info[0]["Width"] > rows["x"].max() and info[0]["Frames"] == len(locs)
    assert (tmp_path / "p.yaml").exists()


def test_the_rows_picasso_would_refuse_are_left_out_and_counted(tmp_path):
    locs = table()
    locs.columns["x_err_nm"][1] = np.nan
    locs.columns["photons"][2] = -1
    said = []
    path = formats.writer_named("Picasso").save(locs, tmp_path / "p.hdf5", report=said.append)
    with h5py.File(path, "r") as f:
        assert len(f["locs"]) == len(locs) - 2
    assert said and said[0].startswith("2 localizations left out")


def test_a_picasso_file_reads_in_nanometres_with_its_picks(tmp_path):
    rows = np.zeros(3, dtype=[("frame", "u4"), ("x", "f4"), ("y", "f4"), ("photons", "f4"),
                              ("lpx", "f4"), ("lpy", "f4"), ("sx", "f4"), ("sy", "f4"),
                              ("bg", "f4"), ("net_gradient", "f4"), ("group", "i4")])
    rows["frame"], rows["x"], rows["y"] = [0, 4, 9], [10, 11, 12], [20, 21, 22]
    rows["lpx"], rows["lpy"], rows["sx"], rows["group"] = 0.1, 0.2, 1.2, [1, 1, 2]
    path = tmp_path / "picasso.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("locs", data=rows)
    # an older Picasso: the metadata in the .yaml beside the file
    (tmp_path / "picasso.yaml").write_text("Width: 64\nHeight: 64\nFrames: 10\n---\nPixelsize: 130\n")
    assert formats.reader_for(path).name == "Picasso"
    locs, info = formats.load(path)
    assert info.pixelsize_nm == 130
    assert np.allclose(locs["x_nm"], [1300, 1430, 1560])
    assert np.allclose(locs["x_err_nm"], 13) and np.allclose(locs["xy_err_nm"], np.hypot(13, 26) / np.sqrt(2))
    assert np.allclose(locs["sigma_nm"], 156)
    assert list(locs["pick_id"]) == [1, 1, 2] and "net_gradient" in locs
    assert list(locs["frame"]) == [0, 4, 9]


def test_this_programs_own_hdf5_is_not_taken_for_picassos(tmp_path):
    path = formats.save(table(), tmp_path / "mine.hdf5")
    assert formats.reader_for(path).name == "smappy HDF5"


# ----------------------------------------------------------------- DECODE

def decode_file(path, xy_unit="px"):
    """As decode.utils.emitter_io.save_h5 writes one: an empty dataset for
    what was not predicted."""
    with h5py.File(path, "w") as f:
        f.create_group("meta").attrs.update({"xy_unit": xy_unit,
                                             "px_size": np.array([100.0, 120.0])})
        f.create_group("decode").attrs.update({"version": "0.10.0"})
        g = f.create_group("data")
        g["xyz"] = np.array([[1.0, 2.0, -50.0], [3.0, 4.0, 60.0]], np.float32)
        g["xyz_sig"] = np.array([[0.1, 0.1, 20.0], [0.2, 0.2, 30.0]], np.float32)
        g.create_dataset("xyz_cr", data=h5py.Empty("f"))
        g["phot"] = np.array([1000.0, 2000.0], np.float32)
        g.create_dataset("phot_cr", data=h5py.Empty("f"))
        g["phot_sig"] = np.array([30.0, 40.0], np.float32)
        g["frame_ix"] = np.array([0, 3])
        g["id"] = np.array([-1, -1])
        g["prob"] = np.array([0.9, 0.5], np.float32)
        g["bg"] = np.array([5.0, 6.0], np.float32)
        g.create_dataset("bg_cr", data=h5py.Empty("f"))
        g["bg_sig"] = np.array([1.0, 1.0], np.float32)
    return path


def test_a_decode_file_reads_x_and_y_from_pixels_and_z_as_it_is(tmp_path):
    path = decode_file(tmp_path / "fit.h5")
    assert formats.reader_for(path).name == "DECODE"
    locs, info = formats.load(path)
    assert np.allclose(locs["x_nm"], [100, 300]) and np.allclose(locs["y_nm"], [240, 480])
    assert np.allclose(locs["z_nm"], [-50, 60])
    assert np.allclose(locs["x_err_nm"], [10, 20]) and np.allclose(locs["y_err_nm"], [12, 24])
    assert np.allclose(locs["z_err_nm"], [20, 30])
    assert list(locs["frame"]) == [0, 3] and np.allclose(locs["prob"], [0.9, 0.5])
    assert np.allclose(locs["photons_err"], [30, 40])
    assert "x_crlb_nm" not in locs and "photons_crlb" not in locs   # empty in the file
    assert info.pixelsize_nm == 110


def test_decode_crlbs_are_kept_as_standard_deviations(tmp_path):
    path = decode_file(tmp_path / "fit.h5", xy_unit="nm")
    with h5py.File(path, "a") as f:
        del f["data/xyz_cr"]
        f["data/xyz_cr"] = np.array([[4.0, 9.0, 100.0], [16.0, 25.0, 400.0]], np.float32)
    locs, _ = formats.load(path)
    assert np.allclose(locs["x_nm"], [1, 3])            # already in nm
    assert np.allclose(locs["x_crlb_nm"], [2, 4]) and np.allclose(locs["z_crlb_nm"], [10, 20])


# ----------------------------------------------------------------- export

def export(ctx, tmp_path, name="out.csv", **settings):
    plugin = plugins.get("File/Export/Localizations")()
    return plugin.run(ctx, plugin.Settings(path=str(tmp_path / name), **settings))


def test_export_writes_what_is_shown_or_everything(tmp_path):
    locs = table()
    mask = np.array([True, False, True, False, True, False])
    ctx = Context(locs=locs, selection=Selection(mask))
    export(ctx, tmp_path, "shown.csv")
    export(ctx, tmp_path, "all.csv", localizations="all")
    assert len(formats.load(tmp_path / "shown.csv")[0]) == 3
    assert len(formats.load(tmp_path / "all.csv")[0]) == 6


def test_export_writes_the_ticked_columns_and_picasso_by_the_name(tmp_path):
    ctx = Context(locs=table())
    export(ctx, tmp_path, "few.tsv", columns=["x_nm", "y_nm", "frame"])
    assert (tmp_path / "few.tsv").read_text().splitlines()[0] == "x_nm\ty_nm\tframe"
    result = export(ctx, tmp_path, "p.hdf5", group="cluster_id")
    assert "Picasso" in result.text
    assert formats.reader_for(tmp_path / "p.hdf5").name == "Picasso"


def test_export_shows_columns_for_text_and_a_group_for_picasso():
    plugin = plugins.get("File/Export/Localizations")
    describe = plugin().describe
    assert describe(None, plugin.Settings(path="a.csv")).shown == \
        {"columns": True, "group": False}
    assert describe(None, plugin.Settings(path="a.hdf5")).shown == \
        {"columns": False, "group": True}


def test_export_writes_merged_blinks_when_asked(tmp_path):
    from smappy.session import Session
    rng = np.random.default_rng(0)
    n = 40                         # 20 blinks of two frames each
    x = np.repeat(rng.uniform(0, 5000, 20), 2) + rng.normal(0, 2, n)
    y = np.repeat(rng.uniform(0, 5000, 20), 2) + rng.normal(0, 2, n)
    frame = np.repeat(np.arange(0, 60, 3), 2) + np.tile([0, 1], 20)
    session = Session(Localizations({"x_nm": x, "y_nm": y, "frame": frame,
                                     "photons": np.full(n, 500.0),
                                     "xy_err_nm": np.full(n, 10.0)}))
    ctx = session.context()
    export(ctx, tmp_path, "blinks.csv", localizations="all", grouping="grouped")
    export(ctx, tmp_path, "locs.csv", localizations="all", grouping="ungrouped")
    assert len(formats.load(tmp_path / "blinks.csv")[0]) == 20
    assert len(formats.load(tmp_path / "locs.csv")[0]) == 40


def test_load_is_a_button_for_the_window_only():
    plugin = plugins.get("File/Load/Load")
    assert plugin.window == "open files"
    with pytest.raises(ValueError, match="File/Load/Auto"):
        plugin().run(Context(), plugin.Settings())


def test_the_export_panel_lists_the_tables_columns_to_tick(tmp_path):
    """All ticked is [] -- every column, also one a later table adds --
    and unticking one leaves the rest; the list is hidden for Picasso."""
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.session import Session
    panel = PluginPanel(plugins.get("File/Export/Localizations"), Session(table()))
    field = panel.form.fields["columns"]
    names = [field.list.item(i).data(Qt.UserRole) for i in range(field.list.count())]
    assert set(names) >= set(table().keys())       # and what merging blinks wrote
    assert field.value() == []
    field.list.item(names.index("chi2")).setCheckState(Qt.Unchecked)
    assert "chi2" not in field.value() and "x_nm" in field.value()
    assert panel.form.value().columns == field.value()
    panel.form.fields["path"].set(str(tmp_path / "p.hdf5"))
    panel._react("path")
    assert panel.form.row_layouts["columns"].isRowVisible(field) is False
    assert panel.form.row_layouts["group"].isRowVisible(panel.form.fields["group"])
