"""Which columns a file is read with, and at what width (`io.column_types`,
`import_columns`, `io.formats.load`'s ``columns``)."""
import h5py
import numpy as np
import pytest

from smappy.filter import LocFilter
from smappy.import_columns import ALL, ASK, STANDARD, plan, standard_names
from smappy.io.column_types import is_time, narrow, offset_heavy
from smappy.io.formats import catalog, estimate_bytes, load
from smappy.locs import Localizations, concat


def _sml(path, n=1000):
    """A v7.3 SMAP file with three of the columns *standard* leaves out."""
    with h5py.File(path, "w") as f:
        loc = f.create_group("saveloc/loc")
        loc["xnm"] = np.linspace(0, 5e4, n)[None]                # double, as MATLAB writes
        loc["ynm"] = np.linspace(0, 3e4, n)[None]
        loc["frame"] = np.arange(1, n + 1, dtype=np.float64)[None]
        loc["phot"] = np.round(np.linspace(100, 5000, n))[None]  # whole, and not an id
        loc["locprecnm"] = np.full((1, n), 8.0)
        loc["xnmf"] = np.zeros((1, n))
        loc["xpixerr"] = np.zeros((1, n))
        loc["PSFypix"] = np.ones((1, n))
        loc["filenumber"] = np.ones((1, n))
        f.create_group("saveloc/file/info")["cam_pixelsize_um"] = np.array([[0.1]])
    return path


# ------------------------------------------------------------ the widths

def test_time_by_its_name_stays_float64_and_everything_else_becomes_float32():
    assert is_time("time_s") and is_time("on_time_s") and is_time("tim")
    assert not is_time("photons")
    t = np.linspace(0, 3600, 1001)
    assert narrow("time_s", t).dtype == np.float64
    assert narrow("x_nm", np.linspace(0, 5e4, 1001)).dtype == np.float32
    # a whole-numbered float that is no id stays a float: photons of a simulation
    assert narrow("photons", np.round(np.linspace(1, 1e4, 1001))).dtype == np.float32


def test_a_whole_numbered_id_or_frame_becomes_an_integer_and_integers_stay():
    assert narrow("frame", np.arange(10.0)).dtype == np.int32
    assert narrow("cluster_id", np.array([-1.0, 0, 3])).dtype == np.int32
    assert narrow("tid", np.array([0.0, 2 ** 40])).dtype == np.int64
    assert narrow("frame", np.arange(10, dtype=np.int64)).dtype == np.int64
    # an id with a fraction or a NaN in it is not one: left a float
    assert narrow("cluster_id", np.array([0.5, 1.0])).dtype.kind == "f"


def test_a_column_far_from_zero_against_its_spread_stays_float64():
    clock = 1.7e9 + np.linspace(0, 100, 1001)            # seconds since 1970
    assert offset_heavy(clock)
    assert narrow("clock", clock).dtype == np.float64
    assert not offset_heavy(np.linspace(0, 5e4, 1001))   # positions: float32 is plenty
    # float32 would have kept only steps of 128 s
    assert np.unique(clock.astype(np.float32)).size < 5


# ------------------------------------------------------------ the readers

def test_standard_leaves_out_the_smap_columns_nothing_reads(tmp_path):
    path = _sml(tmp_path / "run_sml.mat")
    standard, _ = load(path)
    assert {"x_nm", "y_nm", "frame", "photons", "xy_err_nm"} <= set(standard)
    assert not {"xnmf", "xpixerr", "PSFypix"} & set(standard)
    every, _ = load(path, columns=ALL)
    assert {"xnmf", "xpixerr", "PSFypix"} <= set(every)
    # read column by column at the width it is kept at
    assert standard["x_nm"].dtype == np.float32 and standard["photons"].dtype == np.float32
    assert standard["frame"].dtype == np.int64 and standard["frame"][0] == 0


def test_a_list_reads_those_columns_and_the_ones_every_table_needs(tmp_path):
    path = _sml(tmp_path / "run_sml.mat")
    locs, _ = load(path, columns=["photons"])
    assert set(locs) == {"x_nm", "y_nm", "frame", "photons"}
    with pytest.raises(ValueError, match="standard"):
        load(path, columns="some")


def test_the_catalog_lists_a_files_columns_and_their_memory_before_reading(tmp_path):
    path = _sml(tmp_path / "run_sml.mat", n=1000)
    found = catalog(path)
    names = {c.name: c for c in found}
    assert set(names) == {"x_nm", "y_nm", "frame", "photons", "xy_err_nm",
                          "xnmf", "xpixerr", "PSFypix"}
    assert names["x_nm"].nbytes == 4000 and names["x_nm"].standard
    assert not names["xnmf"].standard
    assert estimate_bytes(found, ALL) == 8 * 4000
    assert estimate_bytes(found, STANDARD) == 5 * 4000
    assert set(standard_names(found)) == {"x_nm", "y_nm", "frame", "photons", "xy_err_nm"}


def test_a_smappy_file_comes_back_exactly_as_it_was_saved(tmp_path):
    from smappy.io.hdf5 import save_localizations
    locs = Localizations({"x_nm": np.linspace(0, 1, 5), "y_nm": np.zeros(5),
                          "frame": np.arange(5)}, {})
    path = save_localizations(tmp_path / "t.hdf5", locs, {})
    back, _ = load(path)
    assert back["x_nm"].dtype == np.float64            # never narrowed
    only, _ = load(path, columns=["x_nm"])
    assert set(only) == {"x_nm", "y_nm", "frame"}


# ------------------------------------------------------------ the plan

def test_standard_asks_only_when_the_standard_columns_would_not_fit(tmp_path):
    path = _sml(tmp_path / "run_sml.mat", n=1000)      # 20 kB standard
    assert not plan(path, STANDARD, available=10 ** 9).ask
    tight = plan(path, STANDARD, available=30_000)
    assert tight.ask and "needs" in tight.why and tight.found
    assert plan(path, ASK, available=10 ** 9).ask
    assert plan(path, ALL, available=1).columns == ALL and not plan(path, ALL).ask


def test_a_format_that_cannot_list_its_columns_is_read_unasked(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("x_nm,y_nm,frame\n1,2,0\n3,4,1\n")
    assert not plan(path, ASK, available=1).ask


# ------------------------------------------------------------ downstream

def test_joining_float32_and_float64_keeps_float32_but_time_stays_wide():
    a = Localizations({"x_nm": np.zeros(3, np.float32), "time_s": np.zeros(3)}, {})
    b = Localizations({"x_nm": np.ones(2), "time_s": np.linspace(1e3, 1e3 + 1e-4, 2)}, {})
    joined = concat([a, b])
    assert joined["x_nm"].dtype == np.float32
    assert joined["time_s"].dtype == np.float64


def test_a_bound_at_a_float32_value_keeps_that_value():
    locs = Localizations({"photons": np.array([0.1, 0.2, 0.3], np.float32)}, {})
    edge = np.float64(0.1)                     # 0.1 read back as a NumPy scalar
    kept = LocFilter(locs).set("photons", edge, np.float64(0.2)).mask
    assert kept.tolist() == [True, True, False]


# ------------------------------------------------------------ the dialog

def test_the_dialog_starts_on_the_standard_columns_and_keeps_the_required(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui.columns_dialog import ColumnsDialog
    found = catalog(_sml(tmp_path / "run_sml.mat"))
    dialog = ColumnsDialog(found, "why", available=10 ** 6)
    assert set(dialog.columns()) == set(standard_names(found))
    dialog.tick("none")
    assert set(dialog.columns()) == {"x_nm", "y_nm", "frame"}
    dialog.tick("all")
    assert len(dialog.columns()) == len(found)
    assert "available" in dialog.total.text()
    dialog.deleteLater()
