"""The File tab: loading, saving, exporting and simulating as plugins."""
import numpy as np
import pytest

from smappy import plugins
from smappy.io import formats
from smappy.io.hdf5 import load_gui_state, load_localizations, save_gui_state
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.session import Session, read_localizations


def table(n=50):
    rng = np.random.default_rng(0)
    return Localizations({"x_nm": rng.uniform(0, 1000, n), "y_nm": rng.uniform(0, 1000, n),
                          "frame": np.arange(n), "photons": np.full(n, 300.0),
                          "xy_err_nm": np.full(n, 10.0)})


@pytest.fixture
def saved(tmp_path):
    path = tmp_path / "run.hdf5"
    formats.save(table(), path)
    return path


# ------------------------------------------------------------- the registry

def test_every_file_plugin_is_in_the_tree():
    found = plugins.refs("File/")
    assert set(found) == {
        "File/Load/Auto", "File/Load/Load", "File/Save/smappy HDF5",
        "File/Export/Image", "File/Export/Localizations",
        "File/Simulate/Blinking Structure"}


def test_the_one_loader_offers_every_format_and_reads_columns_for_a_csv_only():
    loader = plugins.get("File/Load/Auto")
    spec = loader.specs()["format"]
    assert spec.info.advanced
    assert [v for v, _ in spec.info.choices()] == ["", *formats.reader_names()]
    describe = loader().describe
    # the csv part is shown for a csv only
    assert describe(None, loader.Settings(path="a.csv")).shown == {"csv": True}
    assert describe(None, loader.Settings(path="a.hdf5")).shown == {"csv": False}
    assert describe(None, loader.Settings(path="a.dat", format="csv")).shown == \
        {"csv": True}


def test_the_writer_registry_mirrors_the_readers(tmp_path):
    assert [w.name for w in formats.WRITERS] == ["smappy HDF5", "csv", "Picasso"]
    # a suffix says smappy's own; Picasso's is asked for by name
    assert formats.writer_for("a.hdf5").name == "smappy HDF5"
    assert formats.writer_named("Picasso").name == "Picasso"
    with pytest.raises(ValueError, match="no writer"):
        formats.writer_for("a.xyz")
    path = formats.save(table(3), tmp_path / "w.hdf5")
    assert len(load_localizations(path)) == 3
    assert "smappy HDF5" in formats.save_filter()


# ----------------------------------------------------------------- loading

def test_load_auto_reads_and_links_in_one_place(saved):
    session = Session()
    plugin = plugins.get("File/Load/Auto")()
    result = session.run(plugin, plugin.Settings(path=str(saved)))
    assert len(session.locs) == 50 and session.files[0].format == "smappy"
    assert "50 localizations" in result.text


def test_a_loader_hands_files_back_rather_than_touching_the_session(saved):
    """It runs in a worker thread, so it must not add the file itself."""
    plugin = plugins.get("File/Load/Auto")()
    result = plugin.run(Context(), plugin.Settings(path=str(saved)))
    assert result.locs is None                      # not a table replacement
    (locs, info, grouped) = result.files[0]
    assert len(locs) == 50 and info.format == "smappy"


def test_appending_keeps_the_first_file(saved):
    session = Session()
    plugin = plugins.get("File/Load/Auto")()
    session.run(plugin, plugin.Settings(path=str(saved)))
    session.run(plugin, plugin.Settings(path=str(saved), append=True))
    assert len(session.files) == 2 and len(session.locs) == 100


def test_a_named_format_refuses_the_wrong_file(saved):
    plugin = plugins.get("File/Load/Auto")()
    with pytest.raises(Exception):
        plugin.run(Context(), plugin.Settings(path=str(saved), format="SMAP"))


def test_a_missing_file_says_so_before_reading(tmp_path):
    plugin = plugins.get("File/Load/Auto")()
    with pytest.raises(FileNotFoundError):
        plugin.run(Context(), plugin.Settings(path=str(tmp_path / "nope.hdf5")))
    with pytest.raises(ValueError, match="choose a file"):
        plugin.run(Context(), plugin.Settings())


def test_csv_guesses_its_columns(tmp_path):
    """ThunderSTORM's headers: x and y found by name, and its frames, which
    count from 1, made to count from 0 (the shipped conversion)."""
    path = tmp_path / "t.csv"
    path.write_text("x [nm],y [nm],frame\n1,2,1\n3,4,2\n")
    plugin = plugins.get("File/Load/Auto")()
    result = plugin.run(Context(), plugin.Settings(path=str(path)))
    locs = result.files[0][0]
    assert len(locs) == 2 and list(locs["x_nm"]) == [1.0, 3.0]
    assert list(locs["frame"]) == [0, 1]


def test_read_localizations_is_the_one_implementation_and_merges_nothing(saved):
    """The window's LoadTask, Session.load and the plugin all go through it.
    The layer is set to blinks, and they are merged when it is first read:
    the file does not wait for them."""
    locs, info = read_localizations(saved)
    assert len(locs) == 50 and info.format == "smappy"
    assert "group_id" not in locs and "n_in_group" not in locs
    result = plugins.get("File/Load/Auto")().run(
        Context(), plugins.get("File/Load/Auto").Settings(path=str(saved)))
    assert result.files[0][2] is None
    session = Session()
    session.load(saved)
    assert session.layers[0].grouped and session.tables.peek_grouped() is None
    session.layers[0].locs
    assert session.tables.is_current


# ------------------------------------------------------------- the rest

def test_save_writes_the_table_and_the_gui_state(tmp_path):
    session = Session(table())
    session.gui_state_provider = lambda: {"tabs": [{"name": "File"}], "layout": {}}
    plugin = plugins.get("File/Save/smappy HDF5")()
    out = tmp_path / "out.hdf5"
    session.run(plugin, plugin.Settings(path=str(out), gui_state=True))
    assert len(load_localizations(out)) == 50
    assert load_gui_state(out)["tabs"][0]["name"] == "File"


def test_save_can_leave_the_gui_state_out(tmp_path):
    session = Session(table())
    session.gui_state_provider = lambda: {"tabs": [{"name": "File"}]}
    plugin = plugins.get("File/Save/smappy HDF5")()
    out = tmp_path / "out.hdf5"
    session.run(plugin, plugin.Settings(path=str(out), gui_state=False))
    assert load_gui_state(out) is None


def test_a_failing_gui_state_never_loses_the_data(tmp_path):
    def broken():
        raise RuntimeError("no")

    session = Session(table())
    session.gui_state_provider = broken
    with pytest.raises(RuntimeError):
        broken()                                     # the provider really is broken
    out = session.save(tmp_path / "out.hdf5", gui_state=True)
    assert len(load_localizations(out)) == 50        # the data is there anyway


def test_export_writes_a_picture(tmp_path):
    pytest.importorskip("PIL")
    session = Session(table())
    plugin = plugins.get("File/Export/Image")()
    out = tmp_path / "pic.png"
    session.run(plugin, plugin.Settings(path=str(out), pixelsize_nm=40.0))
    assert out.exists() and out.stat().st_size > 0


def test_simulate_makes_a_dataset_to_try():
    session = Session()
    plugin = plugins.get("File/Simulate/Blinking Structure")()
    result = session.run(plugin, plugin.Settings(n_frames=200, seed=1))
    assert len(session.locs) > 100
    assert session.files[0].format == "simulated"
    assert "emitters" in result.text
    assert {"x_nm", "y_nm", "z_nm", "frame", "photons"} <= set(session.locs.keys())


def test_simulated_drift_records_the_truth():
    plugin = plugins.get("File/Simulate/Blinking Structure")()
    result = plugin.run(Context(), plugin.Settings(n_frames=200, seed=1, drift=True))
    assert "drift_truth" in result.files[0][0].metadata


# ------------------------------------------------------- the /gui group

def test_gui_state_is_a_dataset_so_a_big_one_fits(tmp_path):
    """An HDF5 attribute is bounded by the object header, about 64 kB."""
    path = tmp_path / "big.hdf5"
    formats.save(table(3), path)
    big = {"tabs": [{"name": "T", "instances": [{"plugin": "a/b", "values": {"k": i}}
                                                for i in range(2000)]}]}
    assert len(str(big)) > 64_000
    save_gui_state(path, big)
    assert len(load_gui_state(path)["tabs"][0]["instances"]) == 2000
    assert len(load_localizations(path)) == 3        # the data is untouched


def test_gui_state_can_be_cleared_and_is_never_fatal(tmp_path):
    path = tmp_path / "x.hdf5"
    formats.save(table(3), path)
    assert load_gui_state(path) is None
    save_gui_state(path, {"a": 1})
    save_gui_state(path, None)
    assert load_gui_state(path) is None
    assert load_gui_state(tmp_path / "missing.hdf5") is None


# ------------------------------------------------------------------ the GUI

def test_the_file_tab_ships_with_load_first_and_what_the_file_menu_does_not_do(
        tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    from smappy import config
    config.load(reload=True)
    QApplication.instance() or QApplication([])
    from smappy.gui.app import ControlWindow, RenderWindow

    session = Session()
    window = ControlWindow(session, RenderWindow(session))
    assert window.tabs.tabText(0) == "File"
    pinned = [i.plugin for i in window.tabs.widget(0).tab.instances]
    # Load opens files as File > Open does, first; saving is the File
    # menu's, and the tree still has it and the loader, for chains and the
    # batch
    assert pinned[0] == "File/Load/Load"
    assert set(pinned) == {"File/Load/Load", "File/Export/Image",
                           "File/Export/Localizations", "File/Simulate/Blinking Structure"}
    assert "open files" in session.window_openers
    # the window is what supplies the state a save embeds, and it leaves
    # geometry out: machine-specific, and worthless as provenance
    state = window.plugin_state()
    assert [t["name"] for t in state["tabs"]][0] == "File"
    assert "geometry" not in state["layout"]
    assert session.gui_state_provider is not None
