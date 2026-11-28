"""A PSF calibration keeps its run in its file (`calibrate.saved_run`): opened again, its pages, table and
settings come back without the bead stacks, and the stacks come back where they are found."""
import shutil

import h5py
import numpy as np
import pytest
import tifffile

from smappy.calibrate import saved_run
from smappy.plugins import Context, get, settings_from


@pytest.fixture(scope="module")
def average(tmp_path_factory):
    """A bead average from two simulated stacks, saved by the plugin: the folder, the calibration."""
    from smappy.simulate.camera import bead_stacks
    folder = tmp_path_factory.mktemp("saved_run")
    (folder / "beads").mkdir()
    stacks, _ = bead_stacks(n_stacks=2, z_range_nm=(-600.0, 600.0))
    for i, stack in enumerate(stacks):
        tifffile.imwrite(folder / "beads" / f"beads_{i}.tif", stack, imagej=True, resolution=(10, 10),
                         metadata={"unit": "um", "axes": "ZYX"})
    cls = get("Localize/PSF calibration")
    cls().run(Context(), settings_from(cls.Settings, {
        "stacks": str(folder / "beads"), "method": "average", "beads.dz_nm": 20,
        "beads.roi_size": 21, "save_as": str(folder / "average.h5")}))
    return folder, folder / "average.h5"


@pytest.fixture(scope="module")
def app():
    from smappy.gui import make_app
    return make_app()


def _draw(page):
    from matplotlib.figure import Figure
    if page.table is not None:
        return len(page.table()[1])
    figure = Figure()
    chosen = {k: (v if isinstance(v, bool) else list(v)[0]) for k, v in (page.controls or {}).items()}
    page.draw(figure, [], set(), chosen) if page.controls else page.draw(figure, [], set())
    return len(figure.axes)


def test_the_tree_keeps_dicts_tuples_strings_and_nan(tmp_path):
    value = {"a": np.arange(3.0), 2: "two", "t": (np.ones(2), [1, 2]), "s": np.array(["used", "rejected"]),
             "n": float("nan"), "nested": [{"x": np.zeros((2, 2), np.float32)}, None]}
    with h5py.File(tmp_path / "t.h5", "w") as f:
        saved_run._put(f, "v", value)
    with h5py.File(tmp_path / "t.h5", "r") as f:
        back = saved_run._get(f["v"])
    assert list(back) == ["a", 2, "t", "s", "n", "nested"]
    assert np.array_equal(back["a"], value["a"]) and back[2] == "two"
    assert isinstance(back["t"], tuple) and back["t"][1] == [1, 2]
    assert list(back["s"] == "used") == [True, False]
    assert np.isnan(back["n"]) and back["nested"][0]["x"].shape == (2, 2) and back["nested"][1] is None


def test_the_stacks_are_found_where_they_were_beside_the_calibration_or_in_a_chosen_folder(tmp_path):
    (tmp_path / "data" / "beads").mkdir(parents=True)
    stack = tmp_path / "data" / "beads" / "b.tif"
    stack.write_bytes(b"x" * 100)
    calibration = tmp_path / "data" / "cal.h5"
    record = {"sources": [saved_run._source(stack, calibration)]}
    assert saved_run.find_sources(record, calibration) == [str(stack.resolve())]
    # the whole folder moved: found beside the calibration as it was
    shutil.move(tmp_path / "data", tmp_path / "moved")
    assert saved_run.find_sources(record, tmp_path / "moved" / "cal.h5") == [
        str(tmp_path / "moved" / "beads" / "b.tif")]
    # the stacks alone moved: only in a folder chosen, by name, below it too, and of the same size
    (tmp_path / "archive" / "deep").mkdir(parents=True)
    shutil.move(tmp_path / "moved" / "beads" / "b.tif", tmp_path / "archive" / "deep" / "b.tif")
    assert saved_run.find_sources(record, tmp_path / "moved" / "cal.h5") == [None]
    assert saved_run.find_sources(record, tmp_path / "moved" / "cal.h5", tmp_path / "archive") == [
        str(tmp_path / "archive" / "deep" / "b.tif")]
    (tmp_path / "archive" / "deep" / "b.tif").write_bytes(b"x" * 99)        # another file of that name
    assert saved_run.find_sources(record, tmp_path / "moved" / "cal.h5", tmp_path / "archive") == [None]


def test_a_saved_bead_average_shows_its_pages_table_and_settings_from_its_file_alone(average):
    folder, path = average
    run = saved_run.load(path)
    record = run.result.record
    assert record["method"] == "average" and record["mode"] == "Single channel"
    assert {p.title: _draw(p) for p in saved_run.ENTRY.pages(run)} == {
        "Model vs data": pytest.approx(21, abs=20), "Fit quality": 4, "CRLB": 1}
    rows = saved_run.ENTRY.rows(run, {})
    assert rows and len(rows[0][0]) == len(record["columns"]) and "beads used" in saved_run.ENTRY.summary(run)
    assert saved_run.settings_of(record)["detection"].roi_size == 21
    # the simulated cylindrical lens focuses x and y 300 nm either side of focus (`simulate.camera.ASTIGMATISM`)
    focal = record["focal_planes"]["psf"]
    assert focal["separation_nm"] == pytest.approx(600, abs=30) and focal["x_nm"] * focal["y_nm"] < 0
    assert "focal separation" in saved_run.ENTRY.summary(run)
    assert saved_run.find_sources(record, path) == [str((folder / "beads" / f"beads_{i}.tif").resolve())
                                                    for i in range(2)]


def test_a_file_without_a_run_says_only_its_spline_can_be_used(average, tmp_path):
    _, path = average
    bare = tmp_path / "bare.h5"
    shutil.copyfile(path, bare)
    with h5py.File(bare, "a") as f:
        del f[saved_run.GROUP]
    with pytest.raises(ValueError, match="only its spline"):
        saved_run.load(bare)


def test_the_window_opens_a_calibration_with_its_stacks_without_them_or_from_a_chosen_folder(
        app, average, tmp_path):
    from smappy.calibrate.psf_window import PSFCalibrationWindow
    folder, path = average
    window = PSFCalibrationWindow()
    window.choose("beads in a z stack", "pupil")
    assert window.open_calibration(path)
    assert window.method.key == "average" and len(window.paths) == 2
    assert window.bead_table.rowCount() == len(saved_run.load(path).result.record["rows"])
    assert "Model vs data" in [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert not window.toggle_button.isEnabled()          # marks need a detection: Calibrate from the stacks
    # a copy elsewhere, its stacks not beside it: Cancel shows it alone; a folder chosen finds them
    away = tmp_path / "away"
    away.mkdir()
    shutil.copyfile(path, away / "average.h5")
    shutil.move(folder / "beads", tmp_path / "stacks")
    try:
        window._ask = lambda box: [b.click() for b in box.buttons()
                                   if box.buttonRole(b) == box.ButtonRole.RejectRole]
        assert window.open_calibration(away / "average.h5")
        assert window.paths == [] and "not found" in window.status_bar.currentMessage()
        assert window.open_calibration(away / "average.h5", folder=tmp_path / "stacks")
        assert sorted(p.split("/")[-1] for p in window.paths) == ["beads_0.tif", "beads_1.tif"]
    finally:
        shutil.move(tmp_path / "stacks", folder / "beads")
        window.close()


def test_a_bead_diameter_of_zero_is_flagged_for_beads_only():
    from smappy.calibrate.pupil import BeadSettings, bead_size_warning
    from smappy.plugins.psf_calibration import PSFCalibration, PSFCalibrationSettings
    assert "bead diameter is 0" in bead_size_warning(BeadSettings())
    assert bead_size_warning(BeadSettings(bead_diameter_nm=100.0)) is None
    state = PSFCalibration().describe(None, PSFCalibrationSettings(method="pupil"))
    assert "bead diameter is 0" in state.notice.text
    assert PSFCalibration().describe(None, PSFCalibrationSettings(method="insitu")).notice is None
