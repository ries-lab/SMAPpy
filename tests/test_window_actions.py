"""The assistant inside SMAPpy's own windows (`gui/window_actions.py`): the calibration window it
could open and not use (a user's report, 2026-11-12), now the PSF calibration."""
import pytest
import tifffile

pytest.importorskip("PySide6")


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def control(app, tmp_path):
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.session import Session
    session = Session()
    render = RenderWindow(session)
    window = ControlWindow(session, render)
    a = window._assistant().server.assistant
    a.knowledge = tmp_path / "kb"
    a.output = tmp_path / "output"
    a.output.mkdir()
    a.mode = "act"
    yield window, a
    for w in (window.psf_calibration_window,):
        # a job's thread is still winding down after its `done`: one destroyed running aborts
        thread = getattr(w, "_thread", None)
        if thread is not None:
            thread.quit()
            thread.wait()
    for w in (window.psf_calibration_window, render, window):
        if w is not None:
            w.close()


def act(a, name, **arguments):
    return a.call("do_action", {"name": name, "arguments": arguments})


def _bead_folder(tmp_path, n_stacks=2):
    from smappy.simulate.camera import bead_stacks
    folder = tmp_path / "beads"
    folder.mkdir()
    stacks, _ = bead_stacks(n_stacks=n_stacks, z_range_nm=(-600.0, 600.0))
    for i, stack in enumerate(stacks):
        tifffile.imwrite(folder / f"beads_{i}.tif", stack)
    return folder


def test_every_window_the_assistant_opens_has_actions_inside_or_a_reason(control):
    """`open_window` alone opened a calibration window and left the assistant outside it."""
    from smappy.gui.window_actions import WINDOWS_WITH_ACTIONS, WINDOWS_WITHOUT_ACTIONS
    window, a = control
    opened = set(window._assistant_windows())
    assert sorted(opened - set(WINDOWS_WITH_ACTIONS) - set(WINDOWS_WITHOUT_ACTIONS)) == []
    assert sorted((set(WINDOWS_WITH_ACTIONS) | set(WINDOWS_WITHOUT_ACTIONS)) - opened) == []
    assert not set(WINDOWS_WITH_ACTIONS) & set(WINDOWS_WITHOUT_ACTIONS)
    names = {x["name"].split("/")[0] for x in a.call("list_actions", {})}
    assert set(WINDOWS_WITH_ACTIONS.values()) <= names


def test_the_assistant_sets_up_the_psf_calibration_window(control, tmp_path):
    window, a = control
    folder = _bead_folder(tmp_path, n_stacks=1)
    added = act(a, "PSF calibration/Add files", path=str(folder))["answer"]
    assert len(added["files"]) == 1 and "note" not in added
    chosen = act(a, "PSF calibration/Choose", method="average")["answer"]
    assert chosen["method"] == "bead average" and chosen["mode"] == "Single channel"
    assert "beads" in chosen["settings"] and "model" not in chosen["settings"]
    with pytest.raises(Exception, match="method: no 'zernike'"):
        act(a, "PSF calibration/Choose", method="zernike")
    with pytest.raises(Exception, match="part: no 'in situ'"):
        act(a, "PSF calibration/Settings", part="in situ", values={})
    answer = act(a, "PSF calibration/Settings", part="beads", values={"dz_nm": 20})["answer"]
    assert answer["settings"]["beads"]["dz_nm"] == 20
    act(a, "PSF calibration/Settings", part="run", values={"save_model": False})
    w = window.psf_calibration_window
    assert w.run_form.value().save_model is False
    state = act(a, "PSF calibration/State")["answer"]
    assert state["sample"] == "beads in a z stack" and "pupil model" in str(state["choices"])
    assert act(a, "PSF calibration/Remove files")["answer"] == {"files": []}
    with pytest.raises(Exception, match="no files"):
        act(a, "PSF calibration/Detect")


@pytest.mark.slow
def test_the_assistant_sets_a_pages_choices_and_reads_what_each_offers(control, tmp_path):
    """The choices above a page (*section*, *units*, the aberration map's *show*) are the assistant's too."""
    from smappy.plugins import Context, get, settings_from
    window, a = control
    folder = _bead_folder(tmp_path, n_stacks=2)
    cls = get("Localize/PSF calibration")
    cls().run(Context(), settings_from(cls.Settings, {
        "stacks": str(folder), "method": "average", "beads.dz_nm": 20, "beads.roi_size": 21,
        "save_as": str(tmp_path / "average.h5")}))
    act(a, "PSF calibration/Open calibration", path=str(tmp_path / "average.h5"))
    shown = act(a, "PSF calibration/Show page", page="Model vs data", choices={"section": "XZ"})["answer"]
    assert shown["choices"]["section"] == {"chosen": "XZ", "offered": ["XY", "XZ", "YZ"]}
    assert shown["choices"]["average"] is False
    w = window.psf_calibration_window
    title = w.tabs.tabText(w.tabs.currentIndex())
    assert title == "Model vs data"
    with pytest.raises(Exception, match="'ZZ' is not offered"):
        act(a, "PSF calibration/Show page", page="Model vs data", choices={"section": "ZZ"})
    with pytest.raises(Exception, match="no choice 'units'"):
        act(a, "PSF calibration/Show page", page="Model vs data", choices={"units": "nm"})
