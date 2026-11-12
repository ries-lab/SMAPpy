"""The assistant inside SMAPpy's own windows (`gui/window_actions.py`): the bead calibration it
could open and not use (a user's report, 2026-11-12), and the PSF calibration."""
import time

import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def wait_for(condition, seconds=60.0):
    from PySide6.QtWidgets import QApplication
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


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
    for w in (window.calibration_window, window.psf_calibration_window):
        # a job's thread is still winding down after its `done`: one destroyed running aborts
        thread = getattr(w, "_thread", None)
        if thread is not None:
            thread.quit()
            thread.wait()
    for w in (window.calibration_window, window.psf_calibration_window, render, window):
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
    """`open_window` alone opened the bead calibration and left the assistant outside it."""
    from smappy.gui.window_actions import WINDOWS_WITH_ACTIONS, WINDOWS_WITHOUT_ACTIONS
    window, a = control
    opened = set(window._assistant_windows())
    assert sorted(opened - set(WINDOWS_WITH_ACTIONS) - set(WINDOWS_WITHOUT_ACTIONS)) == []
    assert sorted((set(WINDOWS_WITH_ACTIONS) | set(WINDOWS_WITHOUT_ACTIONS)) - opened) == []
    assert not set(WINDOWS_WITH_ACTIONS) & set(WINDOWS_WITHOUT_ACTIONS)
    names = {x["name"].split("/")[0] for x in a.call("list_actions", {})}
    assert set(WINDOWS_WITH_ACTIONS.values()) <= names


def test_the_assistant_calibrates_beads_in_the_bead_calibration_window_and_saves_it(control, tmp_path):
    window, a = control
    names = [x["name"] for x in a.call("list_actions", {"prefix": "Bead calibration"})]
    assert {"Bead calibration/Add files", "Bead calibration/Detect and calibrate",
            "Bead calibration/Save"} <= set(names)
    with pytest.raises(Exception, match="no bead stacks"):
        act(a, "Bead calibration/Detect and calibrate")
    added = act(a, "Bead calibration/Add files", path=str(_bead_folder(tmp_path)))["answer"]
    assert len(added["files"]) == 2
    assert window.calibration_window.isVisible()
    with pytest.raises(Exception, match="has no 'bead_size'"):
        act(a, "Bead calibration/Settings", values={"bead_size": 3})
    settings = act(a, "Bead calibration/Settings", values={"dz_nm": 20})["answer"]["settings"]
    assert settings["dz_nm"] == 20

    started = act(a, "Bead calibration/Detect and calibrate")["answer"]
    assert started["started"]
    w = window.calibration_window
    assert wait_for(lambda: not w.busy, 120)
    state = act(a, "Bead calibration/State")["answer"]
    assert "error" not in state, state
    n = state["result"]["beads"]
    assert n >= 4 and 0 < state["result"]["used"] <= n
    assert state["beads"]["columns"][0] == "bead" and len(state["beads"]["rows"]) == n

    act(a, "Bead calibration/Exclude beads", beads=[0, 1])
    act(a, "Bead calibration/Recalculate")
    assert wait_for(lambda: not w.busy, 120)
    state = act(a, "Bead calibration/State")["answer"]
    assert state["result"]["excluded by hand"] == [0, 1]
    assert act(a, "Bead calibration/Show page", page="bead diag")["answer"] == {
        "page": "Bead diagnostics"}

    # a file is written only by a proposal in suggest mode, and only where the assistant may
    a.mode = "suggest"
    assert act(a, "Bead calibration/Save")["proposed"]["tool"] == "do_action"
    assert act(a, "Bead calibration/State")["answer"]["busy"] is False      # reading is not proposed
    a.mode = "act"
    with pytest.raises(Exception, match="outside the output folder"):
        act(a, "Bead calibration/Save", path=str(tmp_path / "elsewhere" / "cal.h5"))
    seen = []
    w.calibrated.connect(lambda path, dual: seen.append((path, dual)))
    saved = act(a, "Bead calibration/Save", path="cal")["answer"]
    assert saved["saved"] == str(a.output / "cal.h5") and (a.output / "cal.h5").exists()
    assert seen == [(saved["saved"], False)] and "Spline 3D fitter" in saved["fitter"]


def test_a_failed_job_the_assistant_started_is_said_in_state_not_in_a_box(control, tmp_path):
    """A modal box would hold the assistant's queue on the GUI thread until someone closed it."""
    window, a = control
    junk = tmp_path / "junk.tif"
    tifffile.imwrite(junk, np.zeros((3, 4, 4), np.uint16))
    act(a, "Bead calibration/Add files", path=str(junk))
    act(a, "Bead calibration/Detect and calibrate")
    w = window.calibration_window
    assert wait_for(lambda: not w.busy, 60)
    from PySide6.QtWidgets import QApplication
    assert QApplication.activeModalWidget() is None
    assert act(a, "Bead calibration/State")["answer"]["error"]


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
