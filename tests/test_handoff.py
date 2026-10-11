"""*Open in a new window*: a session handed to a new instance of the program.

The table goes through shared memory and the rest as JSON (`smappy.handoff`);
these check that what arrives is the session that was handed over -- table,
layers, filters, name -- and that nothing is left behind, a live fit's table
least of all.
"""
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from smappy import handoff
from smappy.locs import Localizations
from smappy.session import Session


def block(frame, n=400):
    rng = np.random.default_rng(frame)
    return Localizations({"x_nm": rng.uniform(0, 5000, n),
                          "y_nm": rng.uniform(0, 5000, n),
                          "frame": np.full(n, frame, np.int64),
                          "photons": rng.uniform(100, 2000, n).astype(np.float32),
                          "xy_err_nm": rng.uniform(5, 20, n)}, {})


def live_session():
    session = Session()
    session.begin_live((0, 5000, 0, 5000))
    for frame in range(1, 5):
        session.append(block(frame))
    return session


def test_a_live_session_is_handed_over_as_it_was_and_carries_on_growing(tmp_path):
    session = live_session()
    session.add_layer()
    session.layers[1].set_bound("photons", 1000, 2000)
    session.layers[1].name = "bright"
    session.rename("cells", colour="blue")
    name, colour = handoff.snapshot_identity(session)
    assert name == "cells, snapshot at frame 4" and colour == "orange"

    offer = handoff.offer(session, name=name, colour=colour, folder=tmp_path)
    session.append(block(5))                      # the fit goes on meanwhile
    locs, state, grouped = handoff.take(offer.path)
    assert offer.taken
    offer.release()

    copy = Session()
    handoff.apply_state(copy, locs, state, grouped)
    assert len(copy.locs) == 1600 and len(session.locs) == 2000 and session.live
    np.testing.assert_array_equal(copy.locs["x_nm"], session.locs["x_nm"][:1600])
    assert copy.locs["photons"].dtype == np.float32
    assert (copy.name, copy.colour) == (name, "orange")
    assert [l.name for l in copy.layers] == ["layer 1", "bright"]
    assert copy.layers[1].state.sets["ungrouped"].filter.ranges["photons"] == (1000, 2000)
    assert copy.path is None and copy.history[-1]["what"] == "copy"
    assert not copy.live


def test_an_offer_nobody_takes_is_released_and_leaves_nothing(tmp_path):
    from multiprocessing import shared_memory
    offer = handoff.offer(live_session(), folder=tmp_path)
    block_name = offer.block.name
    offer.release()
    offer.release()                               # twice is harmless
    assert not offer.path.exists()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=block_name)


def test_a_handoff_of_another_version_is_refused(tmp_path):
    import json
    offer = handoff.offer(live_session(), folder=tmp_path)
    state = json.loads(offer.path.read_text())
    state["version"] = 99
    offer.path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="version 99"):
        handoff.receive(offer.path)
    offer.release()


@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def wait_for(condition, app, seconds=20.0):
    end = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > end:
            raise AssertionError("timed out")
        app.processEvents()
        time.sleep(0.01)


def test_the_control_window_hands_its_session_to_another(app, tmp_path):
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.gui.identity import uninstall_identity
    session = live_session()
    windows = []
    try:
        render = RenderWindow(session)
        control = ControlWindow(session, render)
        windows += [control, render]
        offer = control.open_in_new_window(start=False)
        assert session.colour == "blue"           # told apart from its copy
        other = Session()
        other_render = RenderWindow(other)
        other_control = ControlWindow(other, other_render)
        windows += [other_control, other_render]
        other_control.take_handoff(str(offer.path))
        wait_for(lambda: len(other.locs) == 1600 and other_control._task is None, app)
        assert other.colour == "orange" and "snapshot" in other.name
        wait_for(lambda: not control._offers, app)
        assert offer.block is None
    finally:
        for w in windows:
            w.close()
        uninstall_identity(app)


@pytest.mark.slow
def test_a_new_instance_starts_and_takes_the_session(tmp_path):
    """The whole path: the program started as *Open in a new window* starts
    it, taking the session it was handed.  Ended once it has."""
    pytest.importorskip("PySide6")
    offer = handoff.offer(live_session(), name="copy", colour="green",
                          folder=tmp_path)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               SMAPPY_CONFIG_DIR=str(tmp_path / "config"))
    process = subprocess.Popen([sys.executable, "-m", "smappy.cli.gui",
                                "--handoff", str(offer.path)], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        end = time.monotonic() + 120
        while not offer.taken:
            assert process.poll() is None, process.stdout.read().decode()[-2000:]
            assert time.monotonic() < end, "the new instance did not take it"
            time.sleep(0.1)
    finally:
        process.kill()
        process.wait()
        offer.release()


def test_a_run_handed_over_is_run_by_the_new_instance(app, tmp_path, monkeypatch):
    """What could not be applied during the live fit is applied in the copy."""
    import dataclasses
    from smappy import plugins
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.gui.identity import uninstall_identity
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    session = live_session()
    cls = plugins.get("Analysis/Process/Math Parser")
    settings = dataclasses.replace(cls.Settings(), field="bright",
                                   expression="photons > 1000")
    windows = []
    try:
        render = RenderWindow(session)
        control = ControlWindow(session, render)
        offer = control.open_in_new_window(start=False, run=(cls(), settings))
        other = Session()
        other_render = RenderWindow(other)
        other_control = ControlWindow(other, other_render)
        windows += [control, render, other_control, other_render]
        other_control.take_handoff(str(offer.path))
        wait_for(lambda: "bright" in other.locs, app)
        assert "bright" not in session.locs and session.live
        expected = np.asarray(other.locs["photons"]) > 1000
        np.testing.assert_array_equal(np.asarray(other.locs["bright"]) > 0.5, expected)
        windows += list(other_control._plugin_windows.values())
    finally:
        for w in windows:
            w.close()
        uninstall_identity(app)
