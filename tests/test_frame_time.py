"""A file's frame time: recorded by the simulators, set by hand where a file
has none, and asked for by the plugins that turn frames into seconds."""
from __future__ import annotations

import numpy as np
import pytest

from smappy.io.formats import FileInfo
from smappy.locs import Localizations
from smappy.metadata import file_frame_time, table_frame_time
from smappy.plugins import NO_FRAME_TIME
from smappy.session import Session


def _bare(n_frames=200, seed=0):
    """A table that says nothing about time: one bead's track, ids 1."""
    rng = np.random.default_rng(seed)
    frame = np.arange(n_frames, dtype=np.int32)
    return Localizations({"x_nm": rng.normal(0, 3, n_frames),
                          "y_nm": rng.normal(0, 3, n_frames),
                          "frame": frame, "photons": np.full(n_frames, 1000.0),
                          "xy_err_nm": np.full(n_frames, 3.0),
                          "track_id": np.ones(n_frames, np.int32)}, {})


def test_the_structure_simulation_records_its_frame_time():
    from smappy.simulate import SimulationSettings, ground_truth, localizations
    settings = SimulationSettings(n_frames=200)
    settings.frame_time_ms = 25.0
    locs = localizations(ground_truth(settings))
    assert table_frame_time(locs) == 25.0


def test_the_simulation_plugin_hands_its_frame_time_to_the_session():
    from smappy.plugins.file import SimulateBlinks
    settings = SimulateBlinks.Settings(n_frames=200)
    settings.frame_time_ms = 30.0
    session = Session()
    session.run(SimulateBlinks(), settings)
    assert session.files[0].frame_time_ms == 30.0
    assert table_frame_time(session.locs) == 30.0


def test_a_frame_time_set_for_a_file_is_read_by_every_table_of_it():
    session = Session()
    session.add_file(_bare(), FileInfo(name="bare", path="bare", format="test"))
    session.add_file(_bare(seed=1), FileInfo(name="second", path="second", format="test"),
                     append=True)
    assert file_frame_time(session.locs.metadata, 0) is None
    grouped = session.tables.grouped()
    session.set_frame_time(1, 50.0)
    assert file_frame_time(session.locs.metadata, 1) == 50.0
    assert file_frame_time(session.locs.metadata, 0) is None
    # the grouped table shares the file list rather than waiting for a relink
    assert file_frame_time(grouped.locs.metadata, 1) == 50.0
    assert session.unsaved
    session.set_frame_time(1, None)
    assert file_frame_time(session.locs.metadata, 1) is None


def test_a_plugin_that_needs_seconds_says_where_the_frame_time_is_set():
    from smappy.plugins.stability import Stability
    session = Session()
    session.add_file(_bare(), FileInfo(name="bare", path="bare", format="test"))
    ctx = session.context()
    state = Stability().describe(ctx, Stability.Settings())
    assert state.notice is not None and state.notice.window == "file properties"
    with pytest.raises(ValueError, match="File properties"):
        Stability().run(ctx, Stability.Settings())
    session.set_frame_time(0, 10.0)
    state = Stability().describe(session.context(), Stability.Settings())
    assert state.notice is None
    assert state.hints["frame_time_ms"] == "10"
    assert "File > File properties" in NO_FRAME_TIME


def test_the_file_properties_dialog_sets_the_frame_times_that_changed():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from smappy.gui.dialogs import FilePropertiesDialog
    QApplication.instance() or QApplication([])
    session = Session()
    session.add_file(_bare(), FileInfo(name="bare", path="bare", format="test"))
    session.add_file(_bare(seed=1), FileInfo(name="second", path="second", format="test",
                                             frame_time_ms=20.0), append=True)
    seen = []
    session.on_change(seen.append)
    dialog = FilePropertiesDialog(session)
    assert dialog.times[0].value() == 0 and dialog.times[1].value() == 20.0
    dialog.times[0].setValue(12.5)
    dialog.apply()
    assert [f.frame_time_ms for f in session.files] == [12.5, 20.0]
    assert seen.count("files") == 1          # the unchanged file is left alone
    dialog.deleteLater()
