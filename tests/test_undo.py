"""Undo: several steps back, forward again, and what the history is allowed to cost."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.plugins import Plugin, Result, register
from smappy.session import Session
from smappy.undo import Edit, UndoStack, table_bytes


def table(n=200, seed=0, photons=100.0):
    rng = np.random.default_rng(seed)
    return Localizations({"x_nm": rng.uniform(0, 1000, n),
                          "y_nm": rng.uniform(0, 1000, n),
                          "frame": np.arange(n, dtype=np.int64),
                          "photons": np.full(n, photons),
                          "loc_precision_nm": np.full(n, 10.0)})


def step(session, name, change):
    """One table change through the same door a plugin uses."""
    session.set_locs(change(session.locs), label=name)


def scale_photons(factor):
    def change(locs):
        columns = dict(locs.columns)
        columns["photons"] = locs["photons"] * factor
        return Localizations(columns, dict(locs.metadata))
    return change


def drop_half(locs):
    keep = np.zeros(len(locs), bool)
    keep[::2] = True
    return locs[keep]


# ------------------------------------------------------------ the session

def test_several_steps_go_back_in_order_and_forward_again():
    session = Session(table())
    first = session.locs
    step(session, "double", scale_photons(2))
    second = session.locs
    step(session, "halve the rows", drop_half)
    third = session.locs

    assert session.undo_labels() == ["halve the rows", "double"]
    session.undo()
    assert session.locs is second
    session.undo()
    assert session.locs is first
    assert not session.can_undo and session.can_redo

    session.redo()
    assert session.locs is second
    session.redo()
    assert session.locs is third
    assert not session.can_redo


def test_the_menu_undoes_as_many_steps_as_the_entry_picked():
    session = Session(table())
    first = session.locs
    for n in range(3):
        step(session, f"step {n}", scale_photons(2))
    session.undo(3)                       # the third entry of the submenu
    assert session.locs is first
    assert not session.can_undo
    session.redo(2)
    assert len(session.redo_labels()) == 1


def test_a_new_change_forgets_the_redo_history():
    session = Session(table())
    step(session, "double", scale_photons(2))
    session.undo()
    assert session.can_redo
    step(session, "triple", scale_photons(3))
    assert not session.can_redo and session.undo_labels() == ["triple"]


def test_undoing_a_change_brings_back_the_filters_and_the_grouped_view():
    locs = table()
    session = Session(locs)
    layer = session.layers[0]
    layer.set_bound("photons", 50.0, None)
    layer.show_grouped(True)
    before = session.locs
    step(session, "halve the rows", drop_half)

    session.undo()
    assert session.locs is before
    layer = session.layers[0]
    assert layer.grouped                              # linked again, as it was
    assert layer.state.sets["ungrouped"].filter.ranges["photons"] == (50.0, None)
    assert "photons" in layer.state.sets["grouped"].filter.ranges


def test_the_log_keeps_every_step_and_says_what_was_undone():
    session = Session(table())
    step(session, "double", scale_photons(2))
    session.set_locs(scale_photons(3)(session.locs), label="triple")
    session.undo(2)
    assert session.history[-1]["what"] == "undo"
    assert session.history[-1]["text"] == "triple, double"
    session.redo()
    assert session.history[-1]["what"] == "redo"


def test_a_plugin_run_is_named_after_the_plugin_in_the_menu():
    @register("Analysis/Measure/Undo test")
    class Doubler(Plugin):
        version = "1"

        def run(self, ctx, settings):
            return Result(locs=scale_photons(2)(ctx.locs), text="doubled")

    session = Session(table())
    session.run(Doubler())
    assert session.undo_labels() == ["Undo test"]
    assert session.undo_entries()[0].text == "doubled"   # the menu's tooltip
    session.undo()
    assert session.locs["photons"][0] == pytest.approx(100.0)


def test_a_new_file_starts_the_history_again():
    from smappy.io.formats import FileInfo

    session = Session(table())
    step(session, "double", scale_photons(2))
    assert session.can_undo
    session.add_file(table(50), FileInfo(name="b.hdf5", path="b.hdf5", format="hdf5"))
    assert not session.can_undo            # a different file: nowhere to go back to


def test_a_second_file_is_one_undo_step_that_restores_the_file_list():
    from smappy.io.formats import FileInfo

    session = Session()
    session.add_file(table(100), FileInfo(name="a.hdf5", path="a.hdf5", format="hdf5"))
    session.add_file(table(50, seed=1),
                     FileInfo(name="b.hdf5", path="b.hdf5", format="hdf5"), append=True)
    assert len(session.locs) == 150 and session.file_names() == ["a.hdf5", "b.hdf5"]

    session.undo()
    assert len(session.locs) == 100 and session.file_names() == ["a.hdf5"]


# -------------------------------------------------------------- the stack

def test_a_shared_column_is_counted_once():
    locs = table(10_000)
    changed = scale_photons(2)(locs)              # one new column, the rest shared
    whole = table_bytes(locs)
    seen = set()
    assert table_bytes(locs, seen) == whole
    one_column = table_bytes(changed, seen)
    assert one_column == locs["photons"].nbytes
    assert one_column < whole / 4


def test_the_cap_drops_the_oldest_but_always_keeps_one_step():
    one = table_bytes(table(10_000))
    stack = UndoStack(limit=one * 2)              # room for two of these
    for n in range(4):
        stack.push(Edit(label=f"step {n}", locs=table(10_000, seed=n)))
    assert stack.labels() == ["step 3", "step 2"]

    stack = UndoStack(limit=1)                    # room for none of them
    for n in range(4):
        stack.push(Edit(label=f"step {n}", locs=table(10_000, seed=n)))
    assert len(stack) == 1 and stack.labels() == ["step 3"]


def test_the_step_count_is_capped_whatever_the_memory_allows():
    stack = UndoStack(limit=10**9, max_steps=3)
    for n in range(6):
        stack.push(Edit(label=f"step {n}", locs=table(10)))
    assert stack.labels() == ["step 5", "step 4", "step 3"]


def test_the_redo_side_is_dropped_before_a_step_that_was_actually_taken():
    stack = UndoStack(limit=10**9, max_steps=4)
    for n in range(3):
        stack.push(Edit(label=f"step {n}", locs=table(10)))
    stack.undo(redo_of=Edit(label="step 2", locs=table(10)))
    stack.limit = 1                               # the machine filled up
    stack.trim()
    assert not stack.can_redo and len(stack) == 1


def test_the_configuration_sizes_the_stack():
    from smappy import config
    from smappy.undo import DEFAULT_FRACTION, configured

    assert configured().fraction == DEFAULT_FRACTION
    config.set("undo_fraction", 0.1)
    config.set("undo_steps", 4)
    try:
        stack = configured()
        assert stack.fraction == 0.1 and stack.max_steps == 4
        config.set("undo_steps", "nonsense")       # a hand-edited file
        assert configured().max_steps == 20
    finally:
        config.set("undo_fraction", None)
        config.set("undo_steps", None)


def test_the_memory_probe_answers_on_this_machine():
    from smappy.undo import budget, memory

    total, available = memory()
    assert total > 0 and 0 < available <= total
    assert budget(0.25) >= 512 * 1024 * 1024


# ---------------------------------------------------------------- the menu

def test_the_file_menu_lists_the_steps_and_undoes_as_many_as_are_picked():
    pytest.importorskip("PySide6")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from PySide6.QtWidgets import QApplication
    from smappy.gui.app import ControlWindow, RenderWindow

    QApplication.instance() or QApplication([])
    session = Session(table())
    first = session.locs
    control = ControlWindow(session, RenderWindow(session))

    assert control.undo_action.text() == "Undo" and not control.undo_action.isEnabled()
    step(session, "double", scale_photons(2))
    step(session, "triple", scale_photons(3))

    assert control.undo_action.text() == "Undo triple"
    entries = control.undo_menu.actions()
    assert [a.text() for a in entries] == ["triple", "double  (2)"]

    entries[1].trigger()                      # "undo the last two"
    assert session.locs is first
    assert not control.undo_action.isEnabled()
    assert control.redo_action.text() == "Redo double"
