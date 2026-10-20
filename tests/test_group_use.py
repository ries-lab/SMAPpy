"""`group_use`: which localizations take part in merging blinks, in a
session -- the merged table follows the column as Math Parser makes,
changes and undoes it, file by file too."""
import numpy as np

from smappy import plugins
from smappy.io.formats import FileInfo
from smappy.locs import Localizations
from smappy.session import Session


def traces(file=0):
    tid = np.repeat([1, 2, 3], [5, 4, 2])
    index = np.concatenate([np.arange(1, 6), np.arange(1, 5), np.arange(1, 3)])
    x = tid * 1000.0 + np.where(index <= 2, 100.0, 0.0) + file * 10000.0
    return Localizations({"x_nm": x, "y_nm": np.zeros(11), "frame": np.arange(11),
                          "tid": tid, "trace_index": index, "xy_err_nm": np.ones(11),
                          "photons": np.full(11, 100.0)})


def test_the_merged_table_follows_group_use_as_math_parser_makes_and_undoes_it():
    session = Session()
    session.add_file(traces(), FileInfo("a", "/a.csv", "csv"))
    before = np.asarray(session.tables.grouped().locs["x_nm"])
    assert len(before) == 3 and before[0] > 1000.0              # the start pulls it
    parser = plugins.get("Analysis/Process/Math Parser")
    session.run(parser(), parser.Settings(field="group_use", expression="trace_index > 2"))
    assert np.allclose(session.tables.grouped().locs["x_nm"], [1000.0, 2000.0])
    assert len(session.locs) == 11
    session.run(parser(), parser.Settings(field="group_use", expression="trace_index > 3"))
    assert list(session.tables.grouped().locs["n_in_group"]) == [2, 1]
    session.undo()
    session.undo()
    assert np.allclose(session.tables.grouped().locs["x_nm"], before)


def test_files_linked_one_by_one_leave_the_same_rows_out():
    session = Session()
    session.add_file(traces(0), FileInfo("a", "/a.csv", "csv"))
    session.add_file(traces(1), FileInfo("b", "/b.csv", "csv"), append=True)
    assert session.tables._per_file()
    parser = plugins.get("Analysis/Process/Math Parser")
    session.run(parser(), parser.Settings(field="group_use", expression="trace_index > 2"))
    grouped = session.tables.grouped(files=[0, 1]).locs
    assert np.allclose(np.sort(grouped["x_nm"]), [1000.0, 2000.0, 11000.0, 12000.0])
    gid = np.asarray(session.locs["group_id"])
    start = np.asarray(session.locs["trace_index"]) <= 2
    assert (gid[start] == 0).all() and (gid[~start] > 0).all()
    # each localization's blink is the grouped row with its id
    rows = {int(g): k for k, g in enumerate(grouped["group_id"])}
    x = np.asarray(grouped["x_nm"])
    assert all(abs(x[rows[int(g)]] - xi) < 1 for g, xi in
               zip(gid[~start], np.asarray(session.locs["x_nm"])[~start]))
