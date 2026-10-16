"""Each file of a session keeps its own metadata.

A session holds one table however many files are open, and the table's
metadata is the first file's.  What a second file said about itself -- its
camera, its frame time, a MINFLUX measurement's sequence -- is kept in its
`FileInfo.metadata`, written with the session (``metadata["files"]``), put
back by the undo, and read with `metadata.file_metadata`.  A file saved from
a session of several opens as those files again.
"""
import numpy as np

from smappy.io.formats import FileInfo
from smappy.locs import Localizations
from smappy.metadata import file_metadata, file_frame_time
from smappy.session import Session


def table(n, who, frame_time):
    return Localizations({"x_nm": np.arange(n, dtype=np.float32),
                          "y_nm": np.zeros(n, np.float32), "frame": np.arange(n)},
                         {"camera": {"frame_time_ms": frame_time}, "who": who,
                          "history": [{"what": "a fit"}]})


def two_files():
    session = Session()
    session.add_file(table(3, "A", 10), FileInfo("a", "a.csv", "csv"))
    session.add_file(table(2, "B", 30), FileInfo("b", "b.csv", "csv"), append=True)
    return session


def test_an_added_file_keeps_its_own_metadata_beside_the_first_files():
    session = two_files()
    meta = session.locs.metadata
    assert meta["who"] == "A"                                  # the table's: the first's
    assert file_metadata(meta, 0)["who"] == "A" and file_metadata(meta, 1)["who"] == "B"
    assert file_frame_time(meta, 1) == 30
    assert "history" not in session.files[1].metadata         # the session's, not the file's


def test_the_files_metadata_comes_back_with_the_undo():
    session = two_files()
    session.remove_file(1)
    assert [f.name for f in session.files] == ["a"]
    session.undo()
    assert [f.name for f in session.files] == ["a", "b"]
    assert file_metadata(session.locs.metadata, 1)["who"] == "B"


def test_a_session_of_several_files_saved_opens_as_those_files(tmp_path):
    session = two_files()
    path = tmp_path / "both.h5"
    session.save(path)
    again = Session()
    again.load(path)
    assert [f.name for f in again.files] == ["a", "b"]
    assert np.bincount(again.locs["filenumber"]).tolist() == [3, 2]
    assert file_metadata(again.locs.metadata, 1)["who"] == "B"
    assert [f.frame_time_ms for f in again.files] == [10, 30]
    # and added to another file, they follow it with their own numbers
    third = Session()
    third.add_file(table(4, "C", 5), FileInfo("c", "c.csv", "csv"))
    third.load(path, append=True)
    assert [f.name for f in third.files] == ["c", "a", "b"]
    assert np.bincount(third.locs["filenumber"]).tolist() == [4, 3, 2]
