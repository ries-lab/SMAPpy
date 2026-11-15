import numpy as np

from smappy import plugins
from smappy.file_ids import keep_apart, own_ids
from smappy.io.formats import FileInfo
from smappy.locs import Localizations
from smappy.region_results import Evaluation, rows
from smappy.session import Session


def traces(n_traces, start=1, seed=0, x0=0.0):
    """A MINFLUX-like file: ``n_traces`` traces of 10 localizations each,
    numbered from ``start``."""
    rng = np.random.default_rng(seed)
    tid = np.repeat(np.arange(start, start + n_traces), 10)
    centres = rng.uniform(0, 5000, (n_traces, 2)) + x0
    xy = centres[tid - start] + rng.normal(0, 3, (len(tid), 2))
    return Localizations({"x_nm": xy[:, 0], "y_nm": xy[:, 1],
                          "time_s": np.arange(len(tid)) * 1e-3,
                          "frame": np.arange(len(tid)),
                          "tid": tid.astype(np.int64)}, {})


def test_overlapping_ids_are_moved_past_the_tables_and_the_files_own_kept():
    old, new = traces(5), traces(3)
    old2, new2, offsets = keep_apart(old, new)
    assert offsets == {"tid": 6}
    assert set(new2["tid"]) == {7, 8, 9} and not set(new2["tid"]) & set(old2["tid"])
    assert np.array_equal(new2["tid_file"], new["tid"])
    assert np.array_equal(old2["tid_file"], old["tid"])
    assert np.array_equal(own_ids(new2, "tid"), new["tid"])


def test_ids_that_do_not_overlap_are_left_alone():
    old, new = traces(5), traces(3, start=10)
    _, new2, offsets = keep_apart(old, new)
    assert offsets == {} and new2 is new and "tid_file" not in new2


def test_an_id_of_minus_one_meaning_none_stays_none():
    def table(ids):
        return Localizations({"x_nm": np.zeros(len(ids)),
                              "cluster_id": np.asarray(ids, np.int64)}, {})
    _, new, offsets = keep_apart(table([-1, 0, 2]), table([-1, 0, 1, 3]))
    assert offsets == {"cluster_id": 3}
    assert list(new["cluster_id"]) == [-1, 3, 4, 6]
    assert list(new["cluster_id_file"]) == [-1, 0, 1, 3]


def test_two_files_in_one_layer_give_one_region_per_trace_of_each():
    session = Session()
    session.add_file(traces(5, seed=1), FileInfo("a", "/a.msr", "Abberior MSR"))
    info = session.add_file(traces(3, seed=2, x0=10000), FileInfo("b", "/b.msr", "Abberior MSR"),
                            append=True)
    assert info.metadata["id_offsets"] == {"tid": 6}
    second = np.asarray(session.locs["filenumber"]) == 1
    assert list(np.unique(session.locs["tid_file"][second])) == [1, 2, 3]
    session.show_grouped(0, False)
    session.layers[0].set_files(None)
    cls = plugins.get("Analysis/Measure/Id statistics")
    session.run(cls(), cls.Settings())
    got = list(rows(session.context(), Evaluation("tid", "Id statistics")))
    # eight traces of 10, none of the first file's merged with the second's
    assert len(got) == 8 and all(r["n_locs"] == 10 for r in got)
    assert max(r["std_xy_nm"] for r in got) < 10
