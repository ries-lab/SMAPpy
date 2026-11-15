import numpy as np

from smappy import plugins
from smappy.ids import NONE, assigned, is_id
from smappy.locs import Localizations, concat
from smappy.region_results import Evaluation, rows
from smappy.session import Session


def trace_table(tids):
    """MINFLUX-like rows: one localization a row, ``tids`` the traces, as
    Imspector numbers them (from 0)."""
    tids = np.asarray(tids, np.int64)
    return Localizations({"x_nm": tids * 1000.0, "y_nm": np.zeros(len(tids)),
                          "time_s": np.arange(len(tids)) * 1e-3,
                          "frame": np.arange(len(tids)), "tid": tids}, {})


def test_minus_one_is_none_and_zero_is_an_id():
    assert NONE == -1
    assert assigned(np.array([-1, 0, 3])).tolist() == [False, True, True]
    assert assigned(np.array([np.nan, -1.0, 0.0])).tolist() == [False, False, True]
    assert is_id("cluster_id") and is_id("tid") and is_id("tid_file")
    assert not is_id("channel") and not is_id("frame")


def test_the_first_minflux_trace_is_a_region_of_id_statistics():
    session = Session(trace_table([0, 0, 0, 1, 1, 1]))
    session.show_grouped(0, False)
    cls = plugins.get("Analysis/Measure/Id statistics")
    session.run(cls(), cls.Settings())
    got = {str(r["roi_id"]): r for r in rows(session.context(), Evaluation("tid", "Id statistics"))}
    assert set(got) == {"0", "1"} and got["0"]["n_locs"] == 3


def test_a_missing_id_column_is_padded_with_none_and_any_other_with_zero():
    a = Localizations({"x_nm": np.zeros(2), "frame": np.arange(2),
                       "cluster_id": np.array([0, 1])}, {})
    b = Localizations({"x_nm": np.zeros(3)}, {})
    both = concat([a, b])
    assert both["cluster_id"].tolist() == [0, 1, -1, -1, -1]
    assert both["frame"].tolist() == [0, 1, 0, 0, 0]
