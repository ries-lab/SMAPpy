"""*Compare groups*: a region table's measures compared between files."""
import numpy as np
import pytest

from smappy import plugins
from smappy.io.formats import FileInfo
from smappy.locs import Localizations
from smappy.session import Session


def traces(spread, n_traces=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_traces):
        c = rng.uniform(0, 5000, 2)
        p = c + rng.normal(0, spread, (30, 2))
        rows += [(k + 1, x, y, j * 1e-3) for j, (x, y) in enumerate(p)]
    a = np.array(rows)
    return Localizations({"tid": a[:, 0].astype(np.int64), "x_nm": a[:, 1].astype(np.float32),
                          "y_nm": a[:, 2].astype(np.float32), "time_s": a[:, 3],
                          "frame": np.arange(len(a))}, {})


def test_two_files_spread_differently_and_the_medians_say_so():
    session = Session()
    session.add_file(traces(3.0), FileInfo("tight", "/tight", "test"))
    session.add_file(traces(9.0, seed=1), FileInfo("wide", "/wide", "test"), append=True)
    session.show_grouped(0, False)
    session.layers[0].set_files(None)
    ids = plugins.get("Analysis/Measure/Id statistics")
    session.run(ids(), ids.Settings())
    cls = plugins.get("Analysis/Regions/Summarize/Compare groups")
    result = session.run(cls(), cls.Settings(evaluation="per tid: Id statistics",
                                             fields="std_xy_nm"))
    table = result.data["table"]
    assert set(result.data["groups"]) == {"tight", "wide"}
    assert table["median"]["std_xy_nm"]["tight"] == pytest.approx(3.0, rel=0.2)
    assert table["median"]["std_xy_nm"]["wide"] == pytest.approx(9.0, rel=0.2)
    assert table["n"]["tight"] == 20


def test_file_conditions_from_the_names_group_the_traces():
    from smappy.plugins.file_conditions import as_column, conditions_of
    found = conditions_of(["cell1_10uW", "cell2_40uW", "cell3_40uW"], r"(?P<power>\d+)uW",
                          "cell3: buffer=GLOX\ncell1: buffer=MEA\ncell2: buffer=MEA")
    assert found["power"] == ["10", "40", "40"]
    assert as_column(found["buffer"])[1] == ["MEA", "GLOX"]
    session = Session()
    session.add_file(traces(3.0), FileInfo("cell1_10uW", "/a", "test"))
    session.add_file(traces(9.0, seed=1), FileInfo("cell2_40uW", "/b", "test"), append=True)
    session.show_grouped(0, False)
    session.layers[0].set_files(None)
    cond = plugins.get("Analysis/Process/File conditions")
    session.run(cond(), cond.Settings(pattern=r"(?P<power>\d+)uW",
                                      assign="cell1: buffer=MEA\ncell2: buffer=GLOX"))
    ids = plugins.get("Analysis/Measure/Id statistics")
    session.run(ids(), ids.Settings())
    cls = plugins.get("Analysis/Regions/Summarize/Compare groups")
    result = session.run(cls(), cls.Settings(evaluation="per tid: Id statistics",
                                             group_by="attribute:buffer",
                                             fields="std_xy_nm"))
    assert set(result.data["groups"]) == {"MEA", "GLOX"}
    assert result.data["table"]["median"]["std_xy_nm"]["MEA"] == pytest.approx(3.0, rel=0.2)


def test_the_forms_line_about_the_conditions_is_a_notice_the_panel_can_draw():
    from smappy.plugins import Notice
    session = Session()
    session.add_file(traces(3.0), FileInfo("cell1_10uW", "/a", "test"))
    cls = plugins.get("Analysis/Process/File conditions")
    for pattern in (r"(?P<power>\d+)uW", r"(?P<bad"):
        state = cls().describe(session.context(), cls.Settings(pattern=pattern))
        assert isinstance(state.notice, Notice), state.notice
