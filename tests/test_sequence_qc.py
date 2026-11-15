"""*Sequence QC* (`minflux_qc`, `plugins/sequence_qc.py`): simulated
attempts with known pass rates, reasons, bursts and scatter come back."""
import os
from pathlib import Path

import numpy as np
import pytest

from smappy import minflux_qc as q
from smappy import plugins
from smappy.io import minflux
from smappy.locs import Localizations
from smappy.session import Session
from smappy.simulate import simulate_attempts

SEQ = {"iterations": [{"Mode.pattern": "hexagon", "patGeoFactor": g}
                      for g in (0.79, 0.84, 0.42, 0.28)]}
PASSED = (0.1, 0.9, 0.6)


@pytest.fixture(scope="module")
def table():
    rec = simulate_attempts(6000, passed=PASSED, locs_per_burst=12.0, seed=3)
    locs, _ = minflux.table(rec, sequence=SEQ, every=True)
    return locs


def col(locs, name):
    return np.asarray(locs[name])


def test_the_fraction_of_attempts_passing_each_iteration_comes_back(table):
    ev = q.iteration_events(col(table, "iteration"), col(table, "sta"), col(table, "final"),
                            col(table, "vld"))
    assert ev.final == 3
    expected = np.cumprod(PASSED)
    got = ev.passed_fraction()
    for k, f in enumerate(expected):
        assert got[k] == pytest.approx(f, abs=4 * np.sqrt(f / 6000))
    # the last catching iteration loses 80 % dark and 20 % to the cfr check
    lost = ev.counts["dark"][2] + ev.counts["cfr"][2]
    assert ev.counts["cfr"][2] / lost == pytest.approx(0.2, abs=0.07)


def test_bursts_their_localizations_photons_and_why_they_ended(table):
    tr = q.traces(col(table, "tid"), col(table, "sta"), col(table, "final"),
                  col(table, "time_s"), None, col(table, "photons"))
    valid = tr.valid()
    assert q.exponential_mean(tr.localizations[valid]) == pytest.approx(12.0, rel=0.15)
    assert (tr.reason[valid] == "dark").all()
    assert (tr.photons[valid] > 0).all()


def test_the_catching_iterations_scatter_about_the_localization(table):
    off = q.catching_offsets(col(table, "iteration"), col(table, "final"), col(table, "tid"),
                             col(table, "trace_index"),
                             np.c_[col(table, "x_nm"), col(table, "y_nm")])
    assert off["iterations"].tolist() == [0, 1, 2]
    for k, sigma in enumerate((60.0, 20.0, 8.0)):
        # the localization itself scatters by 4 nm about the molecule too
        assert off["std"][k].mean() == pytest.approx(np.hypot(sigma, 4.0), rel=0.25)
        assert np.abs(off["mean"][k]).max() < 3 * sigma / np.sqrt(max(off["n"][k], 1)) + 1


def test_the_time_divides_among_the_iterations(table):
    times = q.time_per_iteration(col(table, "iteration"), col(table, "time_s"))
    assert times["fraction"].sum() == pytest.approx(1.0)
    assert times["iterations"].tolist() == [0, 1, 2, 3]


def test_the_plugin_reads_every_row_of_the_file_whatever_the_filter(table):
    session = Session(table)
    session.show_grouped(0, False)
    assert int(np.asarray(session.context().selection.mask).sum()) == int(col(table, "final").sum())
    cls = plugins.get("Analysis/MINFLUX/Sequence QC")
    result = session.run(cls(), cls.Settings())
    summary = result.data["summary"]
    assert summary["scouting_events"] == 6000
    assert summary["passed_itr0"] == pytest.approx(0.1, abs=0.02)
    assert "Sequence QC" in session.id_regions["tid"]
    assert "attempts" in result.text


def test_the_plugin_asks_for_every_iteration_on_a_table_without():
    locs = Localizations({"x_nm": np.zeros(3), "y_nm": np.zeros(3), "tid": np.arange(3),
                          "frame": np.arange(3)}, {})
    cls = plugins.get("Analysis/MINFLUX/Sequence QC")
    with pytest.raises(ValueError, match="every iteration"):
        Session(locs).run(cls(), cls.Settings())


def test_both_layouts_codes_mean_the_same():
    assert q.outcome([1, 2, 6, 8, 9, 0, 4089, 4096, 4100]).tolist() == [
        "passed", "passed", "cfr", "dark", "out of range",
        "passed", "cfr", "dark", "out of range"]


MSRS = os.environ.get("SMAPPY_TEST_MSR")


@pytest.mark.skipif(not MSRS, reason="SMAPPY_TEST_MSR=dir of real .msr files")
def test_real_measurements_give_a_cascade_of_attempts():
    for path in sorted(Path(MSRS).glob("*.msr")):
        session = Session()
        session.load(path, measurements=[1], every=True)
        cls = plugins.get("Analysis/MINFLUX/Sequence QC")
        summary = session.run(cls(), cls.Settings()).data["summary"]
        fractions = [v for k, v in summary.items() if k.startswith("passed_itr")]
        assert fractions and all(0 < f <= 1 for f in fractions), path.name
        assert fractions == sorted(fractions, reverse=True), path.name
