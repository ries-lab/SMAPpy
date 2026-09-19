"""The record a file carries: what was done to it, and what it exports as."""
import json

import numpy as np
import pytest

from smappy.group import GroupSettings
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.history import (History, HistorySettings, as_csv, as_text,
                                    entries, write)
from smappy.plugins.math_parser import MathParser, MathSettings
from smappy.session import Session


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    return tmp_path / "config"


def table(n=12):
    return Localizations({
        "x_nm": np.repeat([100.0, 500.0, 900.0], n // 3).astype(np.float32),
        "y_nm": np.zeros(n, np.float32),
        "frame": np.arange(n, dtype=np.int64) % (n // 3),
        "photons": np.linspace(100, 1200, n).astype(np.float32),
    }, {"units": "nm"})


def session_with_a_run(config_dir):
    session = Session(table())
    session.run(MathParser(), MathSettings(field="double", expression="photons * 2"))
    return session


def test_a_run_that_changed_the_table_is_marked_as_one(config_dir):
    """The log is a record of the session; the marked entries are the record
    of the data."""
    session = session_with_a_run(config_dir)
    session.run(History(), HistorySettings())        # shows, changes nothing
    changed = [e["what"] for e in session.history if e["changed"]]
    assert changed == ["Analysis/Process/Math Parser"]
    # and looking at the log is not an entry in it, which would otherwise put
    # a copy of the whole log inside it on every run
    assert [e["what"] for e in session.history] == changed
    assert entries(session, changes_only=True) == [
        e for e in session.history if e["what"].endswith("Math Parser")]


def test_relinking_is_logged_with_the_parameters_it_used():
    """Which dx and dt produced the grouped table cannot be recovered from the
    file afterwards, so it goes in the record."""
    session = Session(table())
    session.set_group_settings(GroupSettings(dx=40.0, dt=2))
    entry = session.history[-1]
    assert entry["what"] == "regroup" and entry["changed"]
    assert entry["settings"]["dx"] == 40.0 and entry["settings"]["dt"] == 2
    assert "dx = 40" in entry["text"]


def test_the_log_is_shown_with_the_settings_of_each_run(config_dir):
    session = session_with_a_run(config_dir)
    result = session.run(History(), HistorySettings())
    assert "Analysis/Process/Math Parser" in result.text
    assert "expression = photons * 2" in result.text
    assert "1 of 1 entries changed" in result.text
    assert result.locs is None                       # it reads, it writes nothing

    without = as_text(entries(session), settings=False)
    assert "Analysis/Process/Math Parser" in without
    assert "expression = photons * 2" not in without


def test_an_empty_log_says_so_rather_than_printing_nothing():
    assert "nothing has been done" in as_text([])
    assert entries(None) == []                       # a script with no session


def test_the_export_writes_the_form_the_extension_asks_for(tmp_path, config_dir):
    session = session_with_a_run(config_dir)
    found = entries(session)

    rows = as_csv(found).splitlines()
    assert rows[0] == "time,what,changed,text,settings"
    assert "Analysis/Process/Math Parser" in rows[1]

    as_json = json.loads(write(found, tmp_path / "log.json").read_text())
    assert as_json[0]["settings"]["expression"] == "photons * 2"

    import yaml
    as_yaml = yaml.safe_load(write(found, tmp_path / "log.yaml").read_text())
    assert as_yaml[0]["what"] == "Analysis/Process/Math Parser"

    assert "1 of 1 entries changed" in write(found, tmp_path / "log.txt").read_text()


def test_the_plugin_exports_when_it_is_given_a_path(tmp_path, config_dir):
    session = session_with_a_run(config_dir)
    path = tmp_path / "provenance.csv"
    result = session.run(History(), HistorySettings(path=str(path)))
    assert result.data["path"] == path and path.exists()
    assert "written to" in result.text
    assert "photons * 2" in path.read_text()


def test_the_log_of_a_file_is_what_the_plugin_shows_after_reopening_it(
        tmp_path, config_dir):
    session = session_with_a_run(config_dir)
    path = session.save(tmp_path / "t.h5")

    reopened = Session()
    reopened.load(path)
    text = reopened.run(History(), HistorySettings(changes_only=True)).text
    assert "expression = photons * 2" in text        # from before it was saved

    # and a script can read it without a session at all
    assert entries(Context(locs=table()).session) == []


# --------------------------------------- what the log carries and what it does not

def test_a_measurement_is_not_logged_but_its_settings_go_with_its_result():
    """The log is the provenance of the data: a measurement can be repeated
    from the file, so logging it would record only that somebody looked.  What
    it worked out is kept with the result instead, settings and all."""
    from dataclasses import dataclass

    from smappy.plugins import Plugin, Result, param

    @dataclass
    class MeasureSettings:
        bins: int = param(50)

    class Measure(Plugin):
        path = "Analysis/Measure/Fake"
        Settings = MeasureSettings

        def run(self, ctx, settings):
            return Result(text="measured", settings=settings)

        def keep(self, result):
            return {"peak": 1.0}

    session = Session(table())
    session.run(Measure(), MeasureSettings(bins=17))
    assert session.history == []
    kept = session.results["Analysis/Measure/Fake"]
    assert kept["settings"]["bins"] == 17 and kept["data"] == {"peak": 1.0}


def test_writing_a_file_is_logged_although_the_table_did_not_change(tmp_path):
    """Where a file went cannot be seen in the file, so the rule is overridden
    (`Plugin.logged`)."""
    from smappy import plugins

    session = Session(table())
    save = plugins.get("File/Save/smappy HDF5")()
    session.run(save, save.Settings(path=str(tmp_path / "out.h5")))
    entry = session.history[-1]
    assert entry["what"] == "File/Save/smappy HDF5" and not entry["changed"]
    assert entry["settings"]["path"].endswith("out.h5")
