"""The user's own plugin folder: always scanned, made ready for an assistant,
and a new plugin started from a template that works.

The folder is the session's config directory's (`isolated_config`), moved
to a test's own with `SMAPPY_CONFIG_DIR` where a test writes into it.
"""
import os
import textwrap
import time

import numpy as np
import pytest

from smappy import config, plugins, user_plugins
from smappy.plugins.discovery import USER_GROUP, USER_ROOT, load_module

PLAIN = """
    from smappy.plugins import Plugin

    class Thing(Plugin):
        '''Counts things.'''
        changes_locs = False
"""


@pytest.fixture
def own(tmp_path, monkeypatch):
    """A config directory of the test's own, and so a user folder."""
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    config.load(reload=True)
    yield config.user_plugin_dir()
    monkeypatch.undo()
    config.load(reload=True)
    plugins.discover(force=True)


def write(folder, relative, text):
    path = folder / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
    return path


def test_a_file_in_the_user_folder_is_a_plugin_under_user(own):
    write(own, "thing.py", PLAIN)
    write(own, "Analysis/Measure/count_more.py", PLAIN)
    found = plugins.discover(force=True)
    assert found[f"{USER_GROUP}/Thing"].root == USER_ROOT
    # in subfolders it takes its place from them, as in any root
    assert found["Analysis/Measure/Count More"].root == USER_ROOT
    assert "Analysis/Drift/COMET" in found               # the shipped ones stay
    assert found[f"{USER_GROUP}/Thing"].description == "Counts things."
    assert plugins.get(f"{USER_GROUP}/Thing").path == f"{USER_GROUP}/Thing"


def test_a_missing_user_folder_is_not_a_problem(own):
    assert not own.exists()
    plugins.discover(force=True)
    assert not any(str(own) in str(d) for d in plugins.problems())


def test_the_folder_is_made_ready_for_an_assistant(own):
    folder = user_plugins.ensure()
    assert folder == own
    names = {p.name for p in folder.iterdir()}
    assert {"AGENTS.md", "CLAUDE.md", "README.md", "_template.py"} <= names
    agents = (folder / "AGENTS.md").read_text()
    assert "@@" not in agents                              # the placeholders filled
    assert "@AGENTS.md" in (folder / "CLAUDE.md").read_text()
    # the template is skipped by the scan: it is the example, not a plugin
    assert not any(r.root == USER_ROOT for r in plugins.discover(force=True).values())


def test_smappys_files_follow_the_installed_version_and_the_users_are_kept(own):
    folder = user_plugins.ensure()
    (folder / "AGENTS.md").write_text("old")
    (folder / "CLAUDE.md").write_text("my notes")
    user_plugins.ensure()
    assert (folder / "AGENTS.md").read_text() != "old"
    assert (folder / "CLAUDE.md").read_text() == "my notes"


def test_a_new_plugin_from_the_template_loads_and_measures(own):
    path = user_plugins.new("nearest distance (2D)")
    assert path.name == "nearest_distance_2d.py"
    found = plugins.discover(force=True)
    ref = found[f"{USER_GROUP}/Nearest Distance 2d"]     # the path: from the file name
    assert ref.name == "Nearest distance (2D)" and ref.per_region
    assert ref.description.startswith("The distance from each localization")
    cls = ref.load()
    assert cls.changes_locs is False
    # points on a grid 10 nm apart: every nearest neighbour is 10 nm away
    from smappy.locs import Localizations
    x, y = np.meshgrid(np.arange(20) * 10.0, np.arange(20) * 10.0)
    locs = Localizations({"x_nm": x.ravel(), "y_nm": y.ravel(),
                          "frame": np.zeros(x.size, dtype=np.int32)})
    result = cls()(locs)
    assert result.values["nn_median_nm"] == pytest.approx(10.0)
    assert result.values["n"] == 400
    with pytest.raises(FileExistsError):
        user_plugins.new("nearest distance (2D)")
    assert user_plugins.check(report=lambda line: None) == 0


def test_check_finds_a_plugin_that_does_not_load(own):
    write(own, "broken.py", """
        from smappy.plugins import Plugin

        class Broken(Plugin):
            pass
    """)
    lines = []
    assert user_plugins.check(report=lines.append) == 1
    assert any("changes_locs" in line for line in lines)


def test_a_changed_plugin_file_is_imported_again(tmp_path):
    path = write(tmp_path, "thing.py", textwrap.dedent(PLAIN) + "\nVALUE = 1\n")
    assert load_module(path).VALUE == 1
    assert load_module(path) is load_module(path)          # unchanged: cached
    path.write_text(textwrap.dedent(PLAIN) + "\nVALUE = 2\n")
    later = time.time() + 5                  # a coarse clock still sees the change
    os.utime(path, (later, later))
    assert load_module(path).VALUE == 2


@pytest.mark.parametrize("title, stem, cls", [
    ("cluster sizes", "cluster_sizes", "ClusterSizes"),
    ("3D spread", "plugin_3d_spread", "Plugin3DSpread"),
    ("class", "plugin_class", "Class"),
])
def test_a_title_becomes_a_module_and_a_class_name(title, stem, cls):
    assert user_plugins.file_name(title) == stem
    assert user_plugins.class_name(title) == cls
