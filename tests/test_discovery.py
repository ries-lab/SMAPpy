"""The scanner: a folder tree becomes a plugin tree, and nothing gets imported."""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from smappy import plugins
from smappy.plugins.discovery import PluginRef, scan, title


def write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
    return path


PLAIN = """
    from smappy.plugins import Plugin

    class MyDrift(Plugin):
        '''Corrects a drift the cheap way.'''
        description = "the one line that wins over the docstring"
"""


def test_folders_name_the_plugin(tmp_path):
    write(tmp_path, "Analysis/Drift/my_drift.py", PLAIN)
    refs, problems = scan([("test", tmp_path)])
    assert problems == []
    ref = refs["Analysis/Drift/My Drift"]
    assert ref.name == "My Drift" and ref.attr == "MyDrift"
    assert ref.description == "the one line that wins over the docstring"
    assert ref.group == "Analysis/Drift" and ref.parts[0] == "Analysis"
    assert ref.scope == "locs" and ref.favorite


def test_docstring_is_the_description_when_nothing_else_says(tmp_path):
    write(tmp_path, "Analysis/quiet.py", """
        from smappy.plugins import Plugin

        class Quiet(Plugin):
            '''Says only this.

            And this part is not the description.
            '''
    """)
    refs, _ = scan([("test", tmp_path)])
    assert refs["Analysis/Quiet"].description == "Says only this."


def test_register_wins_over_the_folder_and_gives_several_per_file(tmp_path):
    write(tmp_path, "Wherever/two.py", """
        from smappy.plugins import Plugin, register

        class _Base(Plugin):
            pass

        @register("Localize/Alpha")
        class Alpha(_Base):
            name = "Alpha"

        @register("Localize/Beta")
        class Beta(_Base):
            scope = "site"
    """)
    refs, problems = scan([("test", tmp_path)])
    assert problems == []
    assert set(refs) == {"Localize/Alpha", "Localize/Beta"}
    assert refs["Localize/Beta"].scope == "site"
    assert "Wherever/Two" not in refs           # the folder no longer names it


def test_arbitrary_depth(tmp_path):
    write(tmp_path, "MyLab/Cluster/2D/dbscan.py", PLAIN)
    refs, _ = scan([("test", tmp_path)])
    assert "MyLab/Cluster/2D/Dbscan" in refs


def test_scope_must_be_one_we_know(tmp_path):
    write(tmp_path, "A/odd.py", """
        from smappy.plugins import Plugin

        class Odd(Plugin):
            scope = "somewhere else"
    """)
    refs, _ = scan([("test", tmp_path)])
    assert refs["A/Odd"].scope == "locs"


def test_computed_attributes_are_left_alone_rather_than_guessed(tmp_path):
    write(tmp_path, "A/computed.py", """
        from smappy.plugins import Plugin

        PREFIX = "auto"

        class Computed(Plugin):
            name = PREFIX + " name"
    """)
    refs, _ = scan([("test", tmp_path)])
    assert refs["A/Computed"].name == "Computed"     # the leaf, not "auto name"


def test_a_helper_module_beside_the_plugins_is_not_a_problem(tmp_path):
    write(tmp_path, "Analysis/Drift/my_drift.py", PLAIN)
    write(tmp_path, "Analysis/Drift/shared.py", "def helper():\n    return 1\n")
    refs, problems = scan([("test", tmp_path)])
    assert problems == [] and len(refs) == 1


def test_private_files_and_folders_are_skipped(tmp_path):
    write(tmp_path, "Analysis/_scratch.py", PLAIN)
    write(tmp_path, "_wip/Analysis/thing.py", PLAIN)
    write(tmp_path, "Analysis/__pycache__/cached.py", PLAIN)
    refs, problems = scan([("test", tmp_path)])
    assert refs == {} and problems == []


def test_several_plugins_without_register_is_reported_not_guessed(tmp_path):
    origin = write(tmp_path, "A/two.py", """
        from smappy.plugins import Plugin

        class One(Plugin):
            pass

        class Two(Plugin):
            pass
    """)
    refs, problems = scan([("test", tmp_path)])
    assert refs == {}
    assert len(problems) == 1 and problems[0].origin == origin
    assert "several plugins" in problems[0].message and "One, Two" in problems[0].message


def test_a_file_that_cannot_be_parsed_is_reported_not_swallowed(tmp_path):
    write(tmp_path, "A/broken.py", "class Nope(Plugin:\n")
    refs, problems = scan([("test", tmp_path)])
    assert refs == {} and len(problems) == 1
    assert "cannot be read" in problems[0].message


def test_a_missing_root_is_reported(tmp_path):
    refs, problems = scan([("test", tmp_path / "nowhere")])
    assert refs == {} and "does not exist" in problems[0].message


def test_a_later_root_shadows_an_earlier_one_and_says_so(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    write(first, "Analysis/Drift/my_drift.py", PLAIN)
    mine = write(second, "Analysis/Drift/my_drift.py", PLAIN)
    refs, problems = scan([("shipped", first), ("mine", second)])
    assert refs["Analysis/Drift/My Drift"].origin == mine
    assert len(problems) == 1 and "shadows" in problems[0].message


def test_title_leaves_deliberate_spelling_alone():
    assert title("my_drift") == "My Drift"
    assert title("drift-rcc") == "Drift Rcc"
    assert title("NPC3D") == "NPC3D"
    assert title("fit_NPC3D") == "Fit NPC3D"


# ------------------------------------------------------------------ loading

def test_load_imports_the_file_and_finds_the_class(tmp_path):
    write(tmp_path, "Analysis/Drift/my_drift.py", PLAIN)
    refs, _ = scan([("test", tmp_path)])
    cls = refs["Analysis/Drift/My Drift"].load()
    assert cls.__name__ == "MyDrift"
    assert cls.path == "Analysis/Drift/My Drift"      # the folder told it where it is
    assert cls.name == "My Drift"


def test_a_module_can_declare_plugins_it_builds_dynamically(tmp_path):
    write(tmp_path, "A/dynamic.py", """
        from smappy.plugins import Plugin, register

        for label in ("Red", "Green"):
            register("Colour/" + label)(type(label, (Plugin,), {}))

        __smappy_plugins__ = ["Colour/Red", "Colour/Green"]
    """)
    refs, problems = scan([("test", tmp_path)])
    assert problems == [] and set(refs) == {"Colour/Red", "Colour/Green"}
    assert refs["Colour/Red"].attr is None
    assert refs["Colour/Red"].load().__name__ == "Red"


def test_a_declaration_the_import_does_not_honour_is_an_error(tmp_path):
    write(tmp_path, "A/lying.py", """
        from smappy.plugins import Plugin

        __smappy_plugins__ = ["Colour/Blue"]
    """)
    refs, _ = scan([("test", tmp_path)])
    with pytest.raises(ImportError, match="registered no such plugin"):
        refs["Colour/Blue"].load()


# ------------------------------------------------- the registry on top of it

def test_the_builtin_plugins_are_found_by_scanning(tmp_path):
    found = plugins.refs()
    assert set(found) >= {"Analysis/Drift/COMET", "Analysis/Drift/RCC",
                          "Localize/Gaussian 2D", "Localize/Spline 3D"}
    assert found["Analysis/Drift/COMET"].origin.name == "drift_comet.py"
    assert plugins.problems() == []


def test_scanning_imports_nothing():
    """In a subprocess: unimporting here would break tests holding a class.

    This is the whole point of the scanner, so it is worth asserting on a clean
    interpreter rather than on whatever this session has already imported.
    """
    code = textwrap.dedent("""
        import sys
        from smappy import plugins
        assert len(plugins.refs()) >= 5
        after_scan = sorted(m for m in sys.modules if m.startswith("smappy.plugins."))
        plugins.get("Analysis/Drift/COMET")
        after_get = sorted(m for m in sys.modules if m.startswith("smappy.plugins."))
        print(after_scan, after_get, sep="|")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1])
    assert out.returncode == 0, out.stderr
    after_scan, after_get = (eval(part) for part in out.stdout.strip().split("|"))
    assert after_scan == ["smappy.plugins.discovery"]
    # one plugin opened, only that plugin and the helper it shares with the
    # other drift plugin imported: not fit.py with its C extensions
    assert after_get == ["smappy.plugins.discovery", "smappy.plugins.drift_comet",
                         "smappy.plugins.drift_result"]


def test_refs_and_tree_do_not_import_but_available_does():
    assert set(plugins.refs("Analysis/")) == {
        "Analysis/Drift/COMET", "Analysis/Drift/RCC",
        "Analysis/Dual-Color/AssignColors",
        "Analysis/Measure/Line Profile",
        "Analysis/Measure/Localization Precision",
        "Analysis/Measure/Localization Statistics",
        "Analysis/Process/History", "Analysis/Process/Math Parser",
        "Analysis/Process/Remove Localizations",
        "Analysis/Register/Calibrate transform"}
    assert isinstance(plugins.tree()["Analysis"]["Drift"]["COMET"], PluginRef)
    classes = plugins.available("Analysis/")
    assert classes["Analysis/Drift/COMET"] is plugins.get("Analysis/Drift/COMET")


def test_get_is_the_same_class_every_time():
    assert plugins.get("Analysis/Drift/RCC") is plugins.get("Analysis/Drift/RCC")
    with pytest.raises(KeyError):
        plugins.get("Nothing/Here")


def test_a_user_root_from_the_config_is_scanned(tmp_path, monkeypatch):
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    from smappy import config
    config.load(reload=True)
    root = tmp_path / "mine"
    write(root, "MyLab/thing.py", PLAIN)
    config.set_plugin_roots([root])
    try:
        found = plugins.discover(force=True)
        assert "MyLab/Thing" in found
        assert found["MyLab/Thing"].root == str(root)
        assert "Analysis/Drift/COMET" in found          # the shipped ones survive
    finally:
        config.set_plugin_roots([])
        config.load(reload=True)
        plugins.discover(force=True)


def test_two_roots_with_the_same_filename_load_as_different_modules(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    write(first, "Alpha/thing.py", """
        from smappy.plugins import Plugin

        class Thing(Plugin):
            description = "the first"
    """)
    write(second, "Beta/thing.py", """
        from smappy.plugins import Plugin

        class Thing(Plugin):
            description = "the second"
    """)
    refs, problems = scan([("a", first), ("b", second)])
    assert problems == []
    one, two = refs["Alpha/Thing"].load(), refs["Beta/Thing"].load()
    assert one is not two
    assert one.description == "the first" and two.description == "the second"
    assert one.path == "Alpha/Thing" and two.path == "Beta/Thing"


def test_a_subclass_of_a_registered_plugin_does_not_inherit_its_path(tmp_path):
    write(tmp_path, "Mine/special.py", """
        from smappy.plugins import Plugin, register

        @register("Elsewhere/Base")
        class Base(Plugin):
            pass
    """)
    write(tmp_path, "Mine/derived.py", """
        from smappy.plugins import Plugin

        class Derived(Plugin):
            pass
    """)
    refs, _ = scan([("test", tmp_path)])
    assert refs["Mine/Derived"].load().path == "Mine/Derived"
