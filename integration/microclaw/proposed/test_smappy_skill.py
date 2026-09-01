"""The smappy skill: what it must promise, and what it must not.

SMAPpy is an optional dependency that carries compiled extensions, so the two
ways this skill can mislead an agent are promising a fit on a machine that
cannot do one, and telling it to open a viewer window inside the server
process.  Both are pinned here.
"""
from pathlib import Path

from microclaw import skills


def body() -> str:
    return skills.load_skill_text("smappy")


def flat() -> str:
    return " ".join(body().split())


def test_the_skill_is_in_the_catalog():
    names = [item.name for item in skills.SKILL_CATALOG]
    assert "smappy" in names
    entry = next(i for i in skills.SKILL_CATALOG if i.name == "smappy")
    assert "localization" in entry.description.lower()


def test_it_checks_that_smappy_is_installed_before_promising_a_fit():
    text = flat()
    assert "find_spec" in text
    assert "optional dependency" in text
    assert "Do not attempt to install or build it mid-session" in text


def test_it_names_the_distribution_rather_than_the_import_name():
    """`smappy` on PyPI is an unrelated package; installing it helps nobody."""
    text = flat()
    assert "pip install smappy-smlm" in text
    assert "never tell a user to install that" in text


def test_it_does_not_send_the_agent_through_a_tiff_export():
    """NDTiff is read directly; an export would copy the whole raw dataset."""
    text = flat()
    assert "NDTiff is read directly" in text
    assert "no `export_dataset_as_tiff` pass" in text
    assert "never to feed smappy" in text


def test_it_keeps_the_viewer_out_of_the_microclaw_process():
    text = flat()
    assert "Do not call it from the microclaw server process" in text
    assert "save_image" in text
    assert "LiveFit" in text


def test_it_asks_for_camera_parameters_rather_than_a_matlab_file():
    text = flat()
    assert "The camera is stated as parameters, not as a file" in text
    assert "rig profile" in text
    assert "do not require the user to produce a MATLAB settings file" in text


def test_it_says_where_hook_delivered_frames_go():
    text = flat()
    assert "queue_source" in text
    assert "analyze_frame" in text


def test_the_smlm_skill_routes_to_it():
    smlm = skills.load_skill_text("smlm")
    assert 'load_skill(name="smappy")' in smlm
    assert "Microclaw itself performs no localization fitting" in smlm


def test_smappy_is_an_optional_dependency_of_microclaw():
    """It must never become a hard dependency: it ships compiled extensions.

    Read as text rather than with tomllib, which is 3.11+ while microclaw
    supports 3.10 (the same reason `credentials.py` parses its one key by hand).
    """
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    required = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "smappy" not in required
    extras = text.split("[project.optional-dependencies]", 1)[1]
    analysis = extras.split("analysis = [", 1)[1].split("]", 1)[0]
    assert "smappy-smlm" in analysis
