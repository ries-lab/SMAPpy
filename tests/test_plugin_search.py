"""How well the assistant finds plugins: the question set, and the parts.

The questions are what users asked (the usage logs, 2026-11: few, and mostly
about the PSF calibration) and what a session wrote to cover the rest -- a
method's other names (Picasso's, SMAP's, the papers'), a mode inside a
plugin, a vague question.  Each is the phrasings an assistant would send
together, and the plugin or page that answers it.  A question that fails
here is a page that does not say the words people use, or a search that
weighs them wrongly: add questions as the logs bring them.
"""
import pytest

from smappy.assistant import index

# (phrasings, the path that answers -- or the paths, any of which does --
# where the question came from)
QUESTIONS = [
    # the usage logs
    (["uiPSF PSF model from bead stack"], "Localize/PSF calibration", "log"),
    (["show the output images, pupil etc, of the bead calibration used in Spline 3D"],
     "Localize/PSF calibration", "log"),
    (["CRLB plot of the PSF calibration"], "Panels/PSF calibration", "log"),
    # a mode inside a plugin, by another program's or the user's words
    (["undrift from picked"], "Analysis/Drift/Fiducials", "Picasso"),
    (["correct drift from selected origamis"], "Analysis/Drift/Fiducials", "user"),
    (["find more structures that look like this one"],
     "Analysis/Regions/Segment regions", "Picasso's pick similar"),
    (["remove sticky binding events"], "Analysis/DNA-PAINT/Binding kinetics", ""),
    (["convert localizations to Picasso hdf5"], "File/Export/Localizations", ""),
    (["drift correction without beads"], "Analysis/Drift/COMET", ""),
    # a method by its name
    (["Ripley's K"], "Analysis/Measure/Spatial statistics", ""),
    (["how many dimers versus monomers"], "Analysis/DNA-PAINT/SPINNA", ""),
    (["diffusion coefficient of tracks"], "Analysis/Tracking/Tracking analysis", ""),
    (["camera read noise per pixel sCMOS"], "Localize/Camera noise calibration", ""),
    (["molecules per nuclear pore"], "Analysis/Measure/NPC Corners", ""),
    (["count binding sites with qPAINT"], "Analysis/DNA-PAINT/qPAINT", ""),
    (["resolution FRC"], "Analysis/Measure/Localization Precision", ""),
    (["vibrations of the stage"], "Analysis/Measure/Stability", ""),
    # vague: found only with the field's words beside the user's
    (["is my sample moving", "stage drift or vibration"],
     "Analysis/Measure/Stability", ""),
    (["align two colour channels", "register the two channels of a split camera",
      "overlay channels from beads"], "Analysis/After fit/Overlay channels", ""),
    (["make a movie of the reconstruction", "save stack movie mp4"],
     "File/Export/Stack", ""),
    (["merge localizations in consecutive frames", "merge blinks grouping"],
     ("Panels/Render tab", "Analysis/Process/Layers"), ""),
]
TOP = 5


@pytest.mark.parametrize("phrasings, answer, source", QUESTIONS,
                         ids=[q[0][0] for q in QUESTIONS])
def test_the_question_set_finds_its_plugin_among_the_first_five(phrasings, answer, source):
    answers = {answer} if isinstance(answer, str) else set(answer)
    assert answers <= {e.path for e in index.entries()}, f"{answers} are not pages"
    found = [e["path"] for e in index.search(phrasings, limit=8)]
    assert answers & set(found[:TOP]), f"{answers} not in the first {TOP}: {found}"


def test_a_hit_says_which_sentence_matched_and_how_many_phrasings_found_it():
    hits = index.search(["undrift from picked", "drift correction DNA origami"])
    fiducials = next(h for h in hits if h["path"] == "Analysis/Drift/Fiducials")
    assert fiducials["found_by"] == 2
    assert any("picked" in m["text"] for m in fiducials["matches"])
    assert all(m["section"] for m in fiducials["matches"])


def test_a_mode_named_only_in_a_setting_is_found():
    """*like examples* is a choice of Segment regions' *method*."""
    hits = index.search("like examples")
    assert hits[0]["path"] == "Analysis/Regions/Segment regions"


def test_word_forms_are_one_word():
    assert index.words("correction corrected correcting") == ["correct"] * 3
    assert index.words("process focus drifts") == ["process", "focus", "drift"]


def test_a_glossary_entry_searches_its_other_names():
    groups = [{w for name in g for w in name} for g in index.synonyms()]
    assert any({"fiducial", "bead"} <= g for g in groups)
    weights = index._weights("fiducial")
    assert weights["fiducial"] == 1.0 and weights["bead"] == index._SYNONYM


def test_the_catalog_has_every_plugin_on_a_line_of_its_own_with_its_methods():
    from smappy import plugins
    lines = index.catalog().splitlines()
    for path in plugins.refs():
        assert any(line.startswith((path + ":", path + " ")) for line in lines), path
    segment = next(l for l in lines if l.startswith("Analysis/Regions/Segment regions"))
    assert "method: circle template" in segment and "like examples" in segment
    assert index.catalog("Analysis/Drift").splitlines()[0].startswith("Analysis/Drift/")


def test_a_summary_is_the_docstrings_first_paragraph_not_its_first_line():
    from smappy import plugins
    from smappy.plugins.discovery import summary
    assert summary("One line\nwrapped on.\n\nMore.") == "One line wrapped on."
    assert plugins.refs()["Analysis/Measure/Spatial statistics"].description.endswith(
        "with envelopes from random placements.")
