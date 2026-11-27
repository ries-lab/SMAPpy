"""The focus and *learn* in one vocabulary (docs/psf-model-parts-plan.md): settings saved in the words before
2026-11-27 read back to what they did then, and the mode a run takes from the two fields."""
from smappy.calibrate.depth import SampleSettings, focus_hint, focus_mode, settled_focus, AutoDepth
from smappy.plugins import settings_from, settings_values
from smappy.plugins.fit import (DualModelSettings, DualSplineFitSettings, SplineFitSettings,
                                SplineModelSettings)


def test_the_old_words_read_back_to_what_they_did():
    # auto learnt nothing whatever *learn* said; in situ learnt what *learn* said; *learn* beside a typed or
    # calibrated focus was never read
    assert settled_focus("auto", "focus") == ("edge", "nothing")
    assert settled_focus("auto", "everything") == ("edge", "nothing")
    assert settled_focus("insitu", "focus") == ("edge", "blur")
    assert settled_focus("insitu", "aberrations") == ("edge", "phase")
    assert settled_focus("insitu", "everything") == ("edge", "apodisation")
    assert settled_focus("typed", "focus") == ("typed", "nothing")
    assert settled_focus("calibrated", "aberrations") == ("calibrated", "nothing")
    # today's words pass through
    for pair in (("edge", "phase"), ("typed", "blur"), ("calibrated", "nothing"), ("edge", "nothing")):
        assert settled_focus(*pair) == pair


def test_the_mode_follows_the_focus_and_learn():
    assert focus_mode("edge", "nothing") == "auto"
    assert focus_mode("edge", "phase") == "insitu"
    assert focus_mode("typed", "nothing") == "typed"
    assert focus_mode("typed", "blur") == "insitu"              # learnt around the typed focus, held
    assert focus_mode("calibrated", "phase") == "calibrated"


def test_spline_3d_settings_saved_before_read_back_through_the_settings_map():
    old = {"model.focus": "auto", "model.learn": "focus"}
    m = settings_from(SplineFitSettings, old).model
    assert (m.focus, m.learn, m.mode) == ("edge", "nothing", "auto")
    m = settings_from(SplineFitSettings, {"model.focus": "insitu", "model.learn": "aberrations"}).model
    assert (m.focus, m.learn, m.mode) == ("edge", "phase", "insitu")
    m = settings_from(SplineFitSettings, {"model.focus": "typed", "model.learn": "focus",
                                          "model.focus_nm": 400.0}).model
    assert (m.focus, m.learn, m.mode) == ("typed", "nothing", "typed")
    # what is saved today reads back as it was
    today = SplineFitSettings(model=SplineModelSettings(focus="typed", learn="blur", focus_nm=300.0))
    again = settings_from(SplineFitSettings, settings_values(today)).model
    assert (again.focus, again.learn, again.focus_nm) == ("typed", "blur", 300.0)
    assert SplineModelSettings().mode == "auto"                 # the default: the coverslip's edge, nothing learnt


def test_spline_3d_2c_learn_moves_into_the_sample_part():
    old = {"model.sample.focus": "insitu", "model.learn": "aberrations"}
    m = settings_from(DualSplineFitSettings, old).model
    assert (m.sample.focus, m.sample.learn, m.learn) == ("edge", "phase", "")
    m = settings_from(DualSplineFitSettings, {"model.sample.focus": "auto", "model.learn": "focus"}).model
    assert (m.sample.focus, m.sample.learn) == ("edge", "nothing")
    m = settings_from(DualSplineFitSettings, {"model.sample.focus": "typed", "model.learn": "everything"}).model
    assert (m.sample.focus, m.sample.learn) == ("typed", "nothing")
    today = DualSplineFitSettings(model=DualModelSettings(sample=SampleSettings(focus="typed", learn="blur")))
    again = settings_from(DualSplineFitSettings, settings_values(today)).model
    assert (again.sample.focus, again.sample.learn) == ("typed", "blur")
    # a later change of the level is never undone by what was read once
    from dataclasses import replace
    changed = replace(m, sample=replace(m.sample, learn="phase"))
    assert changed.sample.learn == "phase"


def test_the_sample_part_alone_reads_in_situ_as_the_focus_and_the_blur():
    # the DECODE training's part had no *learn*: its in situ learnt the focus and the blur
    assert (SampleSettings(focus="insitu").focus, SampleSettings(focus="insitu").learn) == ("edge", "blur")
    assert SampleSettings(focus="auto").mode == "auto"


def test_the_hint_shows_the_edge_beside_the_focus_learnt():
    assert focus_hint("auto", None) == "edge: found from the movie"
    assert focus_hint("auto", AutoDepth(612.4, 10.0, 0.2, 100)) == "edge: 612 nm"
    learnt = AutoDepth(705.0, 10.0, 0.2, 100, method="in situ", edge_nm=692.0)
    assert focus_hint("insitu", learnt) == "edge 692 nm, learnt 705 nm"
    learnt.fallback = "phase"
    assert focus_hint("insitu", learnt) == "edge 692 nm, learnt 705 nm (the focus alone)"
