"""Measuring: what the statistics, precision and line-profile plugins find.

The plugins tutorial shows how any plugin is run; this one is about reading
three of them.  Localization Statistics -- photons, precision and on-time,
each with the law it should follow.  Localization Precision -- what the fit
believes (CRLB), what the data shows (the pair displacement, NeNA) and what
the picture resolves (FRC).  Line Profile's model comparison, for when the
shape is not known.

On the simulated ring and lines, whose precision is known: the storyboard
checks that the CRLB and the pair displacement agree, as they must for
simulated data with nothing else wrong with it.
"""
from __future__ import annotations

from .common import (expand, field, figure_panels, open_section, place_beside,
                     show_tab, toolbar)

TITLE = "Measuring: statistics, precision and line profiles"
DESCRIPTION = ("Reading Localization Statistics, Localization Precision (CRLB, "
               "NeNA, FRC) and Line Profile's model comparison.")


def _show_tab_of(d, window, name: str) -> None:
    """Bring a result window's figure tab forward; it is drawn when seen."""
    tabs = window.tabs
    for i in range(tabs.count()):
        if tabs.tabText(i).lower().startswith(name.lower()):
            tabs.setCurrentIndex(i)
            d.settle()
            return
    raise LookupError(f"no {name!r} figure; there are "
                      + ", ".join(tabs.tabText(i) for i in range(tabs.count())))


def _run_and_plot(d, panel, fraction: float = 0.8):
    panel.run_button.click()
    d.settle()
    panel.plot()
    d.settle()
    place_beside(d, panel._window, fraction)
    return panel._window


def make(d) -> None:
    session, render = d.session, d.render
    file_tab = show_tab(d, "File")
    _, simulate = open_section(d, file_tab, "Blinking Structure")
    simulate.run_button.click()
    d.settle()

    d.chapter("Three questions")
    d.card(TITLE,
           "<p>How bright are the molecules, and how long do they stay on? How "
           "precisely were they placed, and what can the picture resolve? What "
           "shape is under a line?</p>"
           "<p>One plugin for each, on the simulated data.</p>",
           say="Three questions, and a plugin for each: how the molecules "
               "behave, how precise the data is, and what shape a structure has.")

    d.chapter("Statistics")
    analysis = show_tab(d, "Analysis")
    _, stats = open_section(d, analysis, "Localization Statistics")
    window = _run_and_plot(d, stats)
    d.shot("Localization Statistics draws each distribution with the law it "
           "should follow, so a glance says whether the data behaves.",
           spot=[d.window_rect(window)])
    panels = figure_panels(d, window)       # photons, precision, (z), on-time
    assert len(panels) >= 3, f"{len(panels)} statistics panels"
    d.shot("Photons fall off exponentially above the detection threshold. The "
           "fitted constant is the mean photon count of a blink.",
           spot=[panels[0]], zoom=d.around(panels[0], 900))
    d.shot("The precision follows from the photons. Its maximum and rising edge "
           "are landmarks you can read off the histogram.",
           spot=[panels[1]], zoom=d.around(panels[1], 900))
    d.shot("The on-time, from the grouping: how many frames a molecule stays "
           "on. A straight line on this log scale is a constant switching rate.",
           spot=[panels[-1]], zoom=d.around(panels[-1], 900))
    window.hide()

    d.chapter("Precision and resolution")
    _, precision = open_section(d, analysis, "Localization Precision")
    d.card("Three measures of precision",
           "<p><b>CRLB</b>: what the fit believes, from the photons and the "
           "background.</p>"
           "<p><b>Pair displacement</b> (NeNA): a molecule seen in two "
           "consecutive frames moved only by the error, so the data measures "
           "itself.</p>"
           "<p><b>FRC</b>: what the picture resolves, labelling and drift "
           "included.</p>",
           say="Three measures: what the fit believes, what the data shows, and "
               "what the picture resolves. They are meant to disagree.")
    pixel = field(precision, "frc_pixel_nm")
    expand(pixel)
    pixel.set(4.0)
    pixel.changed.emit()
    d.settle()
    d.shot("This data is so precise that FRC needs a finer pixel than it picks "
           "by itself: set it under more.",
           spot=[pixel], point=pixel, zoom=d.around(pixel, 760))
    window = _run_and_plot(d, precision)
    # the words say the two agree: for simulated data with nothing else
    # wrong they must, and if they stop agreeing the words are wrong
    measured = precision.result.data["measurements"][0]
    crlb = next(v["sigma_c"] for v in measured.crlb.values() if "sigma_c" in v)
    assert abs(measured.radial.sigma / crlb - 1) < 0.2, (measured.radial.sigma, crlb)
    import math
    assert measured.frc is not None and math.isfinite(measured.frc.resolution), \
        "FRC found no resolution"
    d.shot("The pair displacement agrees with the CRLB, as it should for "
           "simulated data. In a real experiment it is larger: drift, or a PSF "
           "that is not the model.",
           spot=[d.window_rect(window)])
    for name, say in (("FRC", "FRC splits the data in two and asks up to which "
                              "detail the halves agree: the resolution of the "
                              "picture."),
                      ("CRLB", "The CRLB histogram, with the same law as in "
                               "Statistics fitted to it.")):
        _show_tab_of(d, window, name)
        d.shot(say, spot=[d.window_rect(window)])
    window.hide()

    d.chapter("Which shape")
    _, profile = open_section(d, analysis, "Line Profile")
    from .plugins import _across
    roi, centre = _across(0.15)
    toolbar(d)._set_kind("line", "line")
    session.set_roi(roi)
    render.view.frame_on((centre[0] - 700, centre[0] + 700), (centre[1] - 500, centre[1] + 500))
    for name, value in (("axis", "along"), ("model", "all")):
        f = field(profile, name)
        f.set(value)
        f.changed.emit()
    d.settle()
    d.shot("When the shape under a line is not known, fit all five models: "
           "a Gaussian, two Gaussians, a step, a disk and a ring.",
           spot=[field(profile, "model")], point=field(profile, "model"),
           zoom=d.around(field(profile, "model"), 760))
    window = _run_and_plot(d, profile, 0.75)
    fits = (profile.result.data or {}).get("fits") or {}
    assert fits, "the model comparison fitted nothing"
    d.shot("Each is fitted to the localizations themselves, and compared by AIC: "
           "the lowest wins, and extra parameters have to earn their place.",
           spot=[d.window_rect(window), profile.output])
    window.hide()
    d.shot("With few localizations, bootstrap gives confidence intervals that "
           "do not rely on the fit's own error bars.",
           spot=[field(profile, "bootstrap")], point=field(profile, "bootstrap"),
           zoom=d.around(field(profile, "bootstrap"), 760))

    d.chapter("Next")
    d.card("What to remember",
           "<ul><li><b>Statistics:</b> photons, precision and on-time, each "
           "against its law.</li>"
           "<li><b>Precision:</b> CRLB, pair displacement and FRC; where they "
           "disagree says what limits the data.</li>"
           "<li><b>Line Profile:</b> compare models by AIC when the shape is "
           "not known.</li></ul>",
           say="That is measuring. Next: correcting drift, before any of these "
               "numbers are trusted.")
