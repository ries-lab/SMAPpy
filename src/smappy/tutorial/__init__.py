"""Tutorials: the real GUI, driven by a script, played back as a slide show.

A tutorial is a *storyboard* -- a Python function in `smappy.tutorial.topics`
that clicks through the program the way a person would and says, at every
step, what they are looking at.  `Director` runs it on the offscreen Qt
platform over simulated data, and at each step keeps a screenshot of the
windows and where on it the widgets being talked about are.  `player` turns
those into an HTML page -- the screenshot, a spotlight on the widget, a
pointer that travels to it and a subtitle under the picture -- and, with
Playwright, records that page as an MP4 with a WebVTT subtitle track.

Why generated rather than recorded: the GUI changes every week, and a
screencast is out of date the day a button moves.  A storyboard is re-run
instead -- in CI, if wanted -- and one that no longer runs is a GUI that no
longer does what its tutorial says, which is worth knowing too.  The pointer
and the spotlight are drawn by the page rather than baked into the pictures,
so they move between steps and the pictures stay plain screenshots.

How a tutorial should sound is in ``STYLE.md`` beside this file: the review
comments, kept as directions for the next edit.  Each build writes
``script.md`` (and ``script.html``) with every step's words, numbered, which
is what a reviewer comments on.

    python -m smappy.tutorial layout -o build/tutorials          # HTML
    python -m smappy.tutorial layout -o build/tutorials --video  # and MP4
"""
