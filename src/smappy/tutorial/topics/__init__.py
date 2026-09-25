"""The storyboards, one module per tutorial, each with a `TITLE` and a `make`.

**Read `../STYLE.md` before writing or editing one**: it is the reviewers'
directions for how a tutorial speaks and how fast it goes, and it wins over
anything below.

`make(director)` does to the GUI what the tutorial shows and asks for a shot
at every step.  Say what the user sees and why it matters, in a sentence or
two a biologist reads at a glance: a subtitle longer than that is not read,
and the pointer and the spotlight are there so that the text need not say
where things are.  Numbers come from the GUI as it runs, never typed in, so
a subtitle cannot drift away from the picture it is under.
"""
# The series as the index shows it: sections, each with the tutorials that
# exist (module names, in order) and those planned (titles).  TOPICS follows
# from it, so a tutorial is added in one place.
SERIES = (
    ("Getting started", ("quickstart", "layout"), ()),
    ("Fitting", ("fitting",),
     ("3D: bead calibration and spline fitting",
      "Two colours: dual-channel calibration and fitting")),
    ("Rendering", ("rendering",), ()),
    ("Plugins", ("plugins", "measuring", "drift_correction"), ()),
    ("ROI manager", (), ("The ROI manager: finding and measuring many regions",)),
    ("Many files", (), ("Chains and batch runs over many files",)),
)
TOPICS = tuple(t for _, built, _ in SERIES for t in built)
