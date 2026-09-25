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
TOPICS = ("quickstart", "layout", "fitting", "rendering")

# What the index says is coming, in the order it is planned to be made.
PLANNED = (
    "Drift correction, grouping and filtering",
    "Analysis plugins: statistics, line profiles, resolution",
    "The ROI manager: finding and measuring many regions",
    "Two colours: registration and assignment",
    "Chains and batch runs over many files",
)
