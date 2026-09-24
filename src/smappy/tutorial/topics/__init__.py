"""The storyboards, one module per tutorial, each with a `TITLE` and a `make`.

`make(director)` does to the GUI what the tutorial shows and asks for a shot
at every step.  Say what the user sees and why it matters, in a sentence or
two a biologist reads at a glance: a subtitle longer than that is not read,
and the pointer and the spotlight are there so that the text need not say
where things are.  Numbers come from the GUI as it runs, never typed in, so
a subtitle cannot drift away from the picture it is under.
"""
TOPICS = ("layout",)
