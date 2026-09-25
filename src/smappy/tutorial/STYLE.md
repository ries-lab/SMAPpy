# How the tutorials speak

Directions for whoever writes or edits a storyboard in `topics/`.  They are
the reviewers' taste, written down so that it outlives the session it was
given in: a rebuild never rewrites a subtitle, but the next edit of a
storyboard -- by a person or by Claude -- must read this first and follow it.
These are directions, not wordings; apply them to lines they do not quote.

Add to it when a review says something that would apply beyond the one line
it was about.  Say who asked and when, so a later reader can weigh it.

## The voice

* **Say what something is for, not what it lists.**  "Opens files from
  different sources", not "smappy, SMAP, ThunderSTORM and MINFLUX files".  A
  list read aloud sounds generated and costs time; the screen already shows
  it.  (Jonas Ries, 2026-09-25)
* **Point at numbers, do not read them.**  "Here you see how many
  localizations are displayed", not "25 888 of 29 088".  The spotlight is on
  the count; the viewer reads it.  A number of four digits or more is never
  in a subtitle (`tests/test_tutorial.py` checks).  (Jonas Ries, 2026-09-25)
* One idea per step, one or two short sentences.  The pointer and the
  spotlight say *where*; the words say *what* and *why*.
* Plain words a biologist uses at the bench.  Name a GUI element by the text
  on it ("Run", "grouped"), so it can be found.
* Keyboard shortcuts may be said once, where the thing is first shown.

## The pace

* Keep it moving: the voice leads, and a step ends shortly after the voice
  does.  A card stays up only a little longer than it is spoken; whoever
  wants to read its body pauses.  (Jonas Ries, 2026-09-25: "a bit slow,
  especially the breaks between the slides")

## What a tutorial is

* Aimed at lab members who know SMLM; a short reminder of a concept
  (grouping, the selection) is welcome, a lecture is not.
* Simulated data, so anyone can follow along.
* Menus and shortcuts are mentioned as the alternative to the tabs.
  (Jonas Ries, 2026-09-24)
* Only what is true of the program now: the storyboard drives the real GUI,
  and a number or a name in a subtitle comes from it, not from memory.

## Per tutorial

### layout -- Finding your way around

* (none yet)
