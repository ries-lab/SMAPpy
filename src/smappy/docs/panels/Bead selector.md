---
title: Bead selector
summary: The beads a drift correction chose, in a table with their numbers and their tracks, to include or exclude by hand.
widget: smappy.gui.bead_window.BeadWindow
covers: [smappy.gui.bead_window.BeadWindow, smappy.beads.BeadView, smappy.beads.draw_bead_tracks, smappy.beads.select_beads]
---

## What it does

A drift correction from beads is only as good as its beads.  This window
shows the beads a plugin found -- the [MBM](plugin:Panels/Glossary) beads of
a [MINFLUX](plugin:Panels/Glossary) measurement for
[MBM drift correction](plugin:Analysis/Drift/MBM) -- with what the plugin
measured of each and which ones it would use.  Clicking a bead shows its
track over the others; a bead can then be included or excluded by hand.

It opens from *Select beads...* under the plugin's settings.  It corrects
nothing itself: a choice made here goes into the plugin's *by hand* field
at once, and the plugin's Run applies it.  So the choice is kept with the
plugin's settings, in a saved workspace and in a chain.

## The table

A row per bead.  *use* is whether the plugin would use it, with "(by hand)"
where that was chosen here.  *noise*, *wander* and *jump* are the bead's
deviation from the curve the other beads give -- its scatter from point to
point, its slow motion of its own and its largest step -- *seen* the
fraction of the time it was measured in and *points* how many times.  The
plugin can add its own columns: for MBM, whether Imspector used the bead
and Imspector's precision for it (*sigpos*).  *why* says why a bead is not
used.  A bead not used is greyed.

## The figure

Left, a row per axis: every bead's track lined up on the drift curve
(black), the used beads grey, the others red and dashed, and the selected
bead in colour over them.  Right: each bead's deviation from the curve.  A
good bead scatters evenly about zero; a step, a slope or a wider scatter
than the others' is a bead that moved.  Clicking a line selects its bead.

## Controls

### beads of
The measurement whose beads are shown, when the selection holds several.

### Include
Use the selected bead whatever the rule says.

### Exclude
Never use the selected bead.  Space toggles between the two.

### Automatic
Let the rule decide for the selected bead again.

### Centre in view
Move the render view to the selected bead, at the same zoom.
