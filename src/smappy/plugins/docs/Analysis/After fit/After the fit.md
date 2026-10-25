---
version: "1"
covers: [smappy.plugins.after_fit.AfterFit, smappy.plugins.after_fit.AfterFitSettings, smappy.plugins.after_fit.finish_localizations, smappy.plugins.after_fit.applicable, smappy.plugins.after_fit.table_can, smappy.plugins.after_fit.shown, smappy.plugins.after_fit.chain_plugin, smappy.plugins.fit._FitPlugin.finish]
---

## What it does

A fit is rarely the end.  An astigmatic fit wants its widths turned into z,
almost every acquisition wants the [drift](plugin:Panels/Glossary) taken out,
and a two-colour fit wants each localization given its colour.  None of that
can be done while the frames are still coming in: a drift curve is measured
across the whole acquisition, and the colours are told apart by a histogram
of the whole sample.

So every fitter has a section *after the fit*, folded at the foot of its
form, whose title says which steps are on.  Once the last frame is fitted,
the fitter runs these steps over the whole table, and only then saves the
file.  The file on disk is then the finished table, not a raw fit somebody
has to remember to correct.

This plugin is the same section on its own: the same settings, run on the
localizations that are open.  You rarely open it by itself.  Inside every
fitter's *after the fit* there is a button, **Run these now**, which runs it
on the open localizations without fitting again.  Use it when a step was not
ticked during the fit, or to try other settings.  It is one step of the undo
and one line of the history, under the name *After the fit*.

## How it works

**The steps, in this order.**  Each one is a plugin of its own, with that
plugin's settings, so a fit that finishes itself and a table finished by
hand give the same numbers:

1. ***astigmatic z*** -- z from the widths of an elliptical Gaussian
   ([Astigmatic z](plugin:Analysis/After fit/Astigmatic z)).  It comes
   first, so that a drift correction after it can correct z too.
2. ***drift correction*** -- none, [RCC](plugin:Analysis/Drift/RCC) or
   [COMET](plugin:Analysis/Drift/COMET).
3. ***assign colours*** -- a colour per localization from how its photons
   split between the two channels
   ([Assign colours](plugin:Analysis/After fit/Assign colours)).
4. ***then chain*** -- a saved chain: plugins in order, with their
   settings, that run as one.  It is for whatever else your lab does to
   every fit.  A chain is made with the + of the Analysis tab and its *Save
   chain*, and is what the [batch window](plugin:Panels/Batch window) runs
   over many files.

**Only what applies is shown.**  Each step appears only where the fit can
give it what it needs.  *Assign colours* appears only in the two-colour
fitters ([Gaussian 2D 2C](plugin:Localize/Gaussian 2D 2C),
[Spline 3D 2C](plugin:Localize/Spline 3D 2C)).  *Astigmatic z* appears only
in [Gaussian 2D](plugin:Localize/Gaussian 2D), and only while its
*elliptical* is ticked.  A step's settings appear below it only while the
step is on: tick *assign colours* and *colour assignment* appears; choose
COMET and only *COMET drift* is shown.

**What the table cannot take is passed over.**  When this plugin runs on
the open localizations, it looks at the table rather than at a fitter.
Colours need two channels' photons (`photons_ch0` and `photons_ch1`, or
MINFLUX's `dcr`).  Astigmatic z needs `sigma_x_nm` and `sigma_y_nm`.  A step
whose columns are missing is skipped without a word, so the default
*assign colours* does nothing to a one-colour table.

**A step that fails is a line in the output.**  These steps run after a fit
that may have taken twenty minutes.  A colour histogram with one peak, or a
drift estimate on too few localizations, must not cost that fit.  So a step
that fails says why in the output, the steps after it still run, and the
table is saved.

## In detail

Inside a fitter the steps run on the table as the fit made it, with every
localization and no filter.  The drift steps merge blinks as the Render
tab's *merge blinks* does, for the distance and dark frames they leave on
*auto*.  Each step that changed the table goes into the file's history with
the settings it used.  The streamed file is written once more with the
finished table when any step changed it.

From the button, the steps get the session's localizations: all of them,
every file, whatever the layers show.  Drift is estimated per file, as when
RCC or COMET is run on its own.

A chain is read from its file each time it runs, so a chain saved a moment
ago is the one that runs.  It runs with the values saved in the file.  A
chain with a fitting step in it is refused, because it would open the movie
and fit it again.  The other steps still run.

## Parameters

### astigmatic_z
The calibration under *z calibration* is the one Astigmatic z uses: a file
from the [PSF calibration](plugin:Panels/PSF calibration) window, method
*elliptical Gaussian*, or curves typed in.  Saving such a calibration in
that window puts it here, in Gaussian 2D, with *elliptical* and *astigmatic
z* ticked.

### drift
Their pages, [RCC](plugin:Analysis/Drift/RCC) and
[COMET](plugin:Analysis/Drift/COMET), say which to choose.  The settings
under *RCC drift* and *COMET drift* are theirs.  COMET's question before a
long run is not asked after a fit: choosing it is the agreement.  With z in
the table, both correct z as well, unless their *correct z* says otherwise.

### assign_colors
On by default: in a two-colour fit it is seconds, and it is what the fit was
for.  To change the rule afterwards, change *colour assignment* and press
*Run these now*.

### chain
Leave it empty for none.  Its settings are edited in the chain's own panel
and saved to its file.

## Output

The table with what the steps wrote: `z_nm` and `z_err_nm` from Astigmatic z,
positions moved by the drift correction, `channel` from Assign colours, and
whatever the chain added.  The output lists one line per step: what it found,
that it failed and why, or nothing for a step that was not run.  Each step's
figures (the drift curve, the colour histogram, the widths against z) open
with the result, each named after its step.
