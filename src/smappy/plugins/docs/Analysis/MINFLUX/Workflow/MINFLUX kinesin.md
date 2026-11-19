---
version: "1"
covers: [smappy.chain.ChainPlugin.run, smappy.plugins.find_steps.rois_ids, smappy.plugins.region_column.trace_values]
---

## What it does

Takes a two-colour [MINFLUX](plugin:Panels/Glossary) kinesin measurement
from its localizations to the steps of each molecule that walks: MINFLUX's
quality cuts, each trace's unsettled start left out, the two colours put on
top of each other and joined into molecules, an ROI per walking kinesin,
and their steps.

It is a [chain](plugin:Panels/Glossary) (also called a pipeline or
workflow): the plugins it runs, one after the other, each with its own
settings in a section of this panel, folded behind *show the steps*.  Four
settings are shown at the top:

* *offset from* -- where [MINFLUX dual colour](plugin:Analysis/MINFLUX/MINFLUX dual colour)
  takes the colours' offset: *auto* the bead measurement open beside the
  tracking, else the molecules themselves;
* *colours apart* -- how far apart the two colours of one molecule
  may be;
* *kinesins kept* -- which molecules become ROIs, a condition on each
  molecule's track (below);
* *steps from* -- the localizations a molecule needs for its steps.

The pages of the plugins explain the methods; this one what running them
together does.  Because it is a chain, it can be edited, saved under
another name, and run over many files in a batch.

Open the tracking measurement and, beside it, the bead measurement taken
before it (File > Open: both measurements of the `.msr`, or both files),
and run it on a layer that shows the tracking.

## How it works

**1. Localizations.**  The *layers* step shows the localizations, not
merged traces: pairing and steps are made of localizations.  Its filter
stays as the Render tab has it.

**2. Quality.**  [Quality filter](plugin:Analysis/MINFLUX/Quality filter)
sets MINFLUX's cfr and efo cuts from the data.

**3. Trace starts.**  [Trace statistics](plugin:Analysis/Tracking/Trace statistics)
finds each trace's start, while it settles (*tail rule* *moving molecule*:
until the rolling spread of its steps settles), per colour, and writes it
as `in_tail`; the next *layers* step leaves it out.

**4. Colours.**  [MINFLUX dual colour](plugin:Analysis/MINFLUX/MINFLUX dual colour)
measures the offset between the colours, moves the second colour onto the
first, and joins the traces of both colours into molecules, `pair_id`.

**5. Kinesins.**  [Segment regions](plugin:Analysis/Regions/Segment regions)
with the *MINFLUX kinesin* preset makes an ROI of each molecule that walks,
drawn as its track and turned along it.

**6. Steps.**  [Find steps](plugin:Analysis/Tracking/Find steps) on
`pair_id`, each colour with its own steps (*pairs*), on the kinesins' ROIs
only (*only ROIs' traces*): the step sizes and dwell times pooled over
them, and a row per kinesin.

## In detail

**Which molecules walk.**  *kinesins kept* is a condition on each
molecule's track, measured on its localizations in time order: `n_locs`,
`displacement_nm`, `velocity_nm_per_s`, `eccentricity`, `linearity`,
`n_colours` and the others *from an id attribute* lists.  The default,
`n_locs >= 20 and displacement_nm >= 100 and linearity >= 0.8`, asks for a
molecule that moved 100 nm with its position along the track rising
steadily with time.  `linearity` is the squared correlation of that
position with time: near 1 for a motor walking on; near 0 for one stuck,
or diffusing to and fro along its microtubule, which is as elongated.  In
the two-colour kinesin data this was made on, the 17 tracks that walked
100 to 270 nm had 0.8 to 0.98, at 200 to 600 nm/s.  Add
`and n_colours == 2` for the molecules seen in both colours, which a
two-colour step analysis needs (the *MINFLUX kinesin, both colours*
preset); leave out the linearity for every longer track.

**Why not a region filter before the steps.**  An id column's measurements
never see the region filter -- a molecule's own numbers do not depend on
which molecules are shown -- so Find steps takes the ROIs' molecules
through its own *only ROIs' traces*.

**Nothing changes until the end.**  The chain runs on a copy of the
session; the table, the ROIs and the results reach the session only when
the whole chain has succeeded.

## Output

* **The text**: one line per step -- the cuts, the traces and their
  starts, the offset and the molecules in both colours, the ROIs and what
  failed the condition, the steps.
* **The figures** of each step.
* **The table**: colour 1 moved onto colour 0, with `in_tail` and
  `pair_id`.
* **The ROIs**: one per walking kinesin, with its track's values.
* **Per molecule**: MINFLUX dual colour's and Find steps' region tables of
  `pair_id`.
* **The log**: one entry for the chain, with every step's settings.
