---
version: "1"
covers: [smappy.chain.ChainPlugin.run, smappy.chain.evaluate_sites, smappy.plugins.npc.NPCSegment.run, smappy.plugins.npc.npc_rois, smappy.roi_manager.core.ROIProject.evaluate, smappy.roi_manager.core.ROIProject.evaluations]
---

## What it does

The labelling efficiency of nuclear pores takes three steps in the ROI
manager: find the pores, count each one's corners and localizations, and fit
the efficiency to all of them.  This runs the three in one go, on the file
the ROI manager is showing, and shows the results of all three in one
window.

It is a *chain* -- the plugins it runs, one after the other, each with its
own settings in a section of this panel -- rather than a plugin of its own:
[NPC](plugin:ROIManager/Segment/NPC),
[NPC Corners](plugin:ROIManager/Evaluate/NPC Corners) and
[NPC Labeling Efficiency](plugin:ROIManager/Analyze/NPC Labeling Efficiency).
Their pages explain the methods; this one only what running them together
changes.  Because it is a chain, it can be edited (*edit steps*), saved
under another name, and run over many files in a batch.

Use it for the standard analysis, and the three plugins separately to look at
one step more closely -- the ROIs this one makes are ordinary ROIs, and can be
walked through, judged and evaluated again in the ROI manager.  It needs the
localization precision (`xy_err_nm`).

## How it works

**1. Group.**  The *layers* step switches the first layer to grouped, one
localization per blink, as the model expects; its filter stays as the Render
tab has it.

**2. Find the pores.**  The file's earlier NPC ROIs are removed (*replace
earlier pores* is ticked in the *find the pores* section), and the pores are
found with the settings of that section.  ROIs drawn by hand, or made by
another segmenter, are left alone.

**3. Count.**  NPC Corners is run on every ROI the manager includes, with
the settings of the *NPC Corners* section, before the next step starts:
each pore's corners seen with the precise localizations, and its
localizations.  The numbers are stored with the ROIs, as an evaluation in the
ROI manager stores them.

**4. Fit.**  The labelling efficiency and the blinks per copy are fitted to
all the counted ROIs with the settings of the *labelling efficiency* section.

The window shows the pores found, the segmenter's checks and the fit (the
corner and localization histograms with the model).

## In detail

**An evaluator in a chain.**  A plugin that measures one ROI at a time
(NPC Corners here) runs, as a chain step, over every ROI the manager
includes -- every one that is ticked *use*, whoever made it -- and the chain
goes on only when all are done.  A ROI on which it fails (no localizations,
say) loses its own row and is counted in the step's line, not the run.

**The evaluation window is left alone.**  The counts are stored with the
ROIs as an evaluation called *NPC Corners*, like any other, and the
evaluation window's pipeline stays as it was set up -- for this work or for
another.  A result belongs to its ROI, whoever ran it, so the site table
shows the chain's counts beside the pipeline's numbers, and NPC Labeling
Efficiency run again from the ROI tab finds them, with the cutoffs they were
counted with.  Running the chain again replaces its counts; to keep a second
set beside the first -- other cutoffs, to compare -- rename the NPC Corners
step (*edit steps*), and name it in the analysis's *counts from*.

**Nothing changes until the end.**  The chain runs on a copy of the session,
ROI manager included; the ROIs it found, the evaluation run and the pipeline
reach the ROI manager only when the whole chain has succeeded.  A step that
fails leaves the ROI manager as it was.

**Re-running.**  With *replace earlier pores* off, the pores of an earlier
run stay, a new run adds only what is new (a candidate near an existing ROI
is suppressed), and all of them are counted.  Results that are still current
-- the same ROI, data and settings -- are carried forward rather than
measured again.

## Output

* **The text**: one line per step -- the grouping, the pores found among the
  candidates, the ROIs counted, then the labelling efficiency and the
  blinks per copy with their errors, as NPC Labeling Efficiency gives them;
  on a simulation also the simulated efficiency and the truth.
* **The figures**: the pores found (fitted circles, kept in blue, rejected in
  grey), the segmenter's checks, and the fit.
* **The ROIs** of the pores, with their counts, in the ROI manager.
* **The log**: one entry for the chain, with every step's settings.

## Differences from SMAP

SMAP has no single counterpart: its NPC analysis is run as `segmentNPC`,
then an evaluation with `NPCLabelingQuantify_s` in the ROI manager, then
`NPCLabelingEfficiency` ([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
