---
version: "1"
covers: [smappy.plugins.npc.NPCWorkflow.run, smappy.plugins.npc.npc_rois, smappy.plugins.npc.rows_of, smappy.roi_manager.core.ROIProject.evaluate]
---

## What it does

The labelling efficiency of nuclear pores takes three steps in the ROI
manager: find the pores, count each one's corners and localizations, and fit
the efficiency to all of them.  This plugin runs the three in one go, on the
file the ROI manager is showing, and shows the results of all three in one
window.

Each step is the plugin of its own: [NPC](plugin:ROIManager/Segment/NPC),
[NPC Corners](plugin:ROIManager/Evaluate/NPC Corners) and
[NPC Labeling Efficiency](plugin:ROIManager/Analyze/NPC Labeling Efficiency),
with the same settings, in three sections of this panel.  Their pages explain
the methods; this one only what running them together changes.

Use it for the standard analysis, and the three plugins separately to look at
one step more closely -- the ROIs this one makes are ordinary ROIs, and can be
walked through, judged and evaluated again in the ROI manager.  It needs
grouped localizations with their precision (`xy_err_nm`): set the layer to
grouped first.

## How it works

**1. Find the pores.**  The file's earlier NPC ROIs are removed, when
*replace earlier pores* is ticked, and the pores are found with the
settings of the *find the pores* section.  ROIs drawn by hand, or made by
another segmenter, are left alone and not counted.

**2. Count.**  Every pore the segmenter kept is evaluated with NPC Corners,
with the settings of the *count corners* section: its corners seen with the
precise localizations, and its localizations.  The numbers are stored with
the ROIs, as an evaluation in the ROI manager stores them.

**3. Fit.**  The labelling efficiency and the blinks per copy are fitted to
those pores with the settings of the *labelling efficiency* section, with the
cutoffs of the *count corners* section.

The window shows the fit (the corner and localization histograms with the
model), the pores found and the segmenter's checks.

## In detail

**Which pores are counted.**  The ROIs of the current file that the NPC
segmenter made (their origin says so) and that are used.  With *replace
earlier pores* off, the pores of an earlier run stay, the new run adds only
what is new (a candidate near an existing ROI is suppressed), and all of them
are counted.

**The cutoffs.**  Run separately, NPC Labeling Efficiency takes the cutoffs
of NPC Corners from the evaluation pipeline.  Here they come from the *count
corners* section, so the counts and the model always agree, whatever the
evaluation pipeline holds.

**The record.**  The ROIs keep how they were found, and the counts are stored
as an evaluation run with the NPC Corners settings used, so the ROI manager
shows them and they are saved with the file.

**Grouping.**  The model assumes one localization per blink.  On a layer
that is not grouped, the plugin still runs, and its text says so.

## Parameters

### replace
On, a second run with other settings gives a fresh answer.  Off, to add the
pores of a second region or a second pass to those already found.

## Output

* **The text**: the pores found and counted, then the labelling efficiency
  and the blinks per copy with their errors, as NPC Labeling Efficiency gives
  them; on a simulation also the simulated efficiency and the truth.
* **The figures**: the fit, the pores found (fitted circles, kept in blue,
  rejected in grey), and the segmenter's checks.
* **The ROIs** of the pores, with their counts, in the ROI manager.
* **Data**: what NPC Labeling Efficiency returns, and `pores_found`,
  `candidates`, `pores_counted`.

## Differences from SMAP

SMAP has no single counterpart: its NPC analysis is run as `segmentNPC`,
then an evaluation with `NPCLabelingQuantify_s` in the ROI manager, then
`NPCLabelingEfficiency` ([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
