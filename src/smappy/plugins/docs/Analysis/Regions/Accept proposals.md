---
version: "1"
covers: [smappy.plugins.roi_results.AcceptProposals.run, smappy.plugins.roi_results.AcceptProposals.describe, smappy.plugins.roi_results.apply_proposals, smappy.plugins.roi_results.proposal_steps, smappy.plugins.roi_results.write_property, smappy.roi_manager.core.proposal_values, smappy.roi_manager.core.ROIProject.chosen]
---

## What it does

Takes what a per-region step proposed each [ROI](plugin:Panels/Glossary)
(region of interest, kept in the [ROI manager](plugin:Panels/ROI manager))
should be, and makes it the ROI's own: its angle, a range, a named line, a
keyword, its centre.  Some steps measure such a thing on the way: the
[Trace inspector](plugin:Analysis/Tracking/Trace inspector) proposes a
track's direction as the ROI's angle, its axis as the line *axis* and its
centre; [Circle fit](plugin:Analysis/Measure/Circle fit) the fitted centre.

In the Regions tab, *Apply* beside a step's proposals does this by hand.
This plugin is the same for a [chain](plugin:Panels/Glossary) or a batch,
where there is nobody to click: segment the tracks, measure them, accept
the direction, measure again along it.

## How it works

**Measure first.**  The step must have run in the ROIs; each ROI's
proposals are kept with its results, and each ROI takes its own.

**Choose the step and what to take.**  *step* lists the steps that
proposed anything; *property* one of what it proposes, or all of it.

**Take them.**  On the ROIs selected in the ROI manager's list -- all of
them when none is, and always all in a chain or a batch.  A property set
by hand in the ROI manager is left alone unless *change* says otherwise;
a centre is always taken, since one placed by hand is placed by eye.  A
new line name, range attribute or keyword joins the project's.  All of it
is one step of the ROI manager's undo.

## In detail

The properties are named as the ROIs keep who set them (`ROI.set_by`):
`angle`, `centre`, `range:<attribute>`, `line:<name>`, `keyword:<list>`.
A property taken here is noted as set by the step that proposed it, so
running it again, or [ROIs from results](plugin:Analysis/Regions/ROIs from results),
may change it, while one set by hand stays.

An angle, a range, a line or a centre is part of what the ROI's results
were measured in, so those ROIs are measured again by the per-region
steps.  A keyword is a note.

## Parameters

### which
*all but those set by hand* is the usual choice, and the one to run again
after new ROIs are measured.  *all, those set by hand too* starts over --
one undo step takes it back.

## Output

The ROIs' new properties, and a line saying on how many ROIs the
proposals were taken, how many properties were written and how many ROIs
changed, and how many were passed over and why: set by hand, already set,
or nothing proposed.
