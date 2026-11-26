---
version: "1"
covers: [smappy.plugins.region_column.WriteResults.run, smappy.plugins.region_column.WriteResults.describe, smappy.plugins.region_column.RegionsToColumn.run, smappy.plugins.region_column.rows_values, smappy.plugins.region_column.keyword_columns, smappy.plugins.region_column.own_rows, smappy.region_runs.write_values, smappy.region_runs.roi_numbers, smappy.region_runs.region_members, smappy.region_runs.values_onto, smappy.region_runs.merge_pieces, smappy.region_runs.with_region_columns, smappy.plugins.roi_results.ROIsFromResults.run, smappy.plugins.roi_results.ROIsFromResults.describe, smappy.plugins.roi_results.new_values, smappy.plugins.roi_results.apply, smappy.plugins.roi_results.range_value, smappy.plugins.roi_results.line_value, smappy.plugins.roi_results.keyword_values, smappy.roi_manager.core.ROI.set_source, smappy.roi_manager.core.ROIProject.frame]
---

## What it does

Takes what was measured once per [region](plugin:Panels/Glossary) and
writes it where it is used next.  *write to* chooses where:

* **the localizations** -- each region's value written onto its
  localizations as an attribute of the table, to filter, colour or measure
  by;
* **the ROIs** -- each ROI's value written onto the ROI itself, as its
  angle, a range, a line, a keyword or its centre, so the next measurement
  uses it and the ROI manager shows it.

Both read the same results -- what the per-region steps, a block of a
chain or a segmenter measured -- and measure nothing themselves.  What a
step *proposed* for its ROI is set with *Apply proposals* in the Regions
tab, or [Apply proposals to ROIs](plugin:Analysis/Regions/Accept proposals)
in a chain.

### To the localizations

Takes what was measured once per [region](plugin:Panels/Glossary) -- in each
[ROI](plugin:Panels/Glossary) (region of interest, a shape in the ROI
manager), or in each cluster an id attribute such as `cluster_id` names -- and
writes it onto the localizations of that region, as an attribute of the
table.  Every localization in a pore gets the pore's number of corners;
every localization in a cluster gets that cluster's size or precision.

The point is that an attribute is something every other tool already reads.
With the region's result on its localizations

* the layer's **[filter](plugin:Panels/Glossary)** (also called bounds) keeps the localizations of regions that measured
  well -- the pores with 6 or more corners, say -- and each layer filters
  on its own;
* *colour by* set to the attribute in the Render tab colours each region by its
  result;
* the attribute is saved with the table and kept when blinks are merged.

It can also write each ROI's own number, as the
[ROI manager](plugin:Panels/ROI manager) shows it (`roi_id`): the ROIs then
live in the table itself, and a block set to *for each* `roi_id` repeats
over them like over clusters.  An ROI's number is given when the ROI is made
and kept when others are removed, and the results' rows name the ROI by the
same number, so ROI 5 in the list, in the attribute and in the results is the
same ROI.

What you said of the ROIs reaches the table the same way.  Only ROIs ticked
*use* are written, so a `roi_id` of 0 or above is "in a used ROI" (-1 is
none): a filter on it
keeps the tracks or sites you kept and drops the rest, for any plugin that
runs on the view afterwards.  An ROI's ranges (set under
*ranges* in the ROI manager's *annotate*: frames, z or any attribute) are
part of what it holds, so a localization outside them gets -1 too.  And each list of keywords
becomes an attribute, `<list>_keyword`, holding the keyword's place in its list
-- with the lists *quality: good, bad*, `quality_keyword` is 1 on the
localizations of the ROIs called *good* and 2 on the *bad* ones.  With
nothing in *results from* and *values*, the plugin writes `roi_id` and every
keyword attribute some used ROI has.

The results must be there first: the ROIs measured (by the
[per-region steps](plugin:Panels/Regions tab), or a block of a
[chain](plugin:Panels/Glossary) -- plugins run one after another as one --
over the
ROIs), or a block over an id attribute such as `cluster_id`.  A block of a chain can write
the same attributes itself at the end of its run -- its *write values* field --
which keeps them up to date whenever the chain runs; this plugin is for
results that are already there, and the way to do it after the per-region
steps, which do not offer *write values*.

It is a snapshot.  Moving an ROI, re-clustering, or measuring again does not
change the attribute; run it again (or let the block write it) to bring it up
to date.  The run is written into the table's history.

### To the ROIs

Writes what was measured in each [ROI](plugin:Panels/Glossary) (region of
interest: a shape, or a track, kept in the
[ROI manager](plugin:Panels/ROI manager)) onto the ROI itself: as its
**angle**, a **range** of any attribute (frames, z, time, ...), a named **line**, a
**keyword**, or its **centre**.

Some measurements say more than a number in a table.  The direction a
motor's track moved is the frame its steps should be measured in; the frame
where a molecule stops settling is where its analysis should start; a
fitted axis is a line worth seeing on the structure; "elongated" is a
category to sort and compare by.  This plugin turns such a value into the
ROI's own property, so the next measurement uses it, the ROI manager shows
it, and the summaries group by it.

It measures nothing itself.  Each target reads columns of what the ROIs'
steps, or the segmenter that made them, measured -- any plugin that writes
the value, a user's own too, can be the source.  For a motor's direction,
for example, the [Trace inspector](plugin:Analysis/Tracking/Trace inspector),
run in each ROI, measures `angle_deg`, and [Segment regions](plugin:Analysis/Regions/Segment regions)
from an id attribute measures it while it makes the ROIs.

What was set by hand in the ROI manager is worth more than a value
measured again, so by default it is left alone: new ROIs can be given
theirs without undoing corrections.

## How it works

**Write to.**  Only the chosen destination's settings are shown, and only
they are read.

### To the localizations

**Which localizations.**  A region holds the localizations inside its
outline -- whatever the layers' filters -- or, for an id attribute, the rows
with its id: membership is the region alone, as when a block runs.  An ROI
of an id alone (a track) holds that id's localizations, and an ROI's frame
range and z range narrow what it holds.  Only the ROIs ticked *use* are
written.

**The values.**  Each region's number is written over its rows.  A
localization in no region, or in a region with no result (the measurement
failed there, or the region came after it), gets no value (NaN): a filter
on the attribute drops it.  The ROI number is 0 outside every ROI.

**Overlaps.**  Where two ROIs overlap, a localization takes the later ROI's
value -- the one made after the other -- or, with *overlaps*, the first's or
the mean of both.  The regions of an id attribute never overlap.  A block's
*write values* writes by the same rules, the block's own *overlaps*; its
fourth choice, *any*, is for a flag the block writes back and is refused
for written values.

**Merged blinks.**  The attributes are kept when blinks are merged -- [merge
blinks](plugin:Panels/Glossary), also called grouping or linking, combines
the localizations in consecutive frames that come from one blinking event
into one: a blink takes the mean of its localizations' values (all the same
unless the blink crosses two regions), and the highest ROI number.

### To the ROIs

**Measure first.**  Run the steps that measure the value in the ROIs, or
make the ROIs with a segmenter that measures it.  Its columns are then the
choices of each target's fields.

**Choose the target.**  *set* picks one; only its part of the form is
shown.

- *angle*: one column, a direction in degrees.  It is the ROI's own frame:
  every step run in it then gets the localizations turned so the
  structure lies along $x$ (`x_site_nm`), and the ROI manager's *rotate*
  shows it so.
- *a range*: on an *attribute* -- the ROI manager's *ranges*, frame and
  z unless changed, or any other the table has -- from a low and a high
  column, an end left *open* open, or from a centre and a width, the range
  then the centre plus and minus *k* widths.  What is measured in the ROI
  is that part of it.  An attribute the ROI manager does not offer yet
  joins its list.
- *a line*: a named line of the ROI, from its two ends or from a centre,
  an angle and a length -- the line then points the way the angle does,
  from behind the centre to in front of it.
- *a keyword*: a keyword of one of the ROI manager's lists, given where a
  condition on the measured columns holds, in
  [Filter ROIs](plugin:Analysis/Regions/Filter ROIs)' syntax; where it does
  not, another keyword, none, or the keyword left as it was.
- *centre*: two columns.  An ROI of a shape is moved there with its
  outline and lines; an ROI of an id alone only has its centre set, the
  point its own frame turns about.

**Write it.**  The ROIs selected in the ROI manager's list -- all of them
when none is, and always all in a chain or a batch -- are the ones it
changes: *use* is for the analysis, what is changed is chosen by
selecting.  Each ROI it may change gets the value of its row.  An ROI
with no value in a column the target reads is left as it is.  All of it is
one step of the ROI manager's undo.

## In detail

### To the localizations

**Which table.**  The localizations, whatever the layer shows: the
values are written onto every localization of a region, and the merged
blinks take them when they are merged again (*Merged blinks*, above).

**What may be written over.**  The table remembers which attributes were
written from regions (`metadata["region_columns"]`, with where each came
from).  Running again writes over those, but never over an attribute the
localizations had of their own: a value named like one (`photons`, say) is
refused, and *prefix* gives it another name.

**Which ROI results.**  For the ROIs, the values are those of the results
that still describe each ROI as it is now -- the same rows the analyses read
-- so an ROI moved since it was measured gets no value until it is
measured again.

### To the ROIs

**Who set what.**  Every ROI keeps who set each of its properties
(`ROI.set_by`): *hand* for one set in the ROI manager -- the angle turned,
a range typed or dragged, a line drawn, a keyword given -- or the name of
the method that set it, this plugin's, a segmenter's, *Centre ROIs*'.  Each
target is its own property: `angle`, `range:<attribute>`, `line:<name>` and
`keyword:<list>` per attribute, name and list; so a line drawn by hand does
not protect the angle.  *change* reads it.  The centre is never protected:
an ROI centred by a click is placed by eye, roughly, and a measured centre
always replaces it.  An angle loaded from a file that kept no such note, SMAP's
rotation line among them, counts as set by hand.

**Directions** are degrees anticlockwise from $+x$ in the file's
coordinates, $\mathrm{atan2}(dy, dx)$, the convention of the ROI manager's
lines and of `angle_deg` in the tracking plugins, and an angle is kept
between $-180$° and $180$°.  An axis that has no direction -- an ellipse's
long axis, between 0° and 180° -- is taken as it is: which end is on the
left is then arbitrary.

**Whole numbers.**  On `frame`, `tid` or an attribute ending `_id`, the
ends are rounded to whole numbers.  A range whose high end comes before its
low end is not set.  A column must be in the attribute's own units: a time
in seconds is not turned into frames -- set the range on `time_s` instead.

**What goes out of date.**  An angle, a range, a line or a centre is part
of what the ROI's results were measured in, so the ROIs whose value changed
are measured again by the per-region steps; the text says so.  A keyword is
a note, and nothing is measured again.

**A new line name** joins the project's line names, so the ROI manager's
*Line* button offers it.

## Parameters

### to
*the localizations* changes the table, and is undone with it; *the ROIs*
is one step of the ROI manager's undo and changes no localization.  The
ROIs can only take results measured in them: for the regions of an id
attribute, write to the localizations.

### localizations.source
Lists every current set of results on the ROIs by its step's name, then
every current set for an id attribute's regions as *per cluster_id: ...*.  Left empty it
writes only the ROIs' numbers.  Results that no longer describe their regions
-- re-clustered, the table or a filter changed -- are refused with a message
saying so; an attribute written by this plugin is not such a change, so a second
value can be written from the same results.

### localizations.values
The names are those of the results' columns, as *Histogram* on region results
shows them.  The lines drawn on the ROIs in the ROI manager give
`<line>_angle_deg` and `<line>_length_nm` with or without *results from*:
they are the ROIs' own, as `roi_id` is, and need no measurement; so are the
keyword attributes, `<list>_keyword`.  A keyword its list no longer has is not
written (the ROI keeps it, but it has no number).  Writing every number of a set of results adds an attribute for each:
name the few you will filter or colour by.

### localizations.prefix
Needed when a value has a name the table already uses, and useful to keep
two snapshots side by side: `before_`, `after_`.

### localizations.merge
Only matters where ROIs overlap.  *mean* suits a value measured over both;
*last* keeps the newer ROI's, as drawn.

### rois.target
Only one target per run: in a chain, one step per target.

### rois.which
*all but those set by hand* is the usual choice, and the one to run again
after adding ROIs.  *only those not set yet* also keeps what an earlier
run or a segmenter set.  *all, those set by hand too* starts over -- one
undo step takes it back.

### rois.range.low
For a track's tail, the frame (or the time) its analysis starts at.  A
value just measured in the ROI was measured with its old range; measuring
again with the new one can move it.

### rois.range.k
2 takes in about 95 % of a Gaussian spread, 3 nearly all of it.

### rois.line.mode
*a centre, an angle and a length* is how the tracking plugins describe a
track's axis; *its two ends* fits a measurement that reports two points.

### rois.keyword.condition
Every name in it must be a measured column; one no ROI has is refused, as
in Filter ROIs, since it is nearly always a typing error.

### rois.keyword.otherwise
*leave it* changes only the ROIs the condition holds for; *none* clears
the keyword elsewhere.

## Output

**To the localizations.**  The table with the new attributes; the text names them and says over how many
regions and localizations they were written, and how many localizations are
in none.

A histogram of such an attribute in the filter counts **localizations**, not
regions: a large cluster weighs more than a small one.  For the distribution
over regions, use [Histogram](plugin:Analysis/Measure/Histogram) with *source* on
region results.

**To the ROIs.**  The ROIs' new properties, and a line saying how many were set, how many
changed, and how many were passed over and why: set by hand, already set,
not used, nothing measured, or (for a keyword) left as they were.
