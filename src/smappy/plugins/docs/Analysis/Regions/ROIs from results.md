---
version: "1"
covers: [smappy.plugins.roi_results.ROIsFromResults.run, smappy.plugins.roi_results.ROIsFromResults.describe, smappy.plugins.roi_results.new_values, smappy.plugins.roi_results.apply, smappy.plugins.roi_results.range_value, smappy.plugins.roi_results.line_value, smappy.plugins.roi_results.keyword_values, smappy.roi_manager.core.ROI.set_source, smappy.roi_manager.core.ROIProject.frame]
---

## What it does

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

**Write it.**  Each ROI it may change gets the value of its row.  An ROI
with no value in a column the target reads is left as it is.  All of it is
one step of the ROI manager's undo.

## In detail

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

### target
Only one target per run: in a chain, one step per target.

### which
*all but those set by hand* is the usual choice, and the one to run again
after adding ROIs.  *only those not set yet* also keeps what an earlier
run or a segmenter set.  *all, those set by hand too* starts over -- one
undo step takes it back.

### used_only
Off, the excluded ROIs are given the value too, which matters only if they
are used again later.

### range.low
For a track's tail, the frame (or the time) its analysis starts at.  A
value just measured in the ROI was measured with its old range; measuring
again with the new one can move it.

### range.k
2 takes in about 95 % of a Gaussian spread, 3 nearly all of it.

### line.mode
*a centre, an angle and a length* is how the tracking plugins describe a
track's axis; *its two ends* fits a measurement that reports two points.

### keyword.condition
Every name in it must be a measured column; one no ROI has is refused, as
in Filter ROIs, since it is nearly always a typing error.

### keyword.otherwise
*leave it* changes only the ROIs the condition holds for; *none* clears
the keyword elsewhere.

## Output

The ROIs' new properties, and a line saying how many were set, how many
changed, and how many were passed over and why: set by hand, already set,
not used, nothing measured, or (for a keyword) left as they were.
