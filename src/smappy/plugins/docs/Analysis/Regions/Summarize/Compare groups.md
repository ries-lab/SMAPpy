---
version: "1"
covers: [smappy.plugins.compare_groups.CompareGroups.run, smappy.plugins.compare_groups.group_labels, smappy.plugins.compare_groups.group_columns, smappy.plugins.compare_groups.numeric_fields, smappy.plugins.compare_groups.compare, smappy.plugins.compare_groups.attribute_columns, smappy.plugins.compare_groups.attribute_labels]
---

## What it does

Compares what was measured per region between groups of regions: the
traces of one file against another's, the ROIs marked "good" against those
marked "bad", the clusters of one condition against another.  For each
measure it draws a box (or a violin) per group, side by side, and gives a
table of the medians -- a grid of plots, as the MFX scripts draw per
condition, for any [region table](plugin:Panels/Glossary): of the ROIs, or
of an [id attribute](plugin:Panels/Glossary) such as a MINFLUX trace
(`tid`).

The groups are the **files** -- open several measurements, one per
condition, and measure them in one go -- or any column of the results with
few distinct values: an ROI keyword, a value written per region.  For the
regions of an id attribute they can also be an **attribute of their
localizations** with few values: a condition *File conditions* wrote (a
laser power, a buffer), a MINFLUX thread (`thi`) or measurement.

## How it works

```figure-setup
from smappy.plugins.compare_groups import compare

rng = np.random.default_rng(0)
# per trace: its spread; two conditions, 40 traces each
rows = ([{"roi_id": i, "std_xy_nm": rng.normal(3.0, 0.4)} for i in range(40)]
        + [{"roi_id": 40 + i, "std_xy_nm": rng.normal(5.0, 0.8)} for i in range(40)])
groups = ["buffer A"] * 40 + ["buffer B"] * 40
table = compare(rows, groups, ["std_xy_nm"], ["buffer A", "buffer B"])
```

**1. The results.**  The rows of the chosen *results from* -- a region
table, one row per region.

**2. Each row's group.**  With *group by* = file: an ROI's file, or, for a
region of an id attribute, the file its localizations are in.  With an
attribute of the localizations: the value of the region's first
localization, named by its word where *File conditions* coded one.
Otherwise the row's value of that column.

**3. Per measure and group** the values, their median, mean and count;
the medians as a table in the text, the values as a box (median, quartiles,
whiskers to 1.5 times the interquartile range) or a violin per group, one
panel per measure.

```figure Traces of two simulated conditions: their spread per trace, one box per condition.
fig.set_size_inches(3.5, 2.8)
ax = fig.subplots()
ax.boxplot([table["values"]["std_xy_nm"][g] for g in ("buffer A", "buffer B")],
           positions=[0, 1], widths=0.6, showfliers=False)
ax.set_xticks([0, 1])
ax.set_xticklabels(["buffer A", "buffer B"])
ax.set_ylabel("std xy (nm)")
```

## Parameters

### group_by
*file* for measurements opened together, one per condition.  Columns of the
results with at most 12 distinct values are offered (an ROI keyword list, a
value written per region with few levels), and for an id attribute's
regions the localizations' attributes with at most 12 values, marked "of
the localizations".

### fields
Name the few measures that answer the question; every numeric column makes
a large grid.

## Output

A table of the medians per measure and group in the text, the number of
regions per group, and a grid of panels, up to four across, one per
measure.
