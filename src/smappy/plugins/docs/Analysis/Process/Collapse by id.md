---
version: "1"
covers: [smappy.table_edit.collapse, smappy.table_edit.collapse_index, smappy.table_edit.id_candidates, smappy.group.combine, smappy.plugins.collapse.Collapse.run]
---

## What it does

Replaces the table with one row per value of an id column: a row per
[merged blink](plugin:Panels/Glossary) (`group_id`), per track
(`track_id`, from [Link tracks](plugin:Analysis/Tracking/Link tracks)), per
cluster (`cluster_id`, from [Find clusters](plugin:Analysis/Regions/Find clusters)),
or per value of any other whole-number column.  Each column is combined as
merging blinks combines it: positions averaged, weighted by their
precision, photons added up.

Its use is a smaller table -- a file of tracks or clusters rather than of
their localizations, to save or hand on.  The localizations are gone from
the table; the undo brings them back.  To look at merged blinks, merge
them on a layer instead: that keeps the localizations.

## How it works

```figure-setup
from smappy.locs import Localizations
from smappy.simulate import simulate
from smappy.simulate.settings import (SimulationSettings, StructureSettings,
                                      LabellingSettings)
from smappy.plugins.clusters import dbscan
from smappy.table_edit import collapse
locs = simulate(SimulationSettings(n_frames=3000, seed=3,
                                   structure=StructureSettings(preset="npc"),
                                   labelling=LabellingSettings(efficiency=0.6)))
locs = locs[np.asarray(locs["photons"]) > 200]
xy = np.column_stack([np.asarray(locs["x_nm"]), np.asarray(locs["y_nm"])])
labels = dbscan(xy, 50, 10)
table = Localizations({**locs.columns, "cluster_id": labels.astype(np.int64)},
                      dict(locs.metadata))
rows, _ = collapse(table, "cluster_id", "remove")
```

**1. The groups.**  One group per id, separately per file and per channel:
an id numbered per file meets the same number in the next file, and a
cluster that spans both channels of a two-colour table becomes a row per
channel rather than one whose channel is 1.5.

**2. The rows without an id.**  An id of 0 or below means "in none" --
DBSCAN's noise, a localization in no track.  With *rows without an id* on
*keep* each is a row of its own, its values as they were; on *remove* they
are not in the new table.

**3. Each column combined**, by the rule merging blinks uses for it: a
weighted mean for positions and most columns, the sum for `photons` and
`background`, the first frame for `frame`, the combined precision for
`xy_err_nm` and the other errors.  The id column keeps its ids exactly.

**4. The count.**  `n_merged` says how many rows each one was made from.

```figure Simulated nuclear pores (grey), found as clusters by DBSCAN and collapsed by cluster_id: one row per pore, at the weighted mean of its localizations, the circle's size the number it was made from.
fig.set_size_inches(5.0, 4.0)
ax = fig.subplots()
ax.scatter(xy[:, 0], xy[:, 1], s=0.2, color="0.8")
ax.scatter(rows["x_nm"], rows["y_nm"], s=2 + 0.5 * np.asarray(rows["n_merged"]),
           facecolor="none", edgecolor="C3", lw=0.8)
ax.set_aspect("equal")
ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
ax.set_title(f"{len(table)} localizations, {len(rows)} clusters", fontsize=8)
```

## In detail

**What is lost.**  What one row cannot hold: `logl`, which has no value for
a group; the frames of the members but the first; and every column's
spread within the group.  The table's own `group_id` and `n_in_group` are
not kept: they described localizations that are gone, and merging the
blinks of the collapsed table writes its own.  Collapsed by `group_id`,
the id column goes too, for the same reason.

**Derived columns.**  A column [Math Parser](plugin:Analysis/Process/Math Parser)
defines is combined by its recipe's rule, or computed again from its
expression.  One whose expression reads `n_in_group` or `group_id` -- an
on-time in milliseconds, say -- is computed on the groups and then kept
as values: its expression would read columns that are not kept.

**Merging again.**  A layer that merges blinks merges the rows of the new
table as it would localizations: rows in consecutive frames close
together become one.  Collapsed by `group_id`, a row starts where its
blink started, so two blinks of one molecule are merged only when the
first frame of the second follows within *dark frames* of the first
frame of the first.

**The table remembers** that it was collapsed, by which column and from
how many rows, in its metadata (`collapsed`).

## Parameters

### column
The ids offered are the whole-number columns the table has, the usual ids
first.  `group_id` is there once blinks have been merged on a layer.

### unassigned
*keep* when the noise matters to what comes next (a density, a picture);
*remove* for a table of clusters or tracks only.

## Output

The table, one row per id, and a line saying how many rows became how
many, and what was not kept.

## Differences from SMAP

Based on SMAP's `GroupAny` and the compact form of `Connect2Unconnect`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).  The main
changes:

* A row with no id is kept or removed by choice; GroupAny puts all of a
  file's rows with id 0 into one group.  Both group per file and channel.
* Connect2Unconnect replaces one channel's localizations with its merged
  blinks; collapsed by `group_id` this does it for every channel at once.
  Its other direction (*unconnect->connect*) was never written in SMAP.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
