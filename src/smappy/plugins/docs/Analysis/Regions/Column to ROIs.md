---
version: "1"
covers: [smappy.plugins.region_column.ColumnToROIs.run, smappy.plugins.region_column.column_regions, smappy.plugins.region_column.hull]
---

## What it does

Makes an [ROI](plugin:Panels/Glossary) (region of interest: a shape kept in
the ROI manager, inside which things are measured) in the
[ROI manager](plugin:Panels/ROI manager) for every
[region](plugin:Panels/Glossary) an id column of the table names -- usually
every [cluster](plugin:Panels/Glossary) (localizations lying closer to each
other than to the rest) a clusterer found (`cluster_id`, from
[DBSCAN](plugin:Analysis/Regions/Find/DBSCAN) or any other) -- so the
clusters can be looked at one by one, reviewed, excluded and annotated.

It works on any column of whole numbers: an id of 0 or below means "in no
region" and gets no ROI, and so does an id with fewer than *at least*
localizations.  Use it when the clusters need a person's eye or a note; to
measure every cluster, a block of a [chain](plugin:Panels/Glossary) (plugins
run one after another as one) repeating over `cluster_id` does it without
ROIs, and is much faster for thousands of clusters.

## How it works

**The regions.**  The localizations are sorted by their id; each id above
0 with enough localizations is one region.  Membership is the column alone:
the layers' filters do not decide which localizations a cluster has.

**The outline.**  With *the cluster's own outline* ticked (the default), the
ROI is the convex hull of the cluster's positions -- the smallest shape with
no dents that holds them all -- so it fits a cluster of any shape and size.
A localization exactly on the hull's edge or at a corner counts as inside.
Unticked, only the cluster's centre is placed, and the ROI is the circle or
square of the Regions tab, at its size: the right choice when the clusters
are alike (pores, say) and every ROI should be measured in the same window.
Such ROIs follow the Regions tab's shape and size when they are changed
later, and *Clear shape* in the ROI manager turns a hull ROI into one.  A
cluster whose localizations all lie on one line has no hull; it gets the
centre and the Regions tab's shape even with the outline ticked, and the
text says how many did.

**The ROI.**  Its centre is the mean position of the cluster's
localizations, its file the one its localizations came from, and it
remembers the column, the id and the number of localizations (`origin`).
All the ROIs of a run are one step of the ROI manager's undo.

## In detail

**Which table.**  The localizations, whatever the layer shows: the id
column is theirs, and a region's outline is drawn round all of its
localizations.

## Parameters

### column
Any column of whole numbers: `cluster_id`, an id a dropped-in clusterer
wrote, or `roi_id` from *Regions to column* to bring ROIs back from a table.

### min_size
Large clusters are usually the interesting ones and small ones are noise:
a few times the localizations one fluorophore gives.  The column keeps every
cluster either way, and a filter on `n_in_cluster` does the same on the
table.

### outline
The hull follows each cluster but makes ROIs of different sizes; the circle
or square makes them all alike, cutting off any cluster larger than it, and
taking in the localizations around a smaller one.

### replace
Removes only the ROIs this plugin made of the same column, so ROIs drawn by
hand or found by a finder stay.  Without it, running again adds a second set
over the first.

## Output

The ROIs in the ROI manager; the text says how many were made, the smallest
cluster that became one, and how many clusters had no outline.  It cannot run
inside a block, which repeats over regions: make the ROIs after it.
