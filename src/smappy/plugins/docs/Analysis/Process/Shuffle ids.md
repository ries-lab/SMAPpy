---
version: "1"
covers: [smappy.plugins.shuffle_ids.shuffle_ids, smappy.plugins.shuffle_ids.ShuffleIds.run]
---

## What it does

Writes a copy of an integer column with its ids in a random order, for
colouring by.  A clusterer numbers its clusters one after the other --
[DBSCAN](plugin:Analysis/Regions/Find/DBSCAN) by size -- and coloured by
`cluster_id` neighbouring clusters often come out in neighbouring colours of
the colour table ([LUT](plugin:Panels/Glossary), look-up table), so two
touching clusters look like one.  Coloured by the shuffled column, they are
far apart on the LUT.

It takes any whole-number column, not only clusters: an ROI's number
(`roi_id`), a MINFLUX trace (`tid`), a track, the frame.  The column itself
is left as it is -- results and regions are kept by its ids -- and the
shuffled ids go into a new one, `<column>_shuffled`; colour the layer by
that in the Render tab.

## How it works

```figure-setup
from smappy.simulate import simulate
from smappy.plugins.clusters import dbscan
from smappy.plugins.shuffle_ids import shuffle_ids
locs = simulate(n_frames=3000, seed=2)
xy = np.column_stack([np.asarray(locs["x_nm"]), np.asarray(locs["y_nm"])])
labels = dbscan(xy, 40.0, 10)
shuffled = shuffle_ids(labels, seed=0)
inside = labels > 0
```

**1. The ids.**  The distinct ids above 0 in the column.  An id of 0 or
below means "in none" -- DBSCAN's noise, a localization in no ROI -- and is
kept.

**2. A random order.**  The ids are permuted: each id is replaced by
another of the same ids, each used once.  The shuffled column has the same
values as the column, so a colour range set for one suits the other.

**3. The new column**, with the rule that a [merged blink](plugin:Panels/Glossary)
keeps its largest value, as `cluster_id` has.

```figure Clusters coloured by their id, numbered by size, and by the shuffled id: neighbours that shared a colour are told apart.
fig.set_size_inches(7.0, 3.2)
for ax, ids, title in zip(fig.subplots(1, 2), (labels, shuffled),
                          ("cluster_id", "cluster_id_shuffled")):
    ax.scatter(xy[~inside, 0], xy[~inside, 1], s=0.2, color="0.85")
    ax.scatter(xy[inside, 0], xy[inside, 1], s=0.4, c=ids[inside], cmap="viridis")
    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")
```

## Parameters

### seed
Leave it on *auto* for a new order each run, and run again when two
neighbours still came out alike.  The seed used is in the history, so a
picture can be made again with the same colours.

## Output

The table with the new column, and a line saying how many ids were
shuffled and with which seed.

## Differences from SMAP

Based on SMAP's `ShuffelClusterIndex`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)), which also
writes a new column (`<field>_rnd`).  The main changes:

* The ids that are there are permuted among themselves; SMAP permutes all
  the numbers from 1 to the largest id, so a column with gaps -- frames,
  ROI numbers after deletions -- gets values it never had, and its colour
  range no longer fits.
* An id below 0 is kept; SMAP sets it to 0.
* The seed is recorded, so the same colours can be made again, and a merged
  blink keeps one value.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
