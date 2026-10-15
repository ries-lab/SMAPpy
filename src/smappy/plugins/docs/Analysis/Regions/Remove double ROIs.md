---
version: "1"
covers: [smappy.plugins.roi_order.RemoveDoubleROIs.run, smappy.plugins.roi_order.double_ids, smappy.roi_manager.order.doubles]
---

## What it does

A segmenter sometimes puts two [ROIs](plugin:Panels/Glossary) (regions of
interest: shapes kept in the [ROI manager](plugin:Panels/ROI manager),
inside which things are measured) on one structure: a nuclear pore found
twice, a bright cluster split in two peaks, ROIs added by hand over ones
already found.  Measured twice, such a structure counts twice in every
summary.  This plugin finds the used ROIs whose centres are closer than a
distance and, of each such pair, keeps the one with more localizations and
excludes the other: it sets its *use* off.

Run it after segmenting and before measuring, with the distance a little
below the spacing of real neighbours: below a pore's diameter for pores.

## How it works

```figure-setup
from matplotlib.patches import Circle
from smappy.roi_manager.order import doubles
rng = np.random.default_rng(3)
sites = np.array([(500, 500), (1500, 600), (900, 1500), (1900, 1700)], float)
locs = np.vstack([rng.normal(s, 40, (n, 2)) for s, n in zip(sites, (60, 40, 80, 50))])
# ROIs: one on each site, and three doubles a little off a site
centers = np.vstack([sites, sites[[0, 2, 3]] + [(90, -60), (-80, 70), (80, 70)]])
radius = 150
counts = [int((np.hypot(*(locs - c).T) < radius).sum()) for c in centers]
double = doubles(centers, counts, 200)
```

**1. How large each ROI is.**  Each used ROI's localizations are counted,
as the ROI manager shows them: inside the ROI's shape, under the layer's
[filter](plugin:Panels/Glossary) (the lowest and highest values of columns a
localization must lie between), and as merged blinks when the layer merges
them.  An excluded ROI takes no part: it neither stays nor removes another.

**2. Who stays.**  The ROIs are taken from the most localizations down.  One
stays unless an ROI that already stays, of the same file, has its centre
closer than *closer than*; otherwise it is a double and is excluded.  So of
two doubles the larger stays, whichever was found first, and of three in a
row where only neighbours are close, the two ends stay and the middle one
goes.

```figure Four sites (localizations in grey), each with an ROI, and three doubles near three of them.  Of each close pair the ROI that holds more localizations stays (blue); the other is excluded (red, dashed).  The numbers are the localizations in each ROI.
ax = fig.subplots()
ax.scatter(locs[:, 0], locs[:, 1], s=2, color="0.6")
for c, n, d in zip(centers, counts, double):
    ax.add_patch(Circle(c, radius, fill=False, lw=1.5,
                            color="#d62728" if d else "#1f77b4",
                            ls="--" if d else "-"))
    ax.text(c[0] + (radius + 40) * (1 if d else -1), c[1], str(n),
            ha="left" if d else "right", va="center", fontsize=7,
            color="#d62728" if d else "#1f77b4")
ax.set_aspect("equal")
ax.set_xlim(0, 2500); ax.set_ylim(200, 2100)
ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
```

**3. The edit.**  The doubles' *use* is set off, as one step of the ROI
manager's undo (*Undo ROI edit*).  Nothing is deleted: an excluded ROI keeps
its number and its results, and can be ticked again.

## In detail

ROIs of different files are never doubles: the files share coordinates.  Two
ROIs exactly *closer than* apart are not doubles.  Of two with the same
number of localizations, the one earlier in the ROI list (as sorted by
[Sort ROIs](plugin:Analysis/Regions/Sort ROIs)) stays.  The neighbours are
found with a k-d tree, so even 10^5 ROIs take a fraction of a second; the
counting is what costs, a few milliseconds per ROI.  Before the run, the
line under the panel says how many ROIs it would exclude (up to 2000 ROIs).

## Parameters

### distance_nm
Below the closest spacing two real structures can have, above how far two
ROIs on one structure lie apart -- a fraction of the ROI size.  For nuclear
pores, about 100 nm.

## Output

The line under the panel names the ROIs it excluded and how many are used
now.  Excluded ROIs turn grey in the ROI manager and drop out of the
results.

## Differences from SMAP

Ported from SMAP's *RemoveDoubleSites*
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).

* **The larger stays.**  SMAP walks the site list in its order and deletes
  every site near the current one, so the one found first survives, often
  the worse-placed of the two.  Here the ROI with more localizations stays.
* **Excluded, not deleted**, as one undo step, and only among the used ROIs.
* **Per file.**  SMAP compares positions across files too.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
