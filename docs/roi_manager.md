# ROI manager

The ROI manager organizes **files → ROIs**.  An ROI (region of interest) is a
shape -- a circle, a square or a polygon -- around one copy of a structure,
inside which it is measured.  Its window has four quadrants: the
whole file, a zoom, the ROI itself, and the file and ROI lists.  The zoom is a
navigation aid; moving through the file never moves ROIs.

## Open the manager

Start the GUI and open the manager from the Regions tab's **Open ROI manager**, from
**Tools → ROI manager**, or with Ctrl+R:

```sh
python -m smappy.gui.app localizations.h5
```

The files are the ones the session has loaded, so *File → Open* and *Add file*
are how sources arrive; a fitted or drift-corrected table can be used without
saving it first.  Inputs are SMAPpy localization HDF5 files, SMAP `_sml.mat`,
MINFLUX exports and csv, whatever the session can read.  Coordinates must be in
nm, or carry `pixelsize_nm` metadata for conversion.

The matplotlib interface and the `smappy-roi` command that this guide first
described are gone; `ROIProject` and its plugins are unchanged and still
scriptable, including `save` and `load` of a sidecar project file.

## Manual selection

1. Click the file image to move the zoom there, or drag inside the zoom.
2. Click empty space in the zoom to draft an ROI.  The ROI image shows it;
   click there to recentre it, then **Add** or Enter stores it.
3. Click an ROI's *outline* to select it.  A click inside one drafts a new ROI
   instead, which is what makes overlapping ROIs drawable.
4. A region drawn in the 2D view becomes an ROI with **Add the drawn region as
   an ROI** in the tab, keeping its rectangle, line or polygon outline.
5. Select an ROI in the list to jump to its file and centre the zoom on it.
   The **use** checkbox, or the space bar, includes or excludes it; **Remove**
   deletes it, and previous run records stay in the history.

There is no review step: an ROI counts from the moment it is made, and *use* is
what excludes one.

**Polygon** draws an outline in the ROI image: a click per vertex, and the first
vertex or a double-click closes it.  A closed polygon replaces the global circle
or square for that ROI and moves its centre to the centroid; **Clear shape**
gives the global geometry back.  **Direction** takes two clicks, start then end,
and records an arrow the analysis plugins can read; it does not rotate anything.
**Clear line** removes it.  Escape abandons a drawing.  Both apply to the
selected ROI, or to a draft, which then carries them when **Add** stores it.

The default circle has a **300 nm diameter**.  For a square, the global size is
its **side length**; both are set in the Regions tab and apply to every ROI at once.
Polygons keep their own boundaries.  **ROI view** sets what the ROI image shows
around the ROI, independently of the analysis geometry.

The boundary is inclusive for circles, squares and polygon edges. Overlapping
ROIs are independent: a localization in their overlap contributes to both.

## Filters and merged blinks

Filters (also called bounds: the lowest and highest values of columns a
localization must lie between) and *merge blinks* (also called grouping or
linking: the localizations in consecutive frames that come from one blinking
event combined into one) come from the render layer, so what an ROI measures
is what the image shows; there is no separate filter panel.  Change them in
the Render tab and the ROIs follow.

A field absent from a source is not filtered; absent fields are named above the
views. The Python API supports additional range filters. File-selection filters
(`filenumber`, `file_id`) are ignored: each ROI is always extracted from its own
source file. The detail view and preview limits never restrict analysis.

The layer's **merge blinks** switch applies to the ROIs as well: the
localization count then counts blinks, and photons are the photon sums of those
blinks.  The merge settings are session-wide, under *merge settings...* beside
*merge blinks*.

## Find, review, measure, histogram

Candidates are found by the **Density Peaks** segmenter
(`Analysis/Regions/Find/Density Peaks`, pinned in the Regions tab), which operates on
the current file's filtered localizations. It
bins coordinates into a density image, smooths it, finds local peaks, recenters
on nearby localizations, and suppresses nearby centers. Defaults target compact
structures of about 100 nm diameter:

| Setting | Default | Meaning |
| --- | --- | --- |
| bin | 20 nm | Density-image pixel width |
| smoothing | 50 nm | Gaussian smoothing standard deviation |
| separation | 150 nm | Minimum distance from an accepted or existing center |
| count radius | 75 nm | Neighborhood used to recenter and check count |
| minimum count | 10 | Minimum filtered localizations within that radius |

These are detection settings, separate from the analysis ROI size. Outlines are amber, grey when the ROI is excluded, green when it is selected,
and a draft is blue and dashed; a drawing in progress is a dashed cyan rubber
line and a direction is a magenta arrow.

**Run**, in the *Per-region steps...* window, runs the steps on every
included ROI across all files, measuring only what is out of date or
missing. **Localization
Statistics** (`Analysis/Measure/Localization Statistics`, the same plugin as
on the view) returns the localization count, the mean and median photons and
`N0`, the mean and median lateral precision and, when its precision model is
switched on, `sigma_c`.  Nonfinite values are omitted from each mean. Empty
selections return count zero and NaN means.
Missing required measurement columns produce a per-ROI error without stopping
other ROIs. Select the error row to see its message.

The **Histograms** analysis (`Analysis/Regions/Summarize/Histograms`) plots the
current results for included ROIs.  NaN means are
omitted, with the number of missing values shown.  A histogram is a snapshot:
run it again after anything changes.

Synthetic tests cover two 100 nm rings and two blobs, each with 30 localizations.
Detection settings still need checking on experimental NPC data, especially for
background, adjacent pores, and nonuniform labeling. The first finder uses one
dense image with a limit of 16 million pixels; increase the bin width if a file
exceeds that limit.

## Saving and reproducibility

ROIs are part of the session: **File → Save** writes them into the localization
file's metadata, together with the geometry, the use flags, the finder
provenance, the runs of the per-region steps and which ROI was selected.  Opening that file
brings all of it back.  Draft ROIs are not saved until added.

`ROIProject.save` and `ROIProject.load` still write and read the versioned HDF5
sidecar described below, for scripts that want ROIs in a file of their own; the
GUI does not use it.  That sidecar holds source references, geometry, flags,
comments, provenance, settings and run history, with source paths relative to
the project.

Every measurement records its plugin name/version, parameters, timestamp, source
fingerprint, effective filters, merge settings, and geometry. A changed
analysis input makes the latest result outdated; it is excluded from current
histograms until measured again. Global size/shape changes do not invalidate
polygon ROIs. Comments, the *use* tick, contrast, and navigation do not change the
numerical results. Old runs remain available in `project.runs`.

## Python API and plugins

```python
from smappy.plugins.roi import HistogramSettings
from smappy.roi_manager import ROIProject, histograms

project = ROIProject()
source = project.add_file("localizations.h5")
project.set_geometry(300, "circle")
project.set_filters({"xy_err_nm": (None, 25), "photons": (500, None)})

# found ROIs are included as the finder judged them (`use`)
candidates = project.find(source.id, parameters={"min_count": 15})

# Or add one by hand:
roi = project.add_roi(source.id, [1500, 2500])
locs = project.extract(roi)
run = project.evaluate()
rows = project.results()
counts = histograms(rows, HistogramSettings(bins=20))
project.save("experiment.rois.h5")
```

In the GUI the project is a `SessionROIs`, the same model with the session's
files as its sources and the layer's filters and merge settings instead of its own.
The GUI uses the built-in plugins. Scripts can pass another plugin object to
`project.find(..., plugin=...)` or `project.evaluate(plugin=...)`. Plugins declare
stable `name` and `version` strings and use these interfaces:

```python
finder.find(filtered_localizations, parameters, existing_centers)  # list of xy centers
per_roi.evaluate(selected_localizations, geometry, parameters)     # dictionary of results
population.analyze(result_rows, parameters)                       # analysis data
```

No plugin needs a window. Per-region plugins receive absolute nm coordinates and an
explicit geometry dictionary, so such a plugin can derive local/rotated
coordinates when required. Return JSON-compatible values, numeric arrays or
NumPy scalars; saved runs contain data rather than plotting handles. Custom
plugin result rows are available with `project.results(plugin.name)`.

For relocated sources, relink by stable source ID while loading:

```python
project = ROIProject.load("experiment.rois.h5", source_paths={source_id: new_path})
```

The source-content check still applies after relinking.
