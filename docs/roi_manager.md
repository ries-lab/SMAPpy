# ROI manager

The ROI manager organizes **files → ROIs**. The three linked images are a file
overview, a movable detail view, and an ROI preview. The detail view is only a
navigation aid: there are no stored cells, and moving through the file never
moves ROIs.

## Open the manager

Use the existing `viewer` extra for matplotlib. From a source checkout:

```sh
python -m smappy.cli.roi localizations.h5
python -m smappy.cli.roi first.h5 second.h5
python -m smappy.cli.roi --project experiment.rois.h5
```

An installation made after this change also provides `smappy-roi` with the same
arguments. Starting without arguments opens an empty manager; enter a localization
file path in the controls window and click **Add file**. Use **Previous file** and
**Next file** to navigate the loaded files.

Inputs are SMAPpy localization HDF5 files. Coordinates must be in nm, or contain
pixel coordinates and `pixelsize_nm` metadata for conversion. SMAP `_sml.mat`
projects are not imported by this first version.

## Manual selection

1. Click the overview to position the detail view. Alternatively, use **Next
   tile** and **Previous tile** for a serpentine grid with spacing equal to the
   detail width.
2. Click the detail view to create a draft ROI. Click in its preview to recenter
   it, then use **Add ROI** or Enter to store it. Manual additions are reviewed.
3. To create an individual boundary, click **Polygon**, place vertices in the
   preview, and close the polygon by clicking its first vertex. Esc cancels the
   drawing tool. **Clear shape** restores the global circle/square boundary.
4. **Direction** takes two clicks, start then end. It records an arrow available
   to analysis plugins; it does not rotate the global square. **Clear line**
   removes it. Comments are stored with **Set comment**.
5. Click a table row to inspect a saved ROI. Left/right arrow keys step through
   the current file's ROIs. Right-click its preview to translate the ROI,
   including its polygon and direction line. **Remove ROI** removes it from the
   collection; previous run records remain in the project history.

The default circle has a **300 nm diameter**. For a square, the global size is
its **side length**. **Apply geometry** updates all ROIs using global geometry.
Polygons retain their individually drawn boundaries. Detail and preview widths
control the visible neighborhood independently of analysis geometry.

The boundary is inclusive for circles, squares and polygon edges. Overlapping
ROIs are independent: a localization in their overlap contributes to both.

## Filters and grouping

Global range filters apply to rendering, cluster finding and analysis. The
starting bounds match `smappy-view`: precision at most 25 nm, relative
log-likelihood at least -1.5, and z between -500 and 500 nm, wherever those
columns exist. Photons and frame are initially unbounded. Blank fields remove a
bound; click **Apply filters** after editing.

A field absent from a source is not filtered; absent fields are named above the
views. The Python API supports additional range filters. File-selection filters
(`filenumber`, `file_id`) are ignored: each ROI is always extracted from its own
source file. The detail view and preview limits never restrict analysis.

**Grouped** followed by **Apply grouping** switches both rendering and analysis
to grouped localizations using SMAPpy's existing grouping implementation. The
default link distance is 50 nm and frame gap is 1. Filters are applied *after*
grouping. Thus the localization count counts groups, and photons are the photon
sums of those groups. Different grouping settings can be supplied through the
Python API. This manager has its own global filter settings; a separately opened
`smappy-view` window is not synchronized with it.

## Find, review, evaluate, histogram

**Find candidates** operates on the current file's filtered localizations. It
bins coordinates into a density image, smooths it, finds local peaks, recenters
on nearby localizations, and suppresses nearby centers. Defaults target compact
structures of about 100 nm diameter:

| Setting | Default | Meaning |
| --- | --- | --- |
| Bin | 20 nm | Density-image pixel width |
| Sigma | 50 nm | Gaussian smoothing standard deviation |
| Spacing | 150 nm | Minimum distance from an accepted or existing center |
| Count radius | 75 nm | Neighborhood used to recenter and check count |
| Min count | 10 | Minimum filtered localizations within that radius |

These are detection settings, separate from the analysis ROI size. Results are
**unreviewed** until **Accept ROI** or **Accept file ROIs**. **Toggle use** controls
inclusion independently of review; bulk acceptance does not change use flags.
Unreviewed outlines are amber, reviewed outlines green, excluded outlines gray,
and the active ROI blue.

**Evaluate all** runs on all reviewed, included ROIs across all files. The
statistics evaluator returns localization count, arithmetic mean lateral
localization precision (`loc_precision_nm`), and arithmetic mean photons.
Nonfinite measurement values are omitted from each mean; their finite sample
counts are saved separately. Empty selections return count zero and NaN means.
Missing required measurement columns produce a per-ROI error without stopping
other ROIs. Select the error row to see its message.

**Histograms** plots current successful results for reviewed, included ROIs.
NaN means are omitted, with the number of missing values shown. Click a bar to
inspect an ROI in that bin; repeated clicks cycle through its ROIs. A histogram
is a snapshot: if inputs, inclusion or results change, the window asks you to
refresh it with **Histograms**.

Synthetic tests cover two 100 nm rings and two blobs, each with 30 localizations.
Detection settings still need checking on experimental NPC data, especially for
background, adjacent pores, and nonuniform labeling. The first finder uses one
dense image with a limit of 16 million pixels; increase the bin width if a file
exceeds that limit.

## Saving and reproducibility

Enter a project path and click **Save**. **Open** replaces the current project;
save your changes before opening another one. Close does not automatically save.
Draft ROIs are not saved until added to the collection.

A project is a versioned HDF5 sidecar containing source references, ROI geometry,
review and use flags, comments, finder provenance, navigation, global settings,
and evaluation-run history. Source paths are relative to the project, so a
folder containing both can be moved together. Source columns and metadata are
fingerprinted; reopening a project rejects changed source contents. Source data
are held as immutable snapshots while the manager is open. In-memory sources
must have been saved as localization files before saving a project.

Every evaluation records its plugin name/version, parameters, timestamp, source
fingerprint, effective filters, grouping settings, and geometry. A changed
analysis input makes the latest result outdated; it is excluded from current
histograms until evaluated again. Global size/shape changes do not invalidate
polygon ROIs. Comments, review flags, contrast, and navigation do not change the
numerical results. Old runs remain available in `project.runs`.

## Python API and plugins

```python
from smappy.roi_manager import ROIProject, Histograms, show

project = ROIProject()
source = project.add_file("localizations.h5")
project.set_geometry(300, "circle")
project.set_filters({"loc_precision_nm": (None, 25), "photons": (500, None)})

candidates = project.find(source.id, parameters={"min_count": 15})
# Inspect candidates in the GUI before marking them reviewed.
manager = show(project)

# Or add a manually specified ROI:
roi = project.add_roi(source.id, [1500, 2500])
locs = project.extract(roi)
run = project.evaluate()
rows = project.results()
histograms = Histograms().analyze(rows, {"bins": 20})
project.save("experiment.rois.h5")
```

The GUI uses the built-in plugins. Scripts can pass another plugin object to
`project.find(..., plugin=...)` or `project.evaluate(plugin=...)`. Plugins declare
stable `name` and `version` strings and use these interfaces:

```python
finder.find(filtered_localizations, parameters, existing_centers)  # list of xy centers
per_roi.evaluate(selected_localizations, geometry, parameters)     # dictionary of results
population.analyze(result_rows, parameters)                       # analysis data
```

No plugin needs a window. Evaluators receive absolute nm coordinates and an
explicit geometry dictionary, so an evaluator can derive local/rotated
coordinates when required. Return JSON-compatible values, numeric arrays or
NumPy scalars; saved runs contain data rather than plotting handles. Custom
plugin result rows are available with `project.results(plugin.name)`.

For relocated sources, relink by stable source ID while loading:

```python
project = ROIProject.load("experiment.rois.h5", source_paths={source_id: new_path})
```

The source-content check still applies after relinking.
