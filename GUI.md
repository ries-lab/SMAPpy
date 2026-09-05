# smappy GUI and plugin architecture

The first step of porting SMAP from MATLAB.  Decisions taken 2026-09-04; the
reasoning is here so that it does not have to be re-derived.

## Decisions

* **Toolkit: PySide6 + pyqtgraph** (optional extra `gui`).  napari was
  considered and rejected: its Points layer draws hard-edged markers in an
  8-bit clamped framebuffer, so SMAP's rendering (per-localization Gaussian,
  quantile contrast and gamma applied *after* the sum) cannot be faked with it;
  we would render ourselves and push an image anyway, and napari owns the main
  window.  pyqtgraph gives `ImageItem`/`ViewBox`/`ROI` for 2D now and
  `pyqtgraph.opengl` or vispy for 3D later.  PySide6 is LGPL, which fits the
  BSD licence; PyQt would not.
* **GUI and function are separate.**  Every plugin is callable without the GUI
  and the GUI is generated from the plugin's declaration.  Nothing in
  `smappy.plugins` or `smappy.session` imports Qt.
* **Parameters are a dataclass.**  A plugin's `Settings` is the same object a
  script passes to it.  GUI information (label, unit, bounds, choices, which
  fields are shown by default) lives in `field(metadata=...)` via `param()`, or
  -- for a settings class that already exists, like `DriftSettings` -- in the
  plugin's `params` dict, so the settings class stays plain.
* **Plugins receive a `Selection`**, not pre-filtered data: many plugins work
  on the filtered set but write back to all localizations (drift: estimate on
  the good ones, correct everything).  `Selection` is a boolean mask plus the
  layer / ROI it came from; the GUI builds it from the Render tab, a script
  builds it from a `LocFilter` or by hand.
* **Results go back through a `Result`**: new localizations (or none), text,
  a plot callback and the settings used.  The session replaces the table with
  one level of undo and appends to a history, like SMAP's `addhistory`.
* **Layers.**  A `Layer` is a filter plus render and display settings over the
  same table; visible layers are rendered separately and added, as in SMAP.  A
  `Selection` names the layer it was built from.  Each layer keeps its own
  spatial index for now (cheap next to the render; share it if memory bites).
* **Layout**: two windows, the render view (most of the screen) and a compact
  control window with tabs Localize / Render / Analysis / ROI (open/save live
  in the File menu).  Inside a
  tab plugins are collapsible sections, one open at a time.  The Render tab
  filters one field at a time: quick buttons for the usual fields, a drop-down
  for the rest, a histogram whose shaded region is the range (dragged to the
  edge = no bound), and the numbers.  Below it an overview of the whole field
  of view; a click there centres the image.  A plugin section can be detached
  into its own window (the arrow on its title), so several stay open.  The
  render window has a toolbar: *Save* (PNG as shown, or a TIFF re-rendered at
  a chosen pixel size, colour or float intensity, with the pixel size in the
  resolution tags) and *ROI*: left-click draws one (click, click; polygon:
  click per vertex, double-click closes; Escape cancels), right-click picks
  the kind, the line width, or clears.  The ROI is a `Region` (`regions.py`)
  in the session; a plugin's `Selection` is the layer's filter *inside* the
  ROI, and it is saved with the file.  A search box
  filters them; once there are many plugins a tree chooser is added for the
  long tail.  Each plugin tab shows the favourites (the star on a section,
  kept in QSettings) or, with *all* ticked, the whole tree.
* **Defaults**: grouped on, precision <= 25 nm, log-likelihood >= -2,
  z within +-500 nm, PSF size <= 180 nm for a 2D table, rendering sigma =
  0.5 x precision.  Bounds apply to the grouped and ungrouped table alike.

* **Fitters are plugins assembled from parts.**  Source, camera, detection,
  PSF model, fit and output are each a small settings dataclass; a fitter's
  `Settings` has one field per part, which the form shows as a section.  Two
  wrappers so far, Gaussian 2D and Spline 3D; they differ only in the model
  part.  *Live* is a checkbox in the source part: keep watching the file.
* **Long plugins stream.**  `run` gets a `stream(event, payload)` callback:
  `"start"` with the view extent, then `"block"` per finished block.  The GUI
  turns that into `Session.begin_live` / `Session.append`, so the image builds
  up during the fit; a script ignores it.
* **Plugins may react to edits.**  `react(changed, settings)` returns values
  to set: choosing a source fills the camera from its metadata, choosing a
  preset fills it from the YAML.  Presets are `*.yaml` in `~/.smappy/cameras`,
  `$SMAPPY_CAMERAS`, or the checkout's `examples/`.

* **Several files, one table.**  Loading more files (File -> Add file, or
  several at once) concatenates them, as SMAP does, with a ``filenumber``
  column; unlike columns are joined with NaN (`locs.concat`).  Grouping does
  not link across files.  A layer picks files in its filter: the *file*
  quick button shows a tickable list of names instead of a histogram, and
  the mask composes with the bounds and the ROI, so plugins get it too.
* **Formats are a registry** (`io/formats.py`): smappy HDF5, SMAP
  `_sml.mat` (v7.3 and older), MINFLUX exports (npy, zip, json; last
  iteration, valid only, m -> nm, ``frame`` is the rank in time), and csv
  (ThunderSTORM and SMAP headers recognised; otherwise a mapping dialog,
  also for positions in pixels).  A reader returns smappy's columns and a
  `FileInfo`; SMAP frames become 0-based.
* **Image layers** (`images.py`): a layer is ``"locs"`` or ``"image"``; an
  image holds pixels with a pixel size and origin in nm (from the TIFF
  resolution tags, or asked for) and is resampled onto the view, so LUT,
  contrast and gamma apply unchanged.  Added from the layer strip's ``+``
  menu or File -> Open image; the Render tab shows pixel size, origin and
  a frame slider in place of the filter.

## Code map

    smappy/plugins/__init__.py   Plugin, Result, Selection, param(), registry
    smappy/plugins/drift_comet.py  the first plugin: COMET drift correction
    smappy/plugins/drift_rcc.py  RCC, the same contract
    smappy/plugins/fit.py        the parts, and the Gaussian 2D / Spline 3D fitters
    smappy/session.py            Session: table, files, layers, ROI, undo, history (no Qt)
    smappy/io/formats.py         readers: smappy, SMAP, MINFLUX, csv
    smappy/images.py             pixel images as layers
    smappy/regions.py            ROIs and their masks
    smappy/gui/params.py         Settings dataclass -> form widget, and back
    smappy/gui/widgets.py        CollapsibleSection
    smappy/gui/render_view.py    pyqtgraph view that re-renders on pan/zoom
    smappy/gui/render_tab.py     filter ranges, colour, contrast, gamma
    smappy/gui/plugin_panel.py   a plugin as a section: form, Run, result
    smappy/gui/app.py            windows, menus, tabs; `smappy-gui`

## Plugin contract

```python
from dataclasses import dataclass
from smappy.plugins import Plugin, Result, Selection, param, register

@dataclass
class Settings:
    radius_nm: float = param(50.0, label="radius", unit="nm", min=0)
    mode: str = param("fast", choices=("fast", "exact"), advanced=True)

@register("Analysis/Cluster/DBSCAN")
class Dbscan(Plugin):
    description = "..."
    Settings = Settings
    def run(self, locs, selection, settings, progress=None) -> Result:
        ...
```

Scripting: `Dbscan()(locs, selection, radius_nm=30)` or
`Dbscan().run(locs, Selection.all(len(locs)), Settings())`.

## Next steps

1. Several ROIs, and the line ROI as a profile tool (SMAP's GUI Format).
2. Per-layer colour for multi-layer images; filter presets.
3. Localize: a stop button; flush fitted blocks on a timer as `LiveFit`
   does, so a sparse live acquisition shows up before 9000 ROIs are in.
4. Tree chooser over the registry; entry points so other packages register plugins.
5. 3D view (`pyqtgraph.opengl`), then retire `viewer.py`.

## 3D viewer: plan (decided 2026-09-05)

### Model

`Projection` (no Qt): rotation (three angles, kept as a 3x3 matrix), the
point the view turns about, zoom (nm per screen pixel), perspective focal
length (None = orthographic), and the **slab**: a box in data coordinates
with centre, size (x, y, z) and an in-plane rotation, which is the 3D ROI.
The slab and the projection are session state next to the ROI, so a script
can set them and every engine and every panel reads the same object.

`Slab.mask(x, y, z)` selects; `Projection.apply(x, y, z)` returns
`(x', y', depth)` in view coordinates, with perspective as a scale by
`1 / (1 + depth / f)`.

### Engines, one interface

    engine.render(layers, projection, fov, preview=False) -> rgb, planes

* **A (first)**: rotate the slab's selected points, then `render_locs` on
  `(x', y')` with each point's own sigma -- the 2D kernel, the layer's
  render and display settings, the same sum-of-Gaussians image as the 2D
  window.  Depth cues as weights and colour: attenuation
  `exp(-depth / lambda)`, colour by depth; later slice opacity (K renders
  composited front to back) and perspective.  Runs on the render worker;
  while the mouse drags, a preview at half resolution and at most ~2 M
  points, the exact image on release.
* **GPU (later)**: instanced quads with the same erf kernel in the shader,
  accumulated additively into a float framebuffer, display pass or readback.
  Library decided then: vispy (OpenGL, deprecated on macOS) or pygfx/wgpu
  (Metal).  Tested image-for-image against A.
* Point-cloud mode (sprites, alpha) comes with the GPU engine.

### Window and controls

A separate 3D window (toolbar: save, presets top/front/side, reset; a
collapsible side panel).  Layers, filters and display stay in the Render
tab and drive both windows.  Mouse: left-drag rotates, shift + left-drag or
middle-drag pans, wheel zooms, ctrl + wheel moves the slab along the depth
axis, shift + wheel changes its thickness.

The slab has three handles that edit the same box:
1. the 2D window: a rectangle ROI is the footprint; a line ROI is a rotated
   footprint (length x width) whose long axis is x' (SMAP's way); z from
   the layer's z filter until set otherwise;
2. the 3D window: the box is drawn, its faces drag, wheel modifiers above;
3. the side panel: three range sliders with numbers, azimuth / elevation /
   roll dials, presets, and a depth histogram of the slab's localizations.

### Phases

1. `Projection` + slab, slab from the 2D ROI, engine A with rotation, zoom,
   pan and depth attenuation, the window with the mouse, the side panel
   (ranges, dials, presets).  Tests: a rotated render of a known structure
   equals the 2D render of the rotated table; the slab mask; preview vs
   full agree in the limit.
2. Slice opacity, perspective, colour by depth, draggable box faces,
   depth histogram, save at pixel size (TIFF/PNG), a 3D `Selection` for
   plugins (the slab as ROI).
3. GPU engine and point-cloud mode.
