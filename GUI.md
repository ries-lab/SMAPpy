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
  edge = no bound), and the numbers.  A search box
  filters them; once there are many plugins a tree chooser is added for the
  long tail.

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

## Code map

    smappy/plugins/__init__.py   Plugin, Result, Selection, param(), registry
    smappy/plugins/drift_comet.py  the first plugin: COMET drift correction
    smappy/plugins/fit.py        the parts, and the Gaussian 2D / Spline 3D fitters
    smappy/session.py            Session: table, layers, undo, history (no Qt)
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

1. ROI tab: `pg.ROI` rectangles/polygons, saved with the file, into `Selection.roi`.
2. Per-layer colour for multi-layer images; filter presets.
3. Localize: a stop button; flush fitted blocks on a timer as `LiveFit`
   does, so a sparse live acquisition shows up before 9000 ROIs are in.
4. Tree chooser over the registry; entry points so other packages register plugins.
5. 3D view (`pyqtgraph.opengl`), then retire `viewer.py`.
