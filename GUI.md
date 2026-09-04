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
* **Layers are modelled now, one is shown.**  A `Layer` is a filter plus render
  and display settings; a `Selection` names the layer it was built from.
* **Layout**: two windows, the render view (most of the screen) and a compact
  control window with tabs File / Localize / Render / Analysis / ROI.  Inside a
  tab plugins are collapsible sections, one open at a time.  A search box
  filters them; once there are many plugins a tree chooser is added for the
  long tail.

## Code map

    smappy/plugins/__init__.py   Plugin, Result, Selection, param(), registry
    smappy/plugins/drift_comet.py  the first plugin: COMET drift correction
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
2. Layers in the GUI (a layer strip in the Render tab) and grouped display.
3. Localize tab: `smappy.fit` and `smappy.live` as plugins.
4. Tree chooser over the registry; entry points so other packages register plugins.
5. 3D view (`pyqtgraph.opengl`), then retire `viewer.py`.
