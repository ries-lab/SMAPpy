# The plugin architecture

Where plugins come from, how they reach the GUI, and what a plugin has to
declare.  Decisions taken 2026-09-12; this extends `GUI.md`, which fixed the
toolkit and the "a plugin is a settings dataclass plus a `run`" contract.  The
goal is that smappy ships with many plugins and that a user adds one by copying
a file into a folder.

## What is wrong with what we have

Three unrelated plugin systems, and only one of them is a plugin system:

* `smappy.plugins` has the real contract -- `Plugin`, a `Settings` dataclass,
  `ParamInfo`, `Result`, `Selection` -- but its registry is filled by importing
  a hardcoded three-module tuple, `_BUILTIN`.  A file dropped anywhere does
  nothing.
* `smappy.roi_manager.plugins` is a parallel, duck-typed system: a `defaults`
  dict instead of a dataclass, `find` / `evaluate` / `analyze` instead of `run`,
  no registry, wired into the ROI window by name.
* `smappy.io.formats` is a third registry of `Reader`s, invisible to the GUI.

And the tree is not a tree.  `PluginTab.rebuild` does one `rpartition("/")`, so
`Analysis/Drift/COMET` groups but `Analysis/Cluster/DBSCAN/2D` would not; groups
are built expanded; and every panel is constructed when the tab is, which is
fine for six plugins and not for sixty.

## Decisions

### The folder tree is the plugin tree

A plugin's path comes from where its file sits, as in SMAP's
`plugins/+Analyze/+cluster/Foo.m`:

    <root>/Analysis/Drift/comet.py   ->  "Analysis/Drift/COMET"
    <root>/MyLab/Cluster/dbscan.py   ->  "MyLab/Cluster/DBSCAN"

Arbitrary depth.  The file stem gives the default name (`my_drift.py` -> "My
Drift"); a `name` attribute on the class overrides it.  `@register("A/B/C")`
still works and wins over the folder, because one file may declare several
plugins and because the built-ins should not have to rearrange themselves to
keep their paths.

The path decides *where a plugin sits in the tree* and nothing else.  It does
not decide which tab shows it -- see below -- so there is no such thing as a
plugin that is installed but unreachable.

Roots scanned, in order, later winning on a collision:

1. `smappy/plugins/**`, the shipped ones -- discovered the same way as the rest,
   not listed in code.
2. Every folder in the user's preferences.  There is deliberately no implicit
   `~/.smappy/plugins`: a plugin folder is something the user names, so that
   what is loaded is always something they wrote down.

Third-party *packages* (a `smappy.plugins` entry point) are not scanned.  If
that is ever wanted it is one more root, not a new mechanism.

### Discovery reads the files; import waits until first use

At startup the scanner walks the roots and *parses* each `.py` with `ast`
without executing it, reading the class name, its `@register` argument, and any
literal `name`, `description`, `favorite` and `scope` assignments -- falling
back to the first line of the docstring for the description.  The result is a
`PluginRef`:

```python
@dataclass(frozen=True)
class PluginRef:
    path: str            # "Analysis/Drift/COMET"
    name: str
    description: str
    scope: str           # "locs" or "site"
    origin: Path         # the file
    attr: str            # the class in it
    root: str            # which root it came from, for the UI and for errors
    def load(self) -> type[Plugin]: ...   # imports, caches, then returns
```

So the whole tree is populated instantly and correctly, a plugin that imports
torch costs nothing until someone opens it, and there is no cache file to go
stale -- the two things SMAP has to pay for with the generated `plugin.m`.

A module that builds its plugins dynamically says so with a module-level
`__smappy_plugins__ = ["Analysis/Cluster/DBSCAN", ...]`; the ref then has no
class name and resolves through `@register` when the module is imported.

Anything else the parser cannot make sense of becomes a `Diagnostic` -- a file
and a sentence -- collected rather than raised, and shown in the GUI.  The scan
does *not* fall back to importing such a file: importing what we failed to parse
is the riskiest thing available, and it would give back the startup cost the
scanner exists to avoid.  A diagnostic already achieves what matters, which is
that a broken plugin complains instead of silently missing.  There are four:
the file will not parse, several plugin classes share a file with no `@register`
to tell them apart, every candidate class is private or is used as a base, and
a root that does not exist.  A path claimed by two roots is not an error --
the later root wins, which is how a user shadows a shipped plugin -- but it is
reported, because a shadow you did not intend is hard to see otherwise.

### One `run`, with a context

`run(self, locs, selection, settings, progress, stream)` cannot express a
loader (no input localizations), a simulator (no input at all) or an ROI
evaluation (one site at a time).  Everything takes a context instead:

```python
def run(self, ctx: Context, settings) -> Result
```

`Context` carries `session` (None in a script), `locs`, `selection`, `rois`,
`site`, `site_table`, and the two callbacks as `ctx.report(text)` and
`ctx.emit(event, payload)`; `ctx.for_site(site, ...)` derives the per-ROI one.
When a session is given, the table and the selection are read *at construction*,
because the GUI builds the context on its thread and hands it to a worker --
reading them later would race a live fit rebinding them.  A processor still returns `Result(locs=...)`; a
loader calls `ctx.session.add_file(...)`; an ROI finder returns
`Result(data={"rois": [...]})`.  One base class, one panel widget, one tree --
so loaders, savers, the simulator and the ROI plugins all get favourites,
parameter forms, validation and state saving for free.

Plugins that are called once per ROI say so with a class attribute rather than a
different base class:

```python
scope = "locs"   # default: run once over the selection
scope = "site"   # run per ROI; ctx.site is set, Result.data is that site's row
```

`scope` is the only thing that separates an evaluator from any other plugin, and
it is a property of the plugin, not of its folder -- which is what lets the
evaluation window filter correctly under one global tree.

`Plugin.__call__` keeps the scripting shape: `plugin(locs, sel, radius_nm=30)`
builds the context itself, and `locs` may be left out entirely for a plugin that
makes its own.  A plugin still written against the old signature is refused by
`__init_subclass__` with a message saying what to do, rather than being handed a
`Context` as its `locs` and failing somewhere far away.

### A tab is a named list of instances

There is **one** plugin tree, shared by everything.  A tab does not own a path
prefix; it is a curated, ordered list over the whole tree:

```python
@dataclass
class Instance:
    id: str          # stable, so a rename does not lose the values
    plugin: str      # "Localize/Spline 3D"
    label: str       # "Spline 3D (beads)" -- renameable
    values: dict     # this instance's parameters
    enabled: bool = True
```

A pin is an `Instance`, not a plugin path, so the same plugin can be pinned
twice with different parameters -- `Spline 3D (beads)` beside
`Spline 3D (data)`.  This is SMAP's `addmodule`, which renames on collision
(`modulename_2`) for exactly this reason, generalised from the evaluation list
to every tab.  The ROI pipeline below then uses the identical type instead of
being a special case.

Tabs are free: add, rename, reorder, remove.  A tab the user creates starts
empty.  A "special" tab is an ordinary tab with a header widget above the list
-- the ROI tab's *open ROI manager* button and site count, the Localize tab's
calibration buttons -- so File and Analysis are nothing but ordinary tabs, and a
special one costs a header, not a class.

Because a fresh install would otherwise be a blank window, smappy ships a
default workspace seeding File, Localize, Render, Analysis and ROI with sensible
pins.  *Empty by default* applies to tabs the user adds.  Shipping it as a
workspace rather than as code also gives us *Reset to defaults* for free.

### The tab shows favourites; a chooser shows the tree

The tab widget has one mode: the curated list, as collapsible sections in the
user's order, one open at a time, showing only the plugin's main parameters
(`Plugin.main`, `ParamInfo.advanced`) with the rest behind "more".  A context
menu gives move up / move down / rename / duplicate / unpin / detach.

A `+` button opens the **plugin chooser**: the full tree, collapsed, with search
and a description pane.  Picking a leaf pins a new instance to the current tab.
One tree widget exists, built once, instead of every tab rendering its own copy,
and the tab has a single layout to build and to persist.

The ROI tab is one of these, holding `ROIManager/Segment/*` and
`ROIManager/Analyze/*` pins -- ordinary run-once plugins -- plus its header.

### The ROI evaluation pipeline is a separate window

Evaluators are the one thing that does not fit a tab.  A tab's pins are run
individually on demand; an evaluation is an *ordered pipeline* run over every
site at once, each instance contributing columns to that site's row.  Putting
thirty evaluators in the ROI tab would also bury the two or three plugins
actually used to find and analyse sites.

So the ROI manager opens an **Evaluation** window, laid out like SMAP's
`SEEvaluationGui`: the pipeline on the left as a checkbox-and-name list that can
be reordered, renamed, duplicated and removed, and the selected instance's
parameter form on the right.  Its `+` opens the same plugin chooser, filtered to
`scope == "site"`.  The list is `list[Instance]` -- the same type a tab holds --
so ordering, renaming and duplicate handling are one implementation.

A pipeline is saveable to its own named file, so an NPC recipe can be kept with
a project or sent to a colleague without dragging a window layout along.

### State lives with the data

The pipeline that produced a site table is *provenance*: the columns in that
table are meaningless without knowing which evaluators ran, in what order, and
with what parameters.  So it is written into the localization file next to the
ROI results, not only into a workspace.  The file already carries a JSON
`metadata` attribute for exactly this kind of thing.

One caveat on where: an HDF5 *attribute* is limited by the object header, in
practice about 64 kB, which a large pipeline plus a GUI snapshot could exceed.
The state therefore goes in a `/gui` group as a variable-length string dataset,
not in `f.attrs`, and the existing `metadata` attribute is left alone.

Beyond the pipeline, the rest of the GUI state -- tabs, their instances and
values, window layout -- is saved:

* in a **workspace** file, auto-saved to the config directory and restored at
  startup, and saveable under a name;
* and, optionally, **appended to the localization file**, so reopening a dataset
  restores the session that produced it.

Recommendation for the optional part, which is the one thing left open: embed
the plugin state (tabs, instances, values) but *not* the window geometry.
Geometry is machine-specific -- it travels badly to a colleague's screen and is
the only part that is worthless as provenance -- while the plugin state is small
(a few kB of JSON) and is genuinely a record of how the file was produced.  A
preference turns the embedding off for anyone who wants their data files to
contain only data.  Values are read back tolerantly through the `ParamSpec`
types: an unknown key is dropped with a warning and a missing one keeps its
default, so a file survives a plugin renaming a field.

Localizations, layers and ROI geometry are unaffected; they stay where they are.

### The old two systems fold in

`roi_manager/plugins.py` is rewritten: `DensityPeaks`, `Statistics` and
`Histograms` become `Plugin` subclasses with `Settings` dataclasses under
`ROIManager/Segment`, `ROIManager/Evaluate` and `ROIManager/Analyze`, the
`defaults` dicts becoming typed fields with units and bounds, and `Statistics`
gaining `scope = "site"`.  Their current signatures go; nothing outside the repo
calls them.

`io/formats.py` keeps `Reader` as the low-level mechanism -- the CLI and scripts
use it and should not have to go through a plugin -- and gains a matching
`Writer` registry.  A `File/Load/<name>` plugin is generated per reader, so
adding a format stays a one-line `register(Reader(...))`.

### The File tab

Seeded with:

    File/Load/Auto            dispatch by extension; format choice, append,
                              reset view, and a reader's own needs (csv mapping)
    File/Load/<format>        smappy HDF5, SMAP _sml.mat, MINFLUX, csv
    File/Save/smappy HDF5
    File/Export/Image         the rendered view as a picture
    File/Simulate/Blinking Structure   scripts/simulate_blinks.py, into the session

`File -> Open` in the menu runs `File/Load/Auto` rather than reimplementing it,
so there is one loading path; the menu's separate append action becomes the
plugin's `append` checkbox.

## Order of work

1. `PluginRef`, the AST scanner, the roots, the preferences for extra folders.
   Registry keeps its `get`/`available`/`tree` API on top of refs.
2. `Context`, the new `run`, `scope`; port the five built-in plugins.
3. `Instance`, the tab widget, the plugin chooser dialog, lazy panels.
4. The workspace file, the shipped default, the preferences dialog.
5. The ROI rewrite and the Evaluation window.
6. The `Writer` registry, the File tab, and the `/gui` group in the file format.

Each step leaves the GUI working; nothing here needs a flag day.
