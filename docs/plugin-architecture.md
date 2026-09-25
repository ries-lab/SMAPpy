# The plugin architecture

Where plugins come from, how they reach the GUI, and what a plugin has to
declare.  Decisions taken 2026-09-12; this extends `GUI.md`, which fixed the
toolkit and the "a plugin is a settings dataclass plus a `run`" contract.  The
goal is that SMAPpy ships with many plugins and that a user adds one by copying
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

Because a fresh install would otherwise be a blank window, SMAPpy ships a
default workspace.  It is *seeded* rather than listed: each shipped tab names a
path prefix used once, at first run, to pin the installed plugins that ask for
it (`Plugin.favorite`).  So the defaults adapt to what is installed, and after
that run the workspace is data and a tab is whatever the user made of it -- the
prefix has no further meaning.  *Empty by default* applies to tabs the user
adds.  This also gives *Reset tabs to defaults* for free.

### The tab shows favourites; a chooser shows the tree

The tab widget has one mode: the curated list, as collapsible sections in the
user's order, one open at a time, showing only the plugin's main parameters
(`Plugin.main`, `ParamInfo.advanced`) with the rest behind "more".  A context
menu gives move up / move down / rename / duplicate / unpin / detach.

A `+` button opens the **plugin chooser**: the full tree, collapsed, with search
and a description pane, built from `PluginRef`s so that opening it imports
nothing.  Picking a leaf pins a new instance to the current tab.  One tree
widget exists, built once, instead of every tab rendering its own copy, and the
tab has a single layout to build and to persist.  The chooser also shows a count
of any files that could not be read, which is where the scanner's diagnostics
surface.

A section's panel is built the first time it is opened, not when the tab is.
Together with the scanner that means starting SMAPpy imports no plugin at all,
and a tab of thirty costs thirty parsed files.  A pin whose plugin is no longer
installed becomes a section with a message in it rather than an exception during
startup, and a workspace naming plugins that have gone is pruned with a line
saying which.

The ROI tab is one of these, holding `ROIManager/Segment/*` and
`ROIManager/Analyze/*` pins -- ordinary run-once plugins -- plus its header,
which carries the geometry every ROI shares and the ways into the two windows.

The chooser opened from a tab offers only `scope == "locs"` plugins.  That is a
restriction of the contract and not of the folder: an evaluator has no site in a
tab, so pinning one there could only produce a Run button that fails.  Any
*runnable* plugin, from any branch of the tree, can still be pinned to any tab.

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

A step whose plugin raises costs its own columns and not the ROI, nor the ROIs
after it; the failure is recorded against that step.  Steps share a site's row,
and a column keeps its plain name while only one step produces it -- qualified
with the step's label when two would collide, so the common case reads as it
always did and the ambiguous one is never silently lost.

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
preference turns the embedding off (`save_gui_state_in_files`) for anyone who
wants their data files to contain only data.  Reading it back is deliberately
*not* automatic: rearranging someone's tabs because they opened a colleague's
dataset would be a surprise, and their own last session is already restored from
the auto-saved workspace, so it is a menu item, *Restore GUI state from file*.  Values are a flat map of dotted names read off the form
widgets -- so a form left mid-edit with an unparseable number is still saveable,
and a panel never opened keeps the values it was given rather than being
blanked -- and are read back a field at a time: a name the plugin no longer has
costs that field and not the form.

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
`Writer` registry.  The `File/Load/<name>` plugins are written out rather than
generated: a generated set could not be declared to the AST scanner without
duplicating the reader list, and a format has its own settings anyway -- only
csv needs a column mapping.  Adding a format is still a one-line
`register(Reader(...))`; giving it a plugin of its own is optional, and the
`Auto` loader picks it up either way.

### The File tab

Seeded with:

    File/Load/Auto            by extension; append, and whether to link blinks
    File/Load/<format>        smappy HDF5, SMAP _sml.mat, MINFLUX, csv --
                              present but unpinned, since Auto covers them
    File/Save/smappy HDF5     with the GUI state, or without
    File/Export/Image         the rendered view as a picture
    File/Simulate/Blinking Structure   `smappy.simulate`, into the session

A named loader *means* it: `load` takes an explicit reader, because dispatching
by extension underneath would have `File/Load/SMAP` on a `.hdf5` quietly read it
as SMAPpy.

The menu and the plugins share one *reading* implementation,
`session.read_and_group`, which the window's `LoadTask`, `Session.load` and
every loader plugin call.  `File -> Open` still drives it through `LoadTask`
rather than through the plugin: the menu owns a multi-file queue, a progress
line and the csv mapping dialog, none of which a panel expresses, and routing it
through the plugin would add indirection without removing code.  What the tab
adds is the options as checkboxes -- append, and whether to link blinks -- beside
the menu's separate *Add file* action.

A loader runs in the panel's worker thread and must not touch the session, so it
returns its work as `Result.files` and `Session.apply` adds it on the thread
that owns the session -- the same reason `Context` snapshots its table.

## Order of work

1. *Done.* `PluginRef`, the AST scanner, the roots, `smappy.config`.
   The registry keeps its `get`/`available`/`tree` API on top of refs.
2. *Done.* `Context`, the new `run`, `scope`; the five built-in plugins ported.
3. *Done.* `Instance`, the tab widget, the plugin chooser dialog, lazy panels.
4. *Done.* The workspace file, the shipped default, the preferences dialog.
5. *Done.* The ROI rewrite, the pipeline, and the Evaluation window.
6. *Done.* The `Writer` registry, the File tab, and the `/gui` group.

Every step is in.  What is left is the ordinary work of writing more plugins,
which is now a file copy.

Each step leaves the GUI working; nothing here needs a flag day.
