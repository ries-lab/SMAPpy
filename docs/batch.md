# Chains of plugins, and running them over many files

Decisions taken 2026-09-23.  Two things are built from the same parts:

* a **chain** -- an ordered list of plugins with their settings, which runs
  as one plugin, can be saved, and appears in the plugin tree like any other;
* a **batch** -- a chain applied to a list of files, each loaded, processed,
  saved and cleared on its own, headless, with the numbers and figures of every
  file written to an output folder.

The GUI is a client of two plain files, a `*.chain.yaml` and a `*.batch.yaml`,
and so is an AI agent: the most common way of writing a chain is expected to be
asking one to write it, and the section "For agents" at the end is written for
that reader.

## What SMAP did, and what is kept

SMAP had three mechanisms: the fitting `Workflow` (a graph of modules, saved
as a `.mat` of parameters plus a `.txt` of layout), the `Batchprocessor` (a
list of `_batch.mat` or `.tif` files fitted with one workflow), and
`BatchAnalysis` (a tab of Analyze plugins run over a list of `_sml.mat`, figures
per plugin, every return value into one `_results.mat`).  All three drive GUI
objects, so none can run without SMAP open; BatchAnalysis has its per-plugin
`try` commented out, so one failing plugin ends the batch; and numbers went
into a MATLAB struct and never into a table (CSV output was a TODO).

Kept: a chain is plugins with their settings, figures are saved per file and
plugin, and a fit can be the first step.  Not kept: the graph (`NOTES.md`,
"No module chain" -- a chain is linear, every step works on the session's
table as a person clicking through would), the dependence on a running GUI,
and the two-file workflow format.

## The chain

```yaml
# ~/.config/smappy/chains/npc_standard.chain.yaml
schema: smappy-chain-v1
path: Analysis/Chains/NPC standard     # optional; see "Where chains live"
description: filter, drift, statistics
grouping: layer            # layer | grouped | ungrouped -- see "Grouping"
steps:
  - plugin: Chain/Layers
    label: layers
    version: "1"
    values:
      layers:
        - grouped: true
          bounds:
            - {field: loc_precision_nm, hi: 20}
            - {field: photons, lo: 0.01, quantile: true}
  - plugin: Analysis/Drift/RCC
    label: drift
    values: {segments: 10}
  - plugin: Analysis/Measure/Localization Statistics
    label: statistics
    grouping: ungrouped       # auto | grouped | ungrouped
    enabled: true
```

* `values` is the flat dotted map every other saved form uses
  (`settings_values`): `fit.roisize: 13`, not a nested `fit:` block.  A nested
  block is accepted on reading, because the session's log has always written
  settings that way.
* A step's **key** is its label made into an identifier (`"NPC radius"` ->
  `npc_radius`), unique within the chain.  Overrides address a field as
  `<key>.<field>`.
* `version` is the plugin's `Plugin.version` when the chain was saved.  A
  mismatch is a warning on validation, and part of what makes a re-run
  "the same" (see "Skipping what is done").
* A step that is not `enabled` is kept in the file and skipped.
* A step's grouping is `<key>.use_grouping` among the chain's settings (the
  field a step's settings gain; a plugin may not have a field of that name),
  and the chain's own is `grouping`.

### A chain is one plugin

`chain.chain_class(spec)` builds a `Plugin` subclass from a spec.  Its
`Settings` is a generated dataclass with one field per step, typed as that
step's own settings, so the form the GUI already builds from nested dataclasses
draws each step as a collapsible section: collapsed, the step's simple view
when expanded, its "more" one level further in.  The step's `grouping` choice
is a field of the section, under "more".

It runs its steps on a **scratch session**: a headless copy of the session's
table, layers, filters, grouping, ROI and slab (`Session.scratch`), where each
step's result is applied exactly as `Session.apply` would apply it.  That is
what lets step 3 see what step 2 did, and it is also why the chain can run in
the GUI's worker thread like any plugin: nothing touches the real session
until the chain's single result comes back.  So:

* **one undo step**, named after the chain;
* **one log entry**, `what: <chain path>`, whose `steps` list holds every
  step's plugin, version, values, the line it reported, whether it changed the
  table, and how long it took.  A chain is logged whether or not it changed
  the table (`logged = True`): a measurement step is in it, because in a chain
  the measurement is the point.  *Analysis/Process/History* shows the entry
  as its steps, each with its settings;
* the figures of all steps, as tabs of one window, `"<label>: <figure>"`;
* what each step keeps (`Plugin.keep`) goes in the file under the chain's
  path, one entry per step, so two RCC steps in one chain do not overwrite one
  another.

A step fails -> the chain stops, and the file is not saved (in a batch, the
file is marked failed and the batch goes on).

Run in the GUI, a chain that starts with a fit does not stream: the table
appears when the chain is done, not block by block as a fit in the Localize
tab does, because the chain's steps run on the scratch copy.

A plugin's `preflight` is asked for each step before it runs.  In the GUI the
chain asks all of them once, up front, and runs without asking again; in a
batch the job says what an answer is (`preflight: skip | proceed | abort`,
default `skip`: the file is skipped and the report says why).

### Building a chain in the GUI

A chain is always a single collapsed plugin panel.  **New chain** in a tab's
add menu creates an empty one; the panel's **edit steps** toggle shows a small
toolbar -- add a plugin from the tree, move up, move down, delete, enable --
and **Save chain / Save chain as…** writes the `.chain.yaml`.  A chain that has
not been saved lives in the workspace with its steps inline, like any pinned
plugin's values, so nothing is lost on restart.  There is no separate chain
tab: building and using are the same widget.

### Where chains live

Discovery reads `*.chain.yaml` beside `*.py` in every plugin root, and also in
`config_dir()/chains`, which is where **Save chain** writes by default.  That
one implicit folder is deliberate, and differs from the rule that a plugin
folder is always something the user names (`config.plugin_roots`): a chain
holds no code, so reading one runs nothing the user did not write.

A chain's path is its `path:` key, else `Analysis/Chains/<Title of stem>` for
one in the chains folder, else the folder it sits in, as for a `.py`.  Reading
the file is `yaml.safe_load` of a few lines: discovery still imports nothing,
and a chain whose step plugins are missing fails when it is opened, with the
name of what is missing, not when the tree is built.

## Layers, filters and grouping

### The `Chain/Layers` step

A fresh file has one layer with the default bounds (`session.DEFAULT_BOUNDS`)
and the default grouping -- the same as opening it in the GUI -- so a chain
without this step filters as a freshly opened file does.  The step sets up
the layers explicitly, as the Render tab does by hand: one block per layer
with

* `grouped` -- the layer's own grouping;
* `bounds` -- rows of `field`, `lo`, `hi` (either may be empty), `quantile`
  (then `lo` and `hi` are fractions, resolved per file with
  `filter.quantile_range` -- absolute photon counts differ between files),
  and `required` (a field the file lacks aborts the file; otherwise the row is
  skipped and the report says so);
* `start` -- `defaults` (the default bounds, then these rows), `empty`, or
  `keep` (the layer's current bounds, then these rows -- interactively, what
  you set in the Render tab).

Fields are not known before a file is open, so the bounds are a table of rows
rather than one settings field per column, and the GUI's editor offers the
session's columns when there is a session and takes free text when not.  A
**from current layers** button fills the step from what the Render tab shows.

`remove` (off by default) goes further than a filter: localizations that no
layer keeps are dropped from the table, and so from the saved file.  SMAP did
this in the writer ("save visible only"); here it is a step, so it is in the
log.

A layer the step names but the session does not have is created; in the GUI
the step acts on the session's layers, which is how the Render tab then shows
what the chain did.  It hands its layers back as `Result.data["layers"]`,
which `Session.apply` honours, the same way `data["bounds"]` works.

### Grouping

Which table a step reads -- one row per localization, or one per blink -- is
decided, lowest priority first, by:

1. **the layer** -- its own grouping, as in Render.  The default.
2. **the chain** -- `grouping: grouped | ungrouped` overrides every layer.
3. **the step** -- `grouping:` in the chain file, the user's choice for that
   plugin, on every layer.
4. **the plugin** -- `Plugin.grouping = "grouped" | "ungrouped"` for a plugin
   that can only work one way.  This wins, and the GUI greys the step's choice
   out.

A plugin reads the result through `ctx.table(layer)`, which returns
`(locs, selection)` for that layer with the grouping resolved -- the
equivalent of SMAP's `getloc(..., 'grouping', ...)`.  `ctx.locs` stays the
ungrouped table, so a plugin that never calls `ctx.table` behaves as it always
did.  A step that runs on the grouped table and hands back a new table is
refused: a grouped run is for measuring, and there is no way back from one row
per blink to the localizations it came from.

## The batch job

```yaml
schema: smappy-batch-v1
name: NPC 2026-09-20
chain: ~/.config/smappy/chains/npc_standard.chain.yaml   # or a mapping: an inline chain
overrides: {drift.segments: 8}
inputs:
  - folder: /data/2026-09-20
    pattern: "**/*.h5"          # default: by the chain's input kind
    exclude: ["*bead*"]
  - file: /data/extra/cell3.h5
    overrides: {layers.layers: [...]}   # per file
  - file: /data/extra/cell4.h5
    include: false
output:
  folder: /data/2026-09-20/smappy_batch      # default: <job>_output beside the job
  suffix: _proc                 # <stem>_proc.h5
  locs: true                    # save the processed table
  figures: png                  # png | pdf | svg | none
  fitted: beside_image          # image input: beside_image | output
  delete_fitted: false
preflight: skip                 # skip | proceed | abort
rerun: skip_identical           # skip_identical | always
```

* Inputs are explicit files (what the GUI's table writes) and folder rules
  (what an agent writes), both at once.  The runner resolves them to one list
  at the start and writes that list into the report, so the report is the
  record of which files a run saw, whatever the rules would match tomorrow.
  A file named twice -- by a folder rule, and again to give it overrides --
  keeps its place and gains the overrides.  The output folder is never
  searched, so a second run does not take its own results for input.
* A folder rule's default patterns are `*.tif`, `*.tiff` for images and
  `*.h5`, `*.hdf5`, `*_sml.mat`, `*.csv` for localizations, in every
  subfolder.  An acquisition written as a multi-file OME series
  (`..._MMStack_Pos0.ome.tif`, `..._1.ome.tif`, ...) is opened from its first
  file; exclude the rest (`exclude: ["*_[0-9].ome.tif"]`) or name the first
  files one by one.
* Overrides, the job's and each file's, use the chain's settings names,
  `<step key>.<field>` (a part may be given as a mapping, `drift: {n_timepoints: 8}`).
  A file's own override wins over the job's.
* The **input kind** follows from the first step: a plugin that makes its own
  table (`Localize/*`, the simulator) means images, and each file becomes that
  step's `source.path`; anything else means localization files, each opened
  with `File/Load/Auto`.  One input is one file: two-file inputs (a two-colour
  pair in separate stacks, several files per cell) are not supported yet.
* `.json` is read too; the GUI writes `.yaml`, which allows comments.

### What is written

Per file, in `<folder>/<stem>/` (the parent folder's name is prefixed when two
inputs share a stem):

* `<stem><suffix>.h5` -- the processed table with its history, which ends in
  the chain's log entry, and the steps' kept results.  An input `.h5` is never
  overwritten.
* for image input, the fit's own `<stem>_locs.hdf5`, beside the image or in
  this folder (`fitted`), deleted at the end if `delete_fitted`;
* `figures/<key>_<figure>.png`, drawn headless with Agg;
* `results.json` -- status, timings, the fingerprint, and per step its text,
  the JSON-able part of its `data`, and the settings it ran with.

Per batch: `summary.csv` (one row per file, a column per scalar
`<key>.<name>` any step returned) and `batch_report.json` (the resolved job,
the resolved chain, versions, every file's status and traceback).

Files run one after another: the plugins are parallel already, and two fits
on one GPU are slower than one after the other.

### Skipping what is done

A file is skipped when its `results.json` says it finished with the same
**fingerprint**: a hash of the resolved chain (every step's plugin, version
and values, after the overrides and the file's own), the output options that
change what is written, and the input's size and modification time.  A plugin
whose code changed without its `version` changing counts as the same -- bump
`version` when a change moves the numbers.

### Running it

    smappy-batch run job.batch.yaml [--limit N] [--force]
    smappy-batch validate job.batch.yaml | chain.chain.yaml
    smappy-batch plugins [prefix]
    smappy-batch describe Analysis/Drift/RCC | chain.chain.yaml

Progress goes to stdout as one line per event (`SMAPPY_START 3/20 path`,
`SMAPPY_DONE`, `SMAPPY_SKIP`, `SMAPPY_FAILED`, `SMAPPY_REPORT path`), which
is what the batch window reads.  A failed file costs that file; the exit code
is 1 if any failed.

**Validation is strict**, unlike reading a saved form.  `settings_from` drops
names it does not know so that a file written by an older version still opens;
for a file somebody has just written, that turns a typo into a silent default.
`validate` refuses any key the plugin does not have, any choice it does not
offer, any step whose plugin does not exist, and an input list that resolves
to nothing.  `run` validates first.

## The batch window

`smappy-batch-gui`, or *Tools -> Batch…* in SMAPpy: the same window class
either way.  A table of inputs (add files, add folder with a pattern, drag and
drop, remove, an include tick, a status column), the chain panel (the same
widget as in the plugin tabs, so a chain is edited the same way everywhere),
the output options, and **Run**, which writes the job file and starts
`smappy-batch run` as a subprocess -- so a crash or an exhausted memory costs
the batch and never the GUI -- and reads its progress lines.

## Not yet

* **Pooling across files** -- one histogram of NPC radii over every cell.
  `summary.csv` is the only aggregate for now; a later "summary chain" could
  run over every file's `results.json`.
* **Inputs of more than one file.**
* **Parallel files** (`jobs: N`), for analysis-only chains on many cores.
* **Branching chains.**  Deliberately linear; see above.

## For agents

This section is for an agent asked to "analyse these files with ...".  You
have the code base; you do not need the GUI.

**The loop.**

1. `smappy-batch plugins Analysis` (or no prefix for everything) -- what
   exists, by path; saved chains are marked `(chain)`.
2. `smappy-batch describe <path>` -- the plugin's fields as the flat dotted
   names a chain uses, with type, default, unit, range, choices and help;
   whether it only works grouped or ungrouped; whether it makes its own
   table (a fit or a loader, which can only be a first step).
3. Write the chain and the job, starting from `docs/examples/`:
   `filter_drift_statistics.chain.yaml` (localizations in),
   `fit_and_measure.chain.yaml` (images in) and `cells.batch.yaml`.
4. `smappy-batch validate job.batch.yaml` until it prints no errors.  It
   refuses unknown fields and says which ones exist, so a failed validation
   is the quickest way to learn a plugin's names.
5. `smappy-batch describe chain.chain.yaml` -- the values every field will
   run with, under the names overrides use.
6. `smappy-batch run job.batch.yaml --limit 1`, then read
   `<output>/<name>/results.json` (status, each step's text and numbers, the
   traceback if it failed) and look at `<output>/<name>/figures/`.
7. Run the rest; `summary.csv` has a row per file.  Re-running skips what is
   done with the same settings, so fixing one step and running again only
   redoes what changed.  `--force` redoes everything.

**Things that are easy to get wrong.**

* Values are flat: `camera.offset: 100`, not `camera: {offset: 100}` inside a
  step's `values` (the nested form is read, but write the flat one).
* Leave `source.path` and `output.path` of a fit out of the chain: the batch
  sets them per file.
* A step key is its label made into an identifier: `label: NPC radius` ->
  `npc_radius`; without a label it is the plugin's name (`Localization
  Statistics` -> `localization_statistics`).  Overrides use the key.
* Without a `Chain/Layers` step every file gets the default filter
  (precision below 25 nm, `logl_rel` above -2, and so on) and is grouped.
  Say what you want explicitly; `start: empty` gives no bounds but yours.
* Grouping: a plugin that reads `ctx.table()` gets the table the rules in
  "Grouping" decide; one that reads `ctx.locs` always gets the ungrouped
  table, whatever the chain says (the shipped drift corrections and colour
  assignment are of this kind).  A step that read the grouped table and hands
  back a new one stops the chain with a message.
* A step that asks before running (COMET estimating hours) skips the file
  by default; the job's `preflight: proceed` runs it anyway.

**If a step needs a plugin that does not exist**, write it first --
`CLAUDE.md`, "Writing a plugin" -- give it a `version`, and put it in a
plugin folder (`config.plugin_roots`, set in Preferences).  A plugin that
measures should return its numbers as scalars in `Result.data` (nested
dictionaries are fine): that is what reaches `summary.csv`, as
`<step key>.<name>`.  Arrays up to a thousand values reach `results.json`;
larger ones and tables are left out.  If it should work on one row per
blink, read the table with `ctx.table()` rather than `ctx.locs`.
