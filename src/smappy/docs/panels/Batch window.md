---
title: Batch window
summary: Runs one chain of plugins over many files, each on its own, and writes every file's table, numbers and figures to an output folder with a summary of all of them.
widget: smappy.gui.batch_window.BatchWindow
covers: [smappy.gui.batch_window.BatchWindow, smappy.batch.resolve_inputs, smappy.batch.validate, smappy.batch.settings_for, smappy.batch.fingerprint, smappy.batch.run_one, smappy.batch.run, smappy.batch.write_summary, smappy.batch.scalars, smappy.batch.json_safe]
---

## What it does

An analysis that works on one cell usually has to be done on thirty: the
same filter, the same drift correction, the same measurement, with the same
settings, and the numbers collected into one table.  Clicking through it
thirty times is slow, and it is easy to change a setting by accident on the
seventeenth.

The batch window takes a **chain** -- an ordered list of plugins with their
settings, which runs as one -- and a list of **files**, and runs the chain on
every file, one after another.  Each file is opened in a fresh session of
its own, processed, saved and closed, so nothing carries over from one file
to the next, and a file that fails is marked failed while the batch goes on
with the rest.  Every file gets its own output folder, with the processed
table, the figures each step drew and a record of each step's numbers; the
batch as a whole gets a table with one row per file.

The chain can start with a fit (a *Localize* plugin), and then the inputs are
camera images; otherwise the inputs are localization files.  Either way one
input is one file.

It opens from **Tools > Batch...**, or on its own as `smappy-batch-gui`.  It
is a front end to the job file and the command `smappy-batch`: everything it
does can be repeated from a terminal with the file it writes.

## How it works

**1. The job.**  The files, the chain and the output options together are a
*job*, saved as a `.batch.yaml` file.  *Run* saves it first (asking for a
name the first time), then starts `smappy-batch run` on it as a **separate
program**.  A crash or a file that exhausts the memory ends that program,
never this window or the session in the main window behind it.

**2. Validation.**  Before anything runs, the job is checked strictly: every
setting a step names must be one the plugin has, every choice one it offers,
every file and folder must exist, and the files must resolve to at least one
input.  Errors stop the run and are listed in the log; warnings -- a step
saved with an older version of its plugin, a second step that makes a table
of its own -- are listed and the run goes ahead.

**3. The files.**  A row of the *files* table is a file, or a folder with the
patterns its files must match (searched in every subfolder).  The rows are
resolved into one list at the start.  A file named twice -- by a folder and
again on its own -- is processed once, in its first place.  The output
folder is never searched, so a second run does not take its own results for
input.

**4. One file.**  For each file the chain's settings are put together: the
chain's values, then the row's *overrides*.  If the chain starts with a fit,
the file becomes the fit's input; otherwise it is opened as a localization
file.  The chain then runs as it would in a plugin tab -- one step after the
other, each seeing what the steps before it did -- and the result is saved.
A step that would ask before running (*this will take an afternoon*) is
answered by *when a step asks*.

**5. What is re-run.**  Every finished file records a **fingerprint** of how
it was made: every setting of every step after the overrides, the **version
of every step's plugin**, the output options that change what is written,
and the input file's size and modification time.  With *files already done*
on *skip_identical*, a file whose last run finished with the same
fingerprint is skipped.  So editing one step and running again redoes every
file, while running again unchanged redoes nothing -- and a plugin whose
version was bumped because its numbers moved is re-run on every file, while
one whose code changed without a new version is not.

**6. The report.**  The output folder holds, per input, a folder named after
it, and for the whole batch `summary.csv` and `batch_report.json`
(see *Output*).  The table's *status* column follows the run as it goes:
*running*, *done*, *skipped* or *failed*, with the reason in the tooltip,
and a folder row counts its files.

## In detail

**The fingerprint** is the first 16 hex digits of a SHA-256 hash of

* the flat settings of the whole chain for that file, `<step key>.<field>`,
  after the job's and the file's overrides (and with the file itself filled
  in as the first step's input);
* `(plugin path, version)` of every enabled step;
* *save the processed localizations*, *file suffix*, *figures* and
  *fitted file*;
* the input's resolved path, size and modification time.

A file is skipped only if its folder's `results.json` says `done` with the
same fingerprint: a failed file, or one skipped because a step asked, is
tried again.  A file skipped as unchanged keeps its row in `summary.csv`,
read from the `results.json` of the run that made it.  *Delete the fitted
file afterwards* and the output folder are not part of the fingerprint; the
output folder does not need to be, since the record lives in it.

**The summary** has one row per file -- the file, its status, the seconds it
took, the number of localizations and any message -- and a column for every
single number or word a step returned in its data, as `<step key>.<name>`
(nested dictionaries give dotted names).  Columns appear in the order they
are first met.  In `results.json` a step's data is kept as far as it is
JSON: arrays and lists of up to 1000 values, nested dictionaries; tables and
anything larger are left out, and a NaN is written as `null`.

**Names.**  A file's output folder is its name without the extension (and
without `.ome.tif` or `_sml.mat`); two inputs with the same name get their
parent folder's name in front, and a number if even that is not enough.

**The run** exits with 0 when every file was done (or already was), 1 when
any failed or the batch was stopped by a step's *abort*, and 2 when the job
did not validate; the log's last line says which.

## Controls

### files
The inputs, one per row.  *use* includes or leaves out a row without
deleting it.  *input* is a file or a folder; *pattern*, for a folder, is the
comma-separated file patterns to look for (it starts from the usual ones for
the chain's input: `*.tif, *.tiff` for images, `*.h5, *.hdf5, *_sml.mat,
*.csv` for localizations).  *overrides* changes settings for that file only,
written `key.field: value` and separated by `;`, as in the job file --
`drift.n_timepoints: 8` for a step labelled *drift*.  Files, folders, a
`.chain.yaml` or a `.batch.yaml` can also be dropped onto the window.

### add files...
Adds files, one row each.

### add folder...
Adds a folder, asking for the patterns its files must match.

### remove
Removes the selected rows (in the *files* table), or, in the chain's step
list, the selected step.

### clear
Removes every row.

### chain
The chain to run: pick a saved one from the list (the chains in the plugin
folders and the chains folder), open a file, or start a new one.  It shows
the same panel a chain has in a plugin tab, one folded section per step with
that step's settings, so a chain is edited here as it is there; it cannot be
run on the session from here, only saved and run over the files.  To set up
the layers and filters every file is to get, put a
[Chain/Layers](plugin:Chain/Layers) step first: without it each file is
filtered as a freshly opened one is.

### open...
Opens a `.chain.yaml` file.

### new
Starts an empty chain, with its step list open.

### edit steps
Shows the chain's steps, each with a tick that enables it; an unticked
step stays in the chain and is skipped.

### add...
Adds a plugin as a step, after the selected one.

### up
Moves the selected step one place earlier.

### down
Moves the selected step one place later.

### rename...
Gives the selected step a label.  The label, made into an identifier, is the
step's key: what overrides and the summary's columns are named by
(`NPC radius` -> `npc_radius`).

### save chain
Writes the chain to its file; one that has none asks where, starting in the
chains folder.
Saved in a plugin folder or the chains folder, the chain also appears in the
plugin tree.  A job refers to a saved, unchanged chain by its file; a chain
that is not saved or has changes is written into the job itself.

### save as...
Writes the chain to a new file.

### grouping
Whether a step reads the table of localizations or the grouped one (one row
per blink): per step, and for the whole chain under *more*.  The rules are in
`docs/batch.md` ("Grouping"): the plugin's own requirement wins, then the
step's choice, then the chain's, then each layer's.

### output
Where the results go and what is written.

### folder
The output folder.  Empty: `<job>_output` beside the job file.

### file suffix
Added to the input's name for the processed table: `cell3_proc.h5`.  An
input `.h5` is never overwritten.

### save the processed localizations
Writes each file's table, with its history (which ends with the chain's
entry, every step's settings in it) and what each step kept.  Untick it for
a chain that only measures and whose numbers are all you want.

### figures
The format the steps' figures are saved in, one file per figure, or *none*.

### fitted file
For a chain that starts with a fit: where the fit writes its own
localization file, *beside_image* (where the fit would put it anyway) or in
the file's *output* folder.

### delete the fitted file afterwards
Deletes the fit's own file once the processed table is saved, which then
is the only copy.

### when a step asks
What to do with a file a step would ask about before running (COMET's
"this will take hours"): *skip* the file, *proceed* anyway, or *abort* the
whole batch.

### files already done
*skip_identical* skips a file whose last run finished with the same
settings, plugin versions, output options and input file (see *How it
works*); *always* runs every file again.

### open job...
Opens a `.batch.yaml` file into the window.

### save job...
Writes the job to a `.batch.yaml` file, which `smappy-batch run` can run
from a terminal.

### validate
Checks the job and lists every problem in the log, without running
anything.  The quickest way to learn a plugin's setting names is a failed
validation, which lists them.

### first file only
Runs only the first file: a trial run, to check the numbers and figures
before the rest.

### Run
Validates, saves the job (asking for a name the first time) and starts the
batch.  The progress bar counts files; the status bar shows what the
current step says.

### Stop
Ends the batch program.  A file cut off half-way has no finished record,
so the next run does it again; the files already done are kept.

### open output
Opens the output folder in the system's file browser.  Opened from the main
window, it asks for a processed file instead and opens it there.

## Output

In the output folder, per input, a folder of its name with

* `<name><suffix>.h5` -- the processed table (if *save the processed
  localizations*);
* for image input, the fit's own `<name>_locs.hdf5`, if *fitted file* is
  *output* (and not deleted);
* `figures/<step key>_<figure>.<format>`;
* `results.json` -- the status, the time it took, the fingerprint, and per
  step the line it reported, its settings and its numbers; a traceback if it
  failed.

and for the batch

* `summary.csv` -- one row per file, a column per number any step returned:
  the table to take to a statistics program;
* `batch_report.json` -- the job and the chain as they were resolved, the
  plugin versions, the list of files the folders resolved to, and each file's
  status, message and traceback.  It is the record of which files a run saw.

A good run ends with *finished* in the log and every row *done* or
*skipped*.  *finished, with failures*: hover over a *failed* status or read
that file's `results.json` for the reason, fix it, and run again -- only
what failed or changed is redone.

## Differences from SMAP

Based on SMAP's `BatchAnalysis` (Analyze plugins over a list of files) and
its fitting `Batchprocessor`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).  The main changes:

* **One window for fitting and analysis.**  A chain may start with a fit and
  go on to the analysis, where SMAP had separate tools for the two.
* **Nothing needs SMAP's GUI.**  SMAP's batch tools drive the open program's
  own objects.  Here the batch is a separate program reading a plain file,
  so it runs the same from this window, a terminal or a script.
* **One failing file costs that file.**  In `BatchAnalysis` the per-plugin
  error handling is commented out, so one failure ends the batch; here the
  file is marked failed, the reason recorded, and the batch goes on.
* **Numbers into a table.**  SMAP collects every plugin's results into one
  MATLAB `_results.mat` struct; here each file's numbers go into its
  `results.json` and into one `summary.csv`.
* **Re-runs skip what is done** with the same settings and plugin versions;
  SMAP runs every file again.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
