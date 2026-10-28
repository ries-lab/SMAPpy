---
title: Batch window
summary: Runs one chain of plugins over many files, each on its own, and writes every file's table, numbers and figures to an output folder with a summary of all of them.
widget: smappy.gui.batch_window.BatchWindow
covers: [smappy.gui.batch_window.BatchWindow, smappy.batch.resolve_inputs, smappy.batch.validate, smappy.batch.settings_for, smappy.batch.fingerprint, smappy.batch.run_one, smappy.batch.run, smappy.batch.write_summary, smappy.batch.summary_rows, smappy.batch.scalars, smappy.batch.json_safe, smappy.batch.publish, smappy.batch.watch, smappy.batch.is_complete, smappy.batch.stop_on_stdin, smappy.batch_page.write_page, smappy.batch_page.trend_columns, smappy.batch_page.draw_trends]
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

With *watch* ticked, the batch does not end with the files there are: it
keeps looking at the folders and runs the chain on each new file as soon as
the microscope has finished writing it -- for a folder that fills during a
day of acquisitions, or the same bead sample measured every week.  A summary
page in the output folder shows every file with its figures and numbers, and
plots chosen numbers across the files in the order they were taken, so a
change in the instrument shows as a trend.

It opens from **Plugins > Batch...**, or on its own as `smappy-batch-gui`.  It
is a front end to the job file and the command `smappy-batch`: everything it
does can be repeated from a terminal with the file it writes.  The **?** at
its top right (or F1) opens this page.

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
it, and for the whole batch `summary.csv`, `batch_report.json` and the
summary page, `index.html` (see *Output*), all written again after every
file, so they can be read while the batch runs.  The table's *status* column follows the run as it goes:
*running*, *done*, *skipped* or *failed*, with the reason in the tooltip,
and a folder row counts its files.

**7. Watching.**  With *watch* ticked, Run processes the files there are,
then looks at the folders every 2 seconds.  A new file is run once it is
**complete**: a localization file written by SMAPpy's fitter once the fit
has closed it -- the file says so itself, so a live acquisition that pauses
for minutes is never taken half-way -- and any other file once neither its
size nor its time stamp has changed for 10 seconds (for a Micro-Manager
series, of all its files).  What is done already is skipped as in step 5,
so a watch started again on the same folder carries on where it was, and a
file that changes after it was run is run again.  The watch runs until
*Stop*.

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

**The trends** on the summary page are the `summary.csv` columns named in
*trend columns* (patterns such as `stats.*` allowed), one panel each,
against the time each file was taken: Micro-Manager's `StartTime` where the
file's metadata has it, else the input's modification time, else simply the
files in order.  A failed file has no point; the dashed line is the median.
With *trend columns* empty, every number that differs between the files is
plotted, except one that repeats a column before it, at most twelve, the
seconds each file took last.  A number the same in every file is usually a
setting echoed back, and is left out.

**Names.**  A file's output folder is its name without the extension (and
without `.ome.tif` or `_sml.mat`); two inputs with the same name get their
parent folder's name in front, and a number if even that is not enough.

**The run** exits with 0 when every file was done (or already was), 1 when
any failed or the batch was stopped by a step's *abort*, and 2 when the job
did not validate; the log's last line says which.  A watch exits with 0
once stopped.

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
[Layers](plugin:Analysis/Process/Layers) step first: without it each file is
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

### merge blinks
Whether a step reads every localization or the blinks merged -- [merge
blinks](plugin:Panels/Glossary) (also called grouping or linking) combines
the localizations in consecutive frames that come from one blinking event
into one row: per step, and for the whole chain under *more*.  The choices
are *merged*, *not merged* and *auto* (the chain's) or, for the chain, *as
each layer* (as the Render tab has it).  The rules are in
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

### trend columns
Empty plots every number that differs between the files.  Name a few to
follow over time, the ones that say the instrument is as it was -- the
precision, the photons, the number of localizations -- and leave out the
ones that differ because the cells do.

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

### watch
Runs the files there are, then each new one in the folders as soon as it is
complete (see *How it works*), until *Stop*.  The folders may be empty at
the start.  *first file only* does not apply.  Not needed after a live fit:
a fitter's *after the fit* runs its steps there already.

### Run
Validates, saves the job (asking for a name the first time) and starts the
batch.  The progress bar counts files; the status bar shows what the
current step says.

### Stop
Ends the batch program.  A file cut off half-way has no finished record,
so the next run does it again; the files already done are kept.  A watch is
asked to stop instead, and ends once the file in hand is done, with its
summary written; pressing *Stop* again ends it at once.

### open summary
Opens the summary page, `index.html`, in the system's browser.

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
  status, message and traceback.  It is the record of which files a run saw;
* `index.html` -- the summary page: a row per file with its status, when it
  was taken, its figures as thumbnails (click for the full figure) and its
  numbers, and above them `trends.png`, the *trend columns* across the
  files.  It is a plain page with relative links: the output folder can be
  zipped or copied and the page still works.  While a batch or a watch runs
  it reloads itself every 20 seconds.

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
* **A folder can be watched**, and a page shows every file and plots numbers
  across them; SMAP has neither.  The idea is Picasso Server's
  (Strauss 2022), not SMAP's.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
* Strauss MT. Picasso-server: a community-based, open-source processing
  framework for super-resolution data. *Commun Biol* 5, 930 (2022) -- the
  folder watcher and the plots of every run's numbers over time.
  [doi:10.1038/s42003-022-03909-5](https://doi.org/10.1038/s42003-022-03909-5)
