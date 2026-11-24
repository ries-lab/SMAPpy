---
version: "4"
covers: [smappy.plugins.file.Open, smappy.import_columns.plan, smappy.import_columns.available_memory, smappy.gui.columns_dialog.ColumnsDialog]
---

## What it does

The first line of the File tab: two buttons in its title bar, beside the
name, that show the file dialog.  *Open* opens the files chosen in place of the
ones open; *Add* adds them as more files of the same table, for instance the
next field of view.  File > Open, and a file dropped on the window, ask
which of the two when localizations are already open.  The section unfolds
to one setting, *columns*: which columns of a file are read, for these
buttons, File > Open and a drop alike.

Any table that [Load from path](plugin:File/Load/Auto) reads can be chosen:
this program's HDF5, SMAP, Picasso, DECODE, MINFLUX and text tables.  A text
table whose columns are not recognised, and that no saved conversion fits,
asks for its columns in a dialog first.

A TIFF is either an image to show or raw data to fit, and its number of
frames says which: up to 50 frames is an image -- a widefield picture, a
few channels, a z-stack -- and more is a movie of blinking molecules.
Between 20 and 500 frames it could be either (a long time-lapse, a short
raw movie), so you are asked, with the count's answer as the default.  An
image (or a PNG) is opened with either button but not shown: it joins the
images an image layer can show, and the Render tab's *+ > image layer* and
the layer's *source* show it.  Raw data goes to the fitter on the Localize
tab.  A file dropped on the window is sorted the same way.

It is a button for the window only.  A chain or a batch, which has no one
to choose files, reads a file through *Load from path*, with the file's path
and its *add to the open files* tick; it asks nothing.

## How it works

**Which columns.**  A localization file often holds more columns than
anything reads -- a SMAP file 30, of which 13 are used -- and a large file
with all of them may not fit the memory of a microscope PC.  Before a
table is read, its columns are listed from the file's header, with the
memory each will take, and *columns* decides:

* **standard, ask if large** (the default): every column but those the
  format writes and nothing here reads (for SMAP, the pixel copies of
  positions in nm; see [Load from path](plugin:File/Load/Auto)).  If even
  those would take more than half the memory available, a dialog asks
  first: every column, its memory, the standard ones ticked, and the total
  against the memory available.  Untick what you will not use.
* **all**: every column, never asked.
* **choose on import**: the dialog for every file.

`x_nm`, `y_nm` and `frame` cannot be unticked: no table is without them.
Only this program's HDF5 files and SMAP's `-v7.3` files can list their
columns without reading the file, so only they are asked about; the
others are read by the rule alone.  An `.msr` asks its own questions.  On a
Mac the memory available is taken as the whole memory, which the system
does not report otherwise.

A column left out is not in the table, and not in a file saved from it.
Open the file again with *all* to have it back.

## Parameters

### columns
Leave it on *standard, ask if large*.  *all* is for a file whose every
column you need and which fits; *choose on import* for reading a few
columns of large files one after another.

## Output

The files, each opened as a file of its own, on the Render tab's layers.
The window's status line says how many localizations were read.
