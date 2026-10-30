---
version: "3"
covers: [smappy.plugins.file.Open]
---

## What it does

The first line of the File tab, open when the program starts: two buttons
that show the file dialog.  *Open* opens the files chosen in place of the
ones open; *Add* adds them as more files of the same table, for instance the
next field of view.  File > Open, and a file dropped on the window, ask
which of the two when localizations are already open.

Any table that [Load from path](plugin:File/Load/Auto) reads can be chosen:
this program's HDF5, SMAP, Picasso, DECODE, MINFLUX and text tables.  A text
table whose columns are not recognised, and that no saved conversion fits,
asks for its columns in a dialog first.  An image (a TIFF of a few frames, a
PNG) is opened with either button but not shown: it joins the images an
image layer can show, and the Render tab's *+ > image layer* and the layer's
*source* show it.  A long TIFF stack, raw data to fit, goes to the fitter on the
Localize tab -- as when the file is dropped on the window.

It is a button for the window only.  A chain or a batch, which has no one
to choose files, reads a file through *Load from path*, with the file's path
and its *add to the open files* tick; it asks nothing.

## Output

The files, each opened as a file of its own, on the Render tab's layers.
The window's status line says how many localizations were read.
