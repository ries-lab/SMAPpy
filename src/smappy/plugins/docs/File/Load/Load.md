---
version: "2"
covers: [smappy.plugins.file.Open]
---

## What it does

The first line of the File tab, open when the program starts: one click on
*Open* shows the file dialog, as File > Open does, and opens the files
chosen.  When localizations are already open it asks whether the new ones
*Replace* them or are *Add*ed as more files of the same table.

Any table that [Load from path](plugin:File/Load/Auto) reads can be chosen:
this program's HDF5, SMAP, Picasso, DECODE, MINFLUX and text tables.  A text
table whose columns are not recognised, and that no saved conversion fits,
asks for its columns in a dialog first.  An image (a TIFF of a few frames, a
PNG) is added as an image layer beside the localizations, and a long TIFF
stack, raw data to fit, goes to the fitter on the Localize tab -- as when the
file is dropped on the window.

It is a button for the window only.  A chain or a batch, which has no one
to choose files, reads a file through *Load from path*, with the file's path.

## Output

The files, each opened as a file of its own, on the Render tab's layers.
The window's status line says how many localizations were read.
