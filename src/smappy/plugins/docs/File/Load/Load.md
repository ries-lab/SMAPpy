---
version: "1"
covers: [smappy.plugins.file.Open]
---

## What it does

The first line of the File tab: a button that opens the file dialog, as
File > Open does, and reads the files chosen.  With *add* ticked the files
join the ones already open, as File > Add file does; without it the first
one replaces them.

Any format that [Auto](plugin:File/Load/Auto) reads can be chosen: this
program's HDF5, SMAP, Picasso, DECODE, MINFLUX and text tables.  A text
table whose columns are not recognised, and that no saved conversion fits,
asks for its columns in a dialog first.

It is a button for the window only.  A chain or a batch, which has no one
to choose files, reads a file through *Auto*, with the file's path.

## Output

The files, each opened as a file of its own (or added), on the Render tab's
layers.  The window's status line says how many localizations were read.
