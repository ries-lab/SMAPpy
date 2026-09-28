---
version: "2"
covers: [smappy.io.formats.reader_for, smappy.io.formats._is_smappy, smappy.io.formats._has_saveloc, smappy.session.read_and_group]
---

## What it does

Opens a localization file of any format this program reads, choosing the
reader from the file's name and, where the name is not enough, from what
is inside.  It is what File > Open does.  The formats, each with a plugin
of its own for when the choice should be made by hand:

* [smappy HDF5](plugin:File/Load/smappy HDF5) -- `.hdf5`, `.h5`, written
  by this program;
* [SMAP](plugin:File/Load/SMAP) -- `_sml.mat`, and any `.mat` that holds
  SMAP's `saveloc`;
* [MINFLUX](plugin:File/Load/MINFLUX) -- `.npy`, `.zip`, `.json`;
* [csv](plugin:File/Load/csv) -- `.csv`, `.txt`, `.tsv`, with the columns
  guessed from the header.

After reading, the blinks are linked, the same whichever reader was used.

## How it works

```figure-setup
import json
import tempfile
from pathlib import Path
import h5py
import scipy.io
from smappy.io.formats import reader_for
from smappy.io.hdf5 import save_localizations
from smappy.simulate import simulate
table = simulate(n_frames=200, seed=1)
chosen = []
with tempfile.TemporaryDirectory() as folder:
    folder = Path(folder)
    save_localizations(folder / "fitted.hdf5", table)
    with h5py.File(folder / "other_program.h5", "w") as f:
        f.create_dataset("points", data=np.zeros(3))
    saveloc = {"saveloc": {"loc": {"xnm": [1.0], "ynm": [2.0], "frame": [1.0]}}}
    scipy.io.savemat(folder / "cells_sml.mat", saveloc)
    scipy.io.savemat(folder / "renamed.mat", saveloc)
    scipy.io.savemat(folder / "calibration.mat", {"cal": [1.0]})
    np.save(folder / "minflux.npy", np.zeros(1))
    (folder / "minflux.json").write_text(json.dumps([]))
    (folder / "thunderstorm.csv").write_text("x [nm],y [nm]\n1,2\n")
    (folder / "stack.tif").write_bytes(b"")
    for name in ("fitted.hdf5", "other_program.h5", "cells_sml.mat", "renamed.mat",
                 "calibration.mat", "minflux.npy", "minflux.json",
                 "thunderstorm.csv", "stack.tif"):
        path = folder / name
        try:
            chosen.append((path.name, reader_for(path).name))
        except ValueError:
            chosen.append((path.name, None))
```

**1. By the name.**  The readers whose endings match the file's name are
the candidates, in the order of the list above.  Upper and lower case do
not matter.

**2. By the contents**, where one ending could be more than one thing:

* an `.hdf5` or `.h5` file is read as this program's only if it has a
  `/locs` group;
* a `.mat` file is read as SMAP's if its name ends in `_sml.mat` or it
  holds a variable `saveloc`.

The first candidate that passes is used.  A file that none will take is
refused with the list of formats that are known.

**3. Linking.**  The localizations of the same blink in consecutive frames
are linked into one, with the session's linking settings (within 50 nm and
a gap of 1 frame by default; the *parameters* dialog of the Render tab).
The linked table is what a layer shows when it is *grouped*, which is how
it opens.  An added file is not linked here: it is linked together with the
files already open.  A table with a trace id (`tid`, as a MINFLUX export
has) is grouped by trace instead, one row per trace.  To look at every
localization, untick *grouped* on the layer: the ungrouped table is always
kept.

```figure Which reader Auto chose for a set of files: by the ending, and by what is inside where the ending is ambiguous.  Grey: refused.
fig.set_size_inches(5.2, 2.6)
ax = fig.subplots(); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(len(chosen) - 0.4, -1.2)
ax.text(0.40, -0.9, "file", ha="right", fontsize=8, color="0.4")
ax.text(0.62, -0.9, "read as", ha="left", fontsize=8, color="0.4")
for row, (name, reader) in enumerate(chosen):
    ax.text(0.40, row, name, ha="right", va="center", family="monospace",
            fontsize=8, color="black" if reader else "0.6")
    if reader is None:
        ax.text(0.44, row, "no reader", ha="left", va="center", fontsize=7,
                color="0.6", style="italic")
        continue
    ax.annotate("", (0.60, row), (0.42, row),
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8))
    ax.text(0.62, row, reader, ha="left", va="center", fontsize=8)
```

## In detail

**The choice is made before reading**, so a file that is chosen and then
turns out not to be what its name says fails with the reader's own error.
Every `.npy`, `.zip` and `.json` file is taken for a MINFLUX export, and
every `.csv`, `.txt` or `.tsv` for a table of localizations.

**A csv whose headers are not recognised** needs its columns named; the
[csv](plugin:File/Load/csv) plugin has the setting for it, and File > Open
asks in a dialog.

**What opening costs.**  Every reader holds the whole table in memory:
$N \sum_c b_c$ bytes for $N$ localizations, $b_c$ the bytes per value of
column $c$ -- 4 for most columns, 8 for `frame` -- so some 60 bytes per
localization for a fitted 3D table, plus the grouped table.  Linking a
table of tens of millions of localizations takes minutes.  Here it is done while the file is read, off
the window's thread, so the window stays usable and says what it is doing.

## Output

The table, added to the session as a file of its own, or joined to the open
ones with *add to the open files* (a `filenumber` column tells them apart).
The text says how many localizations were read, in which format, and how
many there are after linking.  The load is written into the history with
its settings.
