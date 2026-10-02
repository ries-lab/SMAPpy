---
version: "1"
covers: [smappy.session.Session.save, smappy.io.hdf5.save_localizations, smappy.io.hdf5.save_results, smappy.io.hdf5.save_gui_state]
---

## What it does

Writes the localizations to an HDF5 file (`.hdf5` or `.h5`), together with
the record of what was done to them, so that the file can be opened again
here -- by [smappy HDF5](plugin:File/Load/smappy HDF5) or
[Auto](plugin:File/Load/Auto) -- and still says how it was made.  HDF5 is
an open format; any language can read the columns (in Python with `h5py`,
in MATLAB with `h5read`).

It is the only format this program writes localizations in.  Saving is also
File > Save and File > Save as.

## How it works

```figure-setup
import json
import tempfile
from pathlib import Path
import h5py
from smappy import plugins
from smappy.regions import Region
from smappy.session import Session
from smappy.simulate import simulate
session = Session(simulate(n_frames=2000, seed=2))
math = plugins.get("Analysis/Process/Math Parser")()
session.run(math, math.Settings(field="bright", expression="photons > 1000"))
session.set_roi(Region.rect(6700, 5500, 7700, 6500))
session.gui_state_provider = lambda: {"tabs": []}          # what the GUI hands over
session.results = {"Analysis/Drift/RCC": {"drift": [[0.0, 0.0, 0.0]]}}
with tempfile.TemporaryDirectory() as folder:
    saved = session.save(Path(folder) / "demo.hdf5", gui_state=True)
    with h5py.File(saved, "r") as f:
        columns = sorted(f["locs"])
        metadata = json.loads(f.attrs["metadata"])
        extra = [name for name in ("results/saved", "gui/state") if name in f]
```

**1. The table.**  Every column of the current table is written as it is,
one dataset per column under `/locs`, with its own type, compressed.  It is
the ungrouped table: the grouped one is made again by linking when the file
is opened.  If it has been linked, `group_id` and `n_in_group` are among
the columns.

**2. The record.**  The table's metadata goes into one attribute of the
file, as JSON: the history (every change made to the table, with its
settings), the recipes of derived columns, the bounds a plugin asked to
keep with the table, a drift correction's summary, the camera and the fit
settings of a table that came from a fit, the simulation's settings for a
simulated one, the ROI that is drawn, and the ROI manager's project if it
was used.  Two parts of a fitted table's metadata are too large for an
attribute and go beside it: the image tags (what the microscope recorded per
frame, such as the piezo position) into `/frame_tags`, and the acquisition's
Micro-Manager summary and device properties into `/acquisition`.  Both are
read back into the metadata when the file is opened.

**3. What the tools kept.**  Some plugins keep what they worked out -- a
drift curve, say -- so that its figure can be drawn again later.  That goes
into `/results/saved`.

**4. The GUI, optionally.**  With *save the GUI state too*, the tabs, their
plugins and their settings are written into `/gui/state`, without the
window's size and position.  They are not put back when the file is opened;
File > Restore GUI state from file does that.

```figure What a saved file holds: a simulated table after one Math Parser run, with an ROI drawn.
fig.set_size_inches(7.5, 3.3)
ax = fig.subplots(); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(1, 0)
def box(x, title, lines, color):
    ax.text(x, 0.0, title, fontsize=9, weight="bold", color=color, va="top")
    ax.text(x, 0.09, "\n".join(lines), fontsize=7.5, family="monospace",
            va="top", linespacing=1.35)
box(0.0, "/locs: a column each", columns, "#1f77b4")
recipe = metadata["derived"][0]
count = len(metadata["history"])
lines = [f"history: {count} entr{'y' if count == 1 else 'ies'}",
         f"  {metadata['history'][-1]['what']}",
         f"derived: {recipe['field']} = {recipe['expression']}",
         f"roi: {metadata['roi']['kind']}",
         "units: " + metadata["units"],
         "simulation_settings: {...}",
         f"... and {len(metadata) - 5} more"]
box(0.3, "metadata: one JSON attribute", lines, "#d62728")
box(0.75, "groups", [f"/{name}" for name in extra], "#2ca02c")
```

## In detail

**The history is capped** at the last 500 entries, since all of the
metadata is one attribute.  An entry is a few hundred bytes.

**Writing over a file.**  The file is written from scratch: whatever was in
it before is gone, including tool results and a GUI state the session no
longer has.

**What is not saved.**  The layers -- their filters, their grouping, their
colours -- belong to the session, not to the file; only the bounds a plugin
asked to keep with the table (Remove Localizations' hide flag, for
instance) come back.  The grouped table is not saved either.

**How big the file is.**  Before compression, a table of $N$
localizations takes

$$S = N \sum_c b_c$$

bytes, $b_c$ the bytes per value of column $c$: 4 for most columns, 8 for
`frame`.  A fitted 3D table with a dozen or so columns is some 60 bytes per
localization, 600 MB for ten million; the compression takes part of that
off.

**Compression** is lzf, which is about five times faster than gzip for about
15 % more space.  The writer is the one the fitters use to write their
results block by block while they run.

## Parameters

### path
Empty works only when the open file is itself `.hdf5` or `.h5`: after
opening a `_sml.mat` or a csv, give the new file a name.

### gui_state
Useful for a file that is a record of how it was analysed; it costs little
space.  Nothing is restored unless someone asks for it, so it does no
harm in a file that is shared.

## Output

The file.  The text says how many localizations went where.  The save is
written into the session's history, so that the log says where the table
went.

## Differences from SMAP

SMAP saves `_sml.mat` files, which this program reads but does not write.
The localizations here go to HDF5 instead, which is not tied to MATLAB and
can be read while it is being written.

## References

* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
