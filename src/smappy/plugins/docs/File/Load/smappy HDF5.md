---
version: "1"
covers: [smappy.io.formats._load_smappy, smappy.io.hdf5.load_localizations, smappy.session.Session.add_file, smappy.session.file_history]
---

## What it does

Opens an HDF5 file written by this program: a table saved with
[Save › smappy HDF5](plugin:File/Save/smappy HDF5), or the result of a
fitter.  Everything the file carries about the table comes back with it,
so that the analysis goes on where it was left: the history, the derived
columns' recipes, the bounds kept with the table, the ROI.

Only this program's HDF5 files are read: one from another program --
Picasso's, for instance -- does not open.

## How it works

```figure-setup
import tempfile
from pathlib import Path
from smappy import plugins
from smappy.plugins import Context
from smappy.regions import Region
from smappy.session import Session
from smappy.simulate import simulate
before = Session(simulate(n_frames=2000, seed=2))
math = plugins.get("Analysis/Process/Math Parser")()
before.run(math, math.Settings(field="bright", expression="photons > 1000"))
before.set_roi(Region.rect(6700, 5500, 7700, 6500))
load = plugins.get("File/Load/smappy HDF5")()
with tempfile.TemporaryDirectory() as folder:
    path = before.save(Path(folder) / "demo.hdf5")
    after = Session()
    settings = load.Settings(path=str(path))
    after.apply(load, load.run(Context(session=after), settings))
```

**1. The columns** are read from `/locs`, one dataset each, with the types
they were saved with.  A column saved under a name that has since been
renamed is read under its new name, and so is every mention of it in the
metadata -- a derived column's recipe names the columns it is made from.

**2. The metadata** is read back from the file's attribute, and with it:

* the **history** -- the session's log continues from the file's own, so a
  table says everything that was done to it, over as many sessions as it
  took;
* the **derived columns' recipes**, so that a column made with
  [Math Parser](plugin:Analysis/Process/Math Parser) is still recomputed
  or reduced by its rule when the localizations are grouped;
* the **kept bounds**, which open on every layer -- the filter that hides
  what [Remove Localizations](plugin:Analysis/Process/Remove Localizations)
  flagged, for instance;
* the **ROI** that was drawn, and the ROI manager's project, read when the
  ROI manager is first opened;
* the rest as it was: the camera, the fit settings, a drift correction's
  summary, a simulation's settings.

**3. Linking.**  The grouped table is not in the file; it is made again by
linking, unless *link blinks* says not to (see
[Auto](plugin:File/Load/Auto)).

```figure A simulated table, one Math Parser run and a drawn ROI, saved and opened again with this plugin: what the new session starts from.
fig.set_size_inches(7.5, 1.7)
ax = fig.subplots(); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(1, 0)
recipe = after.locs.metadata["derived"][0]
def said(entry):
    text = entry.get("text", "")
    return Path(text).name if entry["what"] == "load" else text.split(",")[0]
lines = ["history:"] + [f"  {entry['what']:<32} {said(entry)}" for entry in after.history]
lines += ["", f"derived: {recipe['field']} = {recipe['expression']}"
              f"   (grouped: {recipe['grouped']})",
          "ROI: {}, x {:g} to {:g}, y {:g} to {:g} nm".format(
              after.roi.kind, *np.asarray(after.roi.bounds)[[0, 2, 1, 3]]),
          f"{len(after.locs)} localizations, "
          f"{len(after.layers[0].state.sets['grouped'].locs)} after linking"]
ax.text(0, 0, "\n".join(lines), fontsize=7.5, family="monospace", va="top",
        linespacing=1.4)
```

## In detail

**Units.**  A table is read in the units it was saved in.  A fitter can
write its table in camera pixels (`x_pix`, ...) instead of nanometres, and
it is then opened in pixels.  The camera's pixel size, if the file records
it, goes into the session's list of files.

**Memory.**  The whole table is read into memory: $N \sum_c b_c$ bytes for
$N$ localizations, $b_c$ the bytes per value of column $c$ (4 for most, 8
for `frame`) -- about as much as the file before compression.  The grouped
table, one row per blink, comes on top.

**The history** is read tolerantly: an entry that is not what this version
of the program writes is dropped rather than stopping the file from
opening, and at most the last 500 are kept.

**Tool results come back.**  What a tool kept with the file -- a drift
curve, say (`/results/saved`) -- is read with it, however the file is
opened, so its *Plot* works again, and the next save writes it back.

**What is not restored.**  The GUI state that may be in the file (`/gui`)
is left alone: opening a colleague's file does not rearrange your tabs.
File > Restore GUI state from file puts it back on request.

## Output

The table, added to the session as a file of its own.  The text says how
many localizations were read and, if they were linked, how many blinks
they make.  The load goes into the history, after the entries the file
brought with it.
