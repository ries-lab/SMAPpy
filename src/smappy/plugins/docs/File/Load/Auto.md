---
version: "3"
covers: [smappy.io.formats.reader_for, smappy.io.formats._is_smappy, smappy.io.formats._has_saveloc, smappy.session.read_localizations, smappy.io.formats._load_smappy, smappy.io.hdf5.load_localizations, smappy.io.formats._load_sml, smappy.io.formats._load_minflux, smappy.io.formats._minflux_records, smappy.io.formats._minflux_from_json, smappy.group.by_trace, smappy.io.formats._load_csv, smappy.io.formats.guess_csv_mapping, smappy.io.formats.csv_columns]
---

## What it does

Opens a localization file of any format this program reads, choosing the
reader from the file's name and, where the name is not enough, from what
is inside.  It is what File > Open does.  The formats:

* **smappy HDF5** -- `.hdf5`, `.h5`, written by this program: a saved table
  or a fitter's result, with everything it carries;
* **SMAP** -- `_sml.mat`, and any `.mat` that holds SMAP's `saveloc`
  ([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1));
* **MINFLUX** -- `.npy`, `.zip`, `.json`: the localizations an Abberior
  MINFLUX microscope exports
  ([Gwosch et al. 2020](https://doi.org/10.1038/s41592-019-0688-0));
* **csv** -- `.csv`, `.txt`, `.tsv`: a table written as text, with the
  columns guessed from the header -- ThunderSTORM's export
  ([Ovesný et al. 2014](https://doi.org/10.1093/bioinformatics/btu202)) and
  SMAP's names are recognised without being told.

A file whose name does not say what it is -- a SMAP file not called
`_sml.mat` -- is read by choosing its *format* by hand (under *more*).

The file's layer shows its blinks merged, the same whichever reader was
used: [merge blinks](plugin:Panels/Glossary) (also called grouping or
linking) combines the localizations in consecutive frames that come from one
blinking event into one localization.  The file does not wait for that to
open: a large one opens on its single localizations and is merged in the
background.

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
the candidates, in the order of the list above (smappy HDF5, SMAP,
MINFLUX, csv).  Upper and lower case do
not matter.

**2. By the contents**, where one ending could be more than one thing:

* an `.hdf5` or `.h5` file is read as this program's only if it has a
  `/locs` group;
* a `.mat` file is read as SMAP's if its name ends in `_sml.mat` or it
  holds a variable `saveloc`.

The first candidate that passes is used.  A file that none will take is
refused with the list of formats that are known.

**3. Merging blinks.**  The localizations of the same blink in consecutive
frames are merged into one, with the session's merge settings (within 50 nm
and 1 dark frame by default; *merge settings...* in the Render tab).  The
merged blinks are what a layer shows when *merge blinks* is ticked, which is
how it opens.  The merge is not part of reading: a small file is merged when
it is first drawn, and a large one (300,000 localizations or more) opens at
once on its single localizations while it is merged in the background, with
its progress in the status bar; the picture switches to blinks when they are
there.  Headless -- in a script, a chain or a batch -- the blinks are merged
when something first reads them.  An added file is merged together with the
files already open.  A table with a trace id (`tid`, as a MINFLUX export
has) is merged by trace instead, one row per trace.  To look at every
localization, untick *merge blinks* on the layer: the single localizations
are always kept.

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
every `.csv`, `.txt` or `.tsv` for a table of localizations.  With a
*format* chosen, that reader is used whatever the file is called.

**What opening costs.**  Every reader holds the whole table in memory:
$N \sum_c b_c$ bytes for $N$ localizations, $b_c$ the bytes per value of
column $c$ -- 4 for most columns, 8 for `frame` -- so some 60 bytes per
localization for a fitted 3D table, plus the merged blinks.  Reading is done
off the window's thread, so the window stays usable and says what it is
doing.  Merging the blinks -- a few seconds for a few million
localizations, some 25 s for 57 million -- follows in the background, after
the file has opened.

**smappy HDF5.**  The columns are read from `/locs`, one dataset each, with
the types they were saved with, and the file's metadata comes back with
them: the **history** (the session's log continues from the file's own; an
entry this version does not understand is dropped, and at most the last 500
are kept), the **derived columns' recipes** (a column made with
[Math Parser](plugin:Analysis/Process/Math Parser) is still recomputed or
reduced by its rule when blinks are merged), the **kept bounds**, which open
on every layer, the drawn **ROI** and the ROI manager's project, the camera,
the fit settings, and what a tool kept with the file -- a drift curve, say
(`/results/saved`) -- so its figure can be shown again.  A table is opened
in the units it was saved in: a fitter can write camera pixels (`x_pix`,
...) instead of nanometres.  The GUI state that may be in the file (`/gui`)
is left alone: File > Workspace > From data file puts it back on request.
Only this program's HDF5 files are read: one from another program --
Picasso's, for instance -- does not open.

**SMAP.**  SMAP keeps its table in `saveloc.loc`, one field per column.
Each becomes a column, renamed to this program's name where there is one
(`SML_COLUMNS`) and kept under its own name where there is not:

| SMAP | here | | SMAP | here |
| --- | --- | --- | --- | --- |
| `xnm`, `ynm`, `znm` | `x_nm`, `y_nm`, `z_nm` | | `LLrel` | `logl_rel` |
| `phot` | `photons` | | `logLikelihood` | `logl` |
| `bg` | `background` | | `xnmerr`, `ynmerr` | `x_err_nm`, `y_err_nm` |
| `locprecnm` | `xy_err_nm` | | `zerr`, `locprecznm` | `z_err_nm` |
| `PSFxnm`, `PSFynm` | `sigma_nm`, `sigma_y_nm` | | `photerr` | `photons_err` |
| `xpix`, `ypix` | `x_pix`, `y_pix` | | `PSFxpix` | `sigma_pix` |

`filenumber` is left behind (it is given again when the file is added), and
so are the columns nothing here reads (`xnmf`, `ynmf`, `xpixf`, `ypixf`,
`xpixerr`, `ypixerr`, `PSFypix`).  SMAP's `numberInGroup` comes across under
its own name: it is not this program's `n_in_group`, which merging blinks
writes.  MATLAB counts from 1, so
$\mathrm{frame} = \mathrm{frame}_{\mathrm{SMAP}} - 1$, as a 64-bit integer;
`channel` and `iterations` become 32-bit integers and every other column
single precision.  The pixel size (`cam_pixelsize_um`) is read from the
first file's `info`, and from a `-v7.3` file also the frame count, the
conversion, the offset and the EM gain, kept in the metadata under `smap`.
A `.mat` without `saveloc` is refused; one that held several SMAP files is
read as one table.

**MINFLUX.**  MINFLUX ([Balzarotti et al. 2017](https://doi.org/10.1126/science.aak9913))
takes no camera frames: it records every iteration of every localization.
Of each, only the last -- the most precise -- is read, and only where the
instrument marked it valid (`vld`) and it has a position.  Positions go from
metres to nanometres and are not shifted to start at zero.  There are no
frames, so the localizations are sorted by their time (`tim`, kept as
`time_s`) and `frame` is the rank in that order; the trace is `tid`, and
`efo`, `cfr`, `dcr`, `efc`, `ecc`, `fbg` and the iteration number come
across where the export has them.  The photons are the last iteration's
`eco`, and the precision, which the export does not give, is set from them,

$$\sigma_{xy} = \frac{150\,\mathrm{nm}}{\sqrt{N}} ,$$

SMAP's number for the json export -- a stand-in so rendering and the
precision filter have something to work with, usually several times worse
than a MINFLUX localization really is.  `sigma_nm` is 150 nm for all.  With
the default 25 nm filter on `xy_err_nm`, localizations of fewer than 36
photons are hidden when the file opens.  Merging blinks goes **by trace**:
one row per `tid` (its mean position, summed photons, first `frame`, and
`n_in_group`), rather than by distance and frame.  The `.npy`, the `.json`
and a `.zip` holding either are read; MINFLUX data saved as a MATLAB `.mat`
is not.

**csv.**  The first line is the header if any of its cells is not a
number; otherwise the columns are called *column 1*, *column 2*, ...  Cells
are split by commas, or by semicolons if the first line has more of them (a
tab-separated file is not split, and is refused as having no x and y).
Unless *columns* says otherwise, each header is cleaned -- lower case, the
unit in square brackets taken off, spaces made underscores -- and looked up
(`CSV_NAMES`):

| column | headers |
| --- | --- |
| `x_nm`, `y_nm`, `z_nm` | `x`, `y`, `z`; `xnm`, ...; `x_nm`, ... |
| `frame` | `frame`, `t` |
| `photons` | `intensity`, `photons`, `phot`, `n` |
| `xy_err_nm` | `uncertainty`, `uncertainty_xy`, `locprecnm` |
| `z_err_nm` | `uncertainty_z`, `locprecznm` |
| `sigma_nm`, `sigma_y_nm` | `sigma`, `sigma1`, `psfxnm`; `sigma2` |
| `background` | `offset`, `bg`, `background` |
| `background_std` | `bkgstd` |
| `channel`, `logl_rel`, `id` | `channel`; `llrel`; `id` |

When two headers mean the same column the first wins; x and y must be
found.  `frame`, `channel` and `id` are read as whole numbers, the rest as
single precision, and a row whose x or y is missing is left out.  **Units
are not converted**: the unit in a header is dropped unread, so `x [px]` is
read as nanometres -- convert a table in pixels first,
$x_{\mathrm{nm}} = x_{\mathrm{px}}\, p$ with $p$ the pixel size.  Frames are
read as written (ThunderSTORM counts from 1); a file with no frame column
is read as frame 0.

## Parameters

### format
Leave it on *auto*.  Choose a format for a file whose name does not say
what it is -- a SMAP file renamed, a csv saved as `.dat` -- or to make sure
the file is read as that format and refused otherwise.

### mapping
Read only for a csv; greyed for any other file.  Needed only when the
headers are not recognised, or to read a column the list does not know.
Each pair is *header in the file* = *column in the table*, and only the
columns named are read -- so name x and y too.  Opening such a file from
the File menu asks for the mapping in a dialog instead.

## Output

The table, added to the session as a file of its own, or joined to the open
ones with *add to the open files* (a `filenumber` column tells them apart).
The text says how many localizations were read, and in which format.  The
load is written into the history with
its settings, after the entries an HDF5 file brought with it.

## Differences from SMAP

Based on SMAP's `Loader_sml` and `Loader_minflux_json`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)).  The main
changes:

* From a SMAP file only the localizations and the camera numbers are read.
  SMAP also restores the file's history, the ROI manager's sites and the
  GUI parameters; those describe SMAP's own windows and plugins.
* Frames count from 0, and the columns are renamed to this program's names.
* A MINFLUX `.npy` or `.zip` export is read, the MATLAB `.mat` export is
  not; only the last iteration and only valid localizations are read, and
  the positions are not moved to start at zero.  The precision is
  $150\,\mathrm{nm}/\sqrt{N}$ for every export, where SMAP works it out from
  the beam pattern for a `.mat` export.

## References

* Balzarotti F, Eilers Y, Gwosch KC, et al. Nanometer resolution imaging
  and tracking of fluorescent molecules with minimal photon fluxes.
  *Science* 355, 606 (2017).
  [doi:10.1126/science.aak9913](https://doi.org/10.1126/science.aak9913)
  -- MINFLUX.
* Gwosch KC, Pape JK, Balzarotti F, et al. MINFLUX nanoscopy delivers 3D
  multicolor nanometer resolution in cells. *Nat Methods* 17, 217 (2020).
  [doi:10.1038/s41592-019-0688-0](https://doi.org/10.1038/s41592-019-0688-0)
  -- the iterative scheme whose records the export holds.
* Ovesný M, Křížek P, Borkovec J, Svindrych Z, Hagen GM. ThunderSTORM: a
  comprehensive ImageJ plug-in for PALM and STORM data analysis and
  super-resolution imaging. *Bioinformatics* 30, 2389 (2014).
  [doi:10.1093/bioinformatics/btu202](https://doi.org/10.1093/bioinformatics/btu202)
  -- the program whose export is read without a mapping.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
  -- the program whose files this reads.
