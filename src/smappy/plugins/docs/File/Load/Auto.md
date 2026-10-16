---
version: "6"
covers: [smappy.io.formats.reader_for, smappy.io.formats._is_smappy, smappy.io.formats._has_saveloc, smappy.session.read_localizations, smappy.io.formats._load_smappy, smappy.io.hdf5.load_localizations, smappy.io.formats._load_sml, smappy.io.formats._load_minflux, smappy.io.minflux.records, smappy.io.minflux.from_json, smappy.io.minflux.table, smappy.io.minflux._fields, smappy.io.minflux._cfr_iteration, smappy.io.minflux._joined, smappy.io.minflux.crlb_xy, smappy.io.minflux.crlb_z, smappy.io.minflux.photon_split, smappy.io.minflux.pattern_sizes, smappy.io.minflux._bound, smappy.io.minflux.sequence, smappy.io.msr.contents, smappy.io.msr.load, smappy.io.msr.read_files, smappy.io.msr.read_measurement, smappy.io.msr.select, smappy.io.msr.merge, smappy.io.msr.image, smappy.io.obf.stacks, smappy.io.obf.read_stack, smappy.io.zarr_lite.archive_files, smappy.io.zarr_lite.read_array, smappy.io.blosc.decode, smappy.io.blosc.lz4_block, smappy.session._saved_files, smappy.group.by_trace, smappy.io.formats._load_csv, smappy.io.formats.guess_csv_mapping, smappy.io.formats.csv_columns, smappy.io.formats.column_name, smappy.io.conversions.conversion_for, smappy.io.picasso.load, smappy.io.picasso.read_info, smappy.io.decode.load]
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
  ([Gwosch et al. 2020](https://doi.org/10.1038/s41592-019-0688-0)), from
  either version of Imspector (m2205 and m2410);
* **Abberior MSR** -- `.msr`: Imspector's own file, which holds a whole
  session -- several MINFLUX measurements, the confocal or STED images they
  were aimed on, the beads that stabilised them and every acquisition
  setting.  File > Open lists what is in it and asks which to open; each
  measurement becomes a file of its own (or all of them one, with
  *merge into one file*), and its images are read, in the same
  coordinates as the localizations, for an image layer to show;
* **Picasso** -- `.hdf5`, `.h5` written by Picasso
  ([Schnitzbauer et al. 2017](https://doi.org/10.1038/nprot.2017.024)),
  told from this program's own HDF5 by what is inside;
* **DECODE** -- `.h5` written by DECODE
  ([Speiser et al. 2021](https://doi.org/10.1038/s41592-021-01236-x)), a
  deep-learning localization program;
* **csv** -- `.csv`, `.txt`, `.tsv`: a table written as text, every column
  of it, with the meaning of the usual ones guessed from the header --
  ThunderSTORM's export
  ([Ovesný et al. 2014](https://doi.org/10.1093/bioinformatics/btu202)) and
  SMAP's names are recognised without being told -- or taken from a saved
  *conversion*.

Every reader keeps every column of the file: the ones this program has a
name for get that name, and the rest keep their own, so whatever the other
program wrote can still be filtered on, coloured by or exported again.

A file whose name does not say what it is -- a SMAP file not called
`_sml.mat` -- is read by choosing its *format* by hand (under *more*).

The file's layer shows its blinks merged, the same whichever reader was
used: [merge blinks](plugin:Panels/Glossary) (also called grouping or
linking) combines the localizations in consecutive frames that come from one
blinking event into one localization.  The file does not wait for that to
open: a large one opens at once and is merged in the background.

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
    with h5py.File(folder / "picasso.hdf5", "w") as f:
        f.create_dataset("locs", data=np.zeros(3, dtype=[("frame", "u4"), ("x", "f4"),
                                                         ("y", "f4")]))
    with h5py.File(folder / "decode.h5", "w") as f:
        f.create_group("meta"); f.create_group("data").create_dataset("xyz", data=np.zeros((1, 3)))
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
    (folder / "imspector.msr").write_bytes(b"")
    for name in ("fitted.hdf5", "picasso.hdf5", "decode.h5", "other_program.h5", "cells_sml.mat", "renamed.mat",
                 "calibration.mat", "minflux.npy", "minflux.json", "imspector.msr",
                 "thunderstorm.csv", "stack.tif"):
        path = folder / name
        try:
            chosen.append((path.name, reader_for(path).name))
        except ValueError:
            chosen.append((path.name, None))
```

**1. By the name.**  The readers whose endings match the file's name are
the candidates, in the order smappy HDF5, SMAP, MINFLUX, Abberior MSR, csv, Picasso,
DECODE.  Upper and lower case do
not matter.

**2. By the contents**, where one ending could be more than one thing:

* an `.hdf5` or `.h5` file is read as this program's if it has a `/locs`
  group, as Picasso's if `/locs` is a single dataset, and as DECODE's if it
  has a `/data` group with `xyz` and a `/meta`;
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
once and is merged in the background, the picture and the status bar saying
so; the blinks are drawn when they are there, and a plugin run started
meanwhile waits for them.  Headless -- in a script, a chain or a batch -- the blinks are merged
when something first reads them.  An added file is merged together with the
files already open.  A table with a trace id (`tid`, as a MINFLUX export
has) is merged by trace instead, one row per trace.  To look at every
localization, untick *merge blinks* on the layer: the single localizations
are always kept.

```figure Which reader Auto chose for a set of files: by the ending, and by what is inside where the ending is ambiguous.  Grey: refused.
fig.set_size_inches(5.2, 3.6)
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
as it was saved: a fit writes nanometres and the pixel size, and a file from
an earlier version that also has camera-pixel columns (`x_pix`, ...) keeps
them.  The GUI state that may be in the file (`/gui`)
is left alone: File > Workspace > From data file puts it back on request.
An HDF5 file from Picasso or DECODE is read by its own reader, below.

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
takes no camera frames: it homes in on a molecule in a few iterations, each
with a smaller beam pattern, and records every one.  Imspector has written
the records in two ways.  Up to version m2205 a row is one attempt at a
localization, with its iterations inside it; from m2410 a row is one
iteration, and the last iteration of a localization is marked final
(`fnl`).  Either way, only the final iteration -- the most precise -- is
read, and only where the instrument marked it valid (`vld`) and it has a
position.  Positions go from metres to nanometres and are not shifted to
start at zero; z is multiplied by the *z scale*.  There are no frames, so
the localizations are sorted by their time (`tim`, kept as `time_s`) and
`frame` is the rank in that order; the trace -- the localizations of one
molecule while it was on -- is `tid`.

The final iteration's `efo`, `efc`, `ecc`, `fbg`, the iteration number
(`iteration`), why the iteration ended (`sta`), the grid point, thread and
sequence (`gri`, `thi`, `sqi`) come across where the records have them.
`dcr`, the fraction of photons on the first detector, is m2410's first of two
ratios; the second, the fraction on the second detector, is `dcr2`.  `cfr`,
the centre-frequency ratio that tells a single molecule from two, is not
measured in the final iteration: Imspector measures it in one earlier
iteration -- found as the last one whose `cfr` varies -- and writes 0 into
the others.  A localization's `cfr` is that iteration's *of the same
localization*.  How often it is measured depends on the sequence: before
every localization in the 3D sequences looked at, once at the start of a
trace in the 2D ones -- so in a 2D measurement only the first localization
of each trace has a `cfr`, and the others have none (NaN, not a number).
They are not given their trace's: a value they were not measured with would
read as if they had been.  Statistics, histograms and filters leave NaN out,
so a mean of `cfr` is the mean over the localizations that have one.  The
position Imspector corrected with its beam monitoring (MBM, beads followed
during the acquisition) is `x_nm`, `y_nm`, `z_nm`; where the uncorrected one
differs it is kept beside it as `x_nc_nm`, `y_nc_nm`, `z_nc_nm`.

The photons are the final iteration's `eco`.  The records give no
[localization precision](plugin:Panels/Glossary) (also called the CRLB,
the Cramér-Rao lower bound: the best precision any estimator can reach with
these photons), so `xy_err_nm` is that bound worked out from the photons,
as a camera fitter's is.  It is not the scatter of the localizations, which
the precision plugins measure.  MINFLUX probes a molecule with the zero of a
doughnut placed at K points on a circle of diameter L (the pattern size,
`patGeoFactor` × 360 nm in the sequence) and counts the photons there:
$N_s$ from the molecule and $N_b$ from background.  Near the zero the
intensity grows as the square of the distance, and for a molecule at the
centre of the pattern the bound per axis is

$$\sigma_{xy} = \frac{L}{2\sqrt{2N_s}} \sqrt{1 + \frac{N_b}{N_s}} .$$

Without background it is the bound of Balzarotti et al. 2017; with it, the
2D form of [Marin & Ries 2024](https://doi.org/10.48550/arXiv.2410.12427)'s
equation for out-of-focus background.  It does not depend on K -- a
triangle, a square or a hexagon -- nor on an exposure at the centre, which
adds no information there.  A molecule away from the centre is localized
worse than this, and the records do not say where it was.  L is the
sequence's last lateral iteration's: 101 nm for the common 2D sequence
(`patGeoFactor` 0.28), where 25 photons without background give 7 nm.

The background is Imspector's: `eco` is split into $N_s$ and $N_b$ in the
ratio `fbg`/`efo`, its background rate over its photon rate.  This assumes
that `efo` includes the background and that `fbg` is a rate at the same
positions in the same units -- Abberior does not document either.  When
Imspector's background sensing is off, `fbg` is 0 and so is $N_b$; that was
so in every file looked at.  At most 90 % of the photons are taken as
background.

In 3D, z gets the bound of a line of two offsets,
$L_z/(4\sqrt{N_s})\sqrt{1 + N_b/N_s}$ with the last axial (`zline`)
iteration's L, times the *z scale*, as `z_err_nm`.  A 3D sequence
alternates lateral and axial iterations, so `xy_err_nm` uses the photons of
the localization's own last lateral iteration, not of the final one (which
`photons` shows).  An export without its sequence (`.npy`, `.json`, `.zip`)
gets L = 100 nm and no `z_err_nm`.  `sigma_nm` is 150 nm for all.  Merging blinks goes **by trace**:
one row per `tid` (its mean position, summed photons, first `frame`, and
`n_in_group`), rather than by distance and frame.  The `.npy`, the `.json`
and a `.zip` holding either are read; MINFLUX data saved as a MATLAB `.mat`
is not.

**Abberior MSR.**  An `.msr` is a container of *stacks*: images, and a
stack of bytes per MINFLUX measurement, which holds the measurement's
records and settings as Imspector's export to *ZARR* writes them.  Listing
the file reads the stacks' headers and unpacks each measurement's settings,
a tenth of a second for a file of ten; the records are decoded only for the
measurements opened.  A measurement is read as above, and keeps with it,
in the file's metadata: its label, date, version and dimension, the counts,
its **sequence** (every iteration's parameters: pattern, laser and power,
photon limit, dwell time, ...), the lasers and detectors, the region it was
taken in, and its **MBM beads** -- which were used, and their positions as
recorded.  The measurements chosen are a file each; with *merge into one
file* they are one table, one after another in time: each one's `time_s`
is moved by when it started after the first one did (to the second, from
its date), its traces are numbered after the last one's so that every `tid`
is one trace, and a `measurement` column says which it was.  **Images** --
confocal and STED channels -- are read and kept in the session's list of
images, placed in nanometres by the stack's own offset and pixel size, in
the frame the localizations are in: no registration is needed to lay one on
the other.  They get no layer of their own -- an `.msr` holds a dozen -- so
add an image layer and choose the image in it.  Like any image, one is saved
with the session only once a layer shows it.  What Imspector drew
of a measurement (its density and trace renders) is listed, and not opened
unless ticked: the localizations draw it again.  Stacks Imspector writes in
chunks, for very large acquisitions, are not read yet, and say so.

**Picasso.**  Picasso keeps its table as one dataset of records, `/locs`,
where this program keeps a column per dataset; that is how the two are told
apart.  Positions, widths and precisions (`x`, `y`, `sx`, `sy`, `lpx`,
`lpy`) are in camera pixels and become nanometres with the file's
`Pixelsize`; `z` and `lpz` are in nanometres already:

| Picasso | here | | Picasso | here |
| --- | --- | --- | --- | --- |
| `x`, `y` | `x_nm`, `y_nm` | | `bg` | `background` |
| `lpx`, `lpy` | `x_err_nm`, `y_err_nm` | | `z`, `lpz` | `z_nm`, `z_err_nm` |
| `sx`, `sy` | `sigma_nm`, `sigma_y_nm` | | `group` | `pick_id` |

`xy_err_nm` is worked out from `x_err_nm` and `y_err_nm`, and the pixel
columns are kept beside the nanometre ones.  `group` is Picasso's *pick*,
read as `pick_id` so it is not taken for merged blinks (`group_id`); like
any id column it can be measured per region.  Everything else
(`photons`, `frame`, `net_gradient`, `ellipticity`, a linked table's `len`
and `n`, ...) keeps its name; frames count from 0 in both programs.  The
metadata is read from the `/metadata` dataset recent Picasso versions
embed, or else from the `.yaml` file beside the `.hdf5`.  Without a pixel
size there, the default one is assumed and the table says so.

**DECODE.**  DECODE writes `/data`: `xyz`, `phot`, `bg`, `frame_ix`,
`prob`, `id`, and where it predicted them the uncertainties `xyz_sig`,
`phot_sig`, `bg_sig` and the Cramér-Rao bounds `xyz_cr`, `phot_cr`,
`bg_cr`; `/meta` says whether x and y are in pixels or nanometres
(`xy_unit`) and the pixel size (`px_size`, x and y).  z is in nanometres
either way.  `xyz_sig` becomes `x_err_nm`, `y_err_nm`, `z_err_nm` -- the
precision DECODE reports for each localization -- and the bounds are kept
beside it as standard deviations (`x_crlb_nm`, ...; DECODE stores
variances).  `phot` is `photons`, `bg` `background`, `frame_ix` `frame`
(from 0 in both), and `prob`, the probability DECODE gives a detection, is
kept under its name: it is the column to filter DECODE's localizations by.
A quantity DECODE did not predict is an empty dataset in the file and
becomes no column.

**csv.**  The first line is the header if any of its cells is not a
number; otherwise the columns are called *column 1*, *column 2*, ...  Cells
are split by tabs, semicolons or commas, whichever the first line has most
of.  **Every column with a number in it is read.**  Those that mean
something to this program are found by their header -- cleaned to lower
case, the unit in square brackets taken off, spaces made underscores, and
looked up (`CSV_NAMES`):

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

-- and every other column keeps its header as its name, cleaned the same
way (`chi2`, `cluster_id`).  When two headers mean the same column the
first wins; x and y must be found.  `frame`, `channel` and `id` are read as
whole numbers, the rest as single precision, and a row whose x or y is
missing is left out.

**Units and the first frame.**  The lengths -- x, y, z and their errors
and widths -- are read as nanometres unless the file is said to be in
micrometres, millimetres (multiplied by $10^3$ or $10^6$) or camera pixels
(x, y and their errors and widths multiplied by the pixel size $p$,
$x_{\mathrm{nm}} = x_{\mathrm{px}}\, p$; z is left in nanometres, as the
programs that write pixels do).  Frames are made to count from 0: a file
whose first frame is 1, as ThunderSTORM's and SMAP's are, has 1 taken off.
Only the columns this program has a name for are converted; a column kept
under its own name is read as it is.

**Conversions.**  How one program's export is read -- which header is
which, the units, the pixel size, the first frame -- can be saved under a
name, as SMAP's *conversion structures* were, and is then found again by
the headers: the saved conversion whose headers the file all has, the one
with the most of them if several do.  ThunderSTORM's ("x [nm]", "y [nm]",
"frame": frames from 1) and SMAP's export ("xnm", "ynm", "frame": frames
from 1) come with the program; a conversion saved from the File menu's
dialog (*save as...*) goes to the configuration folder and is offered in
*conversion* from then on.

## Parameters

### format
Leave it on *auto*.  Choose a format for a file whose name does not say
what it is -- a SMAP file renamed, a csv saved as `.dat` -- or to make sure
the file is read as that format and refused otherwise.

### msr.measurements
The GUI asks in a dialog, with every measurement and image listed and the
selected one's sequence shown beside it; this field is for a chain or a
batch.  `2, 5` opens the second and the fifth, `150915` the one whose label
has that in it.

### msr.merge
Merge measurements of one sample taken one after another -- ten of 30 s
each, say -- to analyse them as one, keeping `measurement` to tell them
apart.  Leave it off to compare measurements, or when they were taken with
different sequences.

### msr.z_scale
The positions in z are those Imspector computed, which do not correct for
the refractive index mismatch between the immersion oil and the sample: an
axial distance in water is measured too long.  Values of 0.7 to 0.8 are used
for imaging in aqueous samples with oil objectives.  It has no effect on 2D
measurements.

### csv.mapping
Needed only when the headers are not recognised and no saved conversion
fits.  Each pair is *header in the file* = *column in the table*; the
headers not named are still read, under their own names.  Opening such a
file from the File menu asks in a dialog instead, which can also save what
was chosen as a conversion.

### csv.first_frame
Empty takes it from the conversion, or 0.  Set it when a program that
counts frames from 1 is not recognised: a frame number off by one shifts
every time-resolved analysis -- drift, merging blinks against another
channel -- by a frame.

## Output

The table, added to the session as a file of its own, or joined to the open
ones with *add to the open files* (a `filenumber` column tells them apart).
An `.msr` gives a file per measurement (or one, merged), and its images for
the image layers to choose from.  Each file keeps its own metadata -- its
camera, its acquisition,
a measurement's sequence and beads -- however many are open, and a session
of several files saved and opened again opens as those files.
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
* A csv is read by `Loader_csvAndMore`'s idea -- the mapping, the units and
  a saved conversion per program -- with two changes: every column is read,
  not only the ones mapped, and a conversion is found by the file's headers
  rather than chosen from a list each time.
* A MINFLUX `.npy` or `.zip` export is read, and Imspector's `.msr` with
  its images; the MATLAB `.mat` export is not.  Only the last iteration
  is read and, unless *keep invalid* is ticked, only valid localizations;
  `cfr` is taken from the iteration that measures it, and
  the positions are not moved to start at zero.  The precision is the
  Cramér-Rao bound from the photons, the pattern size of the sequence and
  Imspector's background estimate; SMAP gives a json export
  $150\,\mathrm{nm}/\sqrt{N}$ and takes a `.mat` export's own bound.

## References

* Balzarotti F, Eilers Y, Gwosch KC, et al. Nanometer resolution imaging
  and tracking of fluorescent molecules with minimal photon fluxes.
  *Science* 355, 606 (2017).
  [doi:10.1126/science.aak9913](https://doi.org/10.1126/science.aak9913)
  -- MINFLUX.
* Marin Z, Ries J. Approximations of MINFLUX localization precision with
  background. arXiv:2410.12427 (2024).
  [doi:10.48550/arXiv.2410.12427](https://doi.org/10.48550/arXiv.2410.12427)
  -- the bound with background photons, and its equivalence with
  Balzarotti's.
* Gwosch KC, Pape JK, Balzarotti F, et al. MINFLUX nanoscopy delivers 3D
  multicolor nanometer resolution in cells. *Nat Methods* 17, 217 (2020).
  [doi:10.1038/s41592-019-0688-0](https://doi.org/10.1038/s41592-019-0688-0)
  -- the iterative scheme whose records the export holds.
* Schnitzbauer J, Strauss MT, Schlichthaerle T, Schueder F, Jungmann R.
  Super-resolution microscopy with DNA-PAINT. *Nat Protoc* 12, 1198 (2017).
  [doi:10.1038/nprot.2017.024](https://doi.org/10.1038/nprot.2017.024)
  -- Picasso, whose files this reads.
* Speiser A, Müller LR, Hoess P, et al. Deep learning enables fast and
  dense single-molecule localization with high accuracy. *Nat Methods* 18,
  1082 (2021).
  [doi:10.1038/s41592-021-01236-x](https://doi.org/10.1038/s41592-021-01236-x)
  -- DECODE, whose files this reads.
* Ovesný M, Křížek P, Borkovec J, Svindrych Z, Hagen GM. ThunderSTORM: a
  comprehensive ImageJ plug-in for PALM and STORM data analysis and
  super-resolution imaging. *Bioinformatics* 30, 2389 (2014).
  [doi:10.1093/bioinformatics/btu202](https://doi.org/10.1093/bioinformatics/btu202)
  -- the program whose export is read without a mapping.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
  -- the program whose files this reads.
