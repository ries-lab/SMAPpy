---
version: "1"
covers: [smappy.plugins.file.ExportLocalizations, smappy.plugins.file.export_format, smappy.io.formats._save_csv, smappy.io.picasso.records, smappy.io.picasso.save, smappy.io.picasso._linked, smappy.io.picasso.metadata_list, smappy.io.picasso.steps, smappy.group.last_frames, smappy.plugins.file.picasso_groups, smappy.gui.params.ColumnsField]
---

## What it does

Writes the localizations to a file for another program: a text table
(csv), the same table with the column names ThunderSTORM uses
([Ovesný et al. 2014](https://doi.org/10.1093/bioinformatics/btu202)),
or Picasso's HDF5 file
([Schnitzbauer et al. 2017](https://doi.org/10.1038/nprot.2017.024)).

It is not *Save*.  Save writes everything this program knows about the
data -- the table, its history, the ROIs and their results, the GUI -- in
this program's own file, to be opened here again.  Export writes the table
alone, in a form another program reads, and only the part of it asked for:

* *localizations*: the ones the [layer](plugin:Panels/Glossary) shows --
  inside its [filter](plugin:Panels/Glossary) and the drawn
  [ROI](plugin:Panels/Glossary) -- or all of them;
* *table*: as the layer shows it, the localizations, or the merged blinks
  ([merge blinks](plugin:Panels/Glossary): the localizations of one
  blinking event in consecutive frames combined into one);
* *attributes*: for a text table, the attributes to write, ticked in a list.
  Picasso's file has a fixed set of columns, so for it the list is hidden.

## How it works

**1. The table.**  The layer's table, or the localizations or merged blinks
whatever the layer shows, cut to what the layer shows when *localizations*
is *visible*.

**2. The format.**  Chosen, or taken from the file's name: `.csv`, `.tsv`
or `.txt` is a text table, `.hdf5` or `.h5` Picasso's file.

**3. The file.**  A text table has one header line and one row per
localization, separated by commas in a `.csv` and by tabs in a `.tsv` or
`.txt`.  Picasso's file is written as Picasso writes it, with its `.yaml`
beside it.

## In detail

**Text.**  Whole numbers (`frame`, `channel`, ids) are written as whole
numbers, everything else with 9 significant digits -- enough to give back
a single-precision value exactly.  With smappy's names, the attributes are
written under their names here and frames count from 0, so
[Load from path](plugin:File/Load/Auto) reads the file back as it was.  With
ThunderSTORM's names the usual attributes are renamed and frames count from 1,
as ThunderSTORM's own export does; the attributes ThunderSTORM has no name for
keep this program's:

| here | ThunderSTORM |
| --- | --- |
| `x_nm`, `y_nm`, `z_nm` | `x [nm]`, `y [nm]`, `z [nm]` |
| `photons` | `intensity [photon]` |
| `background` | `offset [photon]` |
| `sigma_nm` | `sigma [nm]` |
| `xy_err_nm`, `z_err_nm` | `uncertainty_xy [nm]`, `uncertainty_z [nm]` |
| `frame` | `frame`, plus 1 |

**Picasso.**  Picasso's file is one dataset of records, `/locs`, with
positions, widths and precisions in camera pixels -- so the table must know
its pixel size, which a fitted one does -- and z in nanometres:

| Picasso | from |
| --- | --- |
| `frame` | `frame` |
| `x`, `y` | `x_nm`, `y_nm` over the pixel size |
| `photons`, `bg` | `photons`, `background` |
| `sx`, `sy` | `sigma_nm` (and `sigma_y_nm`) over the pixel size |
| `lpx`, `lpy` | `x_err_nm`, `y_err_nm`, or `xy_err_nm`, over the pixel size |
| `ellipticity` | $(s_{\max} - s_{\min}) / s_{\max}$ of `sx` and `sy`, as Picasso works it out |
| `net_gradient` | 0: a number of Picasso's own spot detection, which this program does not have |
| `z`, `lpz` | `z_nm`, `z_err_nm`, for a 3D table |
| `group` | the *Picasso group* attribute, if one is chosen |
| `len`, `n`, `photon_rate` | for merged blinks only: see below |

A field the table has no attribute for is written as 0, and the text says
which.  The metadata Picasso needs -- `Width`, `Height` and `Frames` --
and the `Pixelsize` go into the file (`/metadata`, as Picasso 0.9 and later
embed it) and into a `.yaml` beside it (for older versions).  The width
and height are the camera region the fit read where the table records it,
and otherwise as many pixels as the positions need -- but never less than
the file the table was read from, if that was Picasso's.  Picasso throws away,
without saying so, every localization with a value missing, a negative
position, width or precision, or a position outside the image; the export
leaves those out itself and says how many.

**Merged blinks** are written as Picasso writes a *linked* table, in which
each row is one binding event: `n` is the number of localizations merged
into it (`n_in_group`), `len` the frames from its first to its last, dark
frames bridged by the merge included -- three localizations in frames 10,
11 and 13 are `n` 3 and `len` 4 -- and `photon_rate` its photons over `n`.
The merged table keeps only where a blink starts, so its last frame is read
from the localizations it was merged from.  The localizations themselves
are written without these three fields, as Picasso's unlinked files are.  A
table read from a linked Picasso file keeps the `len` and `n` it came with.

**The metadata.**  Picasso's metadata is a list, one entry for each step
that made the file, and Picasso reads a value from the last entry that
has it.  A table read from a Picasso file keeps that list, and the export
appends one entry of its own -- *smappy* and its version, the image size
and pixel size, and the steps of the session's history that changed the
table, with their settings -- as Picasso's own steps do.  So a file that
goes from Picasso to here and back still says what Picasso did to it,
followed by what was done here.

## Parameters

### columns
Untick what the other program does not need: a table of x, y and frame is
a third of the size of the whole one.  With every box ticked, every attribute
the table has when it runs is written -- also one a later plugin adds --
which is what a chain or a batch wants.

### group
Picasso shows its *picks* by this attribute.  Writing `cluster_id` there, for
example, makes every cluster a pick in Picasso; leave it on *none* for a
plain table.  Every id attribute is offered -- any whose name ends in `_id`,
such as `cluster_id`, `molecule_id` or a Picasso file's own `pick_id` --
and any other whole-numbered one.  On merged blinks an id is a number per
blink, written as a whole number; a blink with none is written as -1.

## Output

The file, and a line saying how many localizations went where, in which
format; for Picasso, also what was left out or written as 0.  None of these
formats carries the calibration a fit kept with its localizations: smappy's
own HDF5 file does.  The export is
written into the history, since where a file went is not visible in the
file.

## Differences from SMAP

Based on SMAP's `export_coordinates` and `export_picasso_hdf5`
([Ries 2020](https://doi.org/10.1038/s41592-020-0938-1)), one plugin here.
The main changes:

* No `.xls` or `.mat` output; a text table is read by everything that read
  those.
* The localizations visible means the layer's filter and the drawn ROI, and
  the table is the layer's own, the localizations or the merged blinks,
  chosen here rather than by a separate *grouped* tick.
* SMAP's Picasso file has the positions and precisions only (`frame`,
  `x`, `y`, `lpx`, `lpy` both from the lateral precision, `z`, `lpz`); here
  it also has the photons, the background and the PSF widths, so Picasso
  can filter and render by them.  The metadata goes into the file as well as
  into the `.yaml`, which recent Picasso versions read, and the rows Picasso
  would drop are left out and counted rather than written.

## References

* Ovesný M, Křížek P, Borkovec J, Svindrych Z, Hagen GM. ThunderSTORM: a
  comprehensive ImageJ plug-in for PALM and STORM data analysis and
  super-resolution imaging. *Bioinformatics* 30, 2389 (2014).
  [doi:10.1093/bioinformatics/btu202](https://doi.org/10.1093/bioinformatics/btu202)
  -- the column names of the ThunderSTORM text table.
* Ries J. SMAP: a modular super-resolution microscopy analysis platform for
  SMLM data. *Nat Methods* 17, 870 (2020).
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
* Schnitzbauer J, Strauss MT, Schlichthaerle T, Schueder F, Jungmann R.
  Super-resolution microscopy with DNA-PAINT. *Nat Protoc* 12, 1198 (2017).
  [doi:10.1038/nprot.2017.024](https://doi.org/10.1038/nprot.2017.024)
  -- Picasso, whose file this writes.
