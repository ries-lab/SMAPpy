# An Abberior `.msr` / MINFLUX reader: plan for discussion

Status: a plan, nothing implemented.  Written 2026-10-16 from (a) a read of
how SMAPpy loads files today and (b) a read of the format documentation and
the open readers, checked by decoding the m2205 `.msr` files of
etMINFLUX-analysis-public end to end.  The research scripts are in
`studies/minflux_msr/`.  Open questions for the discussion are collected at
the end (**Decisions**).

## Goal

*File > Open* on an `.msr` opens a window that lists what the file holds and
lets the user tick what to import:

* each **MINFLUX measurement** -> a file of its own in the session
  (`Result.files`), with its sequence and acquisition metadata;
* each **confocal / STED image** -> an image layer, placed in nm in the same
  frame as the localizations (optionally written out as a TIFF);
* **MBM** (beads) of a measurement -> kept with that file, for a later MBM
  drift plugin;
* **metadata** (Imspector's parameter tree, the MINFLUX sequence) -> kept with
  the file and saved with it.

The same reading is available headless (`smappy-batch`, scripts):
`smappy.io.msr.contents(path)` and `load(path, parts=...)`.  No specpy, no
zarr-python, no new hard dependency.

## What the format is (confirmed on m2205 files)

**OBF container** (spec: AbberiorInstruments/ImspectorDocs
`docs/fileformat.rst`, MIT).  Little-endian.  File header
`"OMAS_BF\n\xff\xff"`, `u32 version`, `u64 first_stack_pos`, description,
and from v2 a `u64` pointer to a tag dictionary (`ome_xml`).  A linked list of
stacks: header `"OMAS_BF_STACK\n\xff\xff"`, `u32 version` (7 seen), rank,
`res[15]`, `len[15]` (m), `off[15]` (m), data type, compression (0 / zlib),
name, description, `data_len_disk`, `next_stack_pos`; the data; then a
footer (size-prefixed, so unknown newer fields can be skipped: v7's is 1528
bytes) with SI units per axis, axis labels, column positions/labels, flush
points, a **tag dictionary** (`imspector`: Imspector's parameter XML, ~0.5 MB
per image; `minflux`: a small JSON) and, from v6, chunk positions.  Images
are `reshape(res[::-1])` -> (z,) y, x.

**A MINFLUX measurement is one stack**: rank 4, `res=(N,1,1,1)`, uint8,
zlib, empty name, description = the dataset label; its `minflux` tag is
`{"did", "sid", "label", "type": "data"}`.  Sibling stacks
`MF(<label>)/density/loc` (`"type": "density"`) are Imspector's renders;
image stacks (`Ch1 {n}`) have an empty `minflux` tag.

Inflated, the stack is an **`MFXDTA` archive**: a 147-byte header (`MFXDTA`,
`u32 2`, `u32` unix time = `floor(tms)`), then entries `u64 len, path, u8
kind` (1 = directory; 0 = file, followed by `u64 size` and the bytes),
ending at `u64 0`.  The files are a complete **zarr v2** tree:
`zarr\.zattrs` (the ROI), `zarr\mfx\.zarray|.zattrs|0..k`,
`zarr\grd\mbm\...` (empty in the samples), `sync\...` (placeholders).

**`mfx` chunks** are 10000 rows, **blosc1 + LZ4**, shuffle on but
`typesize = 1` (the structured itemsize, 1030, exceeds 255), the last chunk
padded.  MBM arrays may have `typesize > 1` and need the byte unshuffle.
specpy's own binaries contain the same strings (`MFXDTA`, `.zattrs`,
`unpack`): the official route also unpacks this archive to a zarr tree.

**Not confirmed** -- no file available: what an **m2410** Imspector writes
into an `.msr` (presumably the flat dtype below and a filled `grd/mbm`),
and what the `.zarr` / `.zarr.zip` export looks like beside it.  Your files
decide this.

### The two record layouts

* **m2205, nested** (`.npy`, `.msr`): one row per localization attempt,
  `itr[n_itr]` a sub-record per iteration -- `itr, tic, loc[3], lnc[3], eco,
  ecc, efo, efc, sta, cfr, dcr, ext[3], ..., fbg` -- and per row `sqi, gri,
  tim, tid, vld, act, dos, sky`.  `n_itr` 5 (2D) / 10 (3D), more with extra
  iterations.
* **m2410, flat** (one row per iteration): `vld, fnl, bot, eot, sta, tim,
  tid, gri, thi, sqi, itr, loc[3], lnc[3], eco, ecc, efo, efc, fbg, cfr,
  dcr[2]`.  Told apart by `fnl`/`bot`/`eot` or a scalar `itr`.

The localization is the **last iteration** (`fnl` in m2410); `cfr` comes
from the last iteration where it varies (pyMINFLUX: the last with non-zero
std); `loc` is drift-corrected by MBM, `lnc` is not.  `dcr` in m2410 has two
entries, ch1/(ch1+ch2) and ch2/(ch1+ch2).  `sta` is the abort reason.

### Sequence metadata

m2205: `zarr\mfx\.zattrs` -> `seq` with `Itr[]` (`Mode{dim, pattern, ...},
patGeoFactor, patDwellTime, patRepeat, phtLimit, ccrLimit, bgcThreshold,
pwrFactor, wavelength, ...`), plus `fld`, `ori`, `tms`.  m2410:
`measurement.threads[0].sequences[0].Itr[i]` with `_excitation{laser,
wavelength, power}`, `_detection.channels` (from PYME-extra, GPL: schema
only).  Laser powers and detectors of images: the `imspector` tag XML.

### Sources and what may be ported

| source | licence | use |
| --- | --- | --- |
| ImspectorDocs `fileformat.rst` | MIT | the OBF spec |
| `jkfindeisen/python-mix` `imspector/obf/obf_support.py` | MIT | a complete OBF reader incl. chunks: port |
| `bsse-scf/pyMINFLUX` `reader/_msr_reader.py`, `_reader_v2.py`, `util/_util.py` | Apache-2.0 | footer, last/valid iteration, cfr choice, dcr pooling, MBM: port with attribution |
| `hoerlteam/msr-reader` | MIT | second OBF reader to compare |
| Bio-Formats `formats-bsd/.../OBFReader.java` | BSD-2 | chunked/compressed reading |
| PYME, PYME-extra, MBM-inspector | GPL-3 | read for the attrs schema and MBM; reimplement, do not copy |
| etMINFLUX-analysis-public | none | test files locally only |
| specpy | proprietary | optional cross-check in a skipped test |

Not reached (private or renamed): minflux-viewer, MFX_data_analysis.

## Design

### Modules (`src/smappy/io/`)

* **`obf.py`** -- the OBF container, nothing MINFLUX-specific:
  `stacks(path) -> list[StackInfo]` (name, description, shape, dtype,
  pixel sizes and offsets in nm from `len/res`, `off` and the SI units, axis
  labels, tags, file offsets) read from the headers and footers only, and
  `read_stack(path, info) -> ndarray` (zlib, chunked).  Opening a 2 GB file
  to list it must not read the data.
* **`blosc.py`** + **`csrc/blosc.cpp`** -- blosc1 frame decoding with LZ4
  (and the raw/memcpy case, and zlib for completeness), and the byte
  unshuffle.  The decoder loop in C++ (~150 lines; LZ4's block format is
  short and public) because pure Python is ~1 s per 2.4 MB; a pure-Python
  fallback kept for tests.  See **Decisions 1**.
* **`zarr_lite.py`** -- just enough zarr v2 to read an array from a mapping
  of path -> bytes: `.zarray` (shape, chunks, dtype incl. structured dtypes
  as zarr writes them, `fill_value`, order C, compressor blosc / zlib /
  none), `.zattrs`, groups.  The same code reads a `.zarr` directory, a
  `.zarr.zip` and an `MFXDTA` archive, because all three are such a mapping.
* **`minflux.py`** -- moves the MINFLUX code out of `formats.py`: one
  function from either record layout (from `.npy`, `.mat`, `.json`, zarr,
  msr) to a `Localizations` table plus metadata, and `sequence(attrs)` that
  normalises both attrs schemas to one per-iteration table.
* **`msr.py`** -- ties them together: `contents(path) -> MsrContents`
  (per MINFLUX dataset: label, did, row count, valid count, 2D/3D, date,
  has-MBM; per image: name, shape, pixel size, channel/detector; density
  renders flagged; file metadata) and `load(path, parts, options) ->
  MsrImport` (tables with `FileInfo`s, `ImageData`s, MBM, metadata).
  Listing a MINFLUX dataset reads its `.zarray` and `.zattrs` only.

### The table

Per localization (the last iteration of each attempt):

* `x_nm, y_nm, z_nm` from `loc` (z times the *z scale*, recorded in the
  metadata); `x_nc_nm, y_nc_nm, z_nc_nm` from `lnc`;
* `time_s` (`tim`), `tid`, `frame` = rank in time (as now);
* `photons` = `eco`, `efo`, `cfr` (from its iteration), `dcr` (and `dcr2`
  in m2410), `fbg`, `ecc`, `efc`, `iteration`, `sta` (abort reason), `gri`,
  `thi`, `sqi` where present;
* `vld` -- see **Decisions 2**;
* `xy_err_nm` stays the 150/sqrt(N) proxy for now; a trace-based precision
  comes with trace grouping (later step).

### Metadata

`FileInfo.metadata` holds the sequence (per iteration), the acquisition
(date, Imspector version, did/sid, label, ROI, field and origin, dimension),
the *z scale*, the counts (rows, valid, final) and the MBM summary.  **This
needs a fix first**: `FileInfo.to_dict()` drops `metadata`
(`formats.py:39`), so it is neither saved nor restored by the undo
(`session.py:556, 606, 1179`).  It must become JSON-able and round-trip
through the `.h5`.

### MBM

Bead tracks (`grd/mbm/points`: `gri, xyz, tim, str` and the `used` list)
are kept as a small side table of the file -- saved in the `.h5` beside
the localizations -- not as rows of the localization table.  `Drift` is
indexed by frame, and MINFLUX's `frame` is a rank in time, so the MBM
drift plugin (step 2) resamples onto `time_s`.  For the reader it is enough
to read and keep the beads.

### Images

Each OBF image stack -> `ImageData` with `pixelsize` from `len/res`, `x0, y0`
from `off` (m -> nm), `metadata` from the `imspector` tag (laser, detector,
dwell).  Whether the confocal's `off` and MINFLUX's `loc` share an origin
must be checked on a file where a structure is visible in both -- **your
data**.  Density renders are listed but not ticked by default.

### The GUI

* `.msr` in `LOCS_FILTER` (`plugins/file.py:33`) and a `Reader` with a
  `needs=("parts",)`, the way csv has `needs=("mapping",)`.
* `gui/dialogs.py`: **`MsrImportDialog`**, modal like `CsvMappingDialog`,
  opened from `app.load_paths` before the load: a tree with check boxes --
  *MINFLUX* (one row per measurement: label, 2D/3D, n valid, duration, MBM
  yes/no), *Images* (name, size, pixel size), *Density renders*, and a
  metadata pane showing the selected item's sequence / parameters.  Options
  under it: *valid only*, *z scale*, *with MBM*, *images as TIFF too*.
* The load itself goes through a **plugin** -- `File/Load/MSR`, settings
  path + parts + options -- returning `Result(files=[...],
  image_layers=[...])`, so it is undoable, logged and usable in a chain and
  in `smappy-batch` (parts by a label pattern, e.g. *all*).  The GUI's
  plain open path (`LoadTask`) carries one table and bypasses
  `Session.apply`; the dialog hands its selection to this plugin instead.
  Known snag: an image layer added in the same result is not undone with
  the file -- fix or accept.
* Drag and drop of an `.msr` opens the same dialog.

### Tests and data

* A **test writer** for OBF + MFXDTA + zarr in the tests (blosc frames with
  raw streams and a hand-made LZ4 block): small synthetic `.msr` files with
  known localizations, both record layouts, an image with known placement
  and a bead table.  Committed: no external data needed for the fast tier.
* LZ4 / blosc against reference vectors (the pure-Python and C++ decoders
  agree, and both against vectors made by `lz4`/`blosc` where installed).
* Real data, skipped without it: `SMAPPY_TEST_MSR=dir` -- each `.msr`
  against its `.npy` export where one is beside it (etMINFLUX's examples;
  yours); `SMAPPY_TEST_MINFLUX=dir` for pyMINFLUX's `reader_v2.zip` and
  `reader_comparison.zip` (m2205 vs m2410 of the same measurements give the
  same localizations); specpy's `unpack` where specpy is installed.
* Docs: the loader's page (`File/Load/MSR.md`), `Auto.md` and `Load.md`
  updated, the MINFLUX columns into `docs/plugin-reference.md`, a line in
  CLAUDE.md's plugin-shape table.

## Order of work

1. `FileInfo.metadata` saved and restored (small, independent).
2. `obf.py` + tests on synthetic files; list your `.msr` files with it.
3. `blosc.py` / `csrc/blosc.cpp`, `zarr_lite.py`, the MFXDTA archive.
4. `minflux.py`: both layouts, all sources, the table and metadata; the
   comparisons against `.npy` and pyMINFLUX's data.
5. `msr.py`, `File/Load/MSR`, the dialog, images and MBM kept.
6. Pages and docs; `--slow` run.

Then, as separate pieces: MBM drift on `time_s` (with bead QC), trace
grouping and per-trace columns and filters (robust CFR / EFO thresholds),
DCR channels, MBM registration of two acquisitions, precision and FRC,
sequence QC across acquisitions.

## Decisions

1. **blosc/LZ4**: our own decoder in `csrc/` (no dependency; recommended),
   or an optional `blosc`/`numcodecs` with the pure-Python fallback?
2. **Invalid attempts**: drop them (as now), or keep them with a `vld`
   column and a default filter `vld == 1`, so the abort reasons can be
   studied?  Keeping roughly doubles the table.
3. **One file per measurement**, always -- or offer to merge the ticked
   measurements into one file (a `measurement` column)?
4. **Images**: image layers only, or also write TIFFs next to the `.msr`
   (and where)?  Does the confocal's origin match MINFLUX's in your data?
5. **Sequence parameters**: metadata only (proposed), or also joined onto
   every row as columns (`patGeoFactor`, `phtLimit`, ...) to compare across
   acquisitions in one table?
6. **z scale** default: 1.0 (Imspector's numbers, pyMINFLUX's reader), 0.7
   (pyMINFLUX's GUI) or 0.72 (PYME-extra)?
7. **Which Imspector versions** are in your folder -- any m2410 `.msr`
   with MBM, any `.zarr` exports?  That decides how much of the m2410 path
   can be tested now.
8. **The existing MINFLUX reader** takes every `.npy`/`.zip`/`.json`
   without sniffing; add a sniff (the dtype) while moving it?
