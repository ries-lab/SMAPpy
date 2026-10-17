# MINFLUX: what exists elsewhere, and what SMAPpy could take from it

> Copied on 2026-10-17 from `docs/minflux/survey.md` on the branch
> `claude/keen-archimedes-eu88tw` (commit 1e6a15a), unchanged.  Since it was
> written, step 1 of its order of work is on main: the `.msr` / zarr reader
> (`io/msr.py`, `io/obf.py`, `io/zarr_lite.py`, `io/blosc.py`, `io/minflux.py`,
> both layouts, `lnc` and the MBM beads kept) and `xy_err_nm` as the beam
> pattern's Cramér-Rao bound.  "What SMAPpy has today" below describes the
> state before that.

A survey written in 2026-10 as the starting point for MINFLUX support in
SMAPpy: a reader for Abberior's `.msr` files, MBM (reference bead)
handling, and MINFLUX-specific analysis.  The repositories were cloned and
their code read; the facts below are from the code (`path:line` is relative
to each repository) unless marked *unverified*.  Nothing here is
implemented yet.

## What SMAPpy has today

`src/smappy/io/formats.py`, `_load_minflux`: Imspector's `.npy`, `.json`
or a `.zip` of one.  It assumes the **m2205** layout (one row per
localization, iterations nested in `itr`), takes the last iteration of the
valid rows, m -> nm, `eco` as photons, precision 150/sqrt(N).  Gaps:

* **m2410 files (Imspector >= 24.10) are not read**: their `itr` is a scalar
  `int32` column and the rows are iteration events, so `last["loc"]` fails.
* No `.msr`, `.mat`, `.zarr` or `.zarr.zip`; no MBM; `lnc` (the position
  before Imspector's drift correction) is dropped.
* `cfr` is taken from the last iteration, where it is usually 0 -- it is only
  computed at an earlier one (see "Which iteration a field comes from").
* No z-scaling (refractive-index foreshortening) factor.
* Precision: the per-trace std is the measured one; 150/sqrt(N) is a proxy.

## The repositories

| repository | what | language | licence | use for SMAPpy (BSD-3) |
| --- | --- | --- | --- | --- |
| [csoeller/PYME-extra](https://github.com/csoeller/PYME-extra) | Soeller group: PYME plugins; MINFLUX `.npy`/`.zarr.zip` reader (`PYMEcs/IO/MINFLUX.py`), MBM correction, DCR colour, origami/NPC analysis, rate analysis | Python | GPL-3.0 | read, reimplement; do not copy |
| [csoeller/MBM-inspector](https://github.com/csoeller/MBM-inspector) | Soeller group: Dash app to curate MBM beads, writes the settings `MBMcorrection` reads | Python | GPL-3.0 | as above |
| [csoeller/PYME-test-env](https://github.com/csoeller/PYME-test-env) | install env; ships the MINFLUX recipes (`etc/PYME/customrecipes/MINFLUX-3D-basic-recipe-V1.yaml`) | YAML | GPL-3.0 | the filter values |
| [bsse-scf/pyMINFLUX](https://github.com/bsse-scf/pyMINFLUX) | ETH D-BSSE: reader (m2205 + m2410: npy, mat, json, zarr), filters, DCR unmixing, FRC, Ostersehlt drift, Combiner by MBM; tested; on PyPI | Python | Apache-2.0 | port with attribution; **test data** |
| [embl-ic/minflux-viewer](https://github.com/embl-ic/minflux-viewer) | EMBL Imaging Centre: reads `.msr` **without specpy**, MBM re-correction reproducing Imspector, alignment, aggregation, DCR EM, CRLB, NPC fit, tracking; many tests | Python/Qt | MIT | the best model to port |
| [jkfindeisen/minflux-analysis](https://github.com/jkfindeisen/minflux-analysis) | Keller-Findeisen / Politi: `mfx` package (specpy -> zarr, MBM registration across washes, trace splitting), MATLAB import/filter/FRC | Python, MATLAB | MIT | port with attribution |
| [napari-storm/napari-storm](https://github.com/napari-storm/napari-storm) | napari SMLM viewer with a MINFLUX zarr reader (`minflux_v2.py`, `select_iteration` by `fnl`) | Python | BSD-3 | port freely |
| [BIIFSweden/8376_minflux](https://github.com/BIIFSweden/8376_minflux) ("Pyflux") | SciLifeLab: GUI, filters by trace length/EFO/CFR, MBM registration, DBSCAN on trace means | Python | MIT | small; reference |
| [mhke0/matFLUX](https://github.com/mhke0/matFLUX) | Kräusslich lab: 3D MINFLUX GUI (DBSCAN, bead alignment via `resources/zarr_extract.py`, precision), [ACS Nano 2025](https://doi.org/10.1021/acsnano.5c12222) | MATLAB | MIT | reference |
| [jonatanalvelid/etMINFLUX](https://github.com/jonatanalvelid/etMINFLUX) | event-triggered acquisition control through specpy (m2205 and m2410 branches) | Python | GPL-3.0 | acquisition, not analysis |
| [jonatanalvelid/etMINFLUX-analysis-public](https://github.com/jonatanalvelid/etMINFLUX-analysis-public) | analysis of the etMINFLUX paper; vendors `obf_support.py` (MIT); 29 real `.msr` + 58 `.npy` example files | Python | **none** (all rights reserved) | read only; data not redistributable |
| [Zatyrus/parameter-optimization-for-iterative-minflux-microscopy](https://github.com/Zatyrus/parameter-optimization-for-iterative-minflux-microscopy) | data release with small export helpers | Python | GPL-3.0 | – |
| [Nuka-Kola/MFX_data_analysis](https://github.com/Nuka-Kola/MFX_data_analysis) (private) | N. Sergeev, T. Engelhardt: single-molecule *tracking* workflow -- `.msr` through specpy from PyPI (`data_analysis_utils/io.py` `msr_to_df`), the sequence's per-iteration parameters joined onto every row, `sta` abort reasons per trace, trace metrics, track rotation, tail labelling, CFR post-hoc trace cutting, step finding with bnp-step, MBM bead traces (`MBM_analysis.py`), building sequences | Python | **none** | read only; ask the authors |
| [FReina/MFX_data_analysis](https://github.com/FReina/MFX_data_analysis) (private; read from a zip of `master`, 2026-10-15) | a fork of an *older* Nuka-Kola state (Python 3.9; no `fit_steps`, no parallel bnp-step) that adds a **specpy-free `.msr` importer**: `data_analysis_utils/msr_utils.py` `MsrImporter` (datasets, images, `to_df`/`to_npy`/`to_mat`/`to_tiff`) on pyMINFLUX's OBF parser vendored as `_msr_reader.py` (Apache-2.0, unchanged but for blank lines), and a beginner's guide to the format, `MSR_file_structure.md` | Python | **none** (the vendored reader Apache-2.0) | read only; ask the authors |
| [lumasullo/p-minflux](https://github.com/lumasullo/p-minflux) | p-MINFLUX (pulsed, TCSPC) analysis and simulation -- a different instrument | Python | GPL-3.0 | – |
| [ries-lab/SimuFLUX](https://github.com/ries-lab/SimuFLUX) | MINFLUX simulator ([bioRxiv 2025](https://doi.org/10.1101/2025.04.08.647786)) | MATLAB/Python | – | simulated data for tests and pages |

Not public or not found:

* **`PYMEXnf`** (`PYMEXnf.IO.msr_minflux`, imported by
  `PYME-extra/PYMEcs/IO/MINFLUX.py:27,957`): the Soeller group's `.msr`
  reader.  It is in no public repository; ask them.
* **specpy** (presumably what "sprecpy" meant): Abberior's Python API to
  Imspector, closed source.  It has shipped in the Imspector installation
  (`python/specpy`) and, since 2026, is also on
  [PyPI](https://pypi.org/project/specpy/) (`2026.20.24464`: wheels for
  CPython 3.9-3.14 on Windows, Linux x86-64 and macOS arm64; no licence
  stated; the older PyPI 1.2.1 of 2017 lacks the MINFLUX calls).
  `specpy.File(path, specpy.File.Read)` reads an `.msr` **without Imspector
  running**: `.minflux_datasets()` lists the data sets, `.unpack(aid, dir)`
  writes one as a zarr directory, then `zarr.open(dir + "/zarr")`
  (MFX_data_analysis `io.py:429-435`; older builds `minflux_data_sets()` /
  `unpack(sid, ...)`, `minflux-analysis/python/mfx/mfx/mfxdata.py:69-84`).
  The macOS wheel needs its rpaths repaired on macOS 27
  (MFX_data_analysis `tools/fix_specpy_rpath.py`).  So specpy is a possible
  *optional* route, and the reference to check our own reader against; the
  format can also be read directly (next section), which needs no
  closed-source dependency and follows no Imspector version.
* Other readers of the OBF image layer: [obffile](https://www.cgohlke.com/docs/obffile/)
  (Gohlke; images only, explicitly not MSR's non-image data),
  [msr-reader](https://github.com/hoerlteam/msr-reader) on PyPI (pure
  Python, images; no licence in its metadata), and
  [Bio-Formats' OBF reader](https://bio-formats.readthedocs.io/en/stable/formats/imspector-obf.html).

## The `.msr` file

`.msr` is Imspector's OBF container.  The MINFLUX data is a zarr v2 store
packed into one OBF stack.  Read without specpy by minflux-viewer
(`msr/mfxdta.py`, `msr/zarr2.py`, using `msr-reader` for the OBF layer) and
by the FReina fork of MFX_data_analysis (`msr_utils.py` `MsrImporter`, about
100 lines on top of pyMINFLUX's OBF parser); pyMINFLUX's
`reader/_msr_reader.py` itself is a full pure-Python OBF parser but returns
images only.  Verified against the real m2205 files in
etMINFLUX-analysis-public.

**OBF** (little-endian):

* File header `'<10sIQI'`: magic `b"OMAS_BF\n\xff\xff"`, format version (2
  in real files), position of the first stack, description length, the
  description; from version 2 a u64 position of a tag dictionary
  (length-prefixed strings, empty key ends it; normally `ome_xml`).  A
  position of 0 means none (`msr-reader` crashes there; minflux-viewer patches
  it, `msr/obf_compat.py:285-317`).
* Stack header `'<16s17I30d5I3Q'`: magic `b"OMAS_BF_STACK\n\xff\xff"`,
  version, rank, `res[15]` u32, `len[15]` and `off[15]` f64 (metres), data
  type, compression type (1 = zlib), level, name and description lengths,
  reserved, `data_len_disk`, `next_stack_pos` (0 ends the list); then name,
  description, data.
* Data types: 0x1 u8, 0x2 i8, 0x4 u16, 0x8 i16, 0x10 u32, 0x20 i32, 0x40 f32,
  0x80 f64, 0x1000 u64, 0x2000 i64, 0x10000 bool.
* A versioned footer after the data (real files: stack version 7, 1528
  bytes; sizes per version in pyMINFLUX `_msr_reader.py:183-200`), then
  labels, column positions, flush points, a **tag dictionary**, chunk
  positions.  Chunked stacks (v6+) exist (`obf_support._read_stack`); none of
  the real MINFLUX stacks is chunked.

**Finding the MINFLUX data**: a uint8 stack whose zlib-decompressed bytes
start with `b"MFXDTA"` (`minflux-viewer msr/mfxdta.py:257-280`).  Its tag
dictionary has a key `minflux` holding JSON with the data set's `did`,
`sid` and `label` (rendered MINFLUX images carry a `display` JSON instead).
In the newer files the FReina fork reads, that JSON also has a `type`:
`data` for localizations, `trace` and `value` for Imspector's preview
renderings (`msr_utils.py` `MsrImporter.__init__`) -- the simplest test where
it exists.

**MFXDTA** (`msr/mfxdta.py:20-37,122-173`):

```
[0:7]   b"MFXDTA\0"
[7:11]  u32 container version: 2 (m2205, '\' key separator), 3 (later, '/')
[11:15] u32 unix time of saving
zero padding, then records:
        u64 name length, name (latin-1 key), u8 flag (1 group, 0 leaf),
        leaf: u64 data length, data
```

Keys under `zarr\` form a plain zarr v2 key -> bytes store (`sync\...` are
empty placeholders).  `mfx` is the localization array: 1-D structured,
chunks of 10000, Blosc lz4 level 5 with shuffle in the m2205 files.  The
FReina fork, written for files of 2026 (labels `2607..`/`2608..`, `/` keys),
decompresses every chunk with plain **zlib**
(`msr_utils.py` `_read_mfx_from_entries`), so the compressor differs between
Imspector versions: a reader must take it from `.zarray`'s `compressor`, not
assume one.  (*Unverified*: no such file is in any repository.)  The same
reader builds the dtype only for flat fields and ignores the `fill_value`
padding of the last chunk by trimming to `shape`.  **zarr-python 3 cannot open
these structured dtypes**; napari-storm and minflux-viewer pin `zarr<3`, and
minflux-viewer wrote its own ~540-line v2 reader on numcodecs.  For SMAPpy,
a small reader of our own (`.zarray` JSON -> dtype; Blosc via `numcodecs`, or
`blosc2`) avoids the zarr dependency altogether and serves `.zarr`
directories and `.zarr.zip` too.

*Unverified* (no sample file anywhere): an `.msr` written by m2410 (MFXDTA
v3, flat `mfx`, populated `grd/mbm/points`).  We need one from our own
instrument.

## The raw data model

**m2205** (`itr` nested; one row per localization):

```
itr: [itr i4, tic u8, loc f8[3], lnc f8[3], eco i4, ecc i4, efo f8, efc f8, sta i4,
      cfr f8, dcr f8, ext f8[3], gvy, gvx, eoy, eox, dmz, lcy, lcx, lcz, fbg] [n_itr]
sqi u4, gri u4, tim f8, tid i4, vld b1, act b1, dos i4, sky i4
```

`n_itr` 5 (2D), 10 (3D), 3-4 (tracking); one more when the sequence has an
empty first iteration (`PYME-extra IO/MINFLUX.py:141-189`).

**m2410** (flat; one row per iteration event):

```
vld b1, fnl b1, bot b1, eot b1, sta u1, tim f8, tid u4, gri u4, thi u1, sqi u1,
itr i4, loc f8(3), lnc f8(3), eco u4, ecc u4, efo f4, efc f4, fbg f4,
cfr f2, dcr f2(2)          (tracking: dcr f2(1))
```

`fnl` marks a cycle's final iteration, `bot`/`eot` a trace's first and last
row.  A 2D trace runs `itr 0 1 2 3 4 3 4 3 4 ...`, `fnl` on each 4.

* **Telling them apart**: `fnl`/`bot`/`eot` in the dtype, or `itr` a scalar
  integer (pyMINFLUX `util/_util.py:225-325`; napari-storm
  `minflux_v2.py:45`).
* **Selecting localizations** (m2410): the `vld & fnl` rows (napari-storm,
  PYME); not `itr == max(itr)`, which fails in tracking where cycles end at
  different iterations.  PYME also drops incomplete cycles.  m2205: `vld`,
  last iteration.  The pyMINFLUX test data has the same 3D measurement in both
  layouts, `fnl` row count equal to the m2205 row count: the cross-check.
* **Which iteration a field comes from**: position, `efo`, `eco`, `dcr`,
  `fbg` from the final one; `cfr` from the last iteration where it is
  computed -- pyMINFLUX the last whose column has non-zero spread
  (`util/_util.py:26,67-72`), PYME the last with `ccrLimit >= 0` in the
  sequence metadata (`IO/MINFLUX.py:200,742`).
* **Photons**: `eco` of the final iteration, or summed over the cycle (PYME),
  or over the final-scale iterations (minflux-viewer; matches Imspector's
  aggregation counts to ~99 %).
* **Units**: positions in metres, `tim` in seconds from the start.  `loc` is
  Imspector's MBM-corrected position, `lnc` the uncorrected one.
* **z**: the stored z needs a foreshortening factor -- 0.7 (Gwosch 2020;
  pyMINFLUX GUI, Findeisen), 0.72 (PYME), 0.67 (minflux-viewer).  2D data
  has z noise of ~1e-13 m.
* **Metadata** (`mfx/.zattrs`): m2205 `tms` (acquisition time), `seq`, `fld`;
  m2410 `acquisition_date`, `measurement` with the per-iteration sequence
  (`ccrLimit`, `patDwellTime`, `patGeoFactor`, wavelengths) --
  `PYME-extra IO/MINFLUX.py:702-810`.

## MBM: the reference beads

* **Layout, m2410**: `grd/mbm/points`, structured `gri i4, xyz f8(3) [m],
  tim f8 [s], str f8` (`str == 0`: bead not seen); bead names in
  `points/.zattrs["points_by_gri"]`; the beads Imspector used in
  `mbm/.zattrs["used"]`.
* **Layout, recent Imspector** (2026 builds, *unverified* beyond one script):
  numbered groups, `grd/mbm_0/points` and `mbm/mbm_0.attrs["used"]`
  (MFX_data_analysis `MBM_analysis.py:26-27`) -- so a reader should look for
  `mbm*` groups rather than a fixed path.  Beside it the FReina fork's notes
  list `grd/search_0/points` with `gri` and xyz, which they call the "MBM
  grid"; the name suggests the positions of the search grid instead
  (`MSR_file_structure.md`, *unverified*).
* **Layout, m2205**: one array per bead, `grd/mbm/<R-name>`, fields `tim`,
  `pos(3)`, `str`, `tid`; `used` on the group (minflux-viewer
  `msr/legacy_mbm.py`).  MBM-inspector's `assets/*.npz` are this.

What is done with them:

1. **Drift correction, redone from `lnc`.** minflux-viewer
   (`msr/mbm_drift.py`, `analysis/bead_drift.py`) reconstructs Imspector's:
   loc = lnc - (D(t) - D(t_ref)), D the unweighted mean of the selected beads,
   each Gaussian-smoothed over sigma = 1.2 samples and interpolated, dead
   samples (`str == 0`) dropped, t_ref the first localization -- reproducing
   Imspector's `loc` to 0.15 nm rms.  It adds a per-bead quality (rms
   deviation from the median of all beads) and suggests dropping beads above
   max(2.5 x the median deviation, 2 nm).  PYME (`recipes/localisations.py:1772`
   `MBMcorrection`, `Analysis/MBMcollection.py`) does the same with a 1 s
   grid, optional rolling median (5), each bead centred on its own mean, the
   mean over beads, LOWESS (frac 0.1); beads chosen by hand or in
   MBM-inspector.  This is the shape of SMAPpy's `drift_fiducials.py`: a
   `Drift`, with bead selection.
2. **Registering two acquisitions or channels** (sequential labelling,
   Exchange-PAINT).  Each bead's position from its first samples (median of
   10, minflux-viewer `msr/alignment.py`; mean of 3, pyMINFLUX
   `correct/_bead_alignment.py`), matched by name; translation for 1-2 beads,
   rigid (Kabsch) for 3+, with per-bead residuals.  PYME `AlignFromMBMs`:
   mean shift only.  Findeisen (`mfxdata.py:111-370`) chains several washes,
   with a time-dependent rotation (needs 3+ non-collinear beads).  In SMAPpy
   this is a source of a named transformation for *Transform*
   (`Result.transformations`).
3. **QC**: bead std over time as the stability of the instrument -- SMAPpy's
   *Stability* already reads beads by an id column.

## Analysis

What exists, with where SMAPpy already has the generic part.

* **Trace (`tid`) averaging.**  One position per trace; precision from the
  trace's std / sqrt(n).  Plain mean or `eco`-weighted (pyMINFLUX
  `analysis/_analysis.py:1158-1315`); Imspector's *aggregation* -- groups of
  consecutive localizations reaching N photons -- reproduced exactly in
  minflux-viewer `analysis/aggregation.py`.  PYME: std with ddof 0,
  100 nm for single localizations; `SuperClumps` joins consecutive traces of
  one molecule within 5 nm.  In SMAPpy: `tid` is an id column like
  `group_id`, so this is grouping (`group.py`) by `tid`.
* **Filters.**  PYME's recipe: `cfr` in [0, 0.71] (0.75 "minimal"), >= 4
  localizations per trace, trace std < 15 nm (xy), < 20 nm (z).
  pyMINFLUX: CFR upper bound at median + 2 MAD; EFO cut at the histogram
  valley near twice the single-emitter EFO (double emitters).  Findeisen:
  DBSCAN within each trace (eps 10 nm, 3 points) to drop outliers and split
  traces; drop sub-clusters with `efo` > 2 x median.  SMAPpy's filters work
  on any column; what is new is the per-trace columns and the robust
  automatic thresholds.
* **Two colours by DCR** (detection channel ratio).  PYME `DcrColour`:
  threshold on the trace mean, optional majority vote with a confidence.
  pyMINFLUX: threshold or Bayesian GMM, then a per-trace vote.
  minflux-viewer (most complete, `analysis/dcr_em.py`,
  `attribute_channels.py`): two-Gaussian EM, Bayes decision boundary,
  per-trace assignment from the trace mean, and DCR pooled over iterations
  weighted by `eco` -- which narrows the peaks in 3D.  In SMAPpy: a
  *channel* column, the way `assign_colors.py` writes one.
* **Precision.**  Per-trace std (ddof 1, traces of >= 5, after Schmidt et al.
  2022); for tracking std(diff(x))/sqrt(2) (Deguchi 2023; PYME).  A
  per-localization CRLB (minflux-viewer `localization_precision.py:740`):
  sigma_xy = L/sqrt(8N) (1 - L^2/8 sigma_q^2)^-1 sqrt((1+rho)(1+rho(1+c))),
  L the pattern size from the sequence, rho the background ratio from `fbg`.
  FRC on trace means, so a trace never falls in both halves (pyMINFLUX
  `fourier/_fourier.py`, threshold 1/7).
* **Drift without beads**: Ostersehlt et al. 2022 (pyMINFLUX
  `correct/_correct.py`, minflux-viewer) -- SMAPpy's RCC and COMET cover
  this; minflux-viewer also has DME (entropy minimisation).
  PYME's origami-site drift (`OrigamiSiteTrack`): offsets from DNA-origami
  site centroids, binned at 400 s, smoothed, a spline.
* **z-scale estimation** (minflux-viewer `analysis/trace_analysis.py:211-311`,
  after Abberior's `estimate_anisotropy`): sqrt(size_x size_y)/size_z from
  per-trace scatter.  It assumes the scatter is isotropic, which MINFLUX's
  precision is not; minflux-viewer itself defaults to a fixed factor.
* **QC of the sequence**: PYME `Analysis/Paraflux.py` counts per iteration how
  many traces survive and why they fail (`sta`: 6 CFR, 8 no signal, 9 DAC
  out of range).  MFX_data_analysis (`io.py` `_process_combined_df`) names
  each trace's *abort reason* from its last `sta` (1 valid final, 2 valid,
  6 above CFR limit, 8 below background threshold, 9 out of EOD range, 11
  background measurement) and joins the sequence's per-iteration parameters
  (`patGeoFactor`, `phtLimit`, `patDwellTime`, `ccrLimit`, `bgcThreshold`,
  laser power and wavelength; `sequence.py` `extract_seq_params`) onto every
  row -- which is how parameters can be compared across acquisitions.  Rate analysis (`Analysis/MINFLUX.py:232-350`
  `analyse_locrate`): trace durations, time between traces, traces per
  minute.  TCP cycles per localization `eco/(efo * dwell)` (Vogler et al.
  2025).
* **NPC**: PYME `Analysis/NPC.py` (2D segment counting with the binomial
  labelling-efficiency model; 3D ML template fit) used in the Soeller group's
  RyR2 paper ([Nat. Commun. 2025](https://doi.org/10.1038/s41467-025-67801-6)) for the
  labelling efficiency; minflux-viewer `analysis/npc_geomfit.py` (two-ring
  8-fold mixture, ML).  SMAPpy's NPC plugins are the comparison.
* **Tracking**: irregular sampling handled by a consensus time step
  (PYME `Analysis/Tracking.py`); MSD, jump distances (minflux-viewer;
  pyMINFLUX); SMAPpy's *Tracking analysis* already reads `tid` on `time_s`.
  MFX_data_analysis (`processing.py`) is built for motor-protein-like tracks:
  each track rotated so its long axis is x (`rotate_all_tracks`, paired
  traces of two channels together), per-trace displacement, velocity and std
  after skipping the first 20 localizations, a *tail* (the approach before
  the molecule settles) labelled by a displacement criterion
  (`label_trace_tail`, `calc_interval_disp_std`), traces cut after their CFR
  exceeds a limit (`cfr_posthoc_filter`), and steps found by Bayesian
  non-parametric step finding ([bnp-step](https://github.com/LabPresse/bnp-step),
  `fit_steps`).  Trace ids are made unique across acquisitions (`utid`).

## Test data

* **pyMINFLUX `test/data`** (Apache-2.0, can be used with attribution):
  `reader_v2.zip` (13 MB: m2410 `.npy`, `.mat`, `.json`, tracking `.npy`),
  `reader_comparison.zip` (80 MB: the same 2D and 3D measurements in m2205
  and m2410), `2D_ValidOnly.npy.zip`, `3D_ValidOnly.npy.zip` (2-3 MB).
  Too large to commit; fetch on demand, as `SMAPPY_TEST_CAL` does.
* **etMINFLUX-analysis-public `exampledata/`**: real m2205 `.msr` files
  (smallest 7.4 MB) with their `.npy` exports -- ideal for checking an
  `.msr` reader against the `.npy`, but without a licence: use locally, do
  not commit.
* **MBM-inspector `assets/*.npz`**: legacy bead tracks (GPL).
* Missing: an m2410 `.msr` with MBM, and Imspector's own `.zarr` export.
* Synthetic: `smappy.simulate` for the analysis; SimuFLUX for realistic
  MINFLUX traces.

## A possible order of work

1. **Reader** (`io/minflux.py`): both layouts from `.npy`, `.mat`, `.json`,
   `.zarr`, `.zarr.zip` and `.msr` (OBF -> MFXDTA -> zarr v2, no specpy, no
   zarr-python); `vld & fnl` selection; `cfr` from its iteration; `lnc` kept
   (`x_nc_nm` ...); a *z scale* setting recorded in the metadata; the
   sequence metadata kept; MBM beads kept with the file.  Tested on the
   pyMINFLUX data (m2205 vs m2410 give the same localizations), the
   example `.msr` against its `.npy`, and, where specpy is installed,
   against specpy's `unpack` of the same file (`SMAPPY_TEST_MSR`-style,
   skipped without it).
2. **MBM drift**: redo the correction from `lnc` with chosen beads, as a
   `Drift` (`drift_fiducials.py`'s shape), with bead QC.
3. **Trace grouping** by `tid` and per-trace columns (n, std, duration) and
   filters, including robust CFR/EFO thresholds.
4. **DCR channels.**
5. **MBM registration** of two acquisitions as a transformation.
6. Precision (trace std, CRLB), FRC on trace means, sequence QC.

Licences decide how: from the MIT/Apache/BSD repositories code can be ported
with attribution in the module docstring; from the GPL ones (PYME-extra,
MBM-inspector) only the algorithm, reimplemented.
