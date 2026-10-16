# Notes for writing the MINFLUX reader

The brief for the session that writes SMAPpy's MINFLUX reader: what was
learnt about the formats, which existing code to port from, and what is
still unknown.  The longer survey, with the analysis side, is `survey.md`
beside this file.  Paths of other repositories are `repo:path:line`, read in
2026-10; check them again before porting.

## Goal

One reader for Abberior MINFLUX data in all its forms, replacing
`_load_minflux` in `src/smappy/io/formats.py`:

| input | from |
| --- | --- |
| `.msr` | Imspector's own file (OBF container); the main target |
| `.npy`, `.mat`, `.json` (or a `.zip` of one) | Imspector's exports |
| `.zarr` directory, `.zarr.zip` | Imspector's zarr export (zipped by hand) |

Both data layouts in each: **m2205** (Imspector before 24.10) and **m2410**
(24.10 and later).  No specpy and no zarr-python are required.

## Who reads `.msr` today

| code | `.msr` -> localizations? | how | licence |
| --- | --- | --- | --- |
| embl-ic/minflux-viewer `minflux_viewer/msr/` | **yes, the most complete** | `mfxdta.py` (container, both versions), `zarr2.py` (own zarr v2 reader on numcodecs: any compressor the `.zarray` names), `legacy_mbm.py` + `mbm_drift.py` (beads, both layouts), OBF layer through the `msr-reader` package with a patch (`obf_compat.py`); tests `test_mfxdta.py`, `test_zarr2_shim.py`, `test_msr_*.py` | MIT |
| FReina/MFX_data_analysis `data_analysis_utils/msr_utils.py` `MsrImporter` | yes, ~100 lines, newer files only | pyMINFLUX's OBF parser vendored; chunks decompressed with zlib only; flat dtypes only | none (ask) |
| bsse-scf/pyMINFLUX `pyminflux/reader/_msr_reader.py` | no -- the OBF parser returns images only | pure-Python OBF, footer versions 1-7 | Apache-2.0 |
| jkfindeisen `obf_support.py` (vendored in etMINFLUX-analysis-public) | no | pure-Python OBF incl. chunked stacks | MIT (file header) |
| specpy (Abberior, PyPI `2026.20.24464`) | yes, as a zarr directory | `File(path, File.Read).minflux_datasets()`, `.unpack(aid, dir)`; no Imspector needed | closed, no licence stated |
| PYME-extra | through `PYMEXnf.IO.msr_minflux`, not public | -- | GPL |

**Port from minflux-viewer** (MIT: keep the copyright notice in the module
docstring), take pyMINFLUX's or `obf_support`'s OBF parsing where simpler
(Apache-2.0 / MIT), and use specpy only as an optional cross-check in a test.
Nothing from GPL code.

## The layers of an `.msr`

```
.msr (OBF)  ->  stack whose zlib-inflated payload starts b"MFXDTA"
            ->  MFXDTA records  ->  zarr v2 key/bytes store
            ->  mfx/.zarray + chunks  ->  structured numpy array
```

**OBF** (little-endian):
* file header `'<10sIQI'`: magic `b"OMAS_BF\n\xff\xff"`, format version,
  u64 first-stack position, description length + text; version >= 2: u64
  position of a tag dictionary (0 = none -- `msr-reader` crashes on that).
* stack header `'<16s17I30d5I3Q'`: magic `b"OMAS_BF_STACK\n\xff\xff"`,
  version, rank, `res[15]`, `len[15]`, `off[15]`, data type, compression
  type (1 = zlib), level, name length, description length, reserved,
  `data_len_disk`, `next_stack_pos` (0 ends the list); then name,
  description, `data_len_disk` bytes.
* a footer after the data, size by stack version (pyMINFLUX
  `_msr_reader.py:183-200`; real files version 7, 1528 bytes), then labels,
  flush points, the **stack's tag dictionary** (string pairs), chunk
  positions.  Chunked stacks exist from v6 (`obf_support._read_stack`); none
  of the MINFLUX stacks seen was chunked -- handle or refuse explicitly.

**Which stack**: inflate and test for `b"MFXDTA"`
(minflux-viewer `mfxdta.py:257-280`).  The tag dictionary's `minflux` entry
is JSON with `did`, `sid`, `label`; newer files add `type`: `data`
(localizations), `trace`, `value` (Imspector's preview images).  Several
data sets per file are normal; the `label` names each (it begins
`YYMMDD-HHMMSS`).

**MFXDTA** (`mfxdta.py:20-37,122-173`): `b"MFXDTA\0"`, u32 container
version, u32 save time (not acquisition time), zero padding (first record at
147 in v2 files -- find it, don't hard-code it), then records `u64 name_len,
name (latin-1), u8 flag (1 group, 0 leaf), [u64 len, data]`.  Key separator
`\` in container v2, `/` in v3: normalise.  Keys under `zarr/` are the store;
`sync/...` are empty.

**zarr v2**: `mfx/.zarray` (shape, chunks -- 10000 --, structured `dtype` as
JSON, `compressor`, `fill_value` base64), `mfx/0..n`.  The compressor
differs: **Blosc lz4** (level 5, shuffle) in the m2205 files seen, **zlib**
in the 2026 files the FReina fork reads.  Decode by the `.zarray`
`compressor` id (`numcodecs.get_codec`, or zlib/blosc by hand); trim the
last chunk to `shape`.  The m2205 dtype is nested (`itr` a structured
sub-array), so the JSON -> dtype conversion must recurse (minflux-viewer
`zarr2.py` `dtype_from_json`, :73).  zarr-python 3 cannot open these dtypes.
The same store code then serves `.zarr` directories and `.zarr.zip`.

Metadata: root `.zattrs` (`rois`), `mfx/.zattrs`: m2205 `tms`, `seq`, `fld`;
m2410 `acquisition_date`, `measurement.threads[i].sequences[0].Itr[...]`
(per-iteration `ccrLimit`, `patGeoFactor`, `patDwellTime`, `phtLimit`,
lasers) -- keep it in the file's metadata.

## The two data layouts

**m2205**: one row per localization, `itr` a nested array of `n_itr`
iterations (`loc, lnc, eco, ecc, efo, efc, cfr, dcr, fbg, sta, ...`), plus
`tim, tid, vld, sqi, gri, act, dos, sky`.  Take the last iteration of the
`vld` rows.

**m2410**: one row per iteration event, flat:
`vld fnl bot eot sta tim tid gri thi sqi itr loc(3) lnc(3) eco ecc efo efc
fbg cfr dcr(2)` (tracking: `dcr(1)`).  Take the **`vld & fnl`** rows -- not
`itr == max(itr)`, which fails in tracking (napari-storm `select_iteration`,
PYME `mk_seqids_maxpos`).

* Telling them apart: `fnl`/`bot`/`eot` present, or `itr` a scalar integer.
* `cfr` is computed only at some earlier iteration: take it from the last
  iteration where it is (sequence `ccrLimit >= 0`, else the last iteration
  whose `cfr` varies -- pyMINFLUX `util/_util.py:26,67-72`).  In m2410 that is
  an earlier row of the same cycle.
* Units: positions in metres (-> nm), `tim` s.  `loc` is Imspector's
  MBM-corrected position, **`lnc` the uncorrected one -- keep both**.  z needs
  a foreshortening factor (0.7 common; 0.67, 0.72 used): a setting, recorded.
  2D data has z ~ 1e-13 m: drop it.
* `thi` is the thread (channel) in two-colour sequences; `tid` repeats across
  data sets, so a merged table needs the data set too.
* The current SMAPpy loader fails on m2410 and takes `cfr` from the last
  iteration (usually 0).

## MBM beads, in the same store

* m2205: `grd/mbm/<R-name>` arrays (`tim, pos(3), str, tid`), names in
  attributes, `used` on `grd/mbm`.
* m2410: `grd/mbm/points` (`gri, xyz(3), tim, str`), names in
  `points_by_gri`, `mbm/.zattrs["used"]`.
* 2026 builds (one script): `grd/mbm_0/points`, `mbm/mbm_0.attrs["used"]`
  -- search for `mbm*` groups.
* `str == 0`: bead not seen.  Keep the beads with the file; correction is a
  later step (minflux-viewer `mbm_drift.py`).

## Test data

* pyMINFLUX `test/data` (Apache-2.0): `reader_comparison.zip` -- the same 2D
  and 3D measurements as m2205 and m2410 `.npy` (3D: 12709 m2205 rows =
  12709 `fnl` rows); `reader_v2.zip` -- m2410 `.npy`, `.mat`, `.json`,
  tracking.  Fetch on demand; too large to commit.
* etMINFLUX-analysis-public `exampledata/`: 29 real m2205 `.msr` with their
  `.npy` exports (smallest 7.4 MB) -- no licence: local checks only.
* For the tests in the repo: write small synthetic `.msr` files ourselves
  (minflux-viewer has a writer, `msr/writer.py`, to learn the inverse from),
  m2205 and m2410, zlib and Blosc.

## Still unknown -- need a file from our own instrument

* A real m2410 `.msr`: container v3, flat `mfx` inside, the compressor used,
  whether stacks are chunked.
* Real MBM beads inside an `.msr` (the m2205 examples have none).
* `grd/search_0/points` (the FReina notes call it the MBM grid; probably the
  search grid).
