# Two-colour MINFLUX kinesin steps: the plan

Agreed 2026-11-19 (Jonas, in discussion; nothing implemented yet).  The
question: kinesin labelled in two colours, tracked by MINFLUX -- which tracks
walk, which are one kinesin in both colours, the colour offset, the steps in
one and two colours.  Until now a student's agent did it in a private plugin
and a script; it should be a few shipped plugins, then a chain.

Test data: `smappy test/minflux_kinesin` (iCloud, Vienna/Data):
`12_alignment_at1620_before_561.msr` (a TetraSpeck bead in the sample, its
own measurement, 6 min before the tracking) and `13_TD_trk_638_561_.msr`
(the tracking).  The student's report is beside them.

## What the data say

- **The metadata tie each thread to its laser and detector**
  (`measurement.threads[i].lasers` / `detectors`), and the thread order
  differs between measurements: file 12 thi 0 = 560 nm, file 13 thi 0 =
  638 nm.  File 13 thi 1 starts `on_thi0_vfn`: the 560 trace is started
  where thread 0 found a molecule.
- **File 13 has MBM beads in both colours**: threads 2 (640) and 3 (560) on
  PMT1, grids `grd/mbm_0` and `grd/mbm_1`, `zarr/mbm/mbm_k.correcting_thi`
  naming the thread each corrects.  7 beads, 19 points each, 30-370 s of a
  measurement whose localizations run to 1756 s.  Per bead the 640 - 560
  offset ranges from (-36, -25) to (9, -9) nm with 2-7 nm SD per bead: a
  bias per bead, so the absolute MBM colour offset is useless; its change
  is 1-2 nm per 100 s.
- **Neither colour was MBM-corrected** in file 13 (`loc` = `lnc`, `used: []`).
- **The loader reads only `grd/mbm/points`**, the one-colour layout: file 13
  has no beads for *MBM drift correction* today.
- **File 13's MINFLUX data hold no TetraSpeck**; file 12 is the bead, with
  no MBM, so nothing links its time to file 13's.

## Principles

- Stage drift moves both colours together (interleaved per localization) and
  does not change the colour offset; the tracks last < 1 s.  So drift
  correction applies **one curve to both colours** by default and never
  changes the overlay as a side effect.
- The colour offset = an **absolute** offset (beads in the sample, or the
  kinesins) + its **change over time** (MBM colour differences).  The change
  is measured with a free offset per bead, `d_b(t) = o_b + c(t)`
  (`beads.fit_offsets_and_spline`), so beads with different biases, coming
  and going, add no jumps; beads matched across measurements by position,
  never by name (slots are reused).
- Colours are named by laser, not by thread.

## The plugins

| # | plugin | change |
|---|---|---|
| -- | loader (`io/msr.py`, `io/minflux.py`) | `channel` per localization from the thread's MINFLUX laser, numbered by wavelength (longest first), labels in `metadata["channels"]`; both MBM grids read, each with the thread it corrects.  Several measurements loaded together (bead + tracking) as now; the user picks them in the import dialog |
| 1 | *MBM drift correction* | per grid, robust bead selection per colour; one common curve on both colours by default, each colour by its own grid as an option; the 640 - 560 drift shown as a check |
| 2 | *Quality filter* | + the trace start (tails), `traces.tails`' *still* / *moving* rules |
| 3 | *Dual colour* (*Pair colours* extended) | **offset**: the absolute from a chosen bead measurement (default: the bead measurement closest before the tracking), two bracketing bead measurements (before and after, interpolated, MBM's c(t) guiding the shape and the mismatch reported as a consistency check), the average over many kinesin tracks (stuck pairs, and walking pairs with a fit `c + a * direction` that separates the head lead), ROIs, another file's, typed; colour drift over time from MBM; applied, and kept as a named transformation.  **Pairing**: overlap in time (matched localizations, fraction of the shorter trace), fraction of matched localizations within d after the offset, several thread-1 traces to one thread-0 trace; co-movement later.  **Checks**: MBM colour drift, the pairs' residual against time (covers gaps and the time without beads), a warning when bead and tracking measurements share no MBM beads |
| 4 | *Segment regions*, preset *MINFLUX kinesin* | *from an id attribute*; *keep* gains `velocity_nm_per_s`, linearity of along-track position against time, and any current region value of the id column; a keyword list of colours (both / 638 / 560) |
| 5 | step analysis (*Find steps*) | one plugin with a dual-colour option: moving parts (SMAP's piecewise linear x(t), y(t)), steps per colour, head lead, alternation, dc view, display smoothing.  A discussion of its own |

The chain: 1-5 with an interface of a few settings (minimum length,
directed only, both colours only, maximum separation), one click from the
opened measurements to kinesin ROIs and their steps.

## Open

- Bracketing data (beads before and after, ideally with MBM throughout):
  none yet; a simulation (`simulate.minflux.motor_msr` grown to two
  colours, beads and MBM) can test the interpolation meanwhile.
- For Takahiro: why MBM points stop at 370 s; `disConExp`; StepsMINFLUX2C
  = StepsMINFLUX_dc.
