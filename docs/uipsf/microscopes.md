# The lab's microscopes, as their bead stacks record them

What the Micro-Manager metadata of the lab's 2026 bead calibrations says about each microscope, and what
SMAPpy's camera database (`src/smappy/data/cameras.json`) resolves from it through
`smappy.calibrate.pupil.camera_for`: the starting point of a microscope database, the integration step after
the camera database (`docs/uipsf/plan.md` §7.5).  The stacks behind it are the ones of
`benchmarks/uipsf/real/lab_beads.py` (`docs/uipsf/architecture.md`, "Is the blur the microscope's?").
Read from `/Volumes/vbc/STRUCTBIO/RIES/2026` (October 2026); no M6 bead stacks exist there or under 2025.

Every stack: 101 slices 20 nm apart over ±1 µm, 100 ms, one channel.

## M2

| acquisition | date | stacks | camera | readout | ROI (x, y, w, h) | pixel in the file | filter (comment) | sliders |
|---|---|---|---|---|---|---|---|---|
| 260622_AA_BeadsCalZ_638 | 2026-06-22 | 5 | Photometrics Evolve 512 Delta, A14G150010 | EM ("Multiplication Gain"), 10 MHz 16 bit, gain 1, EM gain 25 | 206, 275, 200, 200 | 0 (not set) | – | ELL9 positions 0 / 0 |
| 260829_AA_BeadsCalZ_638_SingleMode | 2026-08-29 | 5 | Andor iXon Ultra DU897_BV, 8172 | EM, 10 MHz, pre-amp Gain 1, EM gain 25 | 312, 176, 150, 150 | 0.127 | – | – |
| 260904_AA_BeadsCalZ_638_SingleMode | 2026-09-04 | 5 | same | same | 300, 193, 150, 150 | 0.127 | 700/100 | – |
| 260915_AASJ_BeadsCalZ_638_665LP | 2026-09-15 | 12 | same | same | 0, 77, 512, 200 | 0.127 | main 700/100, reflected 676/37 | reflected 0, BFP 0 |
| 261012_FF_beadscalib | 2026-10-12 | 8 | same | same | 7, 120, 499, 243 | 0.127 | "600/60" (not the filter used) | reflected 1, BFP 0 |

`camera_for` resolves the Evolve as *M2 Evolve* (6.7 e⁻/ADU, offset 398.6 for EM at 10 MHz) and the iXon
8172 as ***M1 Andor*** (15.8 e⁻/ADU, offset 196 for EM at 10 MHz, Gain 1): **M2's camera changed between
2026-07-09 (the last Evolve stack) and 2026-07-23 (the first iXon one)**, and the database still files that
iXon under M1.  Its pixel there is 0.1315 µm; the files' `PixelSizeUm` of 0.127 overrides it where
Micro-Manager has one, and the Evolve gets the database's fixed 0.127.

**The pixel is 116 × 127 nm on the astigmatic path, and the iXon sits turned by 90°.**  The cylindrical
lens makes the magnification differ between the axes.  Learnt with the same beads, the likelihood picks
116 × 127 (x × y) for the Evolve and 127 × 116 for the iXon, by 28 000 and 38 000 nats
(`benchmarks/uipsf/results/lab_beads_pixel.jsonl`); the astigmatism Z(2, 2) changes sign between them
too (−0.67 and +0.55 rad).  The square 127 nm of the database costs 12 000 and 41 000 nats and moves σ
by 21 nm along the axis it gets wrong.

## M5

| acquisition | date | stacks | camera | readout | ROI (x, y, w, h) | pixel in the file | filter (comment) | filter wheels, sliders |
|---|---|---|---|---|---|---|---|---|
| 260205_HL_beatcalibration_z_640 | 2026-02-05 | 5 | Hamamatsu ORCA-Fusion BT C15440-20UP, S/N 500975 | not recorded (no scope data in the file's summary) | 1236, 988, 304, 304 | 0 (not set) | "100 ms" | – |
| 260818_PM_beadsCal | 2026-08-18 | 3 | same | ScanMode 3, area sensor; conversion 0.23, offset 100 in the camera's own fields | 1508, 968, 312, 324 | 0.1056 | – | wheel 1 "Filter-1" (state 3), wheel 2 "Filter-1" (0); ELL6 position 1; ELL17/20 at 18 235 µm |
| 260826_AA_beadsCalZ_640_new | 2026-08-26 | 7 | same | same | 1504, 968, 312, 324 | 0.1056 | 685/70 | as above, ELL17/20 at 18 151 µm |
| 260901_MA_BeadsCalZ_20nm_-1to1 | 2026-09-01 | 6 | same | same | 1504, 968, 312, 324 | 0.1056 | 685/70 | wheel 1 "Filter-4" (state 3); ELL17/20 at 18 005 µm |

`camera_for` resolves *M5 Fusion BT*: 0.24 e⁻/ADU and offset 100, fixed.  The camera reports 0.23 in its
own `CONVERSION FACTOR COEFF`.  The pixel: the database fixes 102 × 103 nm, the files since August say
105.6 nm square and win; the February file has none and gets the database's.  105.6 is what the runs
used (the user's choice for now).  The beads are astigmatic here too (x and y focus 360 nm apart in the
median).

The eight acquisitions of 2026-10-29 (100 nm beads at 600 and 676 nm, in oil and in buffer, with and
without the 3D lens) are in `m5-media.md`.  There, the likelihood prefers a 104.7 × 106.4 nm pixel, and
the objective's effective NA in matched oil is 1.49–1.52, its nominal 1.50.

## What the database lacks

- **Read noise**, for every camera (`plan.md` §8: proposed as `read_noise_e` per readout state).
- **The microscope**: the iXon 8172 is on M2, not M1, since July 2026.
- **The pixel per axis and per light path**: 116 × 127 nm (Evolve) or 127 × 116 nm (iXon) on M2's
  astigmatic path; 102 × 103 against 105.6 on M5.
- **The optics**: no file names an objective, a dichroic or the cylindrical lens.  Sliders and filter
  wheels are recorded only by position (M2's ELL9 *Reflected Slider* and *BFP*; M5's ELL6, ELL17/20 and two
  Thorlabs wheels as "Filter-N"), and the emission filter only in the free-text comment, sometimes wrong.
  A microscope database needs a table from position to element per microscope, or labelled positions in
  Micro-Manager.  EMU's configurations name them (`docs/microscope-database.md`, which decodes these files with them).
- **The values used** (from the user, 2026-10-20): M2 NA 1.43, oil 1.516; M5 Olympus NA 1.50 TIRF (nominal),
  oil 1.518; TetraSpeck beads of 100 nm in water (1.33), the 676/37 channel, taken as 676 nm.

## The coverslip (October 2026)

uipsf's layered model (`docs/uipsf/architecture.md`, "The layered model and a tilted coverslip") puts the
coverslip between the sample and the immersion, with its own index and thickness, and can learn its tilt.
SMAPpy gives it per objective, from the microscope database, and per calibration:

| `Optics` | from | default |
|---|---|---|
| `n_cov` | the entry's `n_coverslip`, or *coverslip index* typed | the immersion's |
| `d_cov_design_nm` | the entry's `coverslip_design_um` × 1000 | 170 µm |
| `d_cov_nm` | *coverslip thickness* (µm), a setting of the calibration: the sample's, not the objective's | the design thickness |

`OpticsSettings.resolved` returns them beside the NA and the immersion; `OpticsSettings.optics` builds each
channel's `Optics` with them.  The lab's microscopes (M2, M5) have oil objectives and no `n_coverslip`: the
coverslip is the immersion there and their models are as before.  A water or silicone objective needs
`n_coverslip` in its entry (about 1.52), a microscope with several objectives making it follow the device
that holds them.

*learn coverslip tilt* is a model choice (`BeadSettings.fit_tilt`), one field for both methods since the
window's *model* section is theirs: `bead_model` gives `BeadStackGeometry(fit_tilt=…)`, and in situ passes
it to `insitu_model(…, fit_tilt=…)`.  The window offers it always in situ and for beads only when
`n_cov ≠ n_imm`; a bead run drops it otherwise (`pupil.learns_tilt`), since beads on the coverslip of an
oil objective see no tilt.  The learnt `geometry.tilt` is shown in degrees with its CRLB over *Pupil and
Zernikes* and in the run's summary line (`uipsf.diagnostics.coverslip_tilt`), and kept in the calibration's
record as `coverslip_tilt_deg` ([[t_x, CRLB], [t_y, CRLB]]).

