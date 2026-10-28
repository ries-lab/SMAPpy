# The bead scan's z step and range, single-channel astigmatic beads (2026-10-28)

**The question.**  How coarse can the z step of a bead scan be, and how short its range, before the PSF
calibration gets worse?  For uiPSF's pupil model and for smappy's bead average, on real astigmatic beads.
The same study for 4Pi is `docs/4pi.md` §18 (100 nm lost nothing there against 20 nm).

**Answer, for these beads.**
- *Pupil model:* a **200 nm** step loses nothing measurable against 50 nm (the finest these stacks hold).
- *Bead average:* **100 nm** at most.  At 100 nm it costs 4–5 % in z CRLB.  At 150 nm its spline covers only
  part of the range, and at 200 nm it fails.
- *Range (pupil model, 50 nm step):* scan **±750 nm** to localise over about ±850 nm.  It loses nothing
  against ±1000 nm, even judged over ±850 nm.  A ±500 nm scan fits the central ±400 nm as well, a little
  better even.  But its model is measurably worse beyond the scan, and the window's spline stops at the
  scanned range anyway.  So scan at least the depth you will localise over.
- Take bead stacks for the pupil model at 100 nm over ±750–1000 nm: 16–21 frames instead of 41.  If the bead average may be wanted too, keep 50 nm.

## The data and the method

**Data.**  uiPSF's Zenodo `1ch_40nm_bead` (docs/uipsf/data.md): 31 stacks of 40 nm beads, M2's Evolve
EMCCD, 116 × 127 nm pixels.  Each stack has 41 slices **50 nm** apart (Micro-Manager's `z-step_um` 0.05),
±1000 nm.  It is strongly astigmatic: vertical astigmatism (Noll 6) is −0.56 rad in the reference model,
about 53 nm rms at 600 nm.  v1's own model has 0.56 in its transposed frame.  The window drops each
stack's first slice (`pupil.KEEP`), which leaves 40 slices, 975 to −975 nm.  56 beads pass the window's
detection.

So **20 nm cannot be made** from these stacks; the finest step is the native 50 nm.  The ranges are capped
at ±1000 nm; ±1500 nm is not available.  40 slices have no slice at 0, so a thinned scan reaches one side
50 nm further (100 nm: −975 to 925; 200 nm: −975 to 825; 150 nm: −875 to 925).

**Script.**  `benchmarks/uipsf/real/zenodo_1ch_zstep.py`, with its table and figure in
`zenodo_1ch_zstep_figure.py`; results in `benchmarks/uipsf/results/zenodo_1ch_zstep.jsonl`.
- **Splits.**  The beads are split at random, 37 for training and 19 held out, with 5 seeds.
- **Steps.**  Every k-th slice through the focus slice, as a coarser scan at the same exposure per frame:
  the photons fall with the step.
- **Learning.**  uiPSF's pupil model with the window's defaults (`calibrate_beads`; order 6, magnitude and
  blur free, a 40 nm bead; NA 1.43, λ 600, immersion 1.516, medium 1.35 as v1 had).  smappy's bead
  average with the window's defaults (`build_calibration`).
- **The judge.**  Every slice of the held-out beads' full 50 nm stacks is fitted by Spline 3D with each
  calibration.  It uses the best of five z starts, as *Fit quality* does (`pupil.fit_quality`).
- **The pupil's spline.**  Sampled at 50 nm whatever the step, so the learning is compared and not the
  spline's knots.  *pupil, spline at the step* is the same model sampled at the scan's step, as the window
  does on *auto*: it makes no difference.
- **The window.**  Judged over ±400 nm about each bead's own focus, the common central range (the
  reference model puts the beads' foci within ±31 nm).

**Metrics**, all on the held-out beads:
- *z rms about a line*: z against the stage, fitted by a line per bead.
- *z bias*: the systematic curve.  Each bead's centred error against the common slope, the median over
  the beads at each z, as an rms over z.
- *slope*: the median of the beads' slopes.
- *z scatter*: the rms about that curve.
- *x / y rms*: about each bead's mean.  The bead does not move, so this is wander and noise.
- *CRLBs*: medians.
- *deviance per pixel*: Spline 3D's logl is minus half the Poisson deviance.
- *Zernikes from the reference*: the rms over the phase modes Noll ≥ 5, in nm.  The reference is the
  model learnt on all 56 beads at 50 nm over ±1000 nm.

**Noise.**  The split-to-split spread is mostly which beads judge, and it is the same at every setting.
So the test is the *paired* change from the finest step (or widest range) within a split, its mean ± sd
over the 5 splits.  The grey band in the figure is the split-to-split sd of the 50 nm pupil model.
"Loses nothing" means the paired change is inside that band and small beside what matters (0.1 nm in z).

## Results

z step, scanned over ±1000 nm (mean ± sd over 5 splits; in brackets the paired change from 50 nm):

| method | step | z rms about a line (nm) | z bias (nm) | slope | z CRLB (nm) | x rms (nm) | deviance / px | Zernikes from ref. (nm) |
|---|---|---|---|---|---|---|---|---|
| pupil | 50 | 5.57 ± 0.28 | 2.32 ± 0.55 | 1.015 ± 0.005 | 2.455 ± 0.15 | 2.87 ± 0.22 | 1.194 ± 0.03 | 0.090 ± 0.028 |
| pupil | 100 | (+0.004 ± 0.02) | (−0.005 ± 0.03) | (−0.0004 ± 0.0002) | (−0.003 ± 0.002) | (+0.000 ± 0.009) | (+0.001 ± 0.001) | 0.099 (+0.009 ± 0.01) |
| pupil | 150 | (+0.001 ± 0.02) | (+0.001 ± 0.05) | (+0.0000 ± 0.0003) | (+0.004 ± 0.002) | (−0.009 ± 0.01) | (−0.004 ± 0.001) | 0.119 (+0.029 ± 0.01) |
| pupil | 200 | (−0.013 ± 0.005) | (−0.041 ± 0.02) | (−0.0006 ± 0.0006) | (−0.008 ± 0.004) | (+0.005 ± 0.03) | (+0.002 ± 0.001) | 0.137 (+0.046 ± 0.02) |
| average | 50 | 5.50 ± 0.24 | 1.90 ± 0.37 | 1.001 ± 0.007 | 2.291 ± 0.12 | 2.67 ± 0.13 | 1.175 ± 0.02 | |
| average | 100 | (−0.13 ± 0.12) | (+0.01 ± 0.3) | (+0.002 ± 0.002) | (+0.105 ± 0.03) | (−0.003 ± 0.03) | (+0.002 ± 0.01) | |
| average | 150 | 57 % of the window fitted: its spline reaches −100 to +350 nm only | | | (+0.22 ± 0.03) | | (+0.020 ± 0.01) | |
| average | 200 | fails: "too little common z overlap after registration" | | | | | | |

z range, 50 nm step, pupil model; judged over ±400 nm with the window's spline (the scanned range), and
over ±850 nm with the spline sampled over the whole stack (paired change from ±1000 nm):

| scanned | z rms, ±400 | z bias, ±400 | deviance / px, ±400 | z rms, ±850 | z bias, ±850 | slope, ±850 | x rms, ±850 | deviance / px, ±850 | Zernikes from ref. (nm) |
|---|---|---|---|---|---|---|---|---|---|
| ±975 | 5.57 ± 0.28 | 2.32 ± 0.55 | 1.194 ± 0.03 | 7.44 ± 0.21 | 4.03 ± 0.49 | 1.007 ± 0.004 | 7.21 ± 0.39 | 1.197 ± 0.03 | 0.090 |
| ±725 | (−0.07 ± 0.01) | (−0.23 ± 0.1) | (−0.030 ± 0.005) | (−0.26 ± 0.2) | (−0.36 ± 0.3) | (−0.006 ± 0.001) | (−0.40 ± 0.09) | (−0.002 ± 0.001) | 0.45 |
| ±475 | (−0.09 ± 0.1)\* | (−0.09 ± 0.4)\* | (−0.051 ± 0.006)\* | (+0.34 ± 0.2) | (+0.83 ± 0.3) | (−0.018 ± 0.001) | (+0.34 ± 0.3) | (+0.017 ± 0.003) | 0.77 |

\* 93 % of the ±400 nm window fitted: the window's spline stops at the range every bead covers, about
±440 nm.

The figure (`python benchmarks/uipsf/real/zenodo_1ch_zstep_figure.py -o zstep.png`; not checked in) plots
each metric's paired change against the step (left) and against the range (right).

## What it shows

- **The pupil model does not care about the step.**  From 50 to 200 nm, every localisation metric of the
  held-out beads changes by less than 0.05 nm in z and x, and the deviance by 0.002 per pixel.  That is a
  tenth of the band or less.  The Zernikes move by 0.05 nm rms, against 53 nm of astigmatism.
  - It has a few dozen parameters, and ten 200 nm slices of 37 beads determine them; the bead kernel
    integrates over the coarser step.
  - It learns faster with fewer slices: 1 min at 50 nm, 13 s at 200 nm.
  - This agrees with the synthetic `z_spacing.py` study (docs/uipsf/architecture.md): thinning costs
    what the CRLB says for the photons left out, and adds no bias.
- **The bead average does care.**
  - It needs the slices themselves: its spline's knots are the scan's slices, and its registration
    aligns beads by whole-plane shifts.
  - At 100 nm its z CRLB is 0.1 nm (4.5 %) worse.  That is larger than the paired noise, though within
    the split spread.
  - At 150 nm the registered beads overlap only over −100 to +350 nm.  At 200 nm the window would refuse
    to calibrate.
  - Its z slope is 1.00, against the pupil model's 1.015.  The bead average is the stage's z by
    construction; the pupil model's z is the emitter's, through the optics given (n_med 1.35).  That
    1.5 % is the same at every step, so it does not affect this comparison.
- **A shorter scan fits the centre slightly better.**  Judged over ±400 nm, models from ±725 and ±475 nm
  scans have a 0.03–0.05 lower deviance per pixel, a slope nearer 1 and a z bias as small.  A model
  learnt on the out-of-focus slices too must compromise between them and the centre.  The Zernike order
  6 model does not describe the whole ±1000 nm exactly.
- **But it extrapolates worse.**  Judged over ±850 nm, the ±475 nm model has 0.8 nm more z bias, a slope
  1.8 % lower and a higher deviance.  The ±725 nm model is as good as the ±1000 nm one, or slightly
  better (z bias −0.4, x −0.4 nm).
  - The window samples the spline only over the range every used bead covers (`pupil._auto_z_range`), so
    a short scan gives a short calibration.
  - So a scan should cover the depth to be localised.  About ±750 nm is enough for an astigmatic PSF
    used over ±800 nm.

**Caveats.**
- One microscope and one bead set of bright 40 nm beads.  Dim beads have less photon headroom for a
  coarse step.
- 20 nm and ±1500 nm could not be tested from these stacks.
- The held-out beads lie in the same stacks as the training ones (the split is by bead, not by stack).
- The bead average ran with the window's default averaging settings.  A larger *max z shift* or
  *alignment range* might rescue 150 nm; that was not tried.
