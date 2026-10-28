# COMET: precision weights, the width in z, the error bar

A simulation study of COMET's cost (`smappy.drift`, `csrc/drift.hpp`): does
weighting each pair by the two localizations' precisions (DME's weighting)
help, does a wider overlap Gaussian in z (*z width*) help, does the error
bar from halves of the field measure the curve's noise, and what is the
wavy drift COMET finds where there is none.  The results and what was
decided from them are in `docs/drift.md`, "Precision weights and the width
in z" and "Wavy drift where there is none".

## Set-up

* **Structure**: 60 filaments 3 µm long, tilted through 600 nm of z
  (`filaments3d.yaml`), at 0.01-0.08 labels per nm.
* **Blinking and precision**: 4 blinks per label over 20 000 frames, 1500
  photons per blink (spread 750) on 40 photons of background: median
  precision 14 nm laterally and 41 nm in z (the simulation's z is three
  times the lateral), close to the clathrin dataset of docs/drift.md.
* **Drift**: slow sinusoids and a creep, 90-150 nm over the acquisition
  (`truth_curve`); twice that for the stress case.
* **Junk**: *imprecise* -- a fraction given 20-40 nm laterally and 150-300 nm
  in z, recorded and moved so; *pinned* -- out-of-focus fits held at
  z = ±400 nm, which do not follow the z drift, recorded as imprecise.
* **Estimate**: `DriftSettings()` defaults (merged blinks, spline with 2000
  frame knots, target sigma 30 nm, max drift 300 nm) unless a row says
  otherwise; the error is the rms over frames of the curve against the
  drift put in, both made zero mean, root-mean-squared over 3-4 seeds.

Run with smappy installed: `python studies/drift_weights/study.py weights`
(about 5 minutes), `target` (4), `driftless` (5).
