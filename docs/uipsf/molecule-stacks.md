# Molecules on the coverslip in a z stack

A standard sample for PSF calibration beside beads and in situ: single dye molecules stuck to the coverslip in
blinking buffer, imaged through repeated z sweeps of the stage.  The PSF calibration window's sample
*molecules on the coverslip in a z stack*, with the pupil model or the elliptical Gaussian, in single channel or
dual colour.  The user's decision (2 November 2026), from the prototype `benchmarks/uipsf/real/m5_molecules.py`
on M5's AF647 (`benchmarks/uipsf/results/m5_molecules.jsonl`).

## Why

Every bead model carries the bead: a 40 or 100 nm sphere convolved along stage z, its dye in polystyrene up to
its diameter above the glass, and a supercritical (SAF) ring that in buffer did not fit (`m5-media.md`); 40 and
100 nm beads learn different γ and σ (`architecture.md`, "The minimal model").  Molecules on the glass have none
of it: points at h ≈ 0, which is what a single-molecule fit needs, and what tests the SAF model without a bead.

## The design

* **The window's sample comes first** (`psf_methods.SAMPLES`), and narrows the method box
  (`SAMPLE_METHODS`): beads -- bead average, pupil model, elliptical Gaussian; molecules on the coverslip --
  pupil model, elliptical Gaussian; in situ -- the pupil model (in situ).  The mode box offers the modes the
  method has (`for_mode`).  (sample, method, mode) picks one entry; no entry asks which sample it is.  In situ is
  no longer a method of its own; its entry keeps its key, ``insitu``, and its label says *(in situ)*: its spline
  is valid at the imaging focus, z the height in the sample.  The new entries' keys are ``molecules`` and
  ``molecules elliptical``.
* **smappy reads, uipsf learns.**  `smappy.calibrate.molecule_stack` reads the files (NDTiff and OME-TIFF through
  `smappy.io`, the stage z per frame from their metadata, sweeps split where z turns back), makes the width
  curve, detects with a filter matched to each frame's defocus and selects; `molecule_dual` the same on each half
  of a split chip, paired through the registration; `molecule_stack_run.run_molecule_stack` runs it all, Qt-free,
  as `insitu_run.run_insitu` does.  `uipsf.workflows.molecule_stack.learn_molecule_stack` learns in rounds.
* **What is reused from in situ**: the container (`InSituROIs`, `Units`) and its table of counts by state, the
  regions and the first Gaussian (`cut`, `gaussian`), the even draw (`quotas`), the in situ model-vs-data page.
  Not its detection (a sliding median over 21 frames, which in a sweep span 1 µm of defocus; DoG widths fixed;
  blinks linked across frames that z tells apart; a z proxy), and not its rounds
  (`insitu_rounds`): every step after its first round is there to find unknown heights -- starts from the widths,
  the coverslip's edge, a measured focus-per-height ratio, a scalar focus prior.  With z known, the rounds are
  learn, check z, drop what fails, learn again.  Nor its twin test: the stage's z fixes the sign.
* **What is a bead model's**: the learning (`learn_beads`: stage 1 to order 4, the beads' rejection, stage 2),
  and the result -- a spline at the coverslip against defocus, listed for Spline 3D's *auto*, its model file
  beside it for Spline 3D's focus model; a dual calibration for Spline 3D 2C.

## The geometry (`uipsf.model.CoverslipStackGeometry`)

A detection in frame k of sweep s at chip position (u, v) µm from the field's centre is at defocus
z_k − f_s − (a·u + b·v) + δz_e: z_k the stage's nominal z in SMAPpy's sign, f_s a focus per sweep, (a, b) the
focal plane's slope, δz_e the molecule's own offset from the plane, held at 0 while learning (a prior of 0.5 nm)
and freed by the z check.  Every molecule is at one height h above the glass (``geometry.height``), its z in
the sample: a property of the sample, never of the optics, learnt from the glass (0.5 nm), bounded to (0, 200)
nm, held at a value under *more*.

**The SAF sees only h** -- not the plane's slope, the sweep's focus or a molecule's δz, which are all the
defocus's, through the immersion.  And the height enters through the SAF's decay alone, exp(−2πκh)
(`VectorPSF.saf_only_depth`): the propagating phase Re k_z,med·h a molecule at h carries is every molecule's,
and the pupil's phase and the foci take it up.  With the whole physics of the depth instead, h took up the sharp
phase feature at the critical angle, which the Zernikes cannot hold, and came to 27 nm with γ 0.40 on M5's
AF647; with the SAF alone it is 10 nm and γ 0.448, the prototype's (the user's decision, 3 November 2026).

**The model saved is the objective's.**  Its pupil is P_obj = P_Z·exp(−2πi Re k_z,med·h): the learnt Zernike
pupil with the phase at h taken out, kept as the Zernikes and h (``phase_reference_nm``,
`VectorPSF.for_emitters`).  Drawn for an emitter at any height z, the depth term is then the whole physics of
z, exp(2πi k_z,med·z): the SAF and the propagating phase both z's own, one height parameter per molecule, and
at z = h exactly the learnt model.  h is never fitted again and never set to 0: dropped, the phase at h would
be counted twice, most of it a z offset of about h.  Bead and in situ models have no reference; theirs is the
Zernike pupil.  What the split assumes: the phase near the critical angle is the objective's, where its edge
errors are and low Zernikes describe them poorly; the SAF's brightness is the sample's height.

The simulations (`benchmarks/uipsf/real` scratch, 3 November 2026; molecules at 8 nm, γ 0.5): clean, both
the physics and the SAF alone recover γ 0.53 and h 3-5 nm; unmodelled z offsets of ±15 nm per molecule double
h in both, γ kept; a 3 % error of the stage's z scale takes γ to 0.40 in both; a focal plane held flat against
a real slope of (20, −10) nm/µm breaks both and shows, a 22 nm bend.

## The z check

Every molecule learnt from, localised again on the learnt model with its z free: its offset from the plane
against its CRLB.  The median per 100 nm of defocus is the **bend**; the offsets' robust spread about it, in
CRLBs, the **scatter**.  A molecule farther off than max(3 CRLB, 100 nm) is dropped from the next round.

## Memory and time

At chunk 256 the prototype held 31 GB on an M-series GPU: about 120 MB per molecule of a 25-pixel region, a tenth
of what the solver's own cap assumes (`HOST_BYTES_PER_VALUE`).  The chunk is half the free memory at that rate,
8 to 256 (`molecule_chunk`): 84 on a 34 GB laptop with 20 GB free.  A run of 4000 molecules took 12.9 min over
three rounds there, the first most of it.

## M5's AF647, the prototype against the method

The five 3D files of 2 November 2026 (70 sweeps), minimal pupil to order 8, the pixel's blur under σ, 4000
molecules, 11 min:

| | prototype, ring height, round 2 | the method, round 2 | round 3 |
|---|---|---|---|
| Z5 / Z6 / Z11 / Z22 (nm) | −28.55 / 66.58 / 57.63 / 36.02 | −28.62 / 66.40 / 57.54 / 36.18 | −28.58 / 66.35 / 57.62 / 36.23 |
| γ | 0.456 | 0.451 | 0.448 ± 0.004 |
| height (nm) | 5 held + 4.2 ring | 11.0 | 10.9 ± 1.2 |
| σ (nm) | 0 | 0 | 0 |
| z bend, rms over ±600 nm | 5.03 | 7.17 | 6.92 |
| z scatter / CRLB | 1.75 | 1.80 | 1.81 |
| the +900 / +1000 nm bins | 38 / 142 nm | −5 / 22 nm | −5 / 23 nm |

The Zernikes, γ, σ and the height agree.  The bend's differences are the noise of measuring it: bootstrapped over
the molecules, its rms over ±600 nm is ±1.0 nm, and from the medians' noise alone (110-160 molecules a bin) a flat
bend shows 3 nm; another draw of the same code gave 6.5 nm (γ 0.441, h 11.4 nm), the prototype's detection 5.3 and
the prototype 5.0.  What is real is in all of them: a dip of about −10 nm around focus, −100 to +100 nm (−9.9 ±
1.4 and −9.5 ± 1.6 nm), the model's mismatch there, not the selection's.  The +900 / +1000 nm bins differ between
the detections (−5 / 23 nm here, 42 / 130 on the prototype's), where the CRLB is 60 nm.  With the whole physics
of the depth instead of the SAF alone, the same data gave h 27 nm and γ 0.40.

## 4Pi

Not yet: there is no real 4Pi molecule data, and the 4Pi model (`uipsf.model.FourPiModel`) has a cavity phase
per bead rather than a geometry; a variant with a focus and a cavity phase per sweep is the next step.
