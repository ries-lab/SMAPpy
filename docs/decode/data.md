# DECODE in SMAPpy: the data

Nothing here is in git.  Everything lives under `$DECODE_SMLM_DATA`, by default
`~/.cache/decode_smlm/`, and the tests that need a part of it skip without it, as
`SMAPPY_TEST_CAL` does for calibrations.

## What is there

| folder | what | from |
|---|---|---|
| `experimental_data_workflow/` | DECODE's example package: U2OS microtubules on an Evolve 512 EMCCD (`frames.tif`, 20 000 frames of 180 x 179), the 11 bead stacks it was calibrated from (`beads/`), SMAP's spline (`spline_calibration_3dcal.mat`), the trained model and its `param_run.yaml` | DECODE's gateway (`gateway.yaml` of v0.10.2, EMBL's ownCloud, 1.35 GB) |
| `v0102_model/` | the original v0.10.2's localizations of the package's frames, the oracle of milestone (d) | made with `scripts/decode/` in the original's own environment |
| `pixelation/` | the pixelation study's models (B: SMAPpy's bead calibration; C: in situ), fits, figures and results | `benchmarks/decode/pixelation/` |
| `benchmark/` | (i)'s simulated sets, per calibration (`A/`, `B/`), with their fits and scores, and the reference runs' models (`models/`) | `benchmarks/decode/benchmark/` |
| `challenge/` | the SMLM Challenge 2016's astigmatic training sets MT0.N1/N2.LD/HD with their `activations.csv`, the bead z-stack, SMAPpy's calibration of it and the two training jobs | srm.epfl.ch, *challenge-3D-simulation* (downloaded 2026-11-11) |

The tests' own small files are in `tests/decode_tests/fixtures/` (in git: the renderer's and the
method's oracles, made from the original).

## The workstations

On a training host the port runs from a copy of `src/` (or an install); jobs go to
`~/smappy_training/` (the host's *folder there*), the benchmark's trainings to wherever their
script sent them (`~/decode_poc/jobs/` on the lab's H100).  The example package's movie need not
be there: a training simulates its frames.
