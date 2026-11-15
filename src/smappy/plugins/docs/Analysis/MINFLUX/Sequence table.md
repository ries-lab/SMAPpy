---
version: "1"
covers: [smappy.plugins.sequence_table.measured_sequences, smappy.plugins.sequence_table.rows_of, smappy.plugins.sequence_table.differing, smappy.plugins.sequence_table.SequenceTable.run, smappy.io.minflux.iteration_table]
---

## What it does

Shows the MINFLUX *sequence* of every open measurement -- what the
instrument was told to do in each iteration: the beam pattern and its size
(`patGeoFactor`), the dwell time, the photon limit an iteration needs to
pass, the cfr limit, the background threshold, the laser and its power --
one line per iteration, a table per measurement and thread.  Above them it
lists what differs between the measurements, so two measurements that
behave differently can be checked for a different sequence first.

The sequences are those the reader kept when the `.msr` file was opened;
exports (`.npy`, `.json`) carry none.  abberior's wiki explains each
parameter ("MINFLUX Sequence").

## How it works

**1. The sequences.**  Every open file's measurements, and for a
measurement of several threads, each thread's own sequence.

**2. What differs.**  Every global parameter and every parameter of every
iteration is compared across the sequences; those not the same in all are
listed with their values side by side.  Numbers that differ only in their
seventh digit count as the same, and the instrument's internals are left
out: the estimator's residuals (`_lemd`, `_lemn`), the PSF files, the DAC
values the powers map to, the live view's settings.

**3. The tables.**  A line per iteration with *parameters* as columns.

## Parameters

### keys
Any parameter of an iteration, as the tables' headers or the differences
name it: `patGeoFactor`, `Mode.pattern`, `_excitation.power`, `phtLimit`.

## Output

The text: the differences, then a table per measurement and thread.  The
rows -- one per iteration of every sequence, with its file, measurement,
thread and the parameters -- are in the result's data for a script to
write as csv.
