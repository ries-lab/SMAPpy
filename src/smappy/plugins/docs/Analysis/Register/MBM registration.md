---
version: "2"
covers: [smappy.mbm_registration.start_positions, smappy.mbm_registration.match, smappy.mbm_registration.kabsch_2d, smappy.mbm_registration.register, smappy.plugins.mbm_registration.MBMRegistration.run, smappy.plugins.mbm_registration.draw_registration]
---

## What it does

Puts one [MINFLUX](plugin:Panels/Glossary) measurement onto another of the
same field: the second round of a sequential labelling (Exchange-PAINT, a
second antibody after a wash), or two channels measured one after the
other.  Between the two the stage may have been moved and the sample may
have settled, so the tables need not line up.  The
[MBM](plugin:Panels/Glossary) beads are fixed to the coverslip and recorded
in both measurements, so the map between the measurements is the map
between their beads.

It reads the beads from each file's own metadata -- no bead file needs to
be open -- finds which bead of one is which of the other, and fits the
shift between them, or the shift and a rotation from three beads on.  What
it hands back is a named transformation that
[Transform](plugin:Analysis/Process/Transform) applies to the target's
file, as it applies what [Register layers](plugin:Analysis/Register/Register layers)
measures.  Run it after any MBM drift correction: both put a measurement in
the frame of its beads at its start.

## How it works

```figure-setup
from smappy.simulate import simulate_bead_tracks
from smappy.mbm_registration import register, start_positions
from smappy.plugins.mbm_registration import draw_registration

times = np.arange(0.0, 200.0, 2.0)
first = simulate_bead_tracks(times, n_beads=6, field_nm=(-15000, -15000, 15000, 15000),
                             seed=3)
a = np.radians(0.4)
rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
shift = np.array([1200.0, -650.0])
# the second measurement sees the same beads, four of them, rotated and shifted
xy = np.column_stack([first["x_nm"], first["y_nm"]])
seen = np.asarray(first["tid"]) <= 4
moved = (rotation.T @ (xy[seen] - shift).T).T
rng = np.random.default_rng(1)
moved += rng.normal(0, 1.0, moved.shape)
reference = start_positions(first["time_s"], xy, first["tid"])
target = start_positions(first["time_s"][seen], moved, first["tid"][seen])
found = register(reference, target)
```

**1. Each bead's position** in each measurement is where it was when the
measurement began: a corrected measurement is in the instrument's frame at
its start.  Rather than its first points alone, which carry their own
noise, the beads' drift (the curve *MBM drift correction* fits) is
subtracted from all its points and the median is taken: the bead's place
at the start, measured with every point.

**2. Which bead is which** is found from the positions, not the names:
Imspector reuses its bead names and slots, and two beads 20 µm apart have
had one name in two measurements of one file.  Every pair of a bead of the
reference and a bead of the target suggests a shift; the shift the most
pairs agree on, within *tolerance*, is taken, and each bead is paired with
the one that fits it best under that shift.  A bead seen in only one
measurement stays unpaired.

**3. The map** is a translation from one or two beads, and a rigid map --
a rotation and a translation -- from three or more, fitted by least squares.
z, where both measurements are 3D, is shifted by the beads' mean.

```figure Six simulated beads in the reference (circles), four of them seen again in the target, rotated by 0.4° and shifted by 1.2 µm (grey), and mapped back (red): the pairs are found though the target saw two beads fewer, and the map puts each bead back on its partner to the beads' noise.
fig.set_size_inches(5.0, 4.4)
draw_registration(fig.subplots(), found)
```

## In detail

The rigid fit is the Kabsch (1976) solution in x and y: the rotation $R$
and shift $t$ that minimise $\sum_i |R\,p_i + t - q_i|^2$ over the target's
beads $p_i$ and their reference partners $q_i$, from the singular value
decomposition of the beads' cross-covariance about their centres, with the
sign that excludes a mirror.  No scale is fitted: the beads lie on one
coverslip seen through one objective.

The frame is the one *MBM drift correction* corrects into, zero at the
first bead point.  That point carries the beads' noise, so the frame is
offset by it as a constant -- the same for every localization of the
measurement, so the registration of two measurements corrected that way is
exact.  Imspector's own correction is referenced to its beads' first point
too, but per bead and with its own choice of beads: for exact work run MBM
drift correction on both measurements first.

Matching: every reference and target bead within *largest shift* of each
other is a candidate; the candidate shift with the most others within
*tolerance* of it wins, their median is the shift, and pairs within
*tolerance* of it are taken closest first, each bead once.  Beads are tens
of micrometres apart and the shift a micrometre or two, so the vote is not
close; were two beads closer than *tolerance* to each other, lower it.

The text lists each pair and its residual after the map.  A residual well
above the others' is a bead that moved between the measurements (or a
wrong pair): it is worth re-running with *fit* on *translation*, or after
removing that measurement's bad beads in MBM drift correction.  Untried on
real sequential measurements so far (NOTES.md).

## Parameters

### reference
The measurement that stays where it is; every measurement with MBM beads in
the open files is offered, by file and measurement.

### target
The one that moves; *Transform* then applies the map to its file.

### fit
*auto* is right in nearly every case.  Force *translation* when three beads
lie on a line, where a rotation is poorly determined.

### tolerance_nm
Above the beads' scatter and the expected error of the shift, below the
distance between any two beads.

## Output

The text gives the map -- the beads' mean shift and, for a rigid map, the
rotation -- with its residual, each pair with its own residual, how many
target beads found no partner, and the name the transformation is kept
under.  The figure shows the reference beads, the target's before and after
the map, and the pairs.  A good registration has residuals of a few
nanometres, the beads' own noise.

## References

* Kabsch W. A solution for the best rotation to relate two sets of vectors.
  *Acta Cryst* A32, 922 (1976).
  [doi:10.1107/S0567739476001873](https://doi.org/10.1107/S0567739476001873)
