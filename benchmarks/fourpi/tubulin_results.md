# The 2017-11-28 tubulin data (4Pi-SMS), single colour: results (2026-11-24)

What `tubulin_*.py` gave on Gabin's data in `~/.cache/uipsf/4pi_tubulin_2017-11-28`; the discussion is
`docs/4pi.md` §26.  Run from the repository's root with `PYTHONPATH=src`, in this order; each writes to
`smappy_p1/` beside the data.

| script | what |
| --- | --- |
| `tubulin_layout.py` | the registration from Cell06 -- 12 against the authors'; the camera's conversion |
| `tubulin_average.py` | the bead average of Cell03 -- 05, one rate against a rate per scan, and leaving one scan out |
| `tubulin_pupil.py 680 100 held` | uiPSF's pupil model of Cell03 -- 05, the maps held at the registration |
| `tubulin_fit.py MODEL` | Spline 4Pi and 4Pi z on Cell01's first 6000 frames |
| `tubulin_compare.py TABLE` | against the authors' raw fits and final table on the same frames |

**Registration** (86 bead pairs per channel, 0.06 -- 0.08 px rms): scales 1.0098/1.0099, 1.0101/1.0110,
0.9999/1.0013 and rotations 19.9, 13.6, −7.1 mrad; the authors' 1.0102/1.0098, 1.0102/1.0106, 0.9999/1.0013
and 19.3, 12.5, −7.0 mrad.

**Camera**: offset 102 ADU, read variance 9.9 ADU²; conversion 0.40 (median ratio) to 0.50 (20th percentile)
e⁻/ADU; 0.5 and offset 100 used.

**Models** (the beads' own period 291.5 nm; rates per scan 0.431, 0.425, 0.453 rad per 20 nm slice):

| model | period (nm) | objective (nats) | coherence | beads fitted back: z slope, scatter |
| --- | --- | --- | --- | --- |
| bead average, rate per scan | 291.5 | -- | (modulation 0.36 -- 0.49) | 1.00 -- 1.01, 1.3 -- 2.7 nm |
| pupil, 620 nm, 40 nm bead | 265.8 (warned) | 2 450 146 | 0.52 -- 0.69 | |
| pupil, 680 nm, 40 nm bead | 294.0 | 867 614 | 0.74 -- 0.89 | 0.98 -- 0.99, 1.9 -- 2.5 nm |
| pupil, 680 nm, 100 nm bead | 294.2 | 851 590 | 0.82 -- 0.99 | 0.98 -- 0.99, 2.1 -- 2.8 nm |

**Cell01, 6000 frames** (the authors: 320 285 raw fits, 90 850 in their final table; pairs within 0.5 px):

| model | one fit per molecule | ours | expected wrong fringes | theirs paired | z vs their final: spread, other fringe | x, y vs theirs (nm) | photons ours / theirs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bead average | no | 404 154 | 11.0 % | 89 % | 4.6 nm, 0.2 % | −0.6, −6.0 | 1.57 |
| pupil 680 / 40 | no | 404 233 | 8.4 % | 89 % | 4.6 nm, 0.1 % | −1.8, +0.5 | 1.63 |
| pupil 680 / 100 | yes | 369 582 | 7.2 % | 89 % | 4.4 nm, 0.1 % | −1.9, +0.5 | 1.60 |

Their z is counted the other way (compared with theirs turned) and is drift-corrected; their period, 302 nm,
makes it 3.5 -- 7 % longer.  Without one fit per molecule, 9 % of ours were a second fit within a pixel.
