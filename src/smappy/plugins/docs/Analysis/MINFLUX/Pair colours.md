---
version: "1"
covers: [smappy.pairing.trace_intervals, smappy.pairing.pair_traces, smappy.pairing.pair_offsets, smappy.plugins.pair_colours.PairColours.run]
---

## What it does

A [MINFLUX](plugin:Panels/Glossary) measurement can run two *threads* --
two sequences, one per dye, interleaved in time -- and then localizes a
molecule carrying both labels in both, or a motor and its cargo at the same
time.  Every localization says which thread measured it (`thi`), but not
which [trace](plugin:Panels/Glossary) of the other thread belongs with it.
This plugin finds out: a trace of thread 0 and a trace of thread 1
measured at the same time are a pair.

Every localization gets a `pair_id`: the same number for both traces of a
pair, a trace without a partner its own.  Regions made from `pair_id`
(*Segment regions*, *from an id column*; or any per-region step) are then
one track in one colour or both, which is what step finding and the trace
inspector of a two-colour data set need.

In the pairs it also measures the **offset between the colours**: where
thread 1 puts a molecule against where thread 0 puts it.  Per pair, over
time and over the field this is the quality check SMAP's DCoffset showed;
pooled, it is kept as a named transformation, "thi 1 onto thi 0", that
*Transform* applies to thread 1 to put it onto thread 0.  It needs a table
with `tid`, `thi` and `time_s` (a MINFLUX file of two threads) and reads
the localizations the layer's filter keeps.

## How it works

```figure-setup
from smappy import pairing as P

rng = np.random.default_rng(0)
offset = np.array([8.0, -5.0])          # thread 1 against thread 0, nm
rows, t = [], 0.0
for k in range(40):
    pos = rng.uniform(0, 5000, 2)
    for j in range(40):
        th = j % 2                        # the threads alternate in time
        p = pos + (offset if th else 0) + rng.normal(0, 3, 2)
        rows.append((2 * k + 1 + th, th, t + j * 1e-3, p[0], p[1]))
    t += 0.1
a = np.array(rows)
pid = P.pair_traces(a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2])
off = P.pair_offsets(pid, a[:, 1].astype(int), a[:, 2], a[:, 3], a[:, 4])
```

**1. Each trace's time span.**  For every trace, per file and thread, the
time of its first and last localization.

**2. Candidates and pairs.**  A trace of thread 0 and one of thread 1 whose
spans overlap (by at least *overlap at least*) are candidates.  The pairs
are taken one to one, the longest overlap first, so a thread-0 trace
measured beside two thread-1 traces takes the one it was measured with
longer.  The pairs and the traces left alone are numbered by when they
start.

**3. The offset per pair.**  Each localization of thread 1 is matched with
the thread-0 localization of the same pair nearest to it in time (within
*match within*), and the pair's offset is the median of their differences
in x and y.  The median keeps a few bad matches from moving it.

```figure Simulated molecules seen in both threads, thread 1 shifted by (8, −5) nm, each localization scattered by 3 nm: the offset measured in each pair, and their median (cross), which comes back as the shift simulated.
fig.set_size_inches(4.0, 3.2)
ax = fig.subplots()
ax.plot(off.dx, off.dy, "o", color="0.3", ms=4)
dx, dy = off.pooled()
ax.plot([dx], [dy], "x", color="tab:red", ms=10)
ax.plot([8.0], [-5.0], "+", color="k", ms=12)
ax.set_xlabel("dx (nm)")
ax.set_ylabel("dy (nm)")
ax.set_aspect("equal", adjustable="datalim")
ax.set_title(f"median ({dx:.1f}, {dy:.1f}) nm, truth (8, −5)", fontsize=8)
```

**4. The offset over time and over the field.**  Drawn per pair against
when it was measured and where: a drift between the colours shows as a
trend in time, a chromatic difference across the field as arrows that
change with position.

**5. The transformation.**  The median over the pairs is kept as a shift,
under *transformation*'s name, for *Transform* to apply to thread 1.

## In detail

**Matching in time.**  Within a pair, the thread-0 localizations sorted by
time; for each thread-1 localization the one before it and the one after
it in time, the nearer taken.  Matches further apart than *match within*
are left out; a pair with none gives no offset.

**Pairing is per file.**  Traces of different files are never paired.  A
trace's id is numbered per thread by Imspector, so the pairing keys on
file, thread and `tid` together.

**`pair_id`** is written as a recipe with the rule *max*, so a merged
trace keeps its pair.  It is -1 for a localization outside the layer's
filter or without a trace.

The region table of `pair_id`, "per pair_id: Pair colours", holds per
track whether it is a pair (`paired`), the offset (`dx_nm`, `dy_nm`) and
the localizations matched (`n_matched`).

## Parameters

### min_overlap_s
0 pairs any two traces measured together at all.  Raise it where many
molecules blink at once and short chance overlaps would pair the wrong
traces.

### max_dt_ms
A few times the time between localizations of one thread: the threads
alternate, so a localization's partner is one or two dwells away.

### name
Use a name per measurement when several are open, so each keeps its own.

## Output

`pair_id` on the localizations; a region table of `pair_id`; the named
transformation; three panels -- the offset of each pair with their median,
the offset over time, and the offsets as arrows over the field.  A good
two-colour measurement shows the offsets in a tight cloud a few nanometres
across, flat over time and the same over the field.

## Differences from SMAP

Ported in idea from SMAP's MINFLUX plugins Segment_cotracks and
DCoffset_Beads_ROIs (Ries 2020).

* **Pairing**: SMAP takes for each thread-0 trace in a site the most
  frequent thread-1 trace within its time span; here every trace is paired
  at once, one to one, by the longest overlap, without sites.
* **Offsets**: SMAP's *fast* mode (per site, a mean and standard deviation
  over the pairs of localizations nearest in time) is the per-pair median
  here; its *drift* and *FoV* modes are the panels over time and the field.
  The bead-scan grid and the tail mode are not ported.
* **The result** is a `pair_id` and a transformation for *Transform*; SMAP
  showed figures only.

## References

* Ries J (2020).  SMAP: a modular super-resolution microscopy analysis
  platform for SMLM data.  *Nat Methods* 17, 870-872.
  [doi:10.1038/s41592-020-0938-1](https://doi.org/10.1038/s41592-020-0938-1)
