"""Half-split uncertainty: the calibration learnt again on halves of its beads, to see how much it moves.

The CRLB (`solve.result`) is the spread photon noise alone would give if the model were exact.  Beads
differ -- in size, in where they sit on the field, in a defect the rejection let through -- and a model
is never exact, so a calibration learnt from other beads of the same sample differs from it by more
than its CRLB says.  That difference is measured here directly: the used beads are split at random into
two halves, each half is calibrated on its own, and how far the two disagree is the uncertainty.

* **Cold, from the run's own start.**  Each half goes through the whole calibration as the run did --
  the same start (its *start from* model, or none), the same stages -- never from the full fit, which
  would hold every half in the full fit's minimum.  Started alike, halves that end apart show that the
  procedure does not land in one place, which is as much part of a realistic error as the beads.
* **Complementary pairs.**  A split gives two fits on disjoint beads, A and B.  Each has twice the
  variance of the full fit (half the beads), so their difference has four times it, and
  ``((A − B) / 2)²`` is an estimate of the full fit's variance from every split; the root of their mean
  over the K splits is reported (`half_spread`).  Its own accuracy is about ±1/√(2K): ±40 % for 3 splits,
  ±30 % for 5 -- enough for an error bar, and few beads make more splits worth less, since they overlap.
* **No rejection inside a half.**  The halves are drawn from the beads the full fit used, every one of a
  half pinned in by a decision (decisions always win, `rejection.decide`) and every other one out, and
  none held out: each half measures the same beads' scatter, never a rejection that went another way.
* **At least `MIN_BEADS` used beads.**  A half of four beads hardly determines a pupil at the usual order;
  its scatter would be the half's own instability, not the calibration's uncertainty.

What it cannot see: anything every half shares -- a wrong NA, index or wavelength, a model that cannot
represent the PSF.  Those bias all halves alike.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

#: fewer used beads and a half is too small to learn a pupil from
MIN_BEADS = 8


@dataclass
class HalfSplits:
    """K splits of the used beads into two halves, each half calibrated alone."""
    splits: list[tuple[list[int], list[int]]]          # per split, the bead ids of half A and of half B
    values: dict[str, np.ndarray]                      # shared parameter -> [K, 2, *shape], each half's value
    std: dict[str, np.ndarray]                         # shared parameter -> the full fit's sd, `half_spread`
    seed: int = 0

    @property
    def n_splits(self) -> int:
        return len(self.splits)


def split_beads(used, splits: int, seed: int = 0) -> list[tuple[list[int], list[int]]]:
    """``splits`` random splits of the bead ids ``used`` into two halves (the odd one out in B), each split
    different from the others where there are enough beads for that."""
    used = [int(b) for b in used]
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    for _ in range(100 * splits):
        if len(out) == splits:
            break
        order = rng.permutation(used)
        a = sorted(int(b) for b in order[:len(used) // 2])
        b = sorted(int(b) for b in order[len(used) // 2:])
        key = frozenset({frozenset(a), frozenset(b)})        # A | B and B | A are one split
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return out


def half_spread(values: np.ndarray) -> np.ndarray:
    """The full fit's standard deviation from complementary halves: ``values`` [K, 2, ...], each split's
    two halves; the root mean square over the splits of ``(A − B) / 2`` (module docstring)."""
    values = np.asarray(values, float)
    return np.sqrt(np.nanmean(((values[:, 0] - values[:, 1]) / 2) ** 2, axis=0))


def half_splits(batch, optics, settings, used, *, splits: int = 3, seed: int = 0, transforms=None,
                geometry: dict | None = None, start=None, noise=None, progress=None, cancel=None,
                device=None) -> HalfSplits | None:
    """Calibrate the beads ``used`` (a calibration's `BeadTable.used`) again, half of them at a time, in
    ``splits`` complementary pairs: `calibrate_beads` with the calibration's ``batch``, ``optics``,
    ``settings`` (its `BeadSettings`), ``transforms``, ``geometry``, ``start`` and ``noise``.  None if
    cancelled.  ``progress`` hears ``{"message", "fraction"}`` per half."""
    from ..workflows import calibrate_beads
    used = [int(b) for b in used]
    if len(used) < MIN_BEADS:
        raise ValueError(f"{len(used)} used beads: at least {MIN_BEADS} are needed to learn from halves of them")
    drawn = split_beads(used, splits, seed)
    settings = dataclasses.replace(settings, heldout=0.0)       # every bead of a half learns
    n_fits = 2 * len(drawn)
    report = progress or (lambda entry: None)
    values: dict[str, list] = {}
    for k, pair in enumerate(drawn):
        halves = []
        for h, beads in enumerate(pair):
            done = 2 * k + h
            report({"message": f"half {done + 1} of {n_fits}: {len(beads)} beads", "fraction": done / n_fits})
            decisions = {b: ("include" if b in beads else "exclude") for b in range(batch.n_emitters)}
            cal = calibrate_beads(batch, optics, settings, transforms=transforms, geometry=geometry, start=start,
                                  noise=noise, decisions=decisions, cancel=cancel, device=device,
                                  progress=_within(progress, done / n_fits, (done + 1) / n_fits))
            if cal.result.stop_reason == "cancelled":
                return None
            halves.append({n: v.detach().cpu().double().numpy() for n, v in cal.shared().items()})
        for n in halves[0]:
            values.setdefault(n, []).append([halves[0][n], halves[1][n]])
    report({"message": f"{n_fits} halves learnt", "fraction": 1.0})
    arrays = {n: np.asarray(v, float) for n, v in values.items()}
    return HalfSplits(drawn, arrays, {n: half_spread(v) for n, v in arrays.items()}, seed)


def _within(progress, start: float, end: float):
    """A half's own progress entries, their messages passed on and its fraction of the whole kept to its
    share: the solver's iterations say nothing of how far a half is, so only the halves count."""
    if progress is None:
        return None

    def tell(entry):
        message = entry.get("message")
        if message:
            progress({"message": message, "fraction": start})
    return tell
