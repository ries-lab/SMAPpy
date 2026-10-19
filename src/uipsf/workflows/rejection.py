"""Which beads a calibration uses, and why: automatic states, manual decisions, a table (plan §5.7).

**The rule.**  A score counts against a bead when it exceeds a robust
threshold, median + k·1.4826·MAD over the beads in play, or an absolute
floor above the median where the spread is tiny, so that noise alone does
not reject a bead of an excellent set.  Relative thresholds adapt to data
that sit above a deviance of 1 for every bead (model mismatch, a camera's
numbers off); v1 used ratios to the median the same way, with fixed factors
(3 for its MSEs, 1.5 for photons), and SMAPpy mean + 1.5 sd of a shape
score.  Two scores have absolute limits instead: a bead far from its ROI's
centre (a wrong centre or a neighbour), and a bead with too many saturated
pixels.  The relative rules need at least `min_beads` in play; with fewer,
only the absolute ones apply.

**States.**  Every bead ends up *used*, *held out*, *rejected* (with the
reason: the score, its value and its threshold) or *excluded* (by hand).
Scores come from the learning fit (`scoring`), so beads in no fit -- held
out, excluded -- have none (NaN) and are never judged.  Manual decisions
always win: *include* uses a bead whatever its scores,
*exclude* leaves it out.  The rule lives here and only here, so SMAPpy's
window, the CLI and a notebook show the same states for the same beads.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RejectionSettings:
    k: float = 3.0                         # robust standard deviations above the median
    deviance_floor: float = 0.05           # a bead's deviance may exceed the median by this much regardless
    photons_floor: float = float(np.log(1.5))   # a factor 1.5 from the median is never an outlier
    max_xy_offset_px: float = 3.0
    max_masked: float = 0.01               # fraction of a bead's pixels
    min_beads: int = 5
    enabled: bool = True


RELATIVE = {"deviance": "deviance_floor", "photons": "photons_floor"}
UNITS = {"deviance": "", "photons": "", "xy_offset_px": " px", "masked": ""}


def thresholds(scores, pool: np.ndarray, s: RejectionSettings) -> dict[str, float]:
    """Each score's threshold, from the beads in ``pool`` (a boolean mask)."""
    out = {"xy_offset_px": s.max_xy_offset_px, "masked": s.max_masked}
    if pool.sum() >= s.min_beads:
        for name, floor in RELATIVE.items():
            v = getattr(scores, name)[pool]
            v = v[np.isfinite(v)]
            med = float(np.median(v))
            spread = 1.4826 * float(np.median(np.abs(v - med)))
            floor_value = getattr(s, floor) * (med if name == "deviance" else 1.0)
            out[name] = med + max(s.k * spread, floor_value)
    return out


def _reason(name, value, limit):
    label = {"xy_offset_px": "xy offset", "masked": "masked pixels"}.get(name, name)
    return f"{label} {value:.3g}{UNITS[name]} > {limit:.3g}{UNITS[name]}"


@dataclass
class BeadRow:
    bead: int
    stack: int | None
    field_xy: tuple[float, float] | None
    scores: dict                           # the scores the decision was made on
    state: str                             # "used", "held out", "rejected", "excluded"
    reason: str = ""                       # why rejected (or "excluded by hand", "included by hand")
    manual: str | None = None              # None, "include", "exclude"
    final: dict = field(default_factory=dict)   # deviance and photons in the final fit (used beads)


@dataclass
class BeadTable:
    rows: list[BeadRow]
    thresholds: dict

    def ids(self, state: str) -> list[int]:
        return [r.bead for r in self.rows if r.state == state]

    @property
    def used(self) -> list[int]:
        return self.ids("used")

    def as_dicts(self) -> list[dict]:
        """One flat dict per bead, for a table widget or a CSV."""
        out = []
        for r in self.rows:
            d = {"bead": r.bead, "stack": r.stack, "state": r.state, "reason": r.reason, "manual": r.manual}
            if r.field_xy is not None:
                d["field_x"], d["field_y"] = r.field_xy
            d.update(r.scores)
            d.update({f"final_{k}": v for k, v in r.final.items()})
            out.append(d)
        return out


def decide(scores, settings: RejectionSettings, heldout=(), decisions: dict | None = None,
           stack=None, field_xy=None) -> BeadTable:
    """The table of states for every bead: manual decisions, then held out, then the rules."""
    E = len(scores.deviance)
    decisions = decisions or {}
    for b, d in decisions.items():
        if d not in ("include", "exclude"):
            raise ValueError(f"bead {b}: decision must be 'include' or 'exclude', not {d!r}")
    heldout = set(int(b) for b in heldout)
    pool = np.array([e not in heldout and decisions.get(e) != "exclude" for e in range(E)])
    limits = thresholds(scores, pool, settings) if settings.enabled else {}
    rows = []
    for e in range(E):
        sc = scores.row(e)
        manual = decisions.get(e)
        failed = [(n, sc[n], lim) for n, lim in limits.items() if np.isfinite(sc[n]) and sc[n] > lim]
        if manual == "exclude":
            state, reason = "excluded", "excluded by hand"
        elif manual == "include":
            state = "used"
            reason = "included by hand" + (f" ({_reason(*failed[0])})" if failed else "")
        elif e in heldout:                 # in no fit, so not scored and not judged
            state, reason = "held out", ""
        elif failed:
            state = "rejected"
            reason = "; ".join(_reason(*f) for f in failed)
        else:
            state, reason = "used", ""
        rows.append(BeadRow(e, None if stack is None else int(stack[e]),
                            None if field_xy is None else tuple(float(v) for v in field_xy[e]),
                            sc, state, reason, manual))
    return BeadTable(rows, limits)


def held_out(field_xy, n: int, fraction: float) -> list[int]:
    """Every 1/fraction-th bead in order of distance from the beads' centre: spread over the field."""
    if fraction <= 0 or n < 2:
        return []
    every = max(2, int(round(1 / fraction)))
    if field_xy is None:
        order = np.arange(n)
    else:
        f = np.asarray(field_xy, float)
        order = np.argsort(np.hypot(*(f - f.mean(0)).T), kind="stable")
    return sorted(int(b) for b in order[every // 2::every])
