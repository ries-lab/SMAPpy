"""Bead rejection (plan §5.7, M3): the rule on given scores (fast), the whole loop on simulated beads with
planted outliers (slow: about 40 s per calibration on 4 cores)."""
import numpy as np
import pytest
import torch

from outliers import OUTLIERS, bead_set
from uipsf.workflows import RejectionSettings, decide, learn_beads
from uipsf.workflows.rejection import held_out


class Scores:
    NAMES = ("deviance", "photons", "xy_offset_px", "masked")

    def __init__(self, n=10, **overrides):
        rng = np.random.default_rng(0)
        self.deviance = 1 + 0.01 * rng.standard_normal(n)
        self.photons = np.abs(0.1 * rng.standard_normal(n))
        self.xy_offset_px = np.full(n, 0.3)
        self.masked = np.zeros(n)
        for k, (i, v) in overrides.items():
            getattr(self, k)[i] = v

    def row(self, e):
        return {n: float(getattr(self, n)[e]) for n in self.NAMES}


def test_each_score_rejects_its_own_outlier_and_says_why():
    s = Scores(deviance=(1, 1.4), photons=(3, np.log(3)), xy_offset_px=(4, 5.0), masked=(5, 0.05))
    t = decide(s, RejectionSettings())
    assert t.ids("rejected") == [1, 3, 4, 5]
    reasons = {r.bead: r.reason for r in t.rows}
    assert reasons[1].startswith("deviance 1.4 >")
    assert reasons[3].startswith("photons") and "xy offset" in reasons[4] and "masked" in reasons[5]


def test_a_tight_set_is_not_thinned_by_its_own_noise():
    # Deviances 1 ± 0.01: 3 MAD would be 1.03, but the floor keeps beads within 5 % of the median.
    t = decide(Scores(n=40, deviance=(0, 1.04)), RejectionSettings())
    assert t.ids("rejected") == []


def test_manual_decisions_win_and_held_out_beads_are_neither_judged_nor_used():
    s = Scores(deviance=(1, 1.4), photons=(8, 4.0))
    t = decide(s, RejectionSettings(), heldout=[7, 8], decisions={1: "include", 0: "exclude"})
    states = {r.bead: r.state for r in t.rows}
    assert states[1] == "used" and t.rows[1].reason.startswith("included by hand (deviance")
    assert states[0] == "excluded" and states[7] == "held out"
    assert states[8] == "held out" and t.rows[8].reason == ""          # in no fit: not judged
    assert len(t.used) == 7
    with pytest.raises(ValueError, match="include"):
        decide(s, RejectionSettings(), decisions={2: "maybe"})


def test_with_few_beads_only_the_absolute_limits_apply():
    t = decide(Scores(n=4, deviance=(1, 3.0), masked=(2, 0.5)), RejectionSettings())
    assert t.ids("rejected") == [2] and "deviance" not in t.thresholds


def test_held_out_beads_are_spread_over_the_field():
    field = np.c_[np.linspace(-100, 100, 20), np.zeros(20)]
    ids = held_out(field, 20, 0.2)
    assert len(ids) == 4
    radius = np.abs(field[ids, 0])
    assert radius.min() < 30 and radius.max() > 60
    assert held_out(None, 20, 0.0) == []


@pytest.fixture(scope="module")
def calibrated():
    model, batch, values = bead_set()
    return learn_beads(model, batch, heldout=0.0, first_order=3), model, batch, values


# The 300 nm jump shows in the deviance of the stack fit (1.42 against 0.99–1.08): no localisation needed.
EXPECTED = {"double": "deviance", "jump": "deviance", "bright": "photons", "saturated": "masked"}


@pytest.mark.slow
def test_planted_outliers_are_rejected_for_the_right_reason_and_the_fit_reaches_its_crlb(calibrated):
    cal, model, batch, values = calibrated
    rows = {r.bead: r for r in cal.beads.rows}
    for kind, bead in OUTLIERS.items():
        assert rows[bead].state == "rejected", kind
        assert EXPECTED[kind] in rows[bead].reason, (kind, rows[bead].reason)
    assert sorted(cal.beads.used) == list(range(len(OUTLIERS), batch.n_emitters))
    err = cal.result.params.physical()["pupil.phase"] - values["pupil.phase"]
    crlb = cal.result.std["pupil.phase"]
    assert err.pow(2).mean().sqrt() <= 1.5 * crlb.pow(2).mean().sqrt()
    # The used beads fit the final model; the rejected ones were in no final fit and have no final scores.
    assert max(rows[b].final["deviance"] for b in cal.beads.used) < 1.1
    assert all(not rows[b].final for b in cal.beads.ids("rejected"))


@pytest.mark.slow
def test_matched_beads_lose_at_most_one_in_twenty():
    model, batch, _ = bead_set(n_good=20, seed=3, outliers=False)
    cal = learn_beads(model, batch, heldout=0.0, first_order=3)
    assert len(cal.beads.ids("rejected")) <= 1, [r.reason for r in cal.beads.rows if r.reason]


@pytest.mark.slow
def test_a_warm_recalculate_with_manual_decisions_equals_a_cold_fit_of_the_same_beads(calibrated):
    cal, model, batch, _ = calibrated
    decisions = {OUTLIERS["saturated"]: "include", len(OUTLIERS): "exclude"}
    warm = learn_beads(model, batch, decisions=decisions, warm_start=cal)
    cold = learn_beads(model, batch, decisions=decisions, heldout=0.0, first_order=3)
    assert warm.first is None and warm.beads.used == cold.beads.used
    assert OUTLIERS["saturated"] in warm.beads.used and len(OUTLIERS) not in warm.beads.used
    a, b = warm.result.params.physical(), cold.result.params.physical()
    for k in ("pupil.phase", "pupil.magnitude", "psf.sigma"):
        assert ((a[k] - b[k]).double().abs() / cold.result.std[k]).max() < 0.1, k
    assert torch.allclose(a["emitter.xyz"], b["emitter.xyz"], atol=1.0)
