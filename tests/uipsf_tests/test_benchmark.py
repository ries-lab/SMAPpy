"""The review's synthetic bead benchmark (plan §9 M2): bright, dim and very dim beads, fitted with the
default model and procedure and no settings changed.  Slow tier: about 1.5 minutes per fit on 4 cores.

The plan asked for dim beads to reach z ≤ 5 nm and a phase RMS ≤ 0.015 rad, the review's best tuned v1
run.  The phase target holds.  For z the bound is the CRLB, about 7 nm per dim bead in this scenario:
no unbiased fit can promise 5 nm on six beads, so the test holds the spread of the z errors to the
CRLB instead (and v1's default run, at 61 nm, to shame).
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "uipsf", "synthetic"))
import beads  # noqa: E402

pytestmark = [pytest.mark.long, pytest.mark.timeout(3600)]

_cache = {}


def _run(name, scale=None):
    key = (name, tuple(sorted((scale or {}).items())))
    if key not in _cache:
        _cache[key] = beads.run(name, scale)
    return _cache[key]


@pytest.mark.parametrize("name", ["bright", "dim", "very dim"])
def test_the_review_beads_are_learned_to_their_crlb_without_settings(name):
    row, result, _ = _run(name)
    print(row)
    assert result.converged, row["stop"]
    assert row["iterations"] <= 50
    assert row["rms_phase_err_rad"] <= 1.5 * row["rms_phase_crlb_rad"]
    assert row["z_err_nm"] <= 1.5 * row["z_crlb_nm"]
    assert abs(row["mean_deviance"] - 1) < 0.02
    assert abs(row["heldout_deviance"] - 1) < 0.02
    if name == "dim":
        assert row["rms_phase_err_rad"] <= 0.015                       # the review's best tuned v1 run


def test_the_dim_beads_give_the_same_fit_in_any_units():
    # The review's scalings (bead_fit_scaling_lowsnr.py) moved v1's z error from 61 to 5 nm.
    _, ref, _ = _run("dim")
    _, res, _ = _run("dim", {"emitter.photons": 1e-1, "emitter.background": 10.0, "pupil.phase": 0.2,
                             "pupil.magnitude": 5.0})
    a, b = ref.params.physical(), res.params.physical()
    for k, sd in ref.std.items():
        assert ((a[k] - b[k]).double().abs() / sd).max() < 0.05, k
    # Accepted steps: the rejected ones at the end probe the float32 noise floor of the objective.
    accepted = [sum(1 for h in r.history if h.get("accepted")) for r in (ref, res)]
    assert abs(accepted[0] - accepted[1]) <= 2, accepted
    assert res.converged
    assert torch.isfinite(torch.tensor(res.objective))
