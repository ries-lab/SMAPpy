import numpy as np

from smappy.netfit import sigma


def _draw(n=60_000, lie=1.0, seed=0):
    """Errors drawn from true sigmas; claims ``lie`` times the truth (1: honest)."""
    rng = np.random.default_rng(seed)
    true = rng.uniform(5, 60, (n, 3))
    err = rng.normal(0, true)
    return err, true * lie, rng.uniform(-800, 800, n), rng.uniform(500, 20000, n)


def test_an_honest_claim_reads_one_and_covers_68_and_95_percent():
    err, claim, z, ph = _draw()
    s = sigma.scores(err, claim, z, ph)
    for axis in ("lat", "ax"):
        assert abs(s[axis]["width"] - 1) < 0.02
        assert abs(s[axis]["coverage_1"] - 0.683) < 0.01 and abs(s[axis]["coverage_2"] - 0.954) < 0.01
        rel = s[axis]["reliability"]
        assert all(abs(r["error_rms_nm"] / r["claim_rms_nm"] - 1) < 0.05 for r in rel)


def test_a_claim_too_small_by_half_reads_two_and_a_worse_likelihood():
    err, claim, z, ph = _draw()
    honest, small = sigma.scores(err, claim, z, ph), sigma.scores(err, claim * 0.5, z, ph)
    assert abs(small["lat"]["width"] - 2) < 0.05
    assert small["lat"]["nll"] > honest["lat"]["nll"] + 0.5


def test_the_recalibration_learnt_on_one_half_makes_the_other_honest():
    err, claim, z, ph = _draw(lie=1.0)
    claim = claim * np.where(np.abs(z) > 400, 0.5, 1.0)[:, None]      # too small away from the focus
    a = np.arange(len(z)) % 2 == 0
    scale = sigma.fit_scale(err[a], claim[a], z[a], ph[a])
    fixed = sigma.apply_scale(scale, claim[~a], z[~a], ph[~a])
    before, after = sigma.scores(err[~a], claim[~a], z[~a], ph[~a]), sigma.scores(err[~a], fixed, z[~a], ph[~a])
    assert before["ax"]["by_abs_z"][3]["width"] > 1.8
    assert all(abs(b["width"] - 1) < 0.08 for b in after["ax"]["by_abs_z"] if b["width"] is not None)
