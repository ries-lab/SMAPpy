"""A simulated MINFLUX measurement, written as an `.msr` (`simulate.minflux`,
`io.msr_write`): it opens as a measurement from the instrument does, and
the motors' steps, the dwells and the second molecules come back."""
import numpy as np

from smappy.io import msr
from smappy.simulate.minflux import motor_msr, simulate_attempts
from smappy.steps import autostepfinder, split_merge


def test_a_simulated_msr_opens_with_its_measurements_and_their_sequence(tmp_path):
    path = motor_msr(tmp_path / "kinesin.msr", n_events=600)
    found = msr.contents(path)
    assert [m.label for m in found.measurements] == ["kinesin 1000uM ATP",
                                                     "kinesin 20uM ATP"]
    locs, _ = msr.load(path, measurements=[1], every=True)
    meta = locs.metadata["minflux"]["measurements"][0]
    rows = meta["sequence"]["iterations"]
    assert [r["patGeoFactor"] for r in rows] == [0.79, 0.84, 0.42, 0.28]
    assert rows[2]["ccrLimit"] == 0.8 and meta["sequence"]["excitation_nm"] == 640.0
    assert set(np.unique(np.asarray(locs["final"]))) == {0, 1}


def test_a_simulated_motor_steps_by_its_step_size():
    rec = simulate_attempts(400, passed=(0.2, 0.95, 0.9), locs_per_burst=1500.0,
                            precision_nm=2.0, dwell_s=5e-4, step_nm=8.0,
                            step_dwell_s=0.03, seed=1)
    final = rec["fnl"] & rec["vld"]
    tid = rec["tid"][final]
    longest = np.bincount(tid).argmax()
    xy = rec["loc"][final][tid == longest][:, :2] * 1e9
    along = (xy - xy[0]) @ np.linalg.svd(xy - xy.mean(0))[2][0]
    along *= np.sign(along[-1])
    boundaries = split_merge(along, autostepfinder(along).boundaries, 8.0)
    levels = [along[a:b].mean() for a, b in zip(boundaries[:-1], boundaries[1:])]
    assert abs(np.median(np.diff(levels)) - 8.0) < 1.0


def test_two_molecules_at_once_double_the_photon_rate_and_raise_cfr():
    rec = simulate_attempts(3000, passed=(0.3, 0.9, 0.9), two_molecules=0.3,
                            two_cfr=0.6, cfr_limit=0.8, seed=2)
    final = rec["fnl"] & rec["vld"]
    cfr = rec["cfr"][(rec["itr"] == 2) & rec["vld"]].astype(float)
    two = cfr > 0.45
    assert 0.2 < two.mean() < 0.4
    efo_of = {t: e for t, e in zip(rec["tid"][final], rec["efo"][final])}
    tids = rec["tid"][(rec["itr"] == 2) & rec["vld"]]
    one = np.median([efo_of[t] for t in tids[~two] if t in efo_of])
    both = np.median([efo_of[t] for t in tids[two] if t in efo_of])
    assert 1.7 < both / one < 2.3
    assert not (rec["cfr"][rec["vld"]] > 0.8).any()       # the sequence's limit aborts
