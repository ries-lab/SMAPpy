"""MBM registration (`smappy.mbm_registration`): two measurements put onto
each other by the beads both saw, matched by position, not by name."""
import numpy as np
import pytest

from smappy.io.formats import FileInfo
from smappy.locs import Localizations
from smappy.mbm_registration import BeadPositions, match, register, start_positions
from smappy.session import Session

BEADS = np.array([[2000.0, 3000.0], [15000.0, 4000.0], [8000.0, 18000.0],
                  [21000.0, 20000.0], [4000.0, 12000.0]])


def rigid(angle_deg, shift):
    a = np.radians(angle_deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]), np.asarray(shift)


def measurement_meta(positions, names, label, seed=0, drift_from=0.0):
    """A measurement's metadata with beads at ``positions`` at its start,
    then drifting and noisy, named ``names`` -- the slots in another order."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 120.0, 2.0)
    gri, x, y, time = [], [], [], []
    for k, (p, name) in enumerate(zip(positions, names)):
        slot = 61 + 62 * k
        drift = drift_from + 0.2 * t                      # nm, after the start
        gri += [slot] * len(t)
        x += list(p[0] + drift + rng.normal(0, 1.5, len(t)))
        y += list(p[1] - 0.5 * drift + rng.normal(0, 1.5, len(t)))
        time += list(t)
    return {"label": label, "did": label, "dimensionality": 2, "z_scale": 1.0,
            "mbm": {"used": list(names),
                    "points": {"gri": gri, "x_nm": x, "y_nm": y, "time_s": time,
                               "strength": [1.0] * len(gri)},
                    "beads": [{"name": n, "gri": 61 + 62 * k, "used": True}
                              for k, n in enumerate(names)]}}


def two_measurements(angle_deg=0.3, shift=(850.0, -420.0)):
    """The same beads seen in a second measurement whose frame is rotated
    and shifted, named in another order, with one bead fewer and one more."""
    rotation, t = rigid(angle_deg, shift)
    reference = BEADS
    # ref = R tgt + t, so the target sees tgt = R^T (ref - t)
    target = (rotation.T @ (reference[:4] - t).T).T
    target = np.vstack([target, [[30000.0, 30000.0]]])            # a bead of its own
    a = measurement_meta(reference, ["R0", "R1", "R2", "R3", "R4"], "260101-100000_a")
    b = measurement_meta(target, ["R3", "R0", "R4", "R1", "R2"], "260101-110000_b", seed=1)
    return a, b, rotation, t


def test_beads_are_matched_by_one_shared_shift_not_by_their_names():
    shift = np.array([700.0, 300.0])
    ref = BEADS
    tgt = np.vstack([BEADS[[3, 0, 2]] - shift, [[50000.0, 50000.0]]])
    assert match(ref, tgt) == [(0, 1), (2, 2), (3, 0)]


def test_a_rigid_map_comes_back_from_three_beads_and_more():
    a, b, rotation, t = two_measurements()
    ref = BeadPositions(list("ABCDE"), BEADS)
    tgt = BeadPositions(list("abcd"), (rotation.T @ (BEADS[:4] - t).T).T)
    found = register(ref, tgt)
    assert found.kind == "rigid" and len(found.pairs) == 4
    assert np.allclose(found.matrix[:2, :2], rotation, atol=1e-9)
    assert np.allclose(found.matrix[:2, 3], t, atol=1e-6)
    assert found.angle_deg == pytest.approx(0.3, abs=1e-9)
    two = register(BeadPositions(["A", "B"], BEADS[:2]),
                   BeadPositions(["a", "b"], BEADS[:2] - [100.0, 50.0]))
    assert two.kind == "translation" and np.allclose(two.matrix[:2, 3], [100.0, 50.0])


def test_a_beads_position_is_the_median_of_its_first_points():
    t = np.arange(20.0)
    x = np.where(t < 10, 100.0, 500.0)              # it moves after the start
    found = start_positions(t, np.column_stack([x, x]), np.ones(20, int), first=10)
    assert np.allclose(found.xyz, [[100.0, 100.0]])


def test_the_plugin_keeps_a_transformation_that_puts_the_target_onto_the_reference():
    from smappy import plugins
    a, b, rotation, t = two_measurements()
    session = Session()
    for k, meta in enumerate((a, b)):
        locs = Localizations({"x_nm": np.zeros(10), "y_nm": np.zeros(10),
                              "frame": np.arange(10), "time_s": np.arange(10.0)},
                             {"minflux": {"measurements": [meta]}})
        session.add_file(locs, FileInfo(f"m{k}", f"/data/m{k}.msr", "Abberior MSR",
                                        metadata={"minflux": {"measurements": [meta]}}),
                         append=k > 0)
    cls = plugins.get("Analysis/Register/MBM registration")
    form = cls().describe(session.context(), cls.Settings())
    labels = [v for v, _ in form.choices["reference"] if v]
    assert len(labels) == 2
    result = session.run(cls(), cls.Settings(reference=labels[0], target=labels[1]))
    found = result.data["registration"]
    assert found.kind == "rigid" and len(found.pairs) == 4
    # the start positions carry the beads' noise: ~1 nm, ~1e-4 degree
    assert found.rms_nm < 3.0
    assert np.allclose(found.matrix[:2, :2], rotation, atol=2e-4)
    name = next(iter(session.transformations))
    kept = session.transformations[name]
    target = np.column_stack([found.target.xyz[:, :2], np.zeros(len(found.target.names))])
    x, y, _ = kept.apply(target[:, 0], target[:, 1], target[:, 2])
    i, j = zip(*found.pairs)
    assert np.allclose(np.column_stack([x, y])[list(j)], BEADS[list(i)], atol=5.0)
    assert "found no partner" in result.text            # the target's own bead
