"""A simulated scene: components with their own labelling, photophysics and
motion, linked by place or by photophysics, in one ground truth
(`smappy.simulate.scene`, docs/simulation-redesign.md)."""
import numpy as np
import pytest

from smappy.simulate import simulate
from smappy.simulate.localizations import ground_truth, localizations
from smappy.simulate.scene import as_scene, two_colour_scene
from smappy.simulate.settings import (BlinkingSettings, ComponentSettings, MotionSettings,
                                      PlacementSettings, SceneSettings, SimulationSettings,
                                      StructureSettings)


def npc(name="Nup96", **values):
    values.setdefault("placement", PlacementSettings(mode="copies", n=12))
    return ComponentSettings(name=name, structure=StructureSettings(preset="npc"), **values)


def test_a_scene_of_one_component_draws_the_same_numbers_as_one_structure():
    settings = SimulationSettings(n_frames=2000, seed=4)
    alone, scene = simulate(settings), simulate(as_scene(settings))
    for name in ("frame", "x_nm", "y_nm", "photons", "emitter", "copy"):
        np.testing.assert_array_equal(alone[name], scene[name])
    assert set(alone["channel"]) == {1} and set(alone["component"]) == {0}


def test_a_component_placed_same_as_another_takes_its_copies_places_and_turns():
    """A two-colour pore: the second colour's pores where the first's are,
    turned as they are, with labels of their own."""
    first = npc(channel=1)
    second = npc("Nup2", channel=2, placement=PlacementSettings(mode="same", same_as="Nup96"))
    truth = ground_truth(SceneSettings(n_frames=500, seed=2,
                                       components=[second, first]))   # order does not matter
    poses = truth.poses
    for key in ("x_nm", "y_nm", "gamma_deg"):
        assert poses["Nup2"][key] == poses["Nup96"][key]
    assert len(poses["Nup96"]["x_nm"]) == 12
    locs = localizations(truth)
    assert set(locs["channel"]) == {1, 2}
    # the labels are drawn afresh: a different labelling, the same pores
    names = truth.components()
    assert names == ["Nup2", "Nup96"]


def test_photophysics_as_another_component_takes_its_photons():
    bright = npc(blinking=BlinkingSettings(photons=20000, photons_std=0))
    follows = npc("other", photophysics_from="Nup96",
                  blinking=BlinkingSettings(photons=500, photons_std=0))
    truth = ground_truth(SceneSettings(n_frames=1000, seed=1, components=[bright, follows]))
    rate = truth.blinks.rate * BlinkingSettings().on_time       # photons per blink
    owner = truth.fluorophores.component[truth.blinks.owner]
    np.testing.assert_allclose(rate[owner == 1], 20000)


def test_homogeneous_labels_and_beads_are_spread_at_their_density():
    scene = SceneSettings(n_frames=200, seed=5, field_um=20.0, components=[
        ComponentSettings(name="background",
                          structure=StructureSettings(kind="homogeneous", density_um2=3,
                                                      z_range_nm=600)),
        ComponentSettings(name="beads", structure=StructureSettings(kind="beads",
                                                                    density_um2=0.05),
                          blinking=BlinkingSettings(activation="always", photons=30000,
                                                    photons_std=0))])
    truth = ground_truth(scene)
    fl = truth.fluorophores
    background = fl.xyz[fl.component == 0]
    assert len(background) == pytest.approx(3 * 400, rel=0.1)
    assert background[:, :2].min() >= 0 and background[:, :2].max() <= 20000
    assert np.abs(background[:, 2]).max() <= 300
    # a bead is on in every frame, with its photons each frame
    em = truth.emission
    bead = fl.component[em.owner] == 1
    n_beads = int((fl.component == 1).sum())
    assert n_beads == pytest.approx(0.05 * 400, abs=10)
    assert bead.sum() == n_beads * 200
    np.testing.assert_allclose(em.photons[bead], 30000)


def test_diffusing_labels_spread_as_four_d_t():
    """Brownian steps of sqrt(2 D dt) per axis: the mean squared displacement
    in the plane after t is 4 D t."""
    d, frame_time_ms = 0.2, 10.0
    scene = SceneSettings(n_frames=60, seed=7, frame_time_ms=frame_time_ms, components=[
        ComponentSettings(name="particles",
                          structure=StructureSettings(kind="homogeneous", density_um2=20),
                          blinking=BlinkingSettings(activation="always", photons=1000,
                                                    photons_std=0),
                          motion=MotionSettings(on=True, d_um2_s=d))])
    truth = ground_truth(scene)
    em, moved = truth.emission, truth.motion_nm
    last = em.frame == 59
    msd = np.mean(np.sum(moved[last, :2] ** 2, axis=1))            # nm^2
    expected = 4 * d * 1e6 * 59 * frame_time_ms * 1e-3
    assert msd == pytest.approx(expected, rel=0.1)
    assert np.all(moved[em.frame == 0] == 0)


def test_links_that_cannot_be_followed_are_refused_with_their_names():
    loose = npc("Nup2", placement=PlacementSettings(mode="same", same_as="gone"))
    with pytest.raises(ValueError, match="gone"):
        ground_truth(SceneSettings(n_frames=10, components=[npc(), loose]))
    off = npc(on=False)
    linked = npc("Nup2", placement=PlacementSettings(mode="same", same_as="Nup96"))
    with pytest.raises(ValueError, match="switched off"):
        ground_truth(SceneSettings(n_frames=10, components=[off, linked]))
    a = npc("a", placement=PlacementSettings(mode="same", same_as="b"))
    b = npc("b", placement=PlacementSettings(mode="same", same_as="a"))
    with pytest.raises(ValueError, match="circle"):
        ground_truth(SceneSettings(n_frames=10, components=[a, b]))
    with pytest.raises(ValueError, match="two components"):
        ground_truth(SceneSettings(n_frames=10, components=[npc(), npc()]))


def test_the_two_colour_demo_is_the_ring_and_the_cross_each_its_own_colour_and_split():
    scene = two_colour_scene(500, 0, ratios=(0.2, 0.7))
    truth = ground_truth(scene)
    fl = truth.fluorophores
    np.testing.assert_allclose(sorted(set(fl.ratio)), [0.2, 0.7])
    ring = fl.xyz[fl.channel == 1]
    radius = np.hypot(ring[:, 0] - 5000, ring[:, 1] - 5000)
    assert np.median(radius) == pytest.approx(2500, rel=0.05)
    assert truth.components() == ["ring", "cross"]
