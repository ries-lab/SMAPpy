"""SPINNA: structures, the search space, the simulation, the KS score, and
the proportions and labelling efficiencies coming back from simulated data
with known ones."""
import numpy as np
import pytest
from scipy.stats import ks_2samp

from smappy import spinna as sp
from smappy.locs import Localizations
from smappy.plugins import Context, PreflightQuestion, Stopped
from smappy.plugins.spatial_statistics import Window
from smappy.plugins.spinna import (SHIPPED, Spinna, SpinnaSettings, StructureSettings,
                                   preset_values)
from smappy.regions import Region

SIDE = 6000.0
BOX = Window(np.ones((1, 1), bool), 0.0, 0.0, SIDE, SIDE)
MONOMER = sp.parse_structure("monomer", "A 0 0")
DIMER = sp.parse_structure("dimer", "A -10 0; A 10 0")
TRIMER = sp.parse_structure("trimer", "A 0 11.547; A -10 -5.774; A 10 -5.774")


def mixture(dimer_pct, n=4000, efficiency=0.5, unc=5.0, seed=0, window=BOX):
    """``n`` molecules, ``dimer_pct`` of them in dimers 20 nm long."""
    n_dimers = int(round(dimer_pct / 100 * n / 2))
    return sp.simulate([MONOMER, DIMER], [n - 2 * n_dimers, n_dimers], window,
                       [efficiency], [unc], np.random.default_rng(seed))


# ------------------------------------------------------------- structures

def test_a_structure_is_read_from_its_molecules_and_targets():
    s = sp.parse_structure("pair", "A -5 0; b 5, 0, 3\nA 0 1")
    assert np.allclose(s.xyz, [[-5, 0, 0], [5, 0, 3], [0, 1, 0]])
    assert list(s.target) == [0, 1, 0]
    assert sp.parse_structure("one", "0 0").target.tolist() == [0]
    for bad in ("", "A 1", "AB 0 0", "A x y"):
        with pytest.raises(ValueError):
            sp.parse_structure("bad", bad)


def test_the_neighbours_compared_are_the_partners_a_structure_gives():
    assert sp.neighbour_orders([MONOMER, DIMER], 1) == [(0, 0, 1)]
    assert sp.neighbour_orders([MONOMER, DIMER, TRIMER], 1) == [(0, 0, 2)]
    het = sp.parse_structure("A-B", "A 0 0; B 10 0")
    lone_b = sp.parse_structure("B", "B 0 0")
    # A and B never meet themselves; only A to B tells the model apart
    assert sp.neighbour_orders([MONOMER, lone_b, het], 2) == [(0, 1, 1)]
    assert sp.neighbour_orders([MONOMER], 1) == []


def test_every_candidate_uses_up_the_molecules_there_are():
    space = sp.search_space([MONOMER, DIMER], [1000], 21)
    assert np.all(space @ [1, 2] == 1000) and np.all(space >= 0)
    assert len(space) == 21
    props = sp.proportions(space, [MONOMER, DIMER])
    assert np.allclose(props.sum(axis=1), 100)
    assert np.allclose(np.sort(props[:, 1]), np.arange(0, 101, 5))
    het = sp.parse_structure("A-B", "A 0 0; B 10 0")
    lone_b = sp.parse_structure("B", "B 0 0")
    space = sp.search_space([MONOMER, lone_b, het], [800, 500], 11)
    assert np.all(space[:, 0] + space[:, 2] == 800)
    assert np.all(space[:, 1] + space[:, 2] == 500)
    assert space[:, 2].max() == 500
    with pytest.raises(ValueError, match="no structure holds target B"):
        sp.search_space([MONOMER, DIMER], [100, 100], 5)


def test_the_coarse_pass_leaves_no_candidate_out_of_the_fine_ones_reach():
    props = sp.proportions(sp.search_space([MONOMER, DIMER], [1000], 21), [MONOMER, DIMER])
    taken = sp.farthest_points(props, max(2, int(0.1 * len(props))),
                               sp.NEIGHBOURHOOD_PCT / 2)
    # Picasso's 10 % would be two candidates; every one is within half the
    # fine pass's radius of one taken
    gaps = np.abs(props[:, None, :] - props[None, taken, :]).sum(axis=2).min(axis=1)
    assert len(taken) > 2 and gaps.max() <= sp.NEIGHBOURHOOD_PCT / 2


# ------------------------------------------------------------- simulation

def test_the_simulation_keeps_the_efficiency_the_window_and_the_distance():
    rng = np.random.default_rng(1)
    exact = sp.simulate([DIMER], [3000], BOX, [1.0], [0.0], rng)[0]
    assert len(exact) == 6000
    d = sp.nn_distances(exact, exact, 1, True)[:, 0]
    assert np.median(d) == pytest.approx(20.0, abs=1e-6)
    thinned = sp.simulate([MONOMER, DIMER], [1000, 1000], BOX, [0.3], [5.0], rng)[0]
    assert len(thinned) == int(1000 * 0.3) + int(2000 * 0.3)      # Picasso's int(n le)
    assert thinned.min() > -50 and thinned.max() < SIDE + 50
    three = sp.simulate([DIMER], [500], BOX, [1.0], [0.0], rng, z_range=(0, 600),
                        rotation="3D")[0]
    assert three.shape == (1000, 3) and three[:, 2].min() > -20 and three[:, 2].max() < 620
    pair = sp.nn_distances(three, three, 1, True)[:, 0]
    assert np.median(pair) == pytest.approx(20.0, abs=1e-6)


def test_the_ks_statistic_is_scipys():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 500), rng.normal(0.3, 1.2, 700)
    assert sp.ks_statistic(a, b) == pytest.approx(ks_2samp(a, b).statistic)
    assert sp.nnd_score([a[:, None]], [a[:, None]]) == 0.0
    assert sp.nnd_score([np.zeros((0, 1))], [a[:, None]]) == 1.0


# -------------------------------------------------------------- the fit

def test_thirty_percent_dimers_come_back_within_the_bootstrap_error():
    pts = mixture(30.0, seed=104)
    model = sp.Model([MONOMER, DIMER], [0.5], [5.0])
    fit = sp.fit(model, [sp.Field(BOX, pts)], n_sims=5, bootstraps=10, seed=0)
    assert fit.errors is not None and 1.0 < fit.errors[1] < 8.0
    assert abs(fit.proportions[1] - 30.0) <= 2 * fit.errors[1]
    assert fit.proportions.sum() == pytest.approx(100)
    assert (fit.passes == 0).any() and len(fit.tested) < 2 * len(fit.space)


def test_monomers_alone_are_found_by_every_candidate_tried():
    pts = mixture(0.0, seed=5)
    model = sp.Model([MONOMER, DIMER], [0.5], [5.0])
    fit = sp.fit(model, [sp.Field(BOX, pts)], granularity=11, search="brute_force",
                 n_sims=3, bootstraps=0, seed=1)
    assert len(fit.tested) == len(fit.space) == 11 and fit.errors is None
    assert fit.proportions[1] <= 10.0
    assert fit.scores[np.argmax(fit.tested_proportions[:, 1])] > 3 * fit.score


def test_the_labelling_efficiency_of_two_tags_comes_back():
    protein = sp.Structure("protein", [[-5, 0, 0], [5, 0, 0]], [0, 1])
    pts = sp.simulate([protein], [3000], BOX, [0.6, 0.4], [5.0, 5.0],
                      np.random.default_rng(1))
    found = sp.fit_efficiency([sp.Field(BOX, pts)], [10.0], 5.0, n_sims=3,
                              bootstraps=5, seed=1)
    assert found.efficiency == pytest.approx([0.6, 0.4], abs=0.05)
    assert found.distance_nm == 10.0 and found.errors is not None
    assert sp.efficiency_from_proportions([40.0, 20.0, 40.0]) == pytest.approx(
        [0.5, 1 / 3])


def test_files_are_measured_apart_and_pooled():
    a, b = mixture(0.0, n=600, seed=1), mixture(0.0, n=600, seed=2)
    orders = [(0, 0, 1)]
    pooled = sp.data_distances([sp.Field(BOX, a), sp.Field(BOX, b)], orders)[0]
    apart = np.concatenate([sp.nn_distances(a[0], a[0], 1, True),
                            sp.nn_distances(b[0], b[0], 1, True)])
    assert np.array_equal(pooled, apart)


# ------------------------------------------------------------- the plugin

def table(xy, **extra):
    return Localizations({"x_nm": xy[:, 0], "y_nm": xy[:, 1], **extra}, {})


def quick(**changes):
    return SpinnaSettings(**{"simulations": 3, "bootstraps": 5, "granularity": 11,
                             "window": "box", **changes})


def test_the_plugin_hands_back_proportions_and_errors_for_a_region():
    pts = mixture(40.0, n=3000, seed=8)[0]
    region = Region.rect(0, 0, SIDE, SIDE)
    ctx = Context(locs=table(pts), region=region, region_id="1")
    result = Spinna().run(ctx, quick())
    v = result.values
    assert v["monomer_pct"] + v["dimer_pct"] == pytest.approx(100)
    assert abs(v["dimer_pct"] - 40) <= max(2 * v["dimer_err_pct"], 10)
    assert v["n_A"] == len(pts) and v["density_A_um2"] == pytest.approx(len(pts) / 36, rel=0.01)
    assert 0 <= v["ks"] < 0.2 and "ks_err" in v
    assert set(result.plots) == {"scores", "window"} and result.plot.panels == 1
    assert "rect ROI" in result.text and "dimer" in result.text
    empty = Context(locs=table(np.zeros((0, 2))), region=region, region_id="2")
    assert Spinna().run(empty, quick()).values == {}


def test_two_layers_are_two_targets_and_the_efficiency_fit_reads_them():
    from smappy.session import Session
    protein = sp.Structure("protein", [[-5, 0, 0], [5, 0, 0]], [0, 1])
    a, b = sp.simulate([protein], [2500], BOX, [0.7, 0.5], [4.0, 4.0],
                       np.random.default_rng(3))
    xy = np.vstack([a, b])
    session = Session(table(xy, channel=np.repeat([1, 2], [len(a), len(b)]).astype(np.int32),
                            frame=np.arange(len(xy), dtype=np.int64)))
    session.show_grouped(0, False)
    session.layers[0].set_bound("channel", 1, 1)
    second = session.add_layer(like=0)
    second.set_bound("channel", 2, 2)
    result = Spinna().run(session.context(), quick(fit="efficiency", label_unc_nm="4",
                                                   distance_nm="10"))
    v = result.values
    assert v["efficiency_A"] == pytest.approx(0.7, abs=0.06)
    assert v["efficiency_B"] == pytest.approx(0.5, abs=0.06)
    assert v["n_A"] == len(a) and v["n_B"] == len(b)
    assert "labelling efficiency of A" in result.text
    # one model target too many for the layers shown
    second.visible = False
    with pytest.raises(ValueError, match="2 targets"):
        Spinna().run(session.context(), quick(fit="efficiency"))


def test_a_long_run_asks_first_and_offers_a_cheaper_one():
    pts = mixture(30.0, n=4000, seed=9)[0]
    big = SpinnaSettings(window="box", granularity=200, simulations=20)
    question = Spinna().preflight(Context(locs=table(pts)), big)
    assert isinstance(question, PreflightQuestion)
    labels = [c.label for c in question.choices]
    assert labels == ["without the bootstrap", "3 simulations", "11 grid steps"]
    assert question.choices[0].settings.bootstrap is False
    assert Spinna().preflight(Context(locs=table(pts)), quick()) is None


def test_stop_ends_the_fit():
    class Stopping(Context):
        @property
        def stopped(self):
            return True
    pts = mixture(30.0, n=1000, seed=10)[0]
    with pytest.raises(Stopped):
        Spinna().run(Stopping(locs=table(pts)), quick())


def test_every_shipped_model_fills_the_structures():
    names = sorted(p.stem for p in SHIPPED.glob("*.yaml"))
    assert len(names) >= 4
    plugin = Spinna()
    for name in names:
        values = plugin.react("preset", SpinnaSettings(preset=name))
        assert values["preset"] == name and set(values) == {"preset", "structures"}
        structures = [StructureSettings(**s) for s in values["structures"]]
        settings = SpinnaSettings(structures=structures)
        state = plugin.describe(None, settings)
        assert state.notice.kind == "info", (name, state.notice.text)
    two = preset_values("Two proteins, heterodimer (10 nm)")
    assert [s["molecules"] for s in two["structures"]][-1] == "A -5 0; B 5 0"


def test_the_form_shows_the_structures_or_the_tag_distance():
    plugin = Spinna()
    state = plugin.describe(None, SpinnaSettings())
    assert state.shown["structures"] and not state.shown["distance_nm"]
    state = plugin.describe(None, SpinnaSettings(fit="efficiency"))
    assert not state.shown["structures"] and state.shown["distance_nm"]
    bad = SpinnaSettings(structures=[StructureSettings("x", True, "A 1")])
    assert plugin.describe(None, bad).notice.kind == "warning"
