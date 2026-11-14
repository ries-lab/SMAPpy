"""Molecular mapping: RESI's centres and G5M's molecules (smappy.molecules,
ported from Picasso), on simulated DNA-PAINT sites with known positions."""
import numpy as np
import pytest

from smappy import molecules as mol
from smappy.io.formats import FileInfo
from smappy.locs import Localizations
from smappy.plugins.molecular_mapping import MolecularMapping
from smappy.session import Session

PRECISION = 3.0


def sites(centres, n=(30, 80), precision=PRECISION, n_frames=20000, seed=0,
          events=12):
    """Each site's localizations about its centre, in ``events`` binding
    events of a few frames spread over the acquisition; (xy, frame, site)."""
    rng = np.random.default_rng(seed)
    xy, frame, site = [], [], []
    for k, c in enumerate(np.asarray(centres, float)):
        m = int(rng.integers(*n))
        starts = np.sort(rng.integers(0, n_frames - 10, events))
        f = np.sort(starts[rng.integers(0, events, m)] + rng.integers(0, 5, m))
        xy.append(rng.normal(0, precision, (m, 2)) + c)
        frame.append(f)
        site.append(np.full(m, k + 1))
    return np.vstack(xy), np.concatenate(frame), np.concatenate(site)


def grid(origin=(0.0, 0.0), spacing=20.0):
    """A 4 x 3 origami's sites on a ``spacing`` grid."""
    return [(origin[0] + i * spacing, origin[1] + j * spacing)
            for i in range(4) for j in range(3)]


def test_centres_put_each_cluster_at_its_mean_with_std_over_root_n():
    truth = grid(spacing=40.0)
    xy, frame, site = sites(truth)
    c = mol.centres(xy, frame, site)
    assert list(c["label"]) == list(range(1, 13))
    for k in range(12):
        here = xy[site == k + 1]
        assert np.allclose(c["centre"][k], here.mean(0))
        assert np.allclose(c["sem"][k], here.std(0, ddof=1) / np.sqrt(len(here)))
    # the means are as close to the truth as their standard errors say
    z = (c["centre"] - np.asarray(truth)) / c["sem"]
    assert np.abs(z).max() < 4


def test_events_split_at_gaps_of_more_than_three_frames():
    frame = np.array([10, 11, 13, 17, 18, 30, 31, 35])
    #               |  event 0  |  1  |   2   | 3
    assert list(mol.events_in(frame)) == [0, 0, 0, 1, 1, 2, 2, 3]
    c = mol.centres(np.zeros((8, 2)), frame, np.ones(8, int))
    assert c["n_events"][0] == 4


def test_g5m_resolves_two_molecules_three_precisions_apart():
    found = []
    for seed in range(8):
        xy, _, _ = sites([(0, 0), (9, 0)], n=(100, 101), seed=seed)
        model = mol.g5m(xy, np.full(len(xy), PRECISION))
        found.append(len(model))
    assert found.count(2) >= 7


def test_g5m_keeps_one_molecule_one():
    found = []
    for seed in range(8):
        xy, _, _ = sites([(0, 0)], n=(150, 151), seed=seed)
        found.append(len(mol.g5m(xy, np.full(len(xy), PRECISION))))
    assert found == [1] * 8


def test_g5m_finds_an_origamis_twelve_sites_where_they_are():
    truth = np.asarray(grid())
    xy, frame, _ = sites(truth, seed=4)
    model = mol.g5m(xy, np.full(len(xy), PRECISION))
    assert len(model) == 12
    d = np.linalg.norm(model.means[:, None] - truth[None], axis=-1).min(1)
    assert d.max() < 2.0                 # each within 2 nm of a true site
    values, labels = mol.molecule_values(model, xy, np.full(len(xy), PRECISION), frame)
    assert np.all(values["n_events"] >= 6)
    # widths held to the precision
    assert np.all((values["sigma_rel"] >= 0.8 - 1e-9) & (values["sigma_rel"] <= 1.5 + 1e-9))


def test_the_sparrow_test_refuses_two_components_without_a_dip():
    var = np.full((2, 2), 9.0)
    w = np.array([0.5, 0.5])
    assert not mol.resolvable(np.array([[0.0, 0], [5.0, 0]]), w, var)   # < 2 sigma
    assert mol.resolvable(np.array([[0.0, 0], [8.0, 0]]), w, var)


def test_a_cloud_wider_than_its_precision_allows_gets_a_small_p_value():
    rng = np.random.default_rng(2)
    xy = rng.normal(0, 2.5 * PRECISION, (400, 2))          # too wide for one
    model = mol.fit_mixture(xy - xy.mean(0), np.full((len(xy), 2), PRECISION), 1)
    values, _ = mol.molecule_values(model, xy - xy.mean(0), np.full(len(xy), PRECISION),
                                    np.arange(len(xy)) * 40)
    assert values["p_val"][0] < 0.015
    # and a cloud as wide as its precision is one Picasso keeps
    ok = rng.normal(0, PRECISION, (400, 2))
    model = mol.fit_mixture(ok - ok.mean(0), np.full((len(ok), 2), PRECISION), 1)
    values, _ = mol.molecule_values(model, ok - ok.mean(0), np.full(len(ok), PRECISION),
                                    np.arange(len(ok)) * 40)
    assert values["p_val"][0] > 0.015


def session_of(origami=3, seed=0):
    """A session of ``origami`` origami 500 nm apart, each one cluster
    (``cluster_id``), with their sites' truth."""
    xy, frame, site, cluster = [], [], [], []
    for o in range(origami):
        a, b, c = sites(grid((o * 500.0, 0.0)), seed=seed + o)
        xy.append(a)
        frame.append(b)
        site.append(c + 12 * o)
        cluster.append(np.full(len(a), o + 1))
    xy = np.vstack(xy)
    frame = np.concatenate(frame)
    order = np.argsort(frame, kind="stable")
    locs = Localizations({"x_nm": xy[order, 0], "y_nm": xy[order, 1],
                          "frame": frame[order],
                          "xy_err_nm": np.full(len(xy), PRECISION, np.float32),
                          "site_true": np.concatenate(site)[order],
                          "cluster_id": np.concatenate(cluster)[order]},
                         {"units": "nm", "n_frames": 20000})
    session = Session()
    session.add_file(locs, FileInfo(name="origami", path="origami", format="test",
                                    n=len(locs)))
    session.show_grouped(0, False)
    return session


def test_molecular_mapping_adds_a_file_of_molecules_and_writes_the_same_ids():
    session = session_of()
    settings = MolecularMapping.Settings(method="g5m")
    result = session.run(MolecularMapping(), settings)
    assert len(session.files) == 2
    assert session.files[1].name == "molecules of origami"
    table, _, _ = result.files[0]
    assert len(table) == 36                              # 3 origami, 12 sites each
    for column in ("x_nm", "y_nm", "xy_err_nm", "n_events", "p_val", "molecule_id",
                   "cluster_id"):
        assert column in table
    assert np.all(np.asarray(table["xy_err_nm"]) < 1.0)  # a mean of ~50 at 3 nm
    # the same molecules on the localizations, numbered alike
    settings.output = "column"
    again = session.run(MolecularMapping(), settings, layer=0)
    molecule = np.asarray(again.locs["molecule_id"])
    rows = np.asarray(again.locs["filenumber"]) == 0
    site = np.asarray(again.locs["site_true"])[rows]
    molecule = molecule[rows]
    # each molecule holds one site's localizations
    for m in np.unique(molecule[molecule > 0]):
        assert np.bincount(site[molecule == m]).max() / (molecule == m).sum() > 0.95
    mx = np.asarray(table["x_nm"])
    for m in (1, 17, 36):
        assert abs(np.asarray(again.locs["x_nm"])[rows][molecule == m].mean()
                   - mx[m - 1]) < 1.0


def test_centres_need_no_precision_and_g5m_says_it_does():
    session = session_of(origami=1)
    locs = session.locs
    bare = Localizations({k: v for k, v in locs.columns.items() if k != "xy_err_nm"},
                         dict(locs.metadata))
    s = Session()
    s.add_file(bare, FileInfo(name="bare", path="bare", format="test", n=len(bare)))
    s.show_grouped(0, False)
    result = s.run(MolecularMapping(), MolecularMapping.Settings(method="centres"))
    assert len(result.files[0][0]) == 1                 # one cluster, one molecule
    with pytest.raises(ValueError, match="xy_err_nm"):
        s.run(MolecularMapping(), MolecularMapping.Settings(method="g5m"))
