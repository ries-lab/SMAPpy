"""Find clusters' *local maxima* (Picasso's SMLM clusterer) and its *frame
analysis*: DNA-PAINT sites a few nanometres apart found one by one, and a
sticky spot thrown out."""
import numpy as np

from smappy.plugins.clusters import (FindClusters, LocalMaximaSettings, frame_analysis,
                                     local_maxima)
from smappy.session import Session
from smappy.simulate import simulate_binding_sites


def site_pairs(n_side=12, gap_nm=20.0, spread=3.0, seed=0, background=500):
    """Pairs of sites ``gap_nm`` apart on a grid, each 20-60 localizations
    with ``spread`` nm scatter, and a sparse background."""
    rng = np.random.default_rng(seed)
    parts, truth = [], []
    k = 0
    for i in range(n_side):
        for j in range(n_side):
            for dx in (0.0, gap_nm):
                k += 1
                n = int(rng.integers(20, 60))
                parts.append(rng.normal(0, spread, (n, 2)) + [i * 300 + dx, j * 300])
                truth.append(np.full(n, k))
    parts.append(rng.uniform(0, n_side * 300, (background, 2)))
    truth.append(np.zeros(background, int))
    return np.vstack(parts), np.concatenate(truth)


def picasso_reference(points, radius, min_locs):
    """Picasso 0.11.3's `clusterer._cluster`, transcribed with a k-d tree."""
    from scipy.spatial import KDTree
    near = KDTree(points).query_ball_point(points, radius)
    counts = np.array([len(v) for v in near])
    most = np.array([counts[v].max() for v in near])
    labels = np.full(len(points), -1)
    for c, i in enumerate(np.flatnonzero((counts > min_locs) & (counts == most))):
        v = np.asarray(near[i])
        if labels[i] == -1:
            labels[v] = c
        else:
            labels[v[labels[v] == -1]] = labels[i]
    values, sizes = np.unique(labels, return_counts=True)
    labels[np.isin(labels, values[sizes < min_locs])] = -1
    return labels


def purity(labels, truth):
    """The fraction of clustered points whose cluster is mostly their site."""
    hit = 0
    member = labels > 0
    for c in np.unique(labels[member]):
        here = truth[labels == c]
        hit += np.bincount(here).max()
    return hit / max(member.sum(), 1)


def test_local_maxima_finds_each_site_of_pairs_twenty_nanometres_apart():
    # 3 nm precision, a radius about twice that, as Picasso advises
    xy, truth = site_pairs()
    labels = local_maxima(xy, 8.0, 10)
    assert labels.max() == truth.max()                   # 288 sites, 288 clusters
    assert purity(labels, truth) > 0.99
    # and DBSCAN at the same radius joins many pairs, which is why it is there
    from smappy.plugins.clusters import dbscan
    assert dbscan(xy, 8.0, 9).max() < 0.8 * truth.max()


def test_local_maxima_agrees_with_picassos_clusterer():
    xy, _ = site_pairs(n_side=8, seed=3)
    ours = local_maxima(xy, 8.0, 10)
    theirs = picasso_reference(xy, 8.0, 10)
    assert ours.max() == len(set(theirs)) - 1
    # the same points in clusters; a point between two maxima may go to the
    # other one, from the order the maxima are taken in
    assert np.array_equal(ours > 0, theirs >= 0)
    assert purity(ours, theirs + 1) > 0.99


def test_local_maxima_does_not_depend_on_the_order_of_the_table():
    xy, _ = site_pairs(n_side=6, seed=5)
    labels = local_maxima(xy, 10.0, 10)
    perm = np.random.default_rng(1).permutation(len(xy))
    again = local_maxima(xy[perm], 10.0, 10)
    assert np.array_equal(again, labels[perm])


def test_frame_analysis_rejects_a_cluster_bunched_in_time_and_renumbers():
    labels = np.array([-1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    frame = np.array([5, 10, 300, 600, 900,            # 1: spread out
                      500, 501, 502, 503,              # 2: one twentieth
                      10, 20, 30, 40])                  # 3: all at the start
    out, rejected = frame_analysis(labels, frame, None, {0: (0, 999)})
    assert rejected == 2
    assert list(out) == [-1, 1, 1, 1, 1] + [-1] * 8


def test_find_clusters_keeps_the_sites_and_drops_a_sticky_spot():
    # nine spots of three sites each, and a tenth with no sites but an
    # imager stuck for 400 frames -- as many localizations as a site has
    locs = simulate_binding_sites([3] * 9 + [0], n_frames=20000, tau_bright=5.0,
                                  tau_dark=400.0, spacing_nm=300.0, spread_nm=15.0,
                                  precision_nm=3.0, sticky=[(9, 8000, 8400)])
    session = Session(locs)
    session.show_grouped(0, False)      # merged, the stuck imager is one blink
    settings = FindClusters.Settings(
        algorithm="local_maxima", frame_analysis=True,
        local_maxima=LocalMaximaSettings(radius_nm=8.0, min_points=20))
    result = session.run(FindClusters(), settings)
    out = result.locs
    spot = np.asarray(out["spot_true"])
    cluster = np.asarray(out["cluster_id"])
    assert not (cluster[spot == 10] > 0).any()         # the sticky spot: none
    assert result.data["frame_analysis_rejected"] >= 1
    # each true site is mostly one cluster; sites placed 15 nm about their
    # spot's centre sometimes fall within a radius of each other and are one
    site = np.asarray(out["site_true"])
    assert purity(cluster, site) > 0.9
    assert 24 <= cluster.max() <= 27
    without = session.run(FindClusters(), FindClusters.Settings(
        algorithm="local_maxima", frame_analysis=False,
        local_maxima=LocalMaximaSettings(radius_nm=8.0, min_points=20)))
    assert (np.asarray(without.locs["cluster_id"])[spot == 10] > 0).any()
