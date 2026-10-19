"""Bead pairs on a split chip, by SMAPpy's convention (`data/pairing.py`, plan §5.7)."""
import glob
import os

import numpy as np
import pytest
import torch

from uipsf.data.io import BeadStack
from uipsf.data.pairing import extract_pair_rois, fit_projective, pair_beads

H_SMALL = np.array([[0.998, 0.012, 2.4], [-0.011, 1.001, 1.3], [1e-6, -2e-6, 1.0]])
SHIFT = np.array([[1.0, 0, 0], [0, 1, -256], [0, 0, 1]])            # the lower half onto the upper one
MIRROR = np.array([[1.0, 0, 0], [0, -1, 511], [0, 0, 1]])           # mirrored across the split, as SMAPpy's
H_TRUE = H_SMALL @ SHIFT                                            # up-down layouts are


def _apply(H, xy):
    q = np.c_[xy, np.ones(len(xy))] @ H.T
    return q[:, :2] / q[:, 2:]


def _chip(H, seed, n=14, shape=(512, 256), origin=(100, 0)):
    """One stack of Gaussian spots: n beads in the upper half and their partners in the lower one."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    while True:                                     # isolated beads, away from the edges
        main = rng.uniform([30, 30], [shape[1] - 30, shape[0] / 2 - 30], (n, 2)) + origin
        d = np.linalg.norm(main[:, None] - main[None], axis=-1) + 1e3 * np.eye(n)
        if d.min() > 30:
            break
    second = _apply(np.linalg.inv(H), main)
    K = 7
    img = np.full((K, *shape), 10.0)
    for k in range(K):
        w = 1.5 + 0.3 * abs(k - K // 2)
        for x, y in np.r_[main, second] - origin:
            img[k] += 400 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * w ** 2))
    stack = BeadStack(img, np.arange(K) * 100.0, 1.0, origin, None, f"chip{seed}")
    return stack, main


@pytest.mark.parametrize("mirrored", [False, True])
def test_the_layout_the_pairs_and_the_transform_come_back_from_a_simulated_chip(mirrored):
    H = H_SMALL @ MIRROR if mirrored else H_TRUE
    stacks, mains = zip(*[_chip(H, seed) for seed in range(3)], strict=True)
    pairs = pair_beads(list(stacks), roi_size=15)
    assert pairs.layout == ("up-down mirrored" if mirrored else "up-down")
    assert pairs.split == 256 and len(pairs) == 3 * 14
    truth = np.concatenate(mains)
    found = pairs.chip("main")
    assert np.abs(found[:, None] - truth[None]).max(-1).min(1).max() < 0.2      # every pair is a true bead
    assert np.abs(_apply(pairs.transform, pairs.chip("secondary")) - found).max() < 0.3
    assert np.abs(_apply(pairs.transform, truth) - _apply(H, truth)).max() < 0.3
    batch = extract_pair_rois(list(stacks), pairs, 15, slices=slice(None))
    assert batch.data.shape == (42, 2, 7, 15, 15)
    assert torch.allclose(batch.roi_xy[:, 1].double(), torch.tensor(pairs.secondary_px + pairs.chip_origin[pairs.stack],
                                                                    dtype=torch.float64))


def test_the_projective_fit_recovers_a_projective_map():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 500, (20, 2))
    assert np.allclose(fit_projective(src, _apply(H_TRUE, src)), H_TRUE / H_TRUE[2, 2], atol=1e-9)


ZENODO = os.path.join(os.environ.get("UIPSF_DATA", os.path.expanduser("~/.cache/uipsf/zenodo")), "2ch_40nm_bead")


@pytest.mark.skipif(not glob.glob(os.path.join(ZENODO, "*", "*.tif")), reason="needs Zenodo 2ch_40nm_bead")
def test_the_zenodo_pairs_are_smappys():
    # Plan §5.7: pairing follows SMAPpy's convention, as find_beads follows detect_beads.
    pytest.importorskip("smappy")
    from smappy.calibrate.dual import collect_dual_beads
    from smappy.calibrate.input import read_bead_stacks

    from uipsf.data.io import read_stack
    from uipsf.noise import Camera
    files = sorted(glob.glob(os.path.join(ZENODO, "*", "*.tif")))
    ours = pair_beads([read_stack(f, Camera(conversion=1.0, offset=0.0)) for f in files])
    theirs = collect_dual_beads([s for f in files for s in read_bead_stacks(f)])
    assert ours.layout == theirs.settings.registration.layout
    assert ours.split == theirs.geometry["split_position"]
    mine = ours.geometry()
    for k in ("layout", "main_channel", "split_position", "image_shape", "coordinate_system", "mirror_axis_xy"):
        assert mine[k] == theirs.geometry[k], k
    assert [s["roi"] for s in mine["sources"]] == [list(s["roi"]) for s in theirs.geometry["sources"]]
    assert len(ours) == len(theirs.main_points)
    # Pair for pair, stack by stack (some stacks repeat a position), at sub-pixel positions that agree to
    # a fraction of a pixel (SMAPpy fits a Gaussian where uipsf takes a centroid).
    stack = np.array([r["stack"] for r in theirs.records])
    d = np.abs(ours.chip("main")[:, None] - theirs.main_points[None]).max(-1)
    d[ours.stack[:, None] != stack[None]] = np.inf
    match = d.argmin(1)
    assert d.min(1).max() < 0.5 and len(set(match)) == len(ours)
    assert np.abs(ours.chip("secondary") - theirs.secondary_points[match]).max() < 0.5
