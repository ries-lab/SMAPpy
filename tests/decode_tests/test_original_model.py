"""A model the original trained, run in the port: milestone (d) of docs/decode/poc-plan.md.

The model oracle is the example package's model on its first 500 frames through the Fit notebook's
pipeline in DECODE v0.10.2 itself (``scripts/decode/reference_v0102.py model``, in
``$DECODE_SMLM_DATA/v0102_model``): the network's output for the first 100 frames and the
localizations of all 500.  ``decode_smlm.io`` must give the same, and SMAPpy's Spline 3D, fitting
the same frames, must find the same positions where the axes are right.  The data are DECODE's
(1.35 GB, too large for git), so without them the tests skip; they are ``slow`` -- 500 frames take
about 2.5 minutes on the CPU, so the CPU sees fewer of them.
"""
import os
import pathlib

import numpy as np
import pytest

from decode_smlm.io import load_original, training_camera
from decode_smlm.utils.hardware import default_device

DATA = pathlib.Path(os.environ.get("DECODE_SMLM_DATA", pathlib.Path.home() / ".cache" / "decode_smlm"))
PACKAGE, ORACLE = DATA / "experimental_data_workflow", DATA / "v0102_model"
NEEDED = [PACKAGE / "param_run.yaml", PACKAGE / "model.pt", ORACLE / "v0102_model_raw_cpu.npz",
          ORACLE / "v0102_model_emitters_cpu.h5"]
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not all(p.exists() for p in NEEDED),
                                 reason=f"the example package and its model oracle are not in {DATA}")]
GPU = default_device()
DEVICES = ["cpu"] + ([GPU] if GPU != "cpu" else [])
# frames of the oracle each device is checked on: all of them where it is quick
RAW_FRAMES = {"cpu": 16, GPU: 100}
LOC_FRAMES = {"cpu": 40, GPU: 500}


def _frames(n):
    import tifffile
    small = ORACLE / "frames_500.tif"
    if small.exists():
        return tifffile.imread(small)[:n]
    with tifffile.TiffFile(PACKAGE / "frames.tif") as t:
        return np.stack([t.pages[i].asarray() for i in range(n)])


@pytest.fixture(scope="module", params=DEVICES)
def original(request):
    # the package's calibration is of beads taken without EM gain, its frames with: the notebook flipped them
    return load_original(PACKAGE / "param_run.yaml", PACKAGE / "model.pt", frames_mirrored=True, device=request.param)


def _oracle_locs(n_frames):
    import h5py
    with h5py.File(ORACLE / "v0102_model_emitters_cpu.h5") as f:
        d = {k: f["data"][k][()] for k in ("xyz", "frame_ix", "phot", "bg", "prob", "xyz_sig")}
    keep = d["frame_ix"] < n_frames
    return {k: v[keep] for k, v in d.items()}


def test_the_network_output_is_the_originals(original):
    n = RAW_FRAMES[original.device]
    ref = np.load(ORACLE / "v0102_model_raw_cpu.npz")
    # the notebook's camera, the training's: what the oracle was made with
    raw = original.raw(_frames(n), training_camera(original.param)).numpy()
    oracle = ref["raw"][:n]
    for ch in range(oracle.shape[1]):
        span = oracle[:, ch].max() - oracle[:, ch].min()
        assert np.abs(raw[:, ch] - oracle[:, ch]).max() <= 1e-4 * span, ch


def test_the_localizations_are_the_originals(original):
    """Every one found again, at most 2 float32 steps away (3e-5 px at 176 px: 0.004 nm)."""
    n = LOC_FRAMES[original.device]
    em = original.localize(_frames(n), training_camera(original.param))
    ref = _oracle_locs(n)
    assert len(em) == len(ref["xyz"])
    assert np.array_equal(em.frame_ix.numpy(), ref["frame_ix"])     # the same, in the same order
    d = np.abs(em.xyz.numpy() - ref["xyz"])
    assert d[:, :2].max() <= 3e-5 and d[:, 2].max() <= 0.05
    for mine, theirs in ((em.phot, ref["phot"]), (em.bg, ref["bg"]), (em.prob, ref["prob"]),
                         (em.xyz_sig, ref["xyz_sig"])):
        assert np.abs(mine.numpy() - theirs).max() <= 1e-4 * np.abs(theirs).max()


def test_the_columns_are_where_spline_3d_finds_the_molecules():
    """DECODE's x is the rows, and its positions are on the cropped, mirrored frame: put back by
    ``to_columns``, the medians agree with Spline 3D's to about 1 nm -- a half-pixel shift, swapped
    axes or a forgotten mirror (5 nm in x here) would not."""
    import warnings

    from scipy.spatial import cKDTree

    api = pytest.importorskip("smappy.api")

    frames = _frames(100)
    original = load_original(PACKAGE / "param_run.yaml", PACKAGE / "model.pt", frames_mirrored=True, device=GPU)
    camera = {"baseline": 398.6, "e_per_adu": 5.0, "em_gain": 50.0}             # frames_meta.yaml's
    port = original.to_columns(original.localize(frames, camera), frames.shape, (100.0, 100.0))
    # Spline 3D on the frames flipped as the notebook flipped them: the beads were taken without EM
    # gain, so SMAPpy keeps the spline as it is, mirrored against these frames
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = api.fit(np.ascontiguousarray(frames[..., ::-1]), calibration=PACKAGE / "spline_calibration_3dcal.mat",
                      camera={"conversion": 5.0, "offset": 398.6, "pixelsize_um": 0.1, "em_on": True, "emgain": 50.0},
                      cutoff=1.7, cutoff_mode="dynamic")
    sx = (frames.shape[-1] - 1) * 100.0 - np.asarray(fit.columns["x_nm"], np.float64)
    sy, sz = (np.asarray(fit.columns[k], np.float64) for k in ("y_nm", "z_nm"))
    sframe = np.asarray(fit.columns["frame"]).astype(int)
    sframe -= sframe.min()
    dx, dy, dz = [], [], []
    for f in range(len(frames)):
        a = (port["frame"] == f) & (port["prob"] > 0.7)
        b = sframe == f
        if not a.any() or not b.any():
            continue
        dist, j = cKDTree(np.c_[sx[b], sy[b]]).query(np.c_[port["x_nm"][a], port["y_nm"][a]], distance_upper_bound=100)
        ok = np.isfinite(dist)
        dx.append(port["x_nm"][a][ok] - sx[b][j[ok]])
        dy.append(port["y_nm"][a][ok] - sy[b][j[ok]])
        dz.append(port["z_nm"][a][ok] - sz[b][j[ok]])
    dx, dy, dz = (np.concatenate(v) for v in (dx, dy, dz))
    assert len(dx) > 5000
    assert abs(np.median(dx)) < 2.5 and abs(np.median(dy)) < 2.5 and abs(np.median(dz)) < 5
