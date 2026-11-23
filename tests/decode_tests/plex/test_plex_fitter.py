"""DECODE-Plex's published models in SMAPpy's DECODE fitter (`decode_smlm.plex.fitter`; gui-plan I5).

A run of the paper exported to one model file and fitted by *Localize/DECODE* gives the paper's own
localizations: the oracle the port is checked against (`test_plex_port`), on the frames' main half.
"""
import glob

import numpy as np
import pytest
import tifffile

from .test_plex_port import DATA, RUNS

torch = pytest.importorskip("torch")


def _exported(run, tmp_path):
    from decode_smlm.plex.fitter import export
    training, frames_file, trafo, _, _, _ = RUNS[run]
    needed = [DATA / training, DATA / frames_file, DATA / "oracles" / f"plex_{run}_cpu.npz", DATA / trafo]
    if not all(p.exists() for p in needed):
        pytest.skip(f"DECODE-Plex's {run} model, frames or oracle are not in {DATA}")
    return export(DATA / training, DATA / trafo, tmp_path / f"{run}.safetensors"), DATA / frames_file


@pytest.mark.slow
@pytest.mark.parametrize("run", ["fig2c", "fig2a"])
def test_a_published_model_in_the_fitter_gives_the_papers_localizations(run, tmp_path):
    from smappy import plugins
    from smappy.gui.live_session import fit_settings
    from smappy.plugins import Context
    from smappy.training import model_file
    from smappy.training.predictors import describe, load
    path, frames_file = _exported(run, tmp_path)
    meta = model_file.read_metadata(path)
    assert describe(meta) == "DECODE-Plex" and "trafo" in meta
    spec = meta["cfg"]["Camera"][0]["specs"]
    n = 12
    movie = tmp_path / "frames.tif"
    tifffile.imwrite(movie, tifffile.imread(frames_file)[:n], photometric="minisblack")
    px_rows, px_cols = spec["px_size"]
    values = {"source.path": str(movie), "camera.conversion": float(spec["e_per_adu"]),
              "camera.offset": float(spec["baseline"]), "camera.em_on": bool(spec.get("em_gain")),
              "camera.emgain": float(spec.get("em_gain") or 1.0), "camera.pixelsize_um": px_cols / 1000,
              "camera.pixelsize_y_um": px_rows / 1000, "model.path": str(path), "model.device": "cpu",
              "model.threshold": 0.0, "model.batch": 4, "output.path": "", "output.raw_frames": 0,
              "finish.assign_colors": False}
    locs = plugins.get("Localize/DECODE")().run(Context(), fit_settings("Localize/DECODE", values)).locs
    o = np.load(DATA / "oracles" / f"plex_{run}_cpu.npz")
    k = load(path, "cpu").window
    from scipy.spatial import cKDTree
    found = []
    for f in range(k, n - k):        # the movie's ends have a padded window, the oracle's did not
        mine = np.c_[np.asarray(locs["y_nm"]) / px_rows, np.asarray(locs["x_nm"]) / px_cols][np.asarray(locs["frame"]) == f]
        ref = o["em_xyz"][o["em_frame_ix"] == f][:, :2]
        d, _ = cKDTree(ref).query(mine)
        found.append((len(mine), len(ref), float(np.mean(d < 1e-3))))
    for mine, ref, share in found:
        assert abs(mine - ref) <= max(1, ref // 10_000) and share > 0.999, found
    if run == "fig2a":
        assert "photons_ch0" in locs and "photons_ch1" in locs
    # the run's checkpoint chosen as it is: converted once, its registration found near it, the same fit
    ckpt = next((DATA / RUNS[run][0]).glob("*.ckpt"))
    values["model.path"] = str(ckpt)
    direct = plugins.get("Localize/DECODE")().run(Context(), fit_settings("Localize/DECODE", values)).locs
    assert len(direct) == len(locs)
    np.testing.assert_allclose(np.asarray(direct["x_nm"]), np.asarray(locs["x_nm"]))
    assert glob.glob(str(tmp_path / "*.safetensors"))
