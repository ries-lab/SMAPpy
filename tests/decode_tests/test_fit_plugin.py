"""The DECODE fitter (``Localize/DECODE``, `smappy.plugins.fit_decode`): milestone (g).

A tiny network -- randomly initialised, three frames in, 8 features -- localizes 32 x 32 frames in
milliseconds on the CPU, and its detections are what they are: the tests here are of the
plumbing, which must give one table whatever the blocks, live or not.  That the numbers are
DECODE's is the oracle test at the end, the example package's model on its own frames (slow;
skipped without the data, as (d)'s are).
"""
import json
import os
import pathlib
import struct
import threading
import time

import numpy as np
import pytest
import tifffile
import torch

from smappy import plugins
from smappy.plugins import Context

from .test_training import tiny_job

PLUGIN = "Localize/DECODE"
DATA = pathlib.Path(os.environ.get("DECODE_SMLM_DATA", pathlib.Path.home() / ".cache" / "decode_smlm"))
PACKAGE, ORACLE = DATA / "experimental_data_workflow", DATA / "v0102_model"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A three-frame model file, its job's calibration, and 40 simulated frames as a TIFF."""
    import torch

    from decode_smlm.neuralfitter import models
    from decode_smlm.training import parameters, write_model
    from smappy.simulate import LabellingSettings, camera_frames
    from smappy.training import job as jobs

    root = tmp_path_factory.mktemp("decode_fit")
    folder = tiny_job(root / "job")
    job, method = jobs.read(folder)
    method["HyperParameter"]["channels_in"] = 3
    param = parameters(folder, method, "cpu")
    torch.manual_seed(2)            # an initialisation whose probabilities pass 0.3 often (seed 0's never)
    model = models.SigmaMUNet.parse(param)
    write_model(root / "model.safetensors", model, param, job, 0, calibration_em_on=False)
    frames, _ = camera_frames(n_frames=40, size_px=32, seed=3, labelling=LabellingSettings(efficiency=0.1))
    tifffile.imwrite(root / "frames.tif", frames)
    return {"root": root, "model": root / "model.safetensors", "calibration": folder / jobs.CALIBRATION,
            "frames": frames, "tif": root / "frames.tif"}


def settings(path, model, **more):
    from smappy.gui.live_session import fit_settings
    values = {"source.path": str(path), "camera.conversion": 0.5, "camera.offset": 100.0,
              "camera.pixelsize_um": 0.1, "camera.em_on": False, "model.path": str(model),
              "model.device": "cpu", "model.threshold": 0.3, "model.batch": 1, "output.path": "", "output.raw_frames": 0}
    values.update(more)
    return fit_settings(PLUGIN, values)


def fit(path, model, **more):
    return plugins.get(PLUGIN)().run(Context(), settings(path, model, **more)).locs


def _same(a, b):
    """The same table, exactly.  The network runs one frame at a time here (*frames per batch* 1):
    the CPU's convolutions take another path for batches of other sizes, and round differently,
    by 5e-4 of a channel's range in this untrained network -- which would hide a wrong context
    frame among the rounding."""
    assert len(a) == len(b) > 0
    for name in ("frame", "x_nm", "y_nm", "z_nm", "photons", "background", "prob"):
        assert np.array_equal(np.asarray(a[name]), np.asarray(b[name])), name


def test_the_fitter_writes_smappys_columns_on_every_frame(tiny):
    locs = fit(tiny["tif"], tiny["model"])
    for name in ("frame", "x_nm", "y_nm", "z_nm", "photons", "background", "prob", "x_err_nm", "y_err_nm",
                 "xy_err_nm", "z_err_nm", "photons_err"):
        assert name in locs, name
    assert set(np.unique(locs["frame"])) <= set(range(40)) and locs["frame"].min() >= 0
    assert np.all((np.asarray(locs["x_nm"]) > -100) & (np.asarray(locs["x_nm"]) < 3200))


def test_blocks_of_37_frames_give_the_table_of_one_block(tiny):
    whole = fit(tiny["tif"], tiny["model"], **{"source.chunk": 500})
    _same(fit(tiny["tif"], tiny["model"], **{"source.chunk": 7}), whole)
    _same(fit(tiny["tif"], tiny["model"], **{"source.chunk": 37}), whole)
    _same(fit(tiny["tif"], tiny["model"], **{"source.chunk": 1}), whole)


def test_a_first_frame_part_way_in_sees_its_neighbours_before_it(tiny):
    """From frame 10, frame 10 is localized with frame 9 as its earlier neighbour, not repeated."""
    whole = fit(tiny["tif"], tiny["model"])
    part = fit(tiny["tif"], tiny["model"], **{"source.start": 10, "source.chunk": 9})
    later = np.asarray(whole["frame"]) >= 11                 # frame 10 itself has frame 10 repeated
    mine = np.asarray(part["frame"]) >= 11
    assert mine.sum() == later.sum() > 0
    assert np.array_equal(np.asarray(part["x_nm"])[mine], np.asarray(whole["x_nm"])[later])


def test_a_camera_roi_moves_the_positions_by_its_corner(tiny):
    from decode_smlm.predict import load_predictor
    from smappy.metadata import CameraMetadata
    from smappy.window_engine import WindowEngine

    predictor = load_predictor(tiny["model"], "cpu", raw_th=0.3)
    frames = tiny["frames"][:12]
    tables = []
    for roi in (None, (5, 7, 32, 32)):
        engine = WindowEngine(CameraMetadata(conversion=0.5, offset=100, pixelsize_um=(0.1, 0.12), roi=roi), predictor)
        tables.append([engine.push(frames, 0), engine.end()])
    for plain, cut in zip(*tables):
        assert np.allclose(np.asarray(cut["x_nm"]) - np.asarray(plain["x_nm"]), 500, atol=1e-2)
        assert np.allclose(np.asarray(cut["y_nm"]) - np.asarray(plain["y_nm"]), 840, atol=1e-2)


class _GrowingNDTiff:
    """An NDTiff dataset written a frame at a time (tests/test_live_session.py's)."""
    NAME = "Stack_NDTiffStack.tif"

    def __init__(self, folder, shape):
        from smappy.io.ndtiff import INDEX_NAME, SUMMARY_MARKER, SUMMARY_OFFSET
        self.folder, self.n, self.index = folder, 0, folder / INDEX_NAME
        folder.mkdir(parents=True)
        summary = json.dumps({"PixelSize_um": 0.1, "Height": shape[0], "Width": shape[1]}).encode()
        (folder / self.NAME).write_bytes(b"\0" * SUMMARY_OFFSET + struct.pack("<II", SUMMARY_MARKER, len(summary))
                                         + summary)
        self.index.write_bytes(b"")

    def write(self, frames):
        for frame in frames:
            stack = self.folder / self.NAME
            offset = stack.stat().st_size
            pixels = np.ascontiguousarray(frame, dtype=np.uint16)
            meta = json.dumps({"Core-Camera": "Camera", "Camera-Offset": "100",
                               "ROI": f"0-0-{frame.shape[1]}-{frame.shape[0]}"}).encode()
            with stack.open("ab") as f:
                f.write(pixels.tobytes() + meta)
            axes = json.dumps({"time": self.n}).encode()
            record = (struct.pack("<I", len(axes)) + axes + struct.pack("<I", len(self.NAME)) + self.NAME.encode()
                      + struct.pack("<8I", offset, frame.shape[1], frame.shape[0], 1, 0, offset + pixels.nbytes,
                                    len(meta), 0))
            with self.index.open("ab") as f:
                f.write(record)
            self.n += 1


def test_a_live_fit_of_a_growing_file_gives_the_table_of_the_finished_file(tiny, tmp_path):
    frames = tiny["frames"]
    data = _GrowingNDTiff(tmp_path / "ds", frames.shape[1:])
    data.write(frames[:5])
    finished = threading.Event()

    def camera():                  # blocks of whatever has arrived: 5, then 3, 9, 1, ...
        for a, b in ((5, 8), (8, 17), (17, 18), (18, 31), (31, 40)):
            time.sleep(0.15)
            data.write(frames[a:b])
        finished.set()

    writer = threading.Thread(target=camera)
    writer.start()
    live = plugins.get(PLUGIN)().run(Context(writer_finished=finished),
                                     settings(data.folder, tiny["model"], **{"source.live": True,
                                                                              "source.live_timeout": 30.0,
                                                                              "source.chunk": 10}))
    writer.join()
    assert live.data["stats"]["frames"] == len(frames)
    _same(live.locs, fit(data.folder, tiny["model"]))


def test_data_taken_under_other_conditions_than_the_models_beads_are_warned_about(tiny, monkeypatch):
    from smappy.plugins import fit_decode
    from smappy.training import model_file
    tensors, meta = model_file.read(tiny["model"])
    beads = {"name": "M2", "version": 1, "camera": "iXon", "followed": ["3DA"], "states": {"3DA": "on"},
             "channels": [{"name": "red", "wavelength_nm": 670.0}]}
    path = pathlib.Path(tiny["model"]).with_name("conditions.safetensors")
    model_file.write(path, tensors, {**meta, "calibration_microscope": beads, "calibration_source": "/b/beads.h5",
                                     "test_metrics": {"jaccard": 0.37, "rmse_lat_nm": 47.0}})
    data = {**beads, "states": {"3DA": "off"}, "channels": [{"name": "green", "wavelength_nm": 580.0}]}
    monkeypatch.setattr(fit_decode.DecodeFit, "data_conditions", lambda self, s: data)
    plugin = plugins.get(PLUGIN)()
    s = settings(tiny["tif"], path)
    notice = plugin.describe(None, s).notice.text
    assert "3DA on for the beads, off for the data" in notice and "1 more" in notice
    result = plugin.run(Context(), s)
    assert any("another colour: 670 nm" in line for line in result.text.splitlines())
    rows = dict(fit_decode.model_summary(model_file.read_metadata(path), data))
    assert rows["PSF calibration"] == "beads.h5" and "Jaccard 0.37" in rows["test set"]
    # the same conditions: no warning, the model in a line
    monkeypatch.setattr(fit_decode.DecodeFit, "data_conditions", lambda self, s: beads)
    assert plugin.describe(None, s).notice.text.startswith("model: trained on beads.h5, microscope M2")


def test_conditions_differ_by_microscope_device_and_colour_and_never_where_unknown():
    from smappy.calibration_list import differences
    beads = {"name": "M2", "version": 1, "followed": ["3DA"], "states": {"3DA": "on"},
             "channels": [{"name": "red", "wavelength_nm": 670.0}]}
    assert differences(beads, dict(beads)) == []
    assert differences(beads, None) == [] and differences({}, beads) == []
    assert differences(beads, {**beads, "name": "M1"}) == ["another microscope: M2 for the model's beads, M1 for the data"]
    assert differences(beads, {**beads, "states": {"3DA": "off"}}) == ["3DA on for the beads, off for the data"]
    assert "another colour" in differences(beads, {**beads, "channels": [{"wavelength_nm": 580.0}]})[0]


def test_the_batch_on_auto_fills_half_the_free_memory_and_is_fixed_on_the_cpu(monkeypatch):
    from decode_smlm import predict
    assert predict.auto_batch("cpu", 512, 512) == predict.CPU_BATCH
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (40e9, 80e9))
    assert predict.auto_batch("cuda", 512, 512) == int(0.5 * 40e9 // (predict.BYTES_PER_PIXEL * 512 * 512))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (1e6, 80e9))
    assert predict.auto_batch("cuda", 512, 512) == 1
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (1e15, 1e15))
    assert predict.auto_batch("cuda:0", 64, 64) == predict.MAX_BATCH


def test_frames_taken_with_em_gain_are_flipped_for_a_model_of_beads_without(tiny):
    """The tiny model's calibration is of beads without EM gain: EM frames go in mirrored, and the
    positions come back on the frames as they are -- the mirror image of the unflipped run's."""
    from decode_smlm.predict import load_predictor
    plain = load_predictor(tiny["model"], "cpu", raw_th=0.3)
    flipped = load_predictor(tiny["model"], "cpu", raw_th=0.3, frames_mirrored=True)
    photons = (tiny["frames"][:6].astype(np.float32) - 100) * 0.5
    a = plain(photons, 0)
    b = flipped(photons[..., ::-1].copy(), 0)            # the mirror image, flipped back in
    assert len(a["x_pix"]) == len(b["x_pix"]) > 0
    assert np.allclose(np.sort(31 - a["x_pix"]), np.sort(b["x_pix"]), atol=1e-3)
    p = plugins.get(PLUGIN)()
    em = settings(tiny["tif"], tiny["model"], **{"camera.em_on": True, "camera.emgain": 100.0})
    model = p.model(em, em.camera.resolve(p.open_source(em)))
    assert model.frames_mirrored


# ------------------------------------------------------------------------- the original's model
needed = [PACKAGE / "param_run.yaml", PACKAGE / "model.pt", PACKAGE / "spline_calibration_3dcal.mat",
          ORACLE / "v0102_model_emitters_cpu.h5", ORACLE / "frames_500.tif"]


@pytest.mark.slow
@pytest.mark.skipif(not all(p.exists() for p in needed), reason=f"the example package and its oracle are not in {DATA}")
def test_the_fitters_table_of_the_packages_frames_is_the_originals(tmp_path):
    """The example package's model, exported to a model file, through the plugin: every one of the
    original's localizations, on the frames as they are (cropped and flipped back), in order."""
    import h5py

    from decode_smlm.io import crop_of
    from decode_smlm.predict import export_original
    from decode_smlm.utils.hardware import default_device
    from smappy.io.calibration import load_spline_calibration

    device = default_device()
    n = 500 if device != "cpu" else 30
    cal = PACKAGE / "spline_calibration_3dcal.mat"
    model = export_original(PACKAGE / "param_run.yaml", PACKAGE / "model.pt", tmp_path / "package.safetensors",
                            calibration=load_spline_calibration(cal))
    frames = tifffile.imread(ORACLE / "frames_500.tif")[:n]
    tifffile.imwrite(tmp_path / "frames.tif", frames)
    # the oracle was made with the training's camera, the Fit notebook's (plan §8, "(a) as run")
    locs = plugins.get(PLUGIN)().run(Context(), settings(
        tmp_path / "frames.tif", model, **{"camera.conversion": 5.0, "camera.offset": 398.6, "camera.em_on": True,
                                           "camera.emgain": 100.0, "model.device": device,
                                           "model.threshold": 0.1})).locs
    with h5py.File(ORACLE / "v0102_model_emitters_cpu.h5") as f:
        ref = {k: f["data"][k][()] for k in ("xyz", "frame_ix", "phot", "prob")}
    keep = ref["frame_ix"] < n
    ref = {k: v[keep] for k, v in ref.items()}
    row0, col0, _, w = crop_of(frames.shape)
    assert len(locs) == len(ref["xyz"])
    assert np.array_equal(np.asarray(locs["frame"]), ref["frame_ix"])
    x = (w - 1 - ref["xyz"][:, 1] + col0) * 100.0
    y = (ref["xyz"][:, 0] + row0) * 100.0
    assert np.abs(np.asarray(locs["x_nm"]) - x).max() < 0.01           # 1e-4 px
    assert np.abs(np.asarray(locs["y_nm"]) - y).max() < 0.01
    assert np.abs(np.asarray(locs["z_nm"]) - ref["xyz"][:, 2]).max() < 0.05
    assert np.abs(np.asarray(locs["photons"]) - ref["phot"]).max() <= 1e-4 * ref["phot"].max()


def test_a_fit_outside_the_trained_range_says_so_in_its_text_and_its_file(tiny, tmp_path, monkeypatch):
    from smappy.io.hdf5 import load_localizations
    from smappy.plugins import fit_decode
    real = fit_decode.read_model

    def narrow(path):                       # a model that drew backgrounds no fit here has
        meta = real(path)
        return {**meta, "trained_on": {**(meta.get("trained_on") or {}), "background": [1e5, 2e5]}}
    monkeypatch.setattr(fit_decode, "read_model", narrow)
    out = tmp_path / "fit.hdf5"
    result = plugins.get(PLUGIN)().run(Context(), settings(tiny["tif"], tiny["model"], **{"output.path": str(out)}))
    said = [line for line in result.text.splitlines() if "background outside the training's" in line]
    assert len(said) == 1 and said[0].startswith("warning:")             # once, not again by the plugin
    assert any("background outside" in w for w in load_localizations(out).metadata["warnings"])


def test_the_model_part_opens_the_training_window_with_the_fitters_settings_and_the_hosts():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.plugins.fit_decode import DecodeFit
    from smappy.session import Session
    session = Session()
    opened = []
    session.window_openers["decode training"] = lambda settings=None: opened.append(("training", settings))
    session.window_openers["training hosts"] = lambda settings=None: opened.append(("hosts", None))
    panel = PluginPanel(DecodeFit, session)
    assert DecodeFit.description == "Deep-learning fitter for high densities (DECODE)."
    assert DecodeFit.window_buttons == ()
    buttons = panel.part_window_buttons
    assert set(buttons) == {"Model info...", "Train a model...", "Training hosts..."}
    panel.form.set_values({"source.path": "/data/cells.tif"})
    buttons["Train a model..."].click()
    buttons["Training hosts..."].click()
    assert [kind for kind, _ in opened] == ["training", "hosts"]
    assert opened[0][1].source.path == "/data/cells.tif" and opened[0][1].model.run_on == "here"


def test_on_apples_gpu_the_batch_keeps_to_a_fixed_budget_of_the_shared_memory(monkeypatch):
    from decode_smlm import predict
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 27e9, raising=False)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 0, raising=False)
    # half of 27 GB would be 98 frames of 257 x 213; the budget allows 21
    assert predict.auto_batch("mps", 257, 213) == int(predict.MPS_BUDGET // (predict.BYTES_PER_PIXEL * 257 * 213))
    assert predict.auto_batch("mps", 257, 213) < 25
