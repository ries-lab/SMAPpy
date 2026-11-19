"""decode_smlm.plex, the port of DECODE-Plex, against the original's outputs (docs/decode/plex-plan.md §5).

The oracles come from DECODE-Plex itself (``scripts/decode/reference_plex.py``, run in its own
environment on decodebox):

* ``fixtures/plex_arch_{biplane,colour}.npz`` (in git): a biplane and a colour run's network shrunk
  to depth 1 and 8 features, with its seeded weights, its output on fixed inputs, and the
  post-processing of a made output -- the wiring (a U-Net per channel, each frame alone, the offset
  maps into every stage), the activations and ``SpatialIntegration``, without a 90 MB weight file;
* ``$DECODE_SMLM_DATA/plex/oracles/plex_<run>_cpu.npz`` (the cache): three published models on
  the first 200 frames of their movies through their notebooks' steps -- Fig. 1e (the SMLM
  Challenge's biplane ``MT0.N2.HD``, no registration), Fig. 2c (live-cell biplane, a registration
  and a mirror, window 1) and Fig. 2a (two colours, a registration, window 3).  Without the cache
  the test skips.  The CPU's oracle is the reference: the original's run on the 4090 differs from
  it by up to 0.09 in the raw output and by 20 localizations of 339 149 on Fig. 2c, TF32 being
  torch's default for convolutions there.
"""
import json
import math
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

FIXTURES = pathlib.Path(__file__).parents[1] / "fixtures"
DATA = pathlib.Path(os.environ.get("DECODE_SMLM_DATA", pathlib.Path.home() / ".cache" / "decode_smlm")) / "plex"
# run: training folder, frames, registration, lower half upside down, batch, frames the CPU is checked on
RUNS = {
    "fig1e": ("outputs/Fig1e-challenge-2026-05-27_15-24-26-229646",
              "Fig1e-Challenge/sequence-MT0.N2.HD-BP-Exp-as-stack.tif", None, True, 16, 48),
    "fig2c": ("outputs/Fig2c-biplane-live_cell-separated_ch-2026-05-15_13-39-38-146351",
              "Fig2c-240619_NC_LivingCell_HMSir_Tub_i20_2/first300.tif",
              "calibration/Fig1c_2c-Pos0_240621_NC_BeadCal_BifocalPlane_Z_1_MMStack_Default.ome_trafo.mat",
              False, 4, 12),
    "fig2a": ("outputs/Fig2a-dual_color-actin_microtubes-separated_ch-2026-05-14_17-22-25-738645",
              "Fig2a-SmallTest/first300.tif",
              "calibration/Fig2a-Pos0_MM_100ms_638i1_T700(100)_R685(70)_665LP_20nmstep_Z_1_up_noEM_trafo.mat",
              False, 4, 8),
}

EMITTER_FIELDS = ("xyz", "xyz_sig", "phot", "phot_sig", "bg", "prob")


def _arch(name):
    from decode_smlm.plex.neuralfitter.train import setup_cfg

    o = np.load(FIXTURES / f"plex_arch_{name}.npz")
    cfg = json.loads(str(o["cfg"]))
    model = setup_cfg.setup_model(cfg).eval()
    model.load_state_dict({k[2:]: torch.from_numpy(o[k]) for k in o.files if k.startswith("w:")})
    return cfg, model, o


@pytest.mark.parametrize("name", ["biplane", "colour"])
def test_the_networks_wiring_gives_the_originals_output(name):
    """A network per channel, each frame alone through it, the offset maps into every stage, the
    union and the heads: the port's output on the oracle's input and weights is the original's."""
    cfg, model, o = _arch(name)
    with torch.no_grad():
        y = model(torch.from_numpy(o["x"]))
    assert y.shape == o["y"].shape
    # float32 through a network of random weights: the port's output and the original's each lie
    # 2.3e-5 from the same network in float64 (2026-11-19), so they differ by up to 3.3e-5
    np.testing.assert_allclose(y.numpy(), o["y"], rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("name", ["biplane", "colour"])
def test_spatial_integration_finds_the_originals_localizations(name):
    """Peaks alone, side by side and below the threshold, in one photon count (biplane) or two
    (colour): the same localizations, in the same order."""
    from decode_smlm.plex.neuralfitter.train import setup_cfg

    cfg, _, o = _arch(name)
    em = setup_cfg.setup_post_process(cfg, cfg["Test"]).forward(torch.from_numpy(o["y_post"]))
    assert len(em) == len(o["em_xyz"]) == 6
    np.testing.assert_array_equal(em.frame_ix.numpy(), o["em_frame_ix"])
    for field in EMITTER_FIELDS:
        np.testing.assert_allclose(getattr(em, field).numpy(), o[f"em_{field}"], rtol=1e-6, atol=1e-6)


def test_reading_a_model_and_inferring_imports_no_lightning_hydra_or_the_compiled_spline():
    """DECODE-Plex's package imported everything; the port's inference needs torch, numpy, scipy,
    h5py and yaml only (plex-plan §3)."""
    code = ("import sys; import decode_smlm.plex.neuralfitter.inference.functional, decode_smlm.plex.io.model; "
            "heavy = ('pytorch_lightning', 'lightning', 'omegaconf', 'hydra', 'spline', 'seaborn', 'sklearn', "
            "'torchmetrics', 'skimage', 'structlog', 'pydantic', 'deprecated', 'dotted_dict', 'pandas'); "
            "print(sorted(m for m in heavy if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout + out.stderr


def test_importing_decode_smlm_plex_alone_loads_no_torch():
    code = "import sys, decode_smlm.plex; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split()[-1] == "False", out.stdout + out.stderr


def test_a_runs_settings_are_read_as_omegaconf_read_them(tmp_path):
    """OmegaConf's loader reads ``1e3`` as a float and a date as a string, where PyYAML would not."""
    from decode_smlm.plex.io import param

    p = tmp_path / "run.yaml"
    p.write_text("a: 1e3\nb: 1.0e-4\nc: 2026-05-27\nd: ${Paths.experiment}\ne: [1, 2.5]\nf: null\n'y': 3\n")
    cfg = param.load(p)
    assert cfg == {"a": 1000.0, "b": 1e-4, "c": "2026-05-27", "d": "${Paths.experiment}", "e": [1, 2.5],
                   "f": None, "y": 3}
    assert isinstance(cfg["a"], float)


def test_emitter_data_is_coerced_and_checked_as_pydantic_did():
    """``EmitterData`` without pydantic: frame indices to long, 2D coordinates padded with z = 0,
    unequal lengths and non-integer frames refused."""
    from decode_smlm.plex.emitter.emitter import EmitterSet
    from decode_smlm.plex.generic.validated import ValidationError

    em = EmitterSet(xyz=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], phot=[1.0, 2.0], frame_ix=torch.tensor([0.0, 1.0]),
                    xy_unit="px")
    assert em.frame_ix.dtype == torch.long and em.prob is None
    assert EmitterSet(xyz=[[1.0, 2.0]], phot=[1.0], frame_ix=[0], xy_unit="px").xyz.tolist() == [[1.0, 2.0, 0.0]]
    with pytest.raises(ValidationError, match="Unequal length"):
        EmitterSet(xyz=[[1.0, 2.0, 3.0]], phot=[1.0, 2.0], frame_ix=[0, 1], xy_unit="px")
    with pytest.raises(ValidationError, match="not integer"):
        EmitterSet(xyz=[[1.0, 2.0, 3.0]], phot=[1.0], frame_ix=[0.5], xy_unit="px")


def test_validate_arguments_coerces_as_pydantic_did():
    from decode_smlm.plex.generic import validated

    @validated.validate_arguments
    def f(n: int, x: validated.StrictInt = 0, t: tuple[int, int] | None = None):
        return n, x, t

    assert f(np.int64(3)) == (3, 0, None)
    assert f(torch.tensor(4), 2, [1, 2]) == (4, 2, (1, 2))
    assert f(2.7)[0] == 2                       # pydantic 1's int(): truncated, not refused
    with pytest.raises(validated.ValidationError):
        f(1, 2.0)                               # StrictInt takes an int and nothing else
    with pytest.raises(validated.ValidationError):
        f(1, True)


def _device():
    from decode_smlm.utils.hardware import default_device

    return default_device()


def _matched(mine: dict, ref: dict, n_frames: int):
    """Indices pairing the two sets of localizations frame by frame by position (within 0.01 px).

    Order within a frame is the post-processing's and differs between devices, and a spot near the
    threshold, or between two pixels, can come out on one side and not the other."""
    from scipy.spatial import cKDTree

    ia, ib = [], []
    for f in range(n_frames):
        a = np.flatnonzero(mine["frame_ix"] == f)
        b = np.flatnonzero(ref["frame_ix"] == f)
        if len(a) and len(b):
            d, j = cKDTree(ref["xyz"][b, :2]).query(mine["xyz"][a, :2])
            ia += list(a[d < 0.01])
            ib += list(b[j[d < 0.01]])
    return np.array(ia, dtype=int), np.array(ib, dtype=int)


@pytest.mark.slow
@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("run", sorted(RUNS))
def test_a_published_model_gives_the_originals_localizations(run, device):
    """A model of the paper, its checkpoint read without Lightning, on the frames of its figure
    through its notebook's steps: the network's input and output, and the localizations to float
    precision (plex-plan §5: x and y within 1e-4 px, z within 0.05 nm), all but one in 10 000 of
    them found by both."""
    import decode_smlm.plex as dp
    import decode_smlm.plex.neuralfitter.inference.functional as infer_func
    from decode_smlm.plex.neuralfitter.inference.debug import InferenceLogger

    training, frames_file, trafo, flip, batch, n_cpu = RUNS[run]
    oracle = DATA / "oracles" / f"plex_{run}_cpu.npz"
    needed = [DATA / training, DATA / frames_file, oracle] + ([DATA / trafo] if trafo else [])
    if not all(p.exists() for p in needed):
        pytest.skip(f"DECODE-Plex's {run} model, frames or oracle are not in {DATA}")
    dev = _device() if device == "gpu" else "cpu"
    if dev == "cpu" and device == "gpu":
        pytest.skip("no GPU")
    n = 100 if device == "gpu" else n_cpu

    cfg = dp.io.param.load(DATA / training / "param_run.yaml")
    for s in cfg["Hardware"]["device"]:
        cfg["Hardware"]["device"][s] = dev
    for cam in cfg["Camera"]:
        cam["specs"]["flip"]["channel"] = -2             # as the notebooks set it
    window = cfg["Trainer"]["frame_window"]
    frames = dp.io.frames.load_tif(DATA / frames_file, auto_ome=True, memmap=True)[:n].clone()
    if flip:
        h = frames.size(-2) // 2
        frames[:, h:, :] = frames[:, h:, :].flip(1)      # the challenge's lower plane is upside down
    crop = [math.floor(frames.size(-2) / 2 / 8) * 8, math.floor(frames.size(-1) / 8) * 8]
    log = InferenceLogger(interval=1, max_steps=1)
    em, log = infer_func.infer(frames, frame_crop=crop, cfg=cfg, model=next((DATA / training).glob("*.ckpt")),
                               mode="multi", trafo=(DATA / trafo) if trafo else None, mode_camera="rois",
                               roi_shift=None, logger=log, batch_size=batch, num_workers=0, device=[dev])

    o = np.load(oracle)
    k = len(log.model_in[0])
    # the offset maps come from the registration in float32: 3e-6 apart on Fig. 2c
    np.testing.assert_allclose(log.model_in[0].numpy(), o["model_in"][:k], atol=1e-5)
    np.testing.assert_allclose(log.model_out[0].cpu().numpy(), o["model_out"][:k], atol=5e-4)

    # a shortened stack's last frame sees a padded window where the oracle's did not
    last = n - 1 if window > 1 else n
    mine = {f: getattr(em, f).cpu().numpy() for f in ("xyz", "phot", "prob", "frame_ix")}
    ref = {f: o[f"em_{f}"] for f in ("xyz", "phot", "prob", "frame_ix")}
    mine = {f: v[mine["frame_ix"] < last] for f, v in mine.items()}
    ref = {f: v[ref["frame_ix"] < last] for f, v in ref.items()}
    ia, ib = _matched(mine, ref, last)
    allowed = max(1, len(ref["prob"]) // 10_000)
    assert len(mine["prob"]) - len(ia) <= allowed and len(ref["prob"]) - len(ib) <= allowed, \
        (len(mine["prob"]), len(ref["prob"]), len(ia))
    np.testing.assert_allclose(mine["xyz"][ia, :2], ref["xyz"][ib, :2], atol=1e-4)
    np.testing.assert_allclose(mine["xyz"][ia, 2], ref["xyz"][ib, 2], atol=0.05)
    np.testing.assert_allclose(mine["phot"][ia], ref["phot"][ib], rtol=1e-4, atol=0.01)
    np.testing.assert_allclose(mine["prob"][ia], ref["prob"][ib], atol=1e-4)
