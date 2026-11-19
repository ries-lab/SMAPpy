"""The reference outputs of the original DECODE-Plex, which its port (decode_smlm.plex) is checked against.

docs/decode/plex-plan.md §5.  Runs in DECODE-Plex's own environment, with its checkout on the path,
and imports nothing of SMAPpy, so that what it writes is DECODE-Plex's and not ours:

    PYTHONPATH=~/decode_plex/DECODE-Plex python reference_plex.py arch DATA OUT_DIR
    PYTHONPATH=~/decode_plex/DECODE-Plex python reference_plex.py model fig1e DATA OUT_DIR --frames 200

DATA holds the files of the Zenodo release (record 21348704) as its archives name them:
``outputs/<run>/``, ``calibration/``, ``Fig1e-Challenge/``, the movies.

``arch`` writes the small fixture, on the CPU so that two runs are bitwise equal: for a biplane and
a colour run's settings, the network shrunk to depth 1 and 8 features (its wiring unchanged: a
U-Net per channel, each frame alone, the offset maps into every stage), its seeded weights, its
output on fixed inputs, and the post-processing of a made output whose probabilities have peaks,
single and side by side, so that ``SpatialIntegration`` has something to join.

``model`` runs a published model on the first frames of its movie through the steps of its own
notebook (``notebook/biplane/challenge.ipynb`` for Fig. 1e, ``2_inference.ipynb`` for the others):
the network's input and output of the first batches, and all localizations (too large for git:
the cache).

Units are DECODE's: x and y in pixels, the first array axis being x; z in nm; photons.
"""
import argparse
import hashlib
import json
import math
import pathlib
import platform
import sys

import numpy as np
import torch

import decode
import decode.neuralfitter.inference.functional as infer_func
from decode.neuralfitter.inference.debug import InferenceLogger
from decode.neuralfitter.train import setup_cfg

SEED = 20261119

RUNS = {
    # the SMLM Challenge 2016's biplane MT0.N2.HD: both planes in one frame, the second (lower
    # half) upside down; no registration, one grid for both
    "fig1e": dict(training="outputs/Fig1e-challenge-2026-05-27_15-24-26-229646",
                  frames="Fig1e-Challenge/sequence-MT0.N2.HD-BP-Exp-as-stack.tif",
                  trafo=None, flip_lower_half=True, batch=16),
    # the live-cell biplane, window 1, with its registration
    "fig2c": dict(training="outputs/Fig2c-biplane-live_cell-separated_ch-2026-05-15_13-39-38-146351",
                  frames="Fig2c-240619_NC_LivingCell_HMSir_Tub_i20_2/"
                         "240619_NC_LivingCell_HMSir_Tub_i20_2_MMStack_Default.ome.tif",
                  trafo="calibration/Fig1c_2c-Pos0_240621_NC_BeadCal_BifocalPlane_Z_1_MMStack_Default.ome_trafo.mat",
                  flip_lower_half=False, batch=4),
    # dual colour, the small test movie
    "fig2a": dict(training="outputs/Fig2a-dual_color-actin_microtubes-separated_ch-2026-05-14_17-22-25-738645",
                  frames="Fig2a-SmallTest/MM_25ms_638i100_UVi20_focusloci30_1_MMStack_Default_3.ome.tif",
                  trafo="calibration/Fig2a-Pos0_MM_100ms_638i1_T700(100)_R685(70)_665LP_20nmstep_Z_1_up_noEM_trafo.mat",
                  flip_lower_half=False, batch=4),
}

EMITTER_FIELDS = ("xyz", "xyz_sig", "phot", "phot_sig", "bg", "bg_sig", "prob", "frame_ix", "code")


def emitter_arrays(em, prefix: str) -> dict:
    out = {}
    for k in EMITTER_FIELDS:
        v = getattr(em, k, None)
        if v is not None:
            out[f"{prefix}{k}"] = v.detach().cpu().numpy()
    return out


def load_cfg(data: pathlib.Path, run: str, device: str) -> dict:
    cfg = decode.io.param.load(data / RUNS[run]["training"] / "param_run.yaml")
    for s in cfg["Hardware"]["device"]:
        cfg["Hardware"]["device"][s] = device
    return cfg


def arch_case(data: pathlib.Path, run: str, size: int = 32) -> dict:
    """The network of ``run``'s settings at depth 1 and 8 features, on fixed inputs."""
    cfg = load_cfg(data, run, "cpu")
    specs = cfg["Model"]["backbone_specs"]
    specs.update(depth_shared=1, depth_union=1, initial_features=8, inter_features=8)
    cfg = decode.io.param.patch_img_size(cfg, (size, size), scenario=["Simulation", "Test"])

    torch.manual_seed(SEED)
    model = setup_cfg.setup_model(cfg).eval()
    n_ch = cfg["Simulation"]["channels"]
    n_in = cfg["Trainer"]["frame_window"] * n_ch + 2 * n_ch
    g = torch.Generator().manual_seed(SEED + 1)
    x = torch.randn(2, n_in, size, size, generator=g)
    with torch.no_grad():
        y = model(x)

    # a made output for the post-processing: the network's, its probabilities replaced by peaks --
    # alone, two side by side (which SpatialIntegration joins or splits), one below raw_th
    ch_map = setup_cfg.setup_ch_map_model_out(cfg)
    y_post = y.clone()
    p = torch.rand(2, 1, size, size, generator=g) * 0.05
    for b, (i, j, v) in [(0, (5, 7, 0.9)), (0, (20, 11, 0.55)), (0, (20, 12, 0.5)), (0, (9, 25, 0.08)),
                         (1, (3, 3, 0.7)), (1, (16, 16, 0.35)), (1, (17, 16, 0.35)), (1, (30, 29, 0.95))]:
        p[b, 0, i, j] = v
    y_post[:, ch_map.ix_prob] = p
    post = setup_cfg.setup_post_process(cfg, cfg["Test"])
    em = post.forward(y_post)

    state = {f"w:{k}": v.numpy() for k, v in model.state_dict().items()}
    return {"cfg": json.dumps(cfg), "x": x.numpy(), "y": y.numpy(), "y_post": y_post.numpy(),
            **state, **emitter_arrays(em, "em_")}


def run_arch(data: pathlib.Path, out: pathlib.Path):
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"environment": environment(), "files": {}}
    for name, run in (("biplane", "fig1e"), ("colour", "fig2a")):
        path = out / f"plex_arch_{name}.npz"
        np.savez_compressed(path, **arch_case(data, run))
        meta["files"][path.name] = {"run": RUNS[run]["training"], "sha256": sha256(path),
                                    "bytes": path.stat().st_size}
    (out / "plex_oracles.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def run_model(run: str, data: pathlib.Path, out: pathlib.Path, device: str, n_frames: int,
              n_batches: int):
    spec = RUNS[run]
    cfg = load_cfg(data, run, device)
    for cam in cfg["Camera"]:
        cam["specs"]["flip"]["channel"] = -2  # as the notebooks set it
    path_ckpt = next((data / spec["training"]).glob("*.ckpt"))

    frames = decode.io.frames.load_tif(data / spec["frames"], auto_ome=True, memmap=True)
    frames = frames[:n_frames].clone()
    if spec["flip_lower_half"]:
        h = frames.size(-2) // 2
        frames[:, h:, :] = frames[:, h:, :].flip(1)
    frame_size = list(frames.size())
    frame_crop = [math.floor(frame_size[-2] / 2 / 8) * 8, math.floor(frame_size[-1] / 8) * 8]

    log = InferenceLogger(interval=1, max_steps=n_batches)
    em_out, log = infer_func.infer(
        frames, frame_crop=frame_crop, cfg=cfg, model=path_ckpt, mode="multi",
        trafo=(data / spec["trafo"]) if spec["trafo"] else None, mode_camera="rois",
        roi_shift=None, logger=log, batch_size=spec["batch"], num_workers=0, device=[device])

    out.mkdir(parents=True, exist_ok=True)
    arrays = {"model_in": torch.cat([t.cpu() for t in log.model_in[:n_batches]]).numpy(),
              "model_out": torch.cat([t.cpu() for t in log.model_out[:n_batches]]).numpy(),
              "px_size": np.asarray(em_out.px_size, dtype=np.float64),
              **emitter_arrays(em_out, "em_")}
    path = out / f"plex_{run}_{device.split(':')[0]}.npz"
    np.savez_compressed(path, **arrays)
    meta = {"run": spec, "ckpt": path_ckpt.name, "ckpt_sha256": sha256(path_ckpt),
            "frames": n_frames, "frame_size": frame_size, "frame_crop": frame_crop,
            "n_localizations": len(em_out), "environment": environment(), "sha256": sha256(path)}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "environment"}, indent=1))


def environment() -> dict:
    import importlib.metadata as md

    versions = {}
    for p in ("torch", "numpy", "scipy", "pytorch-lightning", "pydantic", "omegaconf", "hydra-core",
              "spline", "tifffile", "h5py"):
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "decode": decode.__version__, "decode_file": decode.__file__,
            "cuda": torch.cuda.get_device_name() if torch.cuda.is_available() else None, **versions}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arch")
    a.add_argument("data", type=pathlib.Path)
    a.add_argument("out", type=pathlib.Path)
    m = sub.add_parser("model")
    m.add_argument("run", choices=sorted(RUNS))
    m.add_argument("data", type=pathlib.Path)
    m.add_argument("out", type=pathlib.Path)
    m.add_argument("--device", default="cpu")
    m.add_argument("--frames", type=int, default=200)
    m.add_argument("--batches", type=int, default=2, help="batches whose raw input and output are kept")
    args = ap.parse_args()
    if args.cmd == "arch":
        run_arch(args.data, args.out)
    else:
        run_model(args.run, args.data, args.out, args.device, args.frames, args.batches)


if __name__ == "__main__":
    main()
