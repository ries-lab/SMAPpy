"""The reference outputs of the original DECODE v0.10.2, which the port is checked against.

Milestone (a) of docs/decode/poc-plan.md.  Runs in the environment of setup_v0102.sh and imports
nothing of SMAPpy, so that what it writes is DECODE's and not ours:

    GLIBC_TUNABLES=glibc.rtld.execstack=2 python reference_v0102.py oracles OUT_DIR
    GLIBC_TUNABLES=glibc.rtld.execstack=2 python reference_v0102.py model PACKAGE_DIR OUT_DIR

``oracles`` writes the two small fixtures, computed on the CPU so that two runs are bitwise equal:

* ``v0102_renderer.npz``: ``CubicSplinePSF``'s ROIs, frames and derivatives for emitters chosen to
  reach the edge cases -- on the frame's border, half out of the frame, outside the calibration's
  z range -- from a synthetic SMAP calibration written by the script.  The frame is not square
  (32 x 24) so that an exchange of DECODE's x (the first axis) and y shows.
* ``v0102_method.npz``: ``GaussianMMLoss`` and its gradient, the targets of
  ``ParameterListTarget`` and the post-processing of ``SpatialIntegration``, on fixed tensors.

``model`` runs the example package's trained model on its first frames through the Fit
notebook's steps and writes the raw output and the localizations (too large for git: the cache).

Units are DECODE's: x and y in pixels, the first array axis being x; z in nm; photons.
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import scipy.io as sio
import torch

import decode
from decode.generic.emitter import EmitterSet
from decode.neuralfitter.coord_transform import Offset2Coordinate
from decode.neuralfitter.loss import GaussianMMLoss
from decode.neuralfitter.post_processing import SpatialIntegration
from decode.neuralfitter.scale_transform import InverseParamListRescale, ParameterListRescale
from decode.neuralfitter.target_generator import ParameterListTarget
from decode.utils.calibration_io import SMAPSplineCoefficient

SEED = 20261110

# The synthetic calibration: small, so the fixture stays under 1 MB, with a z grid of 15 nodes.
CAL_SHAPE = (9, 9, 15)
CAL_DZ = 50.0
CAL_X0 = 5.0                      # MATLAB's 1-based centre: DECODE's ref0 is x0 - 1 = 4
CAL_Z0 = 7.0                      # 0-based, as SMAP writes it

FRAME_SHAPE = (32, 24)            # (x, y) in DECODE's order
XEXTENT = (-0.5, FRAME_SHAPE[0] - 0.5)
YEXTENT = (-0.5, FRAME_SHAPE[1] - 0.5)
N_FRAMES = 16
# the derivatives for every 7th emitter: 148 of the 1032, every kind among them
DRV_SUBSET = slice(0, None, 7)

# The method oracle: a small batch on a small frame, the scaling of the example package.
M_BATCH, M_H, M_W, M_NMAX = 4, 16, 12, 64
PHOT_MAX, Z_MAX, BG_MAX = 15000.0, 960.0, 240.0


def synthetic_calibration(path: pathlib.Path) -> np.ndarray:
    """Write a SMAP ``_3dcal.mat`` with seeded coefficients and return them.

    The coefficients need not make a physical PSF: the oracle is of the evaluation, which is the
    same arithmetic whatever the numbers.  A decaying envelope keeps the values a PSF's order of
    magnitude, so the comparison's tolerances mean something.  Every coefficient is a multiple of
    2**-12 below 2**-4, exact in float32 and few bits each, so the fixture compresses.
    """
    rng = np.random.default_rng(SEED)
    gx = np.exp(-0.5 * ((np.arange(CAL_SHAPE[0]) - CAL_X0 + 1) / 2.0) ** 2)
    env = gx[:, None, None, None] * gx[None, :, None, None]
    coeff = rng.normal(0.0, 4.0, (*CAL_SHAPE, 64)) * env
    coeff[..., 0] += 40.0 * env[..., 0]                    # a positive constant term
    coeff = (np.clip(np.round(coeff), -255, 255) / 4096.0).astype(np.float32)
    sio.savemat(path, {"SXY": {"cspline": {"coeff": coeff, "x0": CAL_X0, "z0": CAL_Z0,
                                           "dz": CAL_DZ}}})
    return coeff


def renderer_emitters(rng: np.random.Generator) -> dict:
    """About 1000 emitters, a quarter of them on purpose where an implementation can differ."""
    n_inside = 760
    z_lo, z_hi = -CAL_Z0 * CAL_DZ, (CAL_SHAPE[2] - 1 - CAL_Z0) * CAL_DZ
    x = rng.uniform(*XEXTENT, n_inside)
    y = rng.uniform(*YEXTENT, n_inside)
    z = rng.uniform(z_lo, z_hi, n_inside)

    def edge(n):  # on or across the frame's border, in x or in y
        u = rng.uniform(-3.0, 3.0, n)
        side = rng.integers(0, 4, n)
        ex = np.where(side == 0, XEXTENT[0] + u, np.where(side == 1, XEXTENT[1] + u, rng.uniform(*XEXTENT, n)))
        ey = np.where(side == 2, YEXTENT[0] + u, np.where(side == 3, YEXTENT[1] + u, rng.uniform(*YEXTENT, n)))
        return ex, ey

    ex, ey = edge(120)
    ez = rng.uniform(z_lo, z_hi, 120)
    ox, oy = rng.uniform(*XEXTENT, 80), rng.uniform(*YEXTENT, 80)   # z outside the calibration
    oz = np.concatenate([rng.uniform(z_lo - 300, z_lo, 40), rng.uniform(z_hi, z_hi + 300, 40)])
    # exactly on pixel centres and voxel boundaries, where a floor or a round can differ
    gx, gy = np.meshgrid(np.arange(0.0, 6.0), np.arange(0.0, 6.0), indexing="ij")
    gz = np.resize(np.array([z_lo, 0.0, z_hi, CAL_DZ * 0.5, -CAL_DZ]), gx.size)
    xyz = np.stack([np.concatenate([x, ex, ox, gx.ravel() + 0.0, gx.ravel() + 0.5]),
                    np.concatenate([y, ey, oy, gy.ravel() + 0.0, gy.ravel() + 0.5]),
                    np.concatenate([z, ez, oz, gz, gz])], 1)
    n = len(xyz)
    phot = rng.uniform(500.0, 8000.0, n)
    frame = rng.integers(0, N_FRAMES, n)
    bg = rng.uniform(0.0, 100.0, n)
    kind = np.concatenate([np.zeros(n_inside), np.ones(120), np.full(80, 2), np.full(2 * gx.size, 3)])
    return {"xyz": xyz.astype(np.float32), "phot": phot.astype(np.float32),
            "frame_ix": frame.astype(np.int64), "bg": bg.astype(np.float32),
            "kind": kind.astype(np.int8)}


def renderer_oracle(out: pathlib.Path) -> dict:
    cal_path = out / "v0102_synthetic_3dcal.mat"
    coeff = synthetic_calibration(cal_path)
    cal = SMAPSplineCoefficient(str(cal_path))
    em = renderer_emitters(np.random.default_rng(SEED + 1))
    xyz, phot = torch.from_numpy(em["xyz"]), torch.from_numpy(em["phot"])
    frame_ix, bg = torch.from_numpy(em["frame_ix"]), torch.from_numpy(em["bg"])

    res = {}
    for name, dev in (("cpu", "cpu"), ("cuda", "cuda:0")):
        if dev != "cpu" and not torch.cuda.is_available():
            continue
        psf = cal.init_spline(XEXTENT, YEXTENT, FRAME_SHAPE, device=dev)
        rois = psf.forward_rois(xyz, phot)
        frames = psf.forward(xyz, phot, frame_ix, 0, N_FRAMES - 1)
        d = DRV_SUBSET                                   # d/d(x, y, z, phot, bg); ROIs with bg
        drv, rois_bg = psf.derivative(xyz[d], phot[d], bg[d])
        res[name] = {"rois": rois.cpu().numpy(), "frames": frames.cpu().numpy(),
                     "drv": drv.cpu().numpy(), "rois_bg": rois_bg.cpu().numpy()}
    cpu = res["cpu"]
    report = {"n_emitters": int(len(xyz)), "roi_shape": list(cpu["rois"].shape[1:])}
    if "cuda" in res:  # how far the GPU is from the CPU, for the record; the fixture is the CPU's
        report["cuda_vs_cpu"] = {k: float(np.abs(res["cuda"][k] - cpu[k]).max() / np.abs(cpu[k]).max())
                                 for k in cpu}
    psf = cal.init_spline(XEXTENT, YEXTENT, FRAME_SHAPE, device="cpu")
    np.savez_compressed(
        out / "v0102_renderer.npz", coeff=coeff, x0=CAL_X0, z0=CAL_Z0, dz=CAL_DZ,
        ref0=np.array(cal.ref0, dtype=np.float64), frame_shape=np.array(FRAME_SHAPE),
        xextent=np.array(XEXTENT), yextent=np.array(YEXTENT), n_frames=N_FRAMES,
        roi_size=np.array(psf.roi_size_px), drv_step=DRV_SUBSET.step, **em, **cpu)
    return report


def method_inputs(rng: np.random.Generator):
    """A model output of the right shape and ranges, and emitters for the targets."""
    gen = torch.Generator().manual_seed(SEED + 2)
    out = torch.empty(M_BATCH, 10, M_H, M_W)
    out[:, 0] = torch.rand(M_BATCH, M_H, M_W, generator=gen) ** 4        # sparse detections
    out[:, 1] = torch.rand(M_BATCH, M_H, M_W, generator=gen)             # phot / phot_max
    out[:, 2:4] = torch.rand(M_BATCH, 2, M_H, M_W, generator=gen) - 0.5  # px offsets
    out[:, 4] = 2 * torch.rand(M_BATCH, M_H, M_W, generator=gen) - 1     # z / z_max
    out[:, 5:9] = 0.02 + 0.3 * torch.rand(M_BATCH, 4, M_H, M_W, generator=gen)  # sigmas
    out[:, 9] = torch.rand(M_BATCH, M_H, M_W, generator=gen)             # bg / bg_max

    n_per = rng.integers(0, 9, M_BATCH)
    n_per[1] = 0                                                         # a frame with nothing
    frame_ix = np.repeat(np.arange(M_BATCH), n_per)
    n = len(frame_ix)
    xyz = np.stack([rng.uniform(-0.5, M_H - 0.5, n), rng.uniform(-0.5, M_W - 0.5, n),
                    rng.uniform(-700.0, 700.0, n)], 1)
    xyz[:2, 0] = [-0.7, M_H - 0.3]            # two outside the frame: RemoveOutOfFOV drops them
    phot = rng.uniform(200.0, 12000.0, n)
    em = EmitterSet(xyz=torch.from_numpy(xyz).float(), phot=torch.from_numpy(phot).float(),
                    frame_ix=torch.from_numpy(frame_ix).long(), xy_unit="px", px_size=(100.0, 100.0))
    bg = torch.rand(M_BATCH, M_H, M_W, generator=gen)
    return out, em, bg, {"xyz": xyz, "phot": phot, "frame_ix": frame_ix}


def method_oracle(out_dir: pathlib.Path) -> dict:
    out, em, bg, em_np = method_inputs(np.random.default_rng(SEED + 3))
    xext, yext = (-0.5, M_H - 0.5), (-0.5, M_W - 0.5)

    # the targets, as setup_trainer composes them for the test set (ix_high is inclusive)
    tar = ParameterListTarget(n_max=M_NMAX, xextent=xext, yextent=yext, ix_low=0,
                              ix_high=M_BATCH - 1, squeeze_batch_dim=False)
    tar_param, tar_mask, tar_bg = tar.forward(em, bg)
    tar_param_s, tar_mask_s, tar_bg_s = ParameterListRescale(
        phot_max=PHOT_MAX, z_max=Z_MAX, bg_max=BG_MAX).forward(tar_param.clone(), tar_mask, bg.clone())

    loss_fn = GaussianMMLoss(xextent=xext, yextent=yext, img_shape=(M_H, M_W), device="cpu",
                             chweight_stat=(1.0, 1.0))
    x = out.clone().requires_grad_(True)
    loss = loss_fn.forward(x, (tar_param_s, tar_mask_s, tar_bg_s), None)
    loss.sum().backward()

    # the post-processing of the Fit notebook and of the validation: rescale, offsets to
    # coordinates, SpatialIntegration; raw_th 0.1 is the notebook's, 0.5 the validation's
    pp = {}
    for th in (0.1, 0.5):
        y = InverseParamListRescale(phot_max=PHOT_MAX, z_max=Z_MAX, bg_max=BG_MAX).forward(out.clone())
        y = Offset2Coordinate(xextent=xext, yextent=yext, img_shape=(M_H, M_W)).forward(y)
        p_int = y[:, 0].clone()
        es = SpatialIntegration(raw_th=th, xy_unit="px", px_size=(100.0, 100.0)).forward(y.clone())
        key = f"th{th:g}".replace(".", "")
        pp[key] = {"xyz": es.xyz.numpy(), "phot": es.phot.numpy(), "frame_ix": es.frame_ix.numpy(),
                   "prob": es.prob.numpy(), "bg": es.bg.numpy(), "xyz_sig": es.xyz_sig.numpy(),
                   "phot_sig": es.phot_sig.numpy()}
    flat = {f"pp_{k}_{f}": v for k, d in pp.items() for f, v in d.items()}
    np.savez_compressed(
        out_dir / "v0102_method.npz", output=out.numpy(), bg_target_in=bg.numpy(),
        em_xyz=em_np["xyz"], em_phot=em_np["phot"], em_frame_ix=em_np["frame_ix"],
        n_max=M_NMAX, phot_max=PHOT_MAX, z_max=Z_MAX, bg_max=BG_MAX,
        tar_param=tar_param.numpy(), tar_mask=tar_mask.numpy(),
        tar_param_scaled=tar_param_s.numpy(), tar_bg_scaled=tar_bg_s.numpy(),
        loss=loss.detach().numpy(), loss_grad=x.grad.numpy(), offset2coord_p=p_int.numpy(), **flat)
    return {"loss": loss.detach().numpy().tolist(),
            "n_localizations": {k: int(len(d["prob"])) for k, d in pp.items()}}


# The model oracle: frames through the Fit notebook's pipeline.  The raw output is 10 channels a
# pixel, 124 MB for 100 frames of 176 x 176, so it is kept for fewer frames than the localizations.
MODEL_FRAMES, MODEL_RAW_FRAMES, MODEL_BATCH = 500, 100, 32


def fit_pipeline(package: pathlib.Path, device: str):
    """The Fit notebook's cells, as they are: camera from frames_meta.yaml, crop 8, mirror, rescale."""
    import yaml
    from decode.neuralfitter import frame_processing, scale_transform
    from decode.neuralfitter.utils.processing import TransformSequence, wrap_callable
    from decode.utils import param_io, model_io

    param = param_io.load_params(package / "param_run.yaml")
    model = decode.neuralfitter.models.SigmaMUNet.parse(param)
    model = model_io.LoadSaveModel(model, input_file=package / "model.pt", output_file=None).load_init(device=device)
    meta = yaml.safe_load((package / "frames_meta.yaml").read_text())
    param = param_io.RecursiveNamespace(**param_io.autofill_dict(meta["Camera"], param.to_dict(),
                                                                  mode_missing="include"))
    camera = decode.simulation.camera.Photon2Camera.parse(param)
    camera.device = "cpu"
    frame_proc = TransformSequence([
        wrap_callable(camera.backward),
        frame_processing.AutoCenterCrop(8),
        frame_processing.Mirror2D(dims=-1),
        scale_transform.AmplitudeRescale.parse(param)])
    return param, model, frame_proc


def run_model(package: pathlib.Path, out: pathlib.Path, device: str):
    from decode.neuralfitter import frame_processing
    from decode.neuralfitter.utils.processing import TransformSequence
    from decode.utils import frames_io

    torch.set_num_threads(8)
    out.mkdir(parents=True, exist_ok=True)
    param, model, frame_proc = fit_pipeline(package, device)
    frames = frames_io.load_tif(package / "frames.tif")[:MODEL_FRAMES]
    size = frame_processing.get_frame_extent(frames.unsqueeze(1).size(), frame_proc.forward)
    extent = ((-0.5, size[-2] - 0.5), (-0.5, size[-1] - 0.5))
    post_proc = TransformSequence([
        InverseParamListRescale.parse(param),
        Offset2Coordinate(xextent=extent[0], yextent=extent[1], img_shape=size[-2:]),
        SpatialIntegration(raw_th=0.1, xy_unit="px", px_size=param.Camera.px_size)])

    common = dict(model=model, ch_in=param.HyperParameter.channels_in, frame_proc=frame_proc,
                  device=device, batch_size=MODEL_BATCH, num_workers=0)
    raw = decode.neuralfitter.Infer(post_proc=None, forward_cat="frames", **common).forward(
        frames[:MODEL_RAW_FRAMES])
    em = decode.neuralfitter.Infer(post_proc=post_proc, **common).forward(frames)
    tag = "cpu" if device == "cpu" else "cuda"
    np.savez(out / f"v0102_model_raw_{tag}.npz", raw=raw.numpy(),
             input=frame_proc.forward(frames[:MODEL_RAW_FRAMES].float()).numpy())
    em.save(out / f"v0102_model_emitters_{tag}.h5")
    report = {"environment": environment(), "device": device, "frames": MODEL_FRAMES,
              "raw_frames": MODEL_RAW_FRAMES, "frame_size_processed": list(size[-2:]),
              "camera": vars(param.Camera) if hasattr(param.Camera, "__dict__") else str(param.Camera),
              "n_localizations": len(em), "raw_shape": list(raw.shape)}
    (out / f"v0102_model_{tag}.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


def environment() -> dict:
    import spline
    return {"decode": decode.__version__, "torch": torch.__version__, "numpy": np.__version__,
            "spline": getattr(spline, "__version__", "?"), "python": sys.version.split()[0],
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_oracles(out: pathlib.Path):
    torch.set_num_threads(1)   # one thread: the CPU sums in one order, so two runs are bitwise equal
    out.mkdir(parents=True, exist_ok=True)
    report = {"environment": environment(), "renderer": renderer_oracle(out), "method": method_oracle(out)}
    report["sha256"] = {p.name: sha256(p) for p in sorted(out.glob("v0102_*.npz"))}
    (out / "v0102_oracles.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oracles")
    o.add_argument("out", type=pathlib.Path)
    m = sub.add_parser("model")
    m.add_argument("package", type=pathlib.Path)
    m.add_argument("out", type=pathlib.Path)
    m.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.cmd == "oracles":
        run_oracles(a.out)
    elif a.cmd == "model":
        run_model(a.package, a.out, a.device)


if __name__ == "__main__":
    main()
