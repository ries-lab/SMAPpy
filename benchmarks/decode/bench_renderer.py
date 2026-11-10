"""Time the spline PSF over one training epoch: DECODE v0.10.2's (the `spline` package) or the port's (torch).

    python bench_renderer.py CALIBRATION_3dcal.mat --impl port --device mps
    python bench_renderer.py CALIBRATION_3dcal.mat --impl original --device cuda:0   # in decode_fast

One epoch as the example package's training draws it (param_run.yaml): 32 768 frames of 40 x 40 with
50 emitters a frame on average (Poisson), x and y uniform over the frame, z uniform in +-800 nm,
1.64 M emitters.  Both build CubicSplinePSF from the raw ``SXY.cspline`` (DECODE's own reading,
without SMAPpy, so the script runs in the original's environment) and call ``forward`` on the whole
epoch, which cuts it into DECODE's chunks of 500 000.  The first call is warm-up; the time is the
median of the rest, with the device synchronised.  Writes one JSON line to stdout.
"""
import argparse
import json
import platform
import statistics
import sys
import time

import numpy as np
import scipy.io as sio
import torch


def build(impl, cal_path, device, frame):
    cs = sio.loadmat(cal_path, struct_as_record=False, squeeze_me=True)["SXY"].cspline
    kw = dict(xextent=(-0.5, frame - 0.5), yextent=(-0.5, frame - 0.5), img_shape=(frame, frame),
              ref0=(cs.x0 - 1, cs.x0 - 1, cs.z0), coeff=torch.from_numpy(np.asarray(cs.coeff, np.float32)),
              vx_size=(1.0, 1.0, float(cs.dz)), device=device)
    if impl == "original":
        from decode.simulation.psf_kernel import CubicSplinePSF
    else:
        from decode_smlm.simulation.psf_kernel import CubicSplinePSF
    return CubicSplinePSF(**kw)


def emitters(n_frames, per_frame, frame, seed=0):
    g = torch.Generator().manual_seed(seed)
    counts = torch.poisson(torch.full((n_frames,), float(per_frame)), generator=g).long()
    frame_ix = torch.repeat_interleave(torch.arange(n_frames), counts)
    n = frame_ix.numel()
    xyz = torch.stack([torch.rand(n, generator=g) * frame - 0.5, torch.rand(n, generator=g) * frame - 0.5,
                       torch.rand(n, generator=g) * 1600 - 800], 1)
    return xyz, torch.rand(n, generator=g) * 6000 + 1000, frame_ix


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(device).startswith("mps"):
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration")
    ap.add_argument("--impl", choices=["original", "port"], required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frames", type=int, default=32768)
    ap.add_argument("--per-frame", type=float, default=50)
    ap.add_argument("--frame", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    psf = build(a.impl, a.calibration, a.device, a.frame)
    xyz, phot, frame_ix = emitters(a.frames, a.per_frame, a.frame)
    times = []
    for i in range(a.repeats + 1):
        sync(a.device)
        t = time.perf_counter()
        out = psf.forward(xyz, phot, frame_ix, 0, a.frames - 1)
        sync(a.device)
        times.append(time.perf_counter() - t)
    print(json.dumps({"impl": a.impl, "device": a.device, "torch": torch.__version__,
                      "gpu": (torch.cuda.get_device_name(0) if str(a.device).startswith("cuda") else
                              platform.processor() or platform.machine()),
                      "threads": torch.get_num_threads(), "frames": a.frames, "emitters": int(xyz.shape[0]),
                      "roi": list(psf.roi_size_px), "warmup_s": round(times[0], 3),
                      "epoch_s": round(statistics.median(times[1:]), 3), "times_s": [round(t, 3) for t in times[1:]],
                      "checksum": float(out.double().sum())}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
