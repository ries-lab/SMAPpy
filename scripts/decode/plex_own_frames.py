"""A published DECODE-Plex model on frames drawn by its own training simulator (plex-plan §14).

    PYTHONPATH=~/decode_plex/DECODE-Plex python plex_own_frames.py DATA RUN N_FRAMES OUT.npz [EMITTER_AVG [ROWSxCOLS]]

Runs in DECODE-Plex's environment (the compiled ``spline`` draws the frames) and imports nothing of
SMAPpy, so the frames, the network and the post-processing are all the original's, exactly as its
validation saw them: ``setup_sampler(cfg, cfg["Simulation"])``, ``pre_val``, the network, ``post``.
The question it answers: does the model's count of false localizations grow with the background on
the frames it was trained on, as it does on the benchmark's sets (plex-plan §11)?  If it does not,
the difference lies in how the sets are drawn.

Writes per frame each channel's background and the localizations (prob, xyz in px of channel 0's
40 x 40 frame) and the truth (filtered as the training's target is: ``setup_em_filter``).
EMITTER_AVG replaces the training's molecules a frame (40), to compare at the benchmark's density.
ROWSxCOLS draws frames of that size at the chip's corner instead of the training's 40 x 40 crops at
random places on it -- the frames the benchmark gives the model (plex-plan §15: does the biplane
model find molecules that are not there on whole-chip frames of its own?).
"""
import pathlib
import sys

import decode
import numpy as np
import torch


def _rows(t):
    """Photons a row per molecule: one number (biplane, linked) or one per channel; frames may hold none."""
    return t[:, None] if t.dim() == 1 else t.flatten(1)

from decode.neuralfitter.train import setup_cfg

data, run, n, out = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), pathlib.Path(sys.argv[4])
folder = next((data / "outputs").glob(f"{run}-*"))
cfg = decode.io.param.load(folder / "param_run.yaml")
for s in cfg["Hardware"]["device"]:
    cfg["Hardware"]["device"][s] = "cpu"
for key in ("calibration", "trafo"):
    if cfg["Paths"].get(key):
        cfg["Paths"][key] = str(data / "calibration" / pathlib.Path(cfg["Paths"][key]).name)
cfg["Simulation"]["samples"] = n
if len(sys.argv) > 5:
    cfg["Simulation"]["emitter_avg"] = float(sys.argv[5])
if len(sys.argv) > 6:
    size = [int(v) for v in sys.argv[6].split("x")]
    sim = cfg["Simulation"]
    sim["img_size"] = size
    for key in ("emitter_extent", "frame_extent", "psf_extent"):
        sim[key]["x"], sim[key]["y"] = [-0.5, size[0] - 0.5], [-0.5, size[1] - 0.5]
    sim["Transformation"]["Pos"]["sampled"]["offset"]["x"] = None     # the chip's corner, as the
    sim["Transformation"]["Pos"]["sampled"]["offset"]["y"] = None     # benchmark's frames are read
torch.manual_seed(20261119)
torch.set_num_threads(8)

sampler = setup_cfg.setup_sampler(cfg, cfg["Simulation"])
sampler.sample()
val = sampler.val_samples()
em_filter = setup_cfg.setup_em_filter(cfg)

model = setup_cfg.setup_model(cfg)
ckpt = torch.load(next(folder.glob("*.ckpt")), map_location="cpu", weights_only=False)
model.load_state_dict({k[len("_model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("_model.")})
model.eval()

bg = sampler.bg if isinstance(sampler.bg, (list, tuple)) else [sampler.bg]
bg = np.stack([b.reshape(len(b), -1).mean(1).cpu().numpy() for b in bg], 1)

loc, tru = {"frame": [], "prob": [], "xyz": [], "phot": []}, {"frame": [], "xyz": [], "phot": []}
B = 32
with torch.no_grad():
    for i0 in range(0, n, B):
        xs, ems = [], []
        for i in range(i0, min(n, i0 + B)):
            x, _, em = val[i]
            xs.append(x); ems.append(em)
        out_em = sampler._proc.post(model(torch.stack(xs)))
        for j, em in enumerate(ems):
            em = em_filter.forward(em)
            tru["frame"].append(np.full(len(em), i0 + j)); tru["xyz"].append(em.xyz_px.numpy())
            tru["phot"].append(_rows(em.phot).sum(1).numpy())
        loc["frame"].append(out_em.frame_ix.numpy() + i0); loc["prob"].append(out_em.prob.numpy())
        loc["xyz"].append(out_em.xyz_px.numpy()); loc["phot"].append(_rows(out_em.phot).numpy())
        print(f"{min(n, i0 + B)} / {n}", flush=True)
np.savez_compressed(out, bg=bg, **{f"loc_{k}": np.concatenate(v) for k, v in loc.items()},
                    **{f"true_{k}": np.concatenate(v) for k, v in tru.items()})
print("wrote", out)
