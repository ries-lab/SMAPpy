"""A published DECODE-Plex model on a C = 2 set, matched to the truth (docs/decode/plex-plan.md §6).

    python benchmarks/decode/suite/plex_on_set.py SET_DIR CONTEXT [PROB]

The check that a set is drawn as the model expects its frames: the model of the set's context, run
through its notebook's steps (decode_smlm.plex), the second channel shifted by the registration's
own offset (``Test`` set to ``Simulation``: the sets use the beads' registration as it is, where a
movie needs the splitter's later move on top), the localizations above PROB (0.5) matched to the
truth within 250 nm laterally and 500 nm in z -- in the area where both channels see the sample,
for the truth and the localizations alike.  Prints recall, Jaccard, bias and RMSE against the
median CRLB, then the errors by z and by position.
"""
import copy
import math
import pathlib
import sys

import numpy as np
import tifffile
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sets_c2 as S

import decode_smlm.plex as dp
import decode_smlm.plex.neuralfitter.inference.functional as infer_func

folder, context = pathlib.Path(sys.argv[1]), sys.argv[2]; pth = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
ctx = S.CONTEXTS[context]; dev = "mps" if torch.backends.mps.is_available() else "cpu"
cfg = dp.io.param.load(ctx["model"] / "param_run.yaml")
for s in cfg["Hardware"]["device"]: cfg["Hardware"]["device"][s] = dev
for cam in cfg["Camera"]: cam["specs"]["flip"]["channel"] = -2
cfg["Test"]["Transformation"]["Pos"]["glob"]["offset"] = copy.deepcopy(cfg["Simulation"]["Transformation"]["Pos"]["glob"]["offset"])
trafo = pathlib.Path(str(ctx["smap"]).replace("_3dcal.mat", "_trafo.mat"))
frames = torch.from_numpy(tifffile.imread(folder / "frames.tif").astype(np.float32))
H, W = frames.shape[-2:]
crop = [math.floor(H / 2 / 8) * 8, math.floor(W / 8) * 8]
em, _ = infer_func.infer(frames, frame_crop=crop, cfg=cfg, model=next(ctx["model"].glob("*.ckpt")), mode="multi",
                         trafo=trafo, mode_camera="rois", roi_shift=None, batch_size=4, num_workers=0, device=[dev])
em = em[em.prob > pth]
xyz = em.xyz_px.cpu().numpy(); f = em.frame_ix.cpu().numpy()
row, col = xyz[:, 0], xyz[:, 1]
if cfg["Camera"][0]["specs"]["flip"]["gain"] is not None:     # DECODE-Plex mirrored the frame in x first
    col = W - 1 - col
t = np.load(folder / "truth.npz")
# score where both channels see the sample: a molecule's partner inside the secondary half, and
# inside the crop the network reads; the same for the localizations
cal = S.calibration(context); _, regions, to_sec, _ = S._layout(cal)
def both(xc, yr):
    ok = S._inside(to_sec(np.c_[xc, yr]), regions[1], margin=3) & (yr > 2) & (yr < crop[0] - 3)
    if cfg["Camera"][0]["specs"]["flip"]["gain"] is not None:
        return ok & (xc > W - crop[1] + 2) & (xc < W - 3)
    return ok & (xc > 2) & (xc < crop[1] - 3)
keep_f = both(col, row); row, col, xyz, f = row[keep_f], col[keep_f], xyz[keep_f], f[keep_f]; em = em[torch.from_numpy(keep_f)]
seen = both(t["x_px"], t["y_px"])
_unused_seen = (t["y_px"] < crop[0] - 1) & (t["x_px"] < crop[1] - 1) & (t["x_px"] > 0)
if cfg["Camera"][0]["specs"]["flip"]["gain"] is not None:
    seen = (t["y_px"] < crop[0] - 1) & (t["x_px"] > W - crop[1]) & (t["x_px"] < W - 1)
px = np.array(S.PIXEL_NM)
dx, dy, dz, n_tp = [], [], [], 0
for k in range(int(t["frame"].max()) + 1):
    a = np.flatnonzero((t["frame"] == k) & seen); b = np.flatnonzero(f == k)
    if not len(a) or not len(b): continue
    lat = np.hypot((t["x_px"][a, None] - col[None, b]) * px[0], (t["y_px"][a, None] - row[None, b]) * px[1])
    ax = np.abs(t["z_nm"][a, None] - xyz[None, b, 2])
    cost = np.where((lat < 250) & (ax < 500), lat, 1e6)
    i, j = linear_sum_assignment(cost); ok = cost[i, j] < 1e6
    i, j = i[ok], j[ok]; n_tp += len(i)
    dx += list((col[b][j] - t["x_px"][a][i]) * px[0]); dy += list((row[b][j] - t["y_px"][a][i]) * px[1]); dz += list(xyz[b][j, 2] - t["z_nm"][a][i])
dx, dy, dz = map(np.array, (dx, dy, dz))
n_true, n_found = int(seen.sum()), int(sum(np.isin(f, np.arange(int(t['frame'].max()) + 1))))
print(f"{folder.name}: truth seen {n_true}, found {len(f)}, matched {n_tp}: recall {n_tp/n_true:.3f}, Jaccard {n_tp/(n_true+len(f)-n_tp):.3f}")
print(f"  bias x {dx.mean():+.1f} y {dy.mean():+.1f} z {dz.mean():+.1f} nm | RMSE x {dx.std():.1f} y {dy.std():.1f} z {dz.std():.1f} nm | "
      f"CRLB median x {np.nanmedian(t['crlb_x_nm'][seen]):.1f} y {np.nanmedian(t['crlb_y_nm'][seen]):.1f} z {np.nanmedian(t['crlb_z_nm'][seen]):.1f}")
# the unmatched: where are they, and how probable
from scipy.spatial import cKDTree

p = em.prob.cpu().numpy()
inside = (row < crop[0] - 1)
print(f"  found in the rows the truth covers: {inside.sum()} of {len(row)}; prob quantiles of all", np.quantile(p, [0.1, 0.5, 0.9]).round(2))
near = []
for k in range(int(t['frame'].max()) + 1):
    b = np.flatnonzero(f == k); a = np.flatnonzero(t['frame'] == k)
    if len(a) and len(b):
        d, _ = cKDTree(np.c_[t['x_px'][a] * px[0], t['y_px'][a] * px[1]]).query(np.c_[col[b] * px[0], row[b] * px[1]], k=2)
        near.append(np.c_[d, p[b], row[b], col[b]])
near = np.concatenate(near)
print("  nearest true molecule of each localization (nm) quantiles:", np.quantile(near[:, 0], [0.25, 0.5, 0.75, 0.9]).round(0),
      " second nearest:", np.quantile(near[:, 1], [0.25, 0.5]).round(0))
far = near[near[:, 0] > 250]
print("  localizations > 250 nm from any truth:", len(far), "prob median", np.median(far[:, 2]).round(2) if len(far) else None,
      "rows quantiles", np.quantile(far[:, 3], [0.1, 0.5, 0.9]).round(0) if len(far) else None, "cols", np.quantile(far[:, 4], [0.1, 0.5, 0.9]).round(0) if len(far) else None)
# errors by z and by position, on the matched pairs (recomputed with indices)
P = []
for k in range(int(t["frame"].max()) + 1):
    a = np.flatnonzero((t["frame"] == k) & seen); b = np.flatnonzero(f == k)
    if not len(a) or not len(b): continue
    lat = np.hypot((t["x_px"][a, None] - col[None, b]) * px[0], (t["y_px"][a, None] - row[None, b]) * px[1])
    ax = np.abs(t["z_nm"][a, None] - xyz[None, b, 2]); cost = np.where((lat < 250) & (ax < 500), lat, 1e6)
    i, j = linear_sum_assignment(cost); ok = cost[i, j] < 1e6
    for ii, jj in zip(a[i[ok]], b[j[ok]]):
        P.append((t["z_nm"][ii], t["x_px"][ii], t["y_px"][ii], (col[jj] - t["x_px"][ii]) * px[0], (row[jj] - t["y_px"][ii]) * px[1], xyz[jj, 2] - t["z_nm"][ii]))
P = np.array(P)
print("  by z (nm):   z bin | dx mean sd | dy mean sd | dz mean sd")
for lo in np.linspace(P[:, 0].min(), P[:, 0].max(), 6)[:-1]:
    m = (P[:, 0] >= lo) & (P[:, 0] < lo + (P[:, 0].max() - P[:, 0].min()) / 5)
    print(f"   {lo:+6.0f} | {P[m,3].mean():+6.1f} {P[m,3].std():5.1f} | {P[m,4].mean():+6.1f} {P[m,4].std():5.1f} | {P[m,5].mean():+6.1f} {P[m,5].std():5.1f}  n {m.sum()}")
print("  by row (y px): y bin | dx mean | dy mean")
for lo in np.linspace(P[:, 2].min(), P[:, 2].max(), 5)[:-1]:
    m = (P[:, 2] >= lo) & (P[:, 2] < lo + (P[:, 2].max() - P[:, 2].min()) / 4)
    print(f"   {lo:6.0f} | {P[m,3].mean():+6.1f} | {P[m,4].mean():+6.1f}  sd {P[m,4].std():5.1f}")
print("  by col (x px): x bin | dx mean | dy mean")
for lo in np.linspace(P[:, 1].min(), P[:, 1].max(), 5)[:-1]:
    m = (P[:, 1] >= lo) & (P[:, 1] < lo + (P[:, 1].max() - P[:, 1].min()) / 4)
    print(f"   {lo:6.0f} | {P[m,3].mean():+6.1f} | {P[m,4].mean():+6.1f}")
