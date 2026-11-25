"""The mechanism behind §33's hypothesis, from the model alone: how differently two molecules' light is split
over the four channels, as a function of their phase difference (docs/4pi.md §33).

A molecule of phase phi puts the share ``s_c(phi) = sum_disc (I_c + A_c cos phi + B_c sin phi)`` of its light into
channel c (the model's knots at z, summed over a disc of 4 px as `fourpi.modulation` does).  For two molecules of
equal brightness at phases phi and phi + dphi, the channel where the second is dimmest against the first has the
ratio ``max_c s_c(phi) / s_c(phi + dphi)``: 1 for dphi = 0 (the astigmatic case, where the two are the same
everywhere), largest at dphi = pi.  Averaged over phi, at the focus and at +-300 nm.  Writes
``results/contrast.json``.

    python contrast.py [MODEL_4PI.h5]
"""
import json
import sys
from pathlib import Path

import numpy as np

from smappy import fourpi

HERE = Path(__file__).resolve().parent
MODEL = Path.home() / ".cache/uipsf/p2/bead_v2_4pi.h5"
RADIUS_PX = 4.0


def shares(cal, z_nm: float, phi: np.ndarray) -> np.ndarray:
    """(len(phi), C): each channel's share of a molecule's light at ``z_nm`` and phase ``phi``."""
    k = round(float(cal.z_nm_to_index(z_nm)))
    V = cal.volumes[:, :, k]                                   # (C, 3, ny, nx)
    ny, nx = V.shape[-2:]
    yy, xx = np.mgrid[0:ny, 0:nx]
    disc = np.hypot(yy - (ny - 1) / 2, xx - (nx - 1) / 2) <= RADIUS_PX
    I, A, B = (V[:, j][:, disc].sum(1) for j in range(3))
    s = I[None] + A[None] * np.cos(phi)[:, None] + B[None] * np.sin(phi)[:, None]
    return np.clip(s, 1e-9, None) / np.clip(s, 1e-9, None).sum(1, keepdims=True)


def contrast(cal, z_nm: float, dphi: np.ndarray, n: int = 360) -> dict:
    """The mean over phi of the best channel's ratio, and its median, for each dphi."""
    phi = np.linspace(-np.pi, np.pi, n, endpoint=False)
    a = shares(cal, z_nm, phi)
    best = []
    for d in dphi:
        b = shares(cal, z_nm, phi + d)
        r = np.maximum(a / b, b / a).max(1)                     # either molecule the dimmer one
        best.append(r)
    best = np.array(best)
    return {"mean": best.mean(1).tolist(), "median": np.median(best, 1).tolist(),
            "p10": np.percentile(best, 10, 1).tolist(), "p90": np.percentile(best, 90, 1).tolist()}


def main():
    cal = fourpi.load_fourpi_calibration(sys.argv[1] if len(sys.argv) > 1 else MODEL)
    dphi = np.linspace(0, np.pi, 37)
    phi = np.linspace(-np.pi, np.pi, 181)
    out = {"dphi": dphi.tolist(), "phi": phi.tolist(), "period_nm": cal.period_nm,
           "modulation": fourpi.modulation(cal, 0.0, RADIUS_PX).tolist()}
    for z in (-300.0, 0.0, 300.0):
        out[f"z{z:+.0f}"] = {"shares": shares(cal, z, phi).tolist(), "ratio": contrast(cal, z, dphi)}
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / "contrast.json", "w") as f:
        json.dump(out, f, indent=1)
    r = out["z+0"]["ratio"]["mean"]
    print("modulation", np.round(out["modulation"], 3), "ratio at dphi 0, pi/4, pi/2, pi:",
          [round(r[i], 2) for i in (0, 9, 18, 36)])


if __name__ == "__main__":
    main()
