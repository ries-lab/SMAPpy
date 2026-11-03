"""The figure of `dh_beads.py`: the learnt pupil and the bead against the model, for each double-helix plate.

    python benchmarks/uipsf/real/dh_beads_figure.py [benchmarks/uipsf/results/dh_beads.jsonl] [-o dh_beads.png]

Each set's best model (Pavani's design with Zernikes to order 8) is fitted again as `dh_beads.py` fitted it.
Per set, one row of pupils and three of planes:

- *pupil*: the mask's phase alone, the learnt Zernikes (the system's aberrations), the whole pupil's phase
  (mask × Zernikes), each inside the aperture, and the magnitude;
- *data*, *model* (the bead, its blur and size included) and *residual* (data − model) at six heights, data and
  model on one scale, the residual on a quarter of it, so that what the model misses shows at a glance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import dh_beads as d  # noqa: E402

from uipsf.model import MaskSpec  # noqa: E402
from uipsf.params import Params  # noqa: E402
from uipsf.workflows import BeadSettings, bead_model  # noqa: E402
from uipsf.workflows.beads import fit_beads  # noqa: E402


def refit(name, row):
    ds = d.load(name)
    roi = int(round(3500.0 / ds.pixel_nm)) | 1
    b, _, _ = d.batch(ds, min(roi, *ds.adu.shape[1:]))
    spec = MaskSpec(kind="double helix", design=row["design"], zones=row["zones"], charge=row["hand"],
                    free=("centre", "rotation", "scale", "alpha"))
    settings = BeadSettings(psf=d.MaskedModel(zernike_order=8, phase_mask=spec), bead_diameter_nm=ds.bead_nm)
    model = bead_model(d.optics(ds), settings, tuple(b.data.shape[-2:]), z_max_nm=float(b.z_nm.abs().max()),
                       device="cpu")
    # started as `dh_beads.py` started it, at the order-4 fit's rotation: started at the order-8 fit's whole
    # placement, DHPSFU's bead stopped 100 000 nats worse, its Zernikes running to ±140 rad at the rim
    values = {"mask.rotation": torch.tensor(row["start_rotation"])}
    start = Params(model.specs(b), b.n_emitters, device="cpu", values=values)
    res = fit_beads(model, b, d.ShiftedPoisson(), start=start, first_order=None, max_dz_nm=None)
    return ds, b, model, res


def pupils(model, p):
    pupil = model.psf.pupil
    with torch.no_grad():
        magnitude, phase = pupil(p)
        t = pupil.transmission(p)
    inside = (magnitude > 0.05 * magnitude.max()).numpy()
    mask = np.angle(t.numpy())
    zern = phase.numpy()
    total = np.angle(np.exp(1j * zern) * t.numpy())

    def cut(a):
        return np.where(inside, a, np.nan)
    return cut(mask), cut(zern), cut(total), cut(magnitude.numpy())


def draw(axes, name, ds, b, model, res, summary):
    p = res.params.physical()
    with torch.no_grad():
        mu = model(p, b)[0, 0].numpy()
    data = b.data[0, 0].numpy()
    K = data.shape[0]
    ks = np.linspace(0.08, 0.92, axes.shape[1]).__mul__(K - 1).round().astype(int)
    vmax = float(data.max())
    titles = ("mask phase", "learnt Zernikes", "whole pupil phase", "magnitude")
    for a, img, t in zip(axes[0, :4], pupils(model, p), titles, strict=True):
        cmap = "twilight" if t != "magnitude" and t != "learnt Zernikes" else ("RdBu_r" if t == "learnt Zernikes"
                                                                                else "viridis")
        lim = {"learnt Zernikes": np.nanmax(np.abs(img))}.get(t)
        a.imshow(img, cmap=cmap, vmin=-np.pi if cmap == "twilight" else (-lim if lim else None),
                 vmax=np.pi if cmap == "twilight" else (lim if lim else None))
        a.set_title(t if t != "learnt Zernikes" else f"learnt Zernikes (±{lim:.1f} rad)", fontsize=8)
    for a in axes[0, 4:]:
        a.set_visible(False)
    axes[0, 0].text(-0.15, 0.5, name, transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right",
                    fontsize=10, weight="bold")
    for i, k in enumerate(ks):
        z = float(b.z_nm[k])
        axes[1, i].imshow(data[k], cmap="magma", vmin=0, vmax=vmax)
        axes[1, i].set_title(f"z {z:+.0f} nm", fontsize=8)
        axes[2, i].imshow(mu[k], cmap="magma", vmin=0, vmax=vmax)
        axes[3, i].imshow(data[k] - mu[k], cmap="RdBu_r", vmin=-vmax / 4, vmax=vmax / 4)
    for r, label in ((1, "data"), (2, "model"), (3, "residual ×4")):
        axes[r, 0].text(-0.08, 0.5, label, transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right",
                        fontsize=8)
    m = summary["mask"]
    axes[0, 4].set_visible(True)
    axes[0, 4].axis("off")
    axes[0, 4].text(0, 0.5, f"deviance/px {summary['deviance_per_pixel']:.1f}\n"
                            f"centre {m['centre'][0]:+.3f}, {m['centre'][1]:+.3f}\n"
                            f"rotation {np.degrees(m['rotation']):.0f}°  scale {m['scale']:.2f}\n"
                            f"α {m['alpha']:.2f}  blur {summary['blur_nm'][0]:.0f} nm",
                    fontsize=7, va="center", transform=axes[0, 4].transAxes)
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", nargs="?", default=os.path.join(os.path.dirname(__file__), "..", "results",
                                                               "dh_beads.jsonl"))
    ap.add_argument("-o", "--out", default="dh_beads.png")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.results)]
    best = {r["set"]: dict(r, start_rotation=next(q["mask"]["rotation"] for q in rows if q["set"] == r["set"]
                                                  and q["model"] == "DH + Zernike 4"))
            for r in rows if r["model"] == "DH + Zernike 8"}
    sets = [s for s in ("moerner", "dhpsfu") if s in best]
    fig, axes = plt.subplots(4 * len(sets), 6, figsize=(9, 6.2 * len(sets)), squeeze=False)
    for i, name in enumerate(sets):
        ds, b, model, res = refit(name, best[name])
        print(name, res.objective, flush=True)
        draw(axes[4 * i:4 * i + 4], name, ds, b, model, res, d.summary(model, res, b))
    fig.savefig(args.out, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
