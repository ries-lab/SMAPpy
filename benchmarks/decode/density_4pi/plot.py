"""§33's figures and table from `density_compare.py`'s results (docs/4pi.md §33).

Every number from the molecules themselves (the ``.npz``), pooled over a row's movies: recall, the RMSEs and the
widths over the CRLB, near the focus (|z| < 300 nm) and over the whole ±500 nm; precision and Jaccard from the
JSON (`metrics.evaluate` per movie, averaged), which needs the false positives the molecules do not hold.  A
molecule counts if it has at least 100 photons in the frame (50 of the half-photon image: the same molecules).

    python plot.py [RESULTS_DIR]      -> results/summary.json, results/*.png
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DENSITIES = (0.5, 1, 2, 5, 10, 25, 50)
ONS = (1, 3)
PX_UM = 0.129
CORE_NM = 300.0
FLOOR = 100.0
PAIR_NM = 250.0
#: fit -> (label, colour, line style, which CRLB)
FITS = {"netfit_4pi": ("netfit, 4Pi", "#1f5fa8", "-", "crlb4"),
        "spline_4pi": ("Spline 4Pi", "#1f5fa8", "--", "crlb4"),
        "netfit_astig": ("netfit, astigmatic", "#d0661c", "-", "crlb_full"),
        "spline_3d": ("Spline 3D, astigmatic", "#d0661c", "--", "crlb_full"),
        "netfit_astig_half": ("netfit, astig., half photons", "#e8a25e", ":", "crlb_half"),
        "spline_3d_half": ("Spline 3D, astig., half photons", "#e8a25e", (0, (1, 3)), "crlb_half")}


def robust_width(v):
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return np.nan
    q = np.percentile(v, [25, 75])
    return float((q[1] - q[0]) / 1.349)


def load(folder: Path, d, on):
    """The row's molecules and the JSON results, the fits from every file of the row merged (the fitters' and
    the networks' passes draw the same molecules: the same seeds)."""
    mol, res = None, {}
    for p in sorted(folder.glob(f"d{d:g}_on{on}*.npz")):
        m = dict(np.load(p))
        if mol is None:
            mol = m
        else:
            if len(m["z"]) != len(mol["z"]) or not np.allclose(m["z"], mol["z"]):
                raise ValueError(f"{p.name} holds other molecules than its row's first file")
            mol.update({k: v for k, v in m.items() if k.startswith(("err_", "crlb"))})
        res.update(json.loads(p.with_suffix(".json").read_text())["results"])
    return mol, res


def scores(mol, res, fit):
    """One fit's numbers in one row."""
    err = mol.get("err_" + fit)
    if err is None:
        return None
    ok = mol["photons"] >= FLOOR
    core = ok & (np.abs(mol["z"]) < CORE_NM)
    found = np.isfinite(err[:, 0])
    crlb = mol[FITS[fit][3]]
    out = {"jaccard": res.get(fit, {}).get("jaccard"), "precision": res.get(fit, {}).get("precision"),
           "seconds": res.get(fit, {}).get("seconds")}
    for name, sel in (("core", core), ("all", ok)):
        f = sel & found
        out[f"{name}_recall"] = float(f.sum() / max(sel.sum(), 1))
        out[f"{name}_n"] = int(sel.sum())
        if f.sum() < 20:
            continue
        e = err[f]
        out[f"{name}_rmse_lat_nm"] = float(np.sqrt(np.mean(e[:, 0] ** 2 + e[:, 1] ** 2)))
        out[f"{name}_rmse_z_nm"] = float(np.sqrt(np.mean(e[:, 2] ** 2)))
        out[f"{name}_width_lat"] = robust_width(np.r_[e[:, 0] / crlb[f, 0], e[:, 1] / crlb[f, 1]])
        out[f"{name}_width_z"] = robust_width(e[:, 2] / crlb[f, 2])
        if fit.endswith("4pi"):
            period = 2 * np.pi / 0.023714926932155617
            wrong = np.abs(e[:, 3]) > period / 2
            out[f"{name}_wrong_fringe"] = float(wrong.mean())
            out[f"{name}_rmse_z_phase_nm"] = float(np.sqrt(np.mean(e[~wrong, 3] ** 2)))
            # the bound of z from the phase: the phase's over the fringe's slope
            zb = mol["phase_crlb4"][f][~wrong] / 0.023714926932155617
            out[f"{name}_width_z_phase"] = robust_width(e[~wrong, 3] / zb)
    return out


def pairs(rows, dense=(5, 10, 25, 50), bins=6):
    """Recall of molecules with a neighbour within `PAIR_NM`, near the focus, by |dphi| -- pooled over the
    ``dense`` rows of each on-time, where close pairs are common enough."""
    edges = np.linspace(0, np.pi, bins + 1)
    out = {}
    for on in ONS:
        acc = {fit: [np.zeros(bins), np.zeros(bins)] for fit in FITS}
        for d in dense:
            mol = rows.get((d, on))
            if mol is None:
                continue
            sel = (mol["photons"] >= FLOOR) & (np.abs(mol["z"]) < CORE_NM) & (mol["nn_nm"] < PAIR_NM)
            b = np.clip(np.digitize(np.abs(mol["nn_dphi"]), edges) - 1, 0, bins - 1)
            for fit in FITS:
                err = mol.get("err_" + fit)
                if err is None:
                    continue
                found = np.isfinite(err[:, 0])
                acc[fit][0] += np.bincount(b[sel], minlength=bins)
                acc[fit][1] += np.bincount(b[sel & found], minlength=bins)
        out[on] = {"edges": edges.tolist(),
                   **{fit: {"n": a[0].tolist(), "recall": (a[1] / np.maximum(a[0], 1)).tolist()}
                      for fit, a in acc.items() if a[0].sum()}}
    return out


def figure_metrics(summary, path):
    import matplotlib.pyplot as plt
    panels = [("jaccard", "Jaccard (all z)", None), ("core_recall", "recall, |z| < 300 nm", None),
              ("core_rmse_lat_nm", "lateral RMSE (nm), |z| < 300", None),
              ("core_rmse_z_nm", "z RMSE (nm), |z| < 300", "core_rmse_z_phase_nm"),
              ("core_width_lat", "lateral error / CRLB", None), ("core_width_z", "z error / CRLB", None)]
    ons = [on for on in ONS if any(f"{d:g}_{on}" in summary for d in DENSITIES)]
    fig, axes = plt.subplots(len(ons), len(panels) + 1, figsize=(3.0 * (len(panels) + 1), 3.0 * len(ons)),
                             constrained_layout=True, squeeze=False)
    for i, on in enumerate(ons):
        for j, (key, title, phase_key) in enumerate(panels):
            ax = axes[i, j]
            for fit, (label, colour, ls, _) in FITS.items():
                x = [d for d in DENSITIES if summary.get(f"{d:g}_{on}", {}).get(fit, {}).get(key) is not None]
                y = [summary[f"{d:g}_{on}"][fit][key] for d in x]
                if x:
                    ax.plot(x, y, color=colour, ls=ls, marker="o", ms=3, label=label)
                if phase_key and fit.endswith("4pi"):
                    y2 = [summary[f"{d:g}_{on}"][fit].get(phase_key) for d in x]
                    ax.plot(x, y2, color="#2a9d4a", ls=ls, marker="s", ms=3,
                            label=label.replace("4Pi", "4Pi, z from the phase"))
            ax.set_xscale("log")
            if key.startswith("core_rmse"):
                ax.set_yscale("log")
            ax.set_title(f"{title}\non-time {on} frame{'s' if on > 1 else ''}", fontsize=9)
            ax.set_xlabel(f"per 40 × 40 px (1 = {1 / (40 * PX_UM) ** 2:.3f} / µm²)", fontsize=8)
            ax.grid(alpha=0.3)
        ax = axes[i, -1]
        for fit in ("netfit_4pi", "spline_4pi"):
            label, colour, ls, _ = FITS[fit]
            x = [d for d in DENSITIES if summary.get(f"{d:g}_{on}", {}).get(fit)]
            ax.plot(x, [100 * summary[f"{d:g}_{on}"][fit].get("core_wrong_fringe", np.nan) for d in x],
                    color=colour, ls=ls, marker="o", ms=3, label=label)
        ax.set_xscale("log")
        ax.set_title(f"4Pi: wrong fringe (%), |z| < 300\non-time {on}", fontsize=9)
        ax.set_xlabel("molecules per 40 × 40 px")
        ax.grid(alpha=0.3)
    handles, labels = axes[0, 3].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.savefig(path, dpi=130)


def figure_mechanism(contrast, pair, summary, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), constrained_layout=True)
    phi = np.array(contrast["phi"])
    s = np.array(contrast["z+0"]["shares"])
    for c in range(s.shape[1]):
        axes[0].plot(phi, s[:, c], label=f"channel {c}")
    axes[0].set_xlabel("phase φ (rad)")
    axes[0].set_ylabel("share of the molecule's light")
    axes[0].set_title("each channel's share, z = 0", fontsize=9)
    axes[0].legend(fontsize=7)
    dphi = np.array(contrast["dphi"])
    for z, ls in (("z+0", "-"), ("z-300", "--"), ("z+300", ":")):
        r = contrast[z]["ratio"]
        axes[1].plot(dphi, r["mean"], ls=ls, color="k", label=z.replace("z", "z = ") + " nm")
        if z == "z+0":
            axes[1].fill_between(dphi, r["p10"], r["p90"], color="k", alpha=0.15)
    axes[1].axhline(1, color="#d0661c", lw=2, label="astigmatic (one image)")
    axes[1].set_xlabel("phase difference Δφ (rad)")
    axes[1].set_ylabel("brightness ratio in the best channel")
    axes[1].set_title("two equal molecules at one place:\nhow much dimmer one is in the best channel", fontsize=9)
    axes[1].legend(fontsize=7)
    for k, on in enumerate(ONS):
        ax = axes[2 + k]
        p = pair[on]
        mid = (np.array(p["edges"][:-1]) + np.array(p["edges"][1:])) / 2
        for fit, (label, colour, ls, _) in FITS.items():
            if fit in p:
                ax.plot(mid, p[fit]["recall"], color=colour, ls=ls, marker="o", ms=3, label=label)
        fits = [k2 for k2 in p if k2 != "edges"]
        if not fits:
            ax.set_visible(False)
            continue
        n = p[fits[0]]["n"]
        ax.set_title(f"recall of close pairs (< {PAIR_NM:.0f} nm), |z| < 300,\n5 -- 50 per 40² px, on-time {on}; "
                     f"{int(sum(n))} molecules", fontsize=9)
        ax.set_xlabel("|Δφ| to the nearest neighbour (rad)")
        ax.set_ylabel("recall")
        ax.grid(alpha=0.3)
    axes[2].legend(fontsize=7)
    fig.savefig(path, dpi=130)


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    summary, rows = {}, {}
    for on in ONS:
        for d in DENSITIES:
            mol, res = load(folder, d, on)
            if mol is None:
                continue
            rows[(d, on)] = mol
            summary[f"{d:g}_{on}"] = {fit: s for fit in FITS if (s := scores(mol, res, fit))}
    pair = pairs(rows)
    with open(folder / "summary.json", "w") as f:
        json.dump({"rows": summary, "pairs": {str(k): v for k, v in pair.items()}}, f, indent=1)
    figure_metrics(summary, folder / "metrics_vs_density.png")
    contrast = json.loads((folder / "contrast.json").read_text())
    figure_mechanism(contrast, pair, summary, folder / "mechanism.png")
    for key in ("core_recall", "jaccard", "core_rmse_lat_nm", "core_rmse_z_nm", "core_width_lat", "core_width_z"):
        print(key)
        for on in ONS:
            for fit in FITS:
                vals = [summary.get(f"{d:g}_{on}", {}).get(fit, {}).get(key) for d in DENSITIES]
                if any(v is not None for v in vals):
                    print(f"  on{on} {fit:18s}", " ".join("  -  " if v is None else f"{v:6.3g}" for v in vals))


if __name__ == "__main__":
    main()
