"""Tables of m5_media.py's results across the eight conditions, as Markdown.

    python benchmarks/uipsf/real/m5_media_table.py benchmarks/uipsf/results/m5_media_window_studio.jsonl [...]

One row per run: the condition (colour, medium, lens), the z window and slice spacing it learnt on, then what
physics says must agree between conditions -- σ beyond the pixel, the phase in nm of path (the main modes),
the magnitude -- and what judges the model: the deviance per pixel and the x/y/z bends of the slice fits.
A second table lines the conditions up against each other: the spread over the four 2D (or 3D) runs of one
window is what the microscope does not explain, the step 2D -> 3D is the lens, 600 -> 676 the colour.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

# Noll modes that carry the phase on M5: astigmatism (5, 6), coma (7, 8), spherical (11, 22, 37)
MODES = (5, 6, 7, 8, 11, 22, 37)


def rows(paths):
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    # m5_media.jsonl predates the fields: each bead on ±1 µm about its focus at 50 nm
                    yield {"window_nm": 1000.0, "dz_nm": 50.0, **json.loads(line)}


def bend(r, half=500.0):
    """The x, y, z bend's rms over |z| <= ``half`` from the stored curve, its median taken out: a range every
    window covers (±500 learns on ±500 only, and `measure` reads ±600 of the others)."""
    z = np.asarray(r["curve_z_nm"], float)
    c = np.asarray(r["curve_nm"], float)
    near = np.isfinite(z) & (np.abs(z) <= half)
    c = c[near] - np.nanmedian(c[near], axis=0)
    return np.sqrt(np.nanmean(c ** 2, axis=0))


def fmt(v, d=1):
    return "–" if v is None else f"{v:.{d}f}"


def table(rs):
    head = ["λ", "medium", "lens", "window", "dz", "beads", "σx, σy (nm)", "magn. radial / asym",
            *[f"Z{j} (nm)" for j in MODES], "deviance", "x, y, z bend rms (nm)", "same, ±500", "z max",
            "z scatter/CRLB", "NA_eff", "min"]
    out = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for r in rs:
        ph = r["phase_nm_path"]
        out.append("| " + " | ".join([
            r["colour"], r["medium"], r["lens"], f"±{r['window_nm']:.0f}", f"{r['dz_nm']:.0f}", str(r["used"]),
            "{}, {}".format(*r["sigma_optics_nm"]),
            f"{r['magnitude_radial_rms']:.3f} / {r['magnitude_asym_rms']:.3f}",
            *[fmt(ph.get(str(j))) for j in MODES], fmt(r["atlas_mean_deviance"], 2),
            ", ".join(fmt(r[f"{a}_systematic_rms_nm"]) for a in "xyz"), ", ".join(fmt(v) for v in bend(r)),
            fmt(r["z_systematic_max_nm"]),
            fmt(r["z_scatter_over_crlb"], 2), fmt(r.get("effective_na"), 3), fmt(r["minutes"])]) + " |")
    return "\n".join(out)


def spread(rs):
    """Per (variant, window, grid NA) and lens: the mean and the spread (max - min) over the four colour x
    medium runs of what must agree -- σ and the phase in nm of path."""
    groups = defaultdict(list)
    for r in rs:
        groups[(r["variant"], r["window_nm"], r["dz_nm"], r["grid_na"], r["lens"])].append(r)
    head = ["variant", "window", "dz", "grid NA", "lens", "runs", "σx mean (range)", "σy mean (range)",
            *[f"Z{j} mean (range)" for j in MODES if j not in (7, 8)], "z bend rms ±500 mean", "deviance mean"]
    out = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for (variant, w, dz, na, lens), g in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        def cell(vals):
            vals = np.asarray(vals, float)
            return f"{vals.mean():.1f} ({np.ptp(vals):.1f})"
        cells = [cell([r["sigma_optics_nm"][0] for r in g]), cell([r["sigma_optics_nm"][1] for r in g])]
        cells += [cell([r["phase_nm_path"].get(str(j), np.nan) for r in g]) for j in MODES if j not in (7, 8)]
        cells.append(f"{np.mean([bend(r)[2] for r in g]):.1f}")
        cells.append(f"{np.mean([r['atlas_mean_deviance'] for r in g]):.2f}")
        out.append("| " + " | ".join([variant, f"±{w:.0f}", f"{dz:.0f}", f"{na:.2f}", lens, str(len(g)), *cells])
                   + " |")
    return "\n".join(out)


def nats(rs):
    """Per acquisition, on one window: the Zernike magnitude on the grid at 1.50 and 1.51, and the bead physics
    at 1.51 (and at 1.50, where m5_media.jsonl is given too), in nats (the objective is ½ deviance + priors;
    the beads are the same, so they compare).  The learnt edge is bounded at ρ 1.05: there the cutoff lies
    outside the grid, and the NA is "the grid's or more", not ρ x the grid's."""
    runs = {(r["acquisition"], r["window_nm"], r["dz_nm"], r["grid_na"], r["variant"]): r for r in rs}
    head = ["λ", "medium", "lens", "window", "Zernike 1.51 over 1.50", "physics over Zernike, 1.50",
            "physics 1.51 over 1.50", "edge ρ at 1.50", "physics over Zernike, 1.51", "edge ρ", "NA_eff",
            "cos γ", "σx, σy physics (nm)", "z bend ±500 Zernike / physics"]
    out = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for (a, w, dz, na, v), p in sorted(runs.items(), key=lambda kv: (kv[1]["lens"], kv[1]["colour"],
                                                                     kv[1]["medium"])):
        if na != 1.51 or v == "zernike" or (a, w, dz, 1.51, "zernike") not in runs:
            continue
        z1 = runs[(a, w, dz, 1.51, "zernike")]
        z0 = runs.get((a, w, dz, 1.5, "zernike"))
        p0 = runs.get((a, w, dz, 1.5, v))
        rho = p["edge"]["edge_rho"]
        out.append("| " + " | ".join([
            p["colour"], p["medium"], p["lens"], f"±{w:.0f}",
            "–" if z0 is None else f"{z0['objective'] - z1['objective']:+.0f}",
            "–" if z0 is None or p0 is None else f"{z0['objective'] - p0['objective']:+.0f}",
            "–" if p0 is None else f"{p0['objective'] - p['objective']:+.0f}",
            "–" if p0 is None else f"{p0['edge']['edge_rho']:.3f}",
            f"{z1['objective'] - p['objective']:+.0f}", f"{rho:.3f}",
            f"≥ {na:.2f} (bound)" if rho >= 1.049 else f"{p['effective_na']:.3f}",
            f"{p['edge'].get('centre_g', float('nan')):.2f}", "{}, {}".format(*p["sigma_optics_nm"]),
            f"{bend(z1)[2]:.1f} / {bend(p)[2]:.1f}"]) + " |")
    return "\n".join(out)


def main():
    rs = list(rows(sys.argv[1:]))
    by_variant = defaultdict(list)
    for r in rs:
        by_variant[(r["variant"], r["grid_na"])].append(r)
    for (variant, na), g in by_variant.items():
        g.sort(key=lambda r: (r["lens"], r["colour"], r["medium"], r["window_nm"]))
        print(f"\n### {variant}, grid NA {na}\n\n{table(g)}")
    print(f"\n### Agreement over colour and medium\n\n{spread(rs)}")
    if any(r["grid_na"] != 1.5 for r in rs):
        print(f"\n### The grid's NA and the bead physics, nats (+: the second better)\n\n{nats(rs)}")


if __name__ == "__main__":
    main()
