"""Every molecule of M2's nucleus movie localised once per model, saved, for the analyses that read heights.

    python heights.py OUTDIR MODEL [MODEL ...] [--states "used,held out,not drawn"]

Writes ``OUTDIR/heights_<MODEL>.npz`` per model: each unit's id and state, its height above the coverslip
``h`` (nm, the in situ geometry's), the height's CRLB ``sd_h``, ½ Σ deviance (the score `cross_score.py` uses),
photons, its place on the chip (``x_px``, ``y_px``) and its frame and movie.  Each unit is localised from
`STARTS_NM` with x, y, z, photons and background free, the best start kept (`cross_score.insitu_scores`'s
procedure).  A MODEL is a saved ``<name>_3dcal.h5`` in OUTDIR; an in situ model uses its own focus, so a stack
model enters through the in situ run that held every aberration of it (``insitu_all_n1345_ab_n1345``).

`movie_rois` with its defaults is deterministic, so every model's file holds the same units in the same order
(``units`` is checked by the readers).  Units the selection rejected are not localised: they are pairs,
neighbours or off centre, and would only add noise to an edge.

The readers: `tilt_from_edge.py` (the coverslip's edge per tile of the field), `edge_vs_time.py` (per time
block) and `score_by_height.py` (two models' scores per height bin).
"""
import argparse
import glob
import os

import numpy as np
import torch

# the data are not in git: M2_DATA names the folder that holds 261112_new's sets
ROOT = os.environ.get("M2_DATA", os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Vienna/Data/"
                                                     "smappy test/261112_new"))
MOVIE = "261112_MR_Intercalib2_BGAAF647_50ms_30i_nucleus_1/*_MMStack_Pos0.ome.tif"
PIXEL_NM = (127.0, 116.0)          # iXon on M2's astigmatic path (docs/uipsf/microscopes.md)
STARTS_NM = (-200.0, 200.0, 600.0, 1000.0, 1400.0)
STATES = ("used", "held out", "not drawn")


def localise_all(model, shared: dict, batch, noise, starts=STARTS_NM) -> dict:
    """Each emitter of ``batch`` localised from every height in ``starts``, the best (lowest ½ Σ deviance)
    kept: {"h", "sd_h", "half_deviance", "photons"}, numpy [E]."""
    from uipsf.solve import Problem
    from uipsf.workflows.insitu import INSITU_CHUNK
    from uipsf.workflows.scoring import localise

    E = batch.n_emitters
    out = {k: np.full(E, np.nan) for k in ("h", "sd_h", "photons")}
    out["half_deviance"] = np.full(E, np.inf)
    for z0 in starts:
        xyz = torch.zeros(E, 3, dtype=torch.float64)
        xyz[:, 2] = z0
        fit, problem = localise(model, shared, batch, noise, start={"emitter.xyz": xyz}, chunk_size=INSITU_CHUNK)
        dev = Problem(model, noise, problem.batch, INSITU_CHUNK).deviance(fit.params)
        e = 0.5 * dev.reshape(E, -1).sum(1).double().cpu().numpy()
        phys = fit.params.physical()
        photons = phys["emitter.photons"] if "emitter.photons" in phys else phys["emitter.photons_per_slice"].mean(1)
        better = e < out["half_deviance"]
        out["half_deviance"][better] = e[better]
        out["h"][better] = phys["emitter.xyz"][:, 2].detach().cpu().double().numpy()[better]
        out["sd_h"][better] = fit.std["emitter.xyz"][:, 2].detach().cpu().double().numpy()[better]
        out["photons"][better] = photons.detach().cpu().double().numpy()[better]
    return out


def unit_table(rois, ids) -> dict:
    """Where and when each unit ``ids`` of ``rois`` (`InSituROIs`) is: its state, chip pixel, first frame,
    movie."""
    u = rois.units
    source = np.zeros(len(u.reason), int) if u.source is None else np.asarray(u.source)
    return {"units": np.asarray(ids), "state": np.array([rois.state[i] for i in ids]),
            "x_px": np.asarray(u.x_px)[ids], "y_px": np.asarray(u.y_px)[ids],
            "frame": np.array([int(np.min(u.frames[i])) for i in ids]), "source": source[ids]}


def path(out: str, name: str) -> str:
    return os.path.join(out, f"heights_{name}.npz")


def load(out: str, name: str) -> dict:
    with np.load(path(out, name), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def coverslip_edge(h, sd):
    """The heights' lower edge (`uipsf.workflows.insitu_rounds.coverslip_edge`, the rounds' own fit), or None."""
    from uipsf.workflows.insitu_rounds import coverslip_edge as edge
    ok = np.isfinite(h) & np.isfinite(sd) & (sd > 0)
    return edge(np.asarray(h)[ok], np.asarray(sd)[ok])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--states", default=",".join(STATES), help="comma-separated unit states to localise")
    a = ap.parse_args()
    from smappy.calibrate.insitu_run import movie_rois
    from smappy.calibrate.pupil import load_start

    rois = movie_rois(sorted(glob.glob(os.path.join(ROOT, MOVIE))))
    states = [s.strip() for s in a.states.split(",")]
    ids = np.sort(np.concatenate([rois.ids(s) for s in states]))
    batch, noise = rois.batch(ids=ids)
    table = unit_table(rois, ids)
    print(f"{len(ids)} units ({', '.join(f'{s}: {len(rois.ids(s))}' for s in states)})", flush=True)
    for name in a.names:
        m = load_start(os.path.join(a.out, f"{name}_3dcal.h5"))
        shared = {k: torch.as_tensor(v) for k, v in m.shared.items()}
        found = localise_all(m.model, shared, batch, noise)
        np.savez(path(a.out, name), **table, **found, focus_nm=float(torch.as_tensor(shared["geometry.focus"])))
        print(f"  {name:44s} heights p5/50/95 {np.nanpercentile(found['h'], [5, 50, 95]).round(0).tolist()} nm, "
              f"½Σdeviance {found['half_deviance'].mean():.2f} per unit -> {path(a.out, name)}", flush=True)


if __name__ == "__main__":
    main()
