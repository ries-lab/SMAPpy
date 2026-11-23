"""netfit on a field-dependent PSF: trained with the field's centre everywhere, trained with the field, and Spline 3D
with the node splines (`docs/decode/fd-plan.md` §7).

    python benchmarks/decode/field/fd_study.py prepare --field synthetic --scale 2 --out runs/fd
    python -m smappy.netfit train runs/fd/synthetic_x2_fd          # on a GPU host, each folder
    python benchmarks/decode/field/fd_study.py score runs/fd/synthetic_x2 [--photons 5000 1500]

``prepare`` writes the truth's node splines as a SMAPpy calibration (`field_correction_study.truth` and
``splines``: the field-dependent vector pupil of `docs/uipsf/field-dependence.md` §11, sampled on the smallest
node grid that keeps to `field_splines.TOLERANCE`) and two training folders on it, the job's settings
(`netfit.job.settings_from_estimates`: GMN7C's ConvGRU over 7 frames, 6 M samples) with the field "off"
(``_centre``) and "on" (``_fd``).

``score`` draws test windows with netfit's own drawing at the field's node splines (the truth), each window a
40 x 40 crop centred on one of a 5 x 5 grid of field places, sparse (2 molecules a frame on average), and
localizes their middle frames with both networks and with Spline 3D (the node splines, each ROI at its own
chip place, three z starts, the best likelihood): a 13 x 13 ROI at every molecule's nearest pixel, no peak
finding, so that what differs is the PSF each method knows.  Molecules nearer than 8 px to another, or on
for less than 90 % of the frame, are not scored.  Reported per method and photon count: the z bias (median of
fitted minus true z) at each field place and z bin, |z| < 300 nm first; the lateral and axial SD at the four
corners and the centre beside the local PSF's CRLB; the networks' recall.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "uipsf" / "synthetic"))

N_XY, FRAME = 5, 40
Z_BINS = np.array([-600, -300, -100, 100, 300, 600.0])
CAMERA = {"px_size": None, "e_per_adu": 1.0, "baseline": 100.0, "em_gain": 0.0, "read_sigma": 1.0, "qe": 1.0}


# ---------------------------------------------------------------- prepare

def prepare(args) -> None:
    from field_correction_study import splines, truth

    from smappy.io.calibration import save_spline_calibration
    from smappy.netfit.job import settings_from_estimates
    name = f"{args.field}_x{args.scale:g}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr = truth(args.field, args.scale)
    fitters, info = splines(tr)
    cal = fitters["node splines"].calibration
    cal.em_on = False
    cal_path = out / f"{name}_3dcal.h5"
    save_spline_calibration(cal_path, cal, overwrite=True)
    est = SimpleNamespace(intensity_mu_sig=(4000.0, 2500.0), background=(10.0, 120.0), z_nm=(-700.0, 700.0),
                          lifetime_frames=1.5, photons_range=None)
    camera = {**CAMERA, "px_size": (tr.pixel_nm, tr.pixel_nm)}
    for mode, suffix in (("off", "centre"), ("on", "fd")):
        s = settings_from_estimates(est, camera, 50.0, calibration=str(cal_path.resolve()), seed=args.seed)
        s.simulation.field = mode
        folder = out / f"{name}_{suffix}"
        folder.mkdir(exist_ok=True)
        (folder / "settings.json").write_text(json.dumps(s.to_dict(), indent=1), encoding="utf-8")
    meta = {"field": args.field, "scale": args.scale, "pixel_nm": tr.pixel_nm, "centre_px": list(tr.centre),
            "px_per_u": tr.px_per_u, "extent": tr.extent, **info}
    (out / f"{name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{name}: node grid {info['grid']} x {info['grid']}, blend error {100 * info['blend_error']:.2f} %; "
          f"folders {name}_centre, {name}_fd in {out}")


# ---------------------------------------------------------------- score

class _Places:
    """A `FieldRecord` whose crops are where the study puts them, not drawn."""

    def __init__(self, record, corners):
        self.record, self.fixed = record, corners

    def corners(self, n, size, g, device):
        return self.fixed[:n].to(device)

    def maps_torch(self, corners, size):
        return self.record.maps_torch(corners, size)


def draw(cal, meta, photons: float, per_place: int, seed: int):
    """Test windows at the 5 x 5 places, ``per_place`` each: (batch, place index per sample)."""
    import torch

    from smappy.netfit import simulate
    from smappy.netfit.field import FieldRecord
    from smappy.netfit.render import SplineRenderer
    from smappy.netfit.settings import CameraModel, SimulationSettings
    record = FieldRecord.from_splines(cal.field_splines, (meta["pixel_nm"],) * 2)
    g = np.linspace(-meta["extent"], meta["extent"], N_XY)
    centres = np.array([[meta["centre_px"][0] + a * meta["px_per_u"], meta["centre_px"][1] + b * meta["px_per_u"]]
                        for b in g for a in g])
    place = np.repeat(np.arange(len(centres)), per_place)
    corners = torch.as_tensor(np.floor(centres[place] - (FRAME - 1) / 2 + 0.5), dtype=torch.float32)
    sim = SimulationSettings(frame_px=FRAME, window=7, density_per_40px=2.0, lifetime_frames=1.0,
                             photons_mean=photons, photons_sd=0.0, background=(20.0, 20.0), z_nm=(-650.0, 650.0),
                             target_photons_min=0.0)
    cam = CameraModel(conversion=1.0, offset=100.0, em_gain=0.0, read_noise_e=1.0, pixel_nm=(meta["pixel_nm"],) * 2)
    renderer = SplineRenderer(cal, field=True)
    batch = simulate.draw_batch(renderer, sim, cam, len(place), torch.Generator().manual_seed(seed),
                                field=_Places(record, corners))
    return batch, place, renderer, cam


def targets(batch):
    """Every molecule on in a middle frame, however briefly: sample, x, y, z, its photons there, and whether it is
    alone (8 px from every other one on)."""
    m = batch.mask.numpy()
    sample = np.nonzero(m)[0]
    xyz = batch.xyz.numpy()[m].astype(float)
    photons = batch.photons.numpy()[m]
    alone = np.ones(len(sample), bool)
    for s in np.unique(sample):
        k = np.flatnonzero(sample == s)
        if len(k) > 1:
            d = np.hypot(*(xyz[k, None, :2] - xyz[None, k, :2]).transpose(2, 0, 1))
            np.fill_diagonal(d, np.inf)
            alone[k] = d.min(1) > 8
    return sample, xyz, photons, alone


def network(model_path, batch, device):
    """The model's detections on every sample's middle frame: sample, x, y, z."""
    import torch

    from smappy.netfit import extract
    from smappy.netfit.predict import Predictor
    p = Predictor(model_path, device, threshold=0.5)
    out = []
    with torch.no_grad():
        for a in range(0, len(batch.frames), 64):
            x = ((batch.frames[a:a + 64] - p.scaling.input_offset) / p.scaling.input_scale).to(p.device)
            o = p.model(x, aux=batch.aux[a:a + 64].to(p.device)) if p.field is not None else p.model(x)
            e = extract.extract(o, p.scaling, p.threshold, p.extraction, p.scale, layout=p.model.layout)
            out.append(np.c_[e["frame"].cpu().numpy() + a, e["x"].cpu().numpy(), e["y"].cpu().numpy(),
                             e["z_nm"].cpu().numpy()])
    return np.concatenate(out) if out else np.zeros((0, 4))


def spline3d(cal, batch, sample, xyz, roi: int = 13):
    """Spline 3D with the node splines on each molecule's ROI at its nearest pixel: x, y (frame), z; NaN where
    the ROI leaves the frame."""
    from smappy import _fit3d
    from smappy.field_splines import bilinear
    mid = batch.frames.shape[1] // 2
    frames = batch.frames[:, mid].numpy()
    half = roi // 2
    cx, cy = np.rint(xyz[:, 0]).astype(int), np.rint(xyz[:, 1]).astype(int)
    inside = (cx - half >= 0) & (cy - half >= 0) & (cx + half < FRAME) & (cy + half < FRAME)
    out = np.full((len(sample), 3), np.nan)
    k = np.flatnonzero(inside)
    rois = np.stack([frames[sample[i], cy[i] - half:cy[i] + half + 1, cx[i] - half:cx[i] + half + 1] for i in k])
    chip = batch.origin.numpy()[sample[k]] + np.c_[cx[k], cy[k]]
    nodes, w = bilinear(cal.field_splines, chip)
    best, logl = None, None
    for z0 in (-400.0, 0.0, 400.0):
        theta, _, ll, _ = _fit3d.fit_cspline_field(rois.astype(np.float32), cal.field_splines.coeff, nodes, w,
                                                   float(cal.z_nm_to_index(z0)), 100, 0)
        if best is None:
            best, logl = theta.copy(), ll.copy()
        else:
            better = ll > logl
            best[better], logl[better] = theta[better], ll[better]
    out[k] = np.c_[best[:, 0] + cx[k] - half, best[:, 1] + cy[k] - half, cal.z_index_to_nm(best[:, 4])]
    return out


def summarise(name, err, place, z, bound, found=None) -> list[dict]:
    rows = []
    corners = {0, N_XY - 1, N_XY * (N_XY - 1), N_XY * N_XY - 1}
    centre = (N_XY * N_XY) // 2
    for p in range(N_XY * N_XY):
        s = place == p
        for lo, hi in pairwise(Z_BINS):
            b = s & (z >= lo) & (z < hi) & np.isfinite(err[:, 2])
            if b.sum() < 5:
                continue
            row = {"method": name, "place": p, "u_index": p % N_XY, "v_index": p // N_XY, "z_lo": lo, "z_hi": hi,
                   "n": int(b.sum()), "z_bias": float(np.median(err[b, 2])),
                   "x_bias": float(np.median(err[b, 0])), "y_bias": float(np.median(err[b, 1]))}
            if p in corners or p == centre:
                mad = lambda a: float(1.4826 * np.median(np.abs(a - np.median(a))))
                row.update(x_mad=mad(err[b, 0]), y_mad=mad(err[b, 1]), z_mad=mad(err[b, 2]))
                row.update(x_sd=float(np.std(err[b, 0])), y_sd=float(np.std(err[b, 1])), z_sd=float(np.std(err[b, 2])),
                           crlb_x=float(np.mean(bound[b, 0])), crlb_y=float(np.mean(bound[b, 1])),
                           crlb_z=float(np.mean(bound[b, 2])))
            if found is not None:
                row["recall"] = float(found[s & (z >= lo) & (z < hi)].mean())
            rows.append(row)
    return rows


def score(args) -> None:
    from smappy.io.calibration import load_spline_calibration
    from smappy.netfit import metrics
    base = Path(args.run)
    meta = json.loads(base.with_suffix(".json").read_text())
    cal = load_spline_calibration(base.parent / f"{base.name}_3dcal.h5")
    px = meta["pixel_nm"]
    rows = []
    for photons in args.photons:
        batch, place_of_sample, renderer, _ = draw(cal, meta, photons, args.per_place, args.seed)
        sample, xyz, on, alone = targets(batch)
        place = place_of_sample[sample]
        chip = xyz[:, :2] + batch.origin.numpy()[sample]
        bound = metrics.crlbs_torch(renderer, np.full(len(sample), photons), np.full(len(sample), 20.0), xyz[:, 2],
                                    (px, px), False, chip=chip)
        use = alone & (on >= 0.9 * photons)          # scored: alone, and on for most of the frame
        fits = {"Spline 3D, node splines": spline3d(cal, batch, sample, xyz)}
        found = {}
        for suffix, label in (("centre", "netfit, centre PSF"), ("fd", "netfit, field")):
            path = base.parent / f"{base.name}_{suffix}" / "model.safetensors"
            if not path.exists():
                print(f"no {path}: skipped")
                continue
            det = network(path, batch, args.device)
            ti, fi = metrics.match(sample, xyz, det[:, 0].astype(int), det[:, 1:], (px, px))
            f = np.full((len(sample), 3), np.nan)
            f[ti] = det[fi, 1:]
            fits[label] = f
            found[label] = np.isfinite(f[:, 0])
        for label, f in fits.items():
            err = np.c_[(f[:, 0] - xyz[:, 0]) * px, (f[:, 1] - xyz[:, 1]) * px, f[:, 2] - xyz[:, 2]]
            r = summarise(label, err[use], place[use], xyz[use, 2], bound[use],
                          found[label][use] if label in found else None)
            for row in r:
                row.update(photons=photons, field=meta["field"], scale=meta["scale"])
            rows += r
        report(rows, photons)
    out = Path(args.out or base.parent / f"{base.name}_scores.jsonl")
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"written {out}")


def report(rows, photons) -> None:
    """The z bias's worst place near focus and over ±600 nm, and the corners' precision against the CRLB."""
    rows = [r for r in rows if r["photons"] == photons]
    for name in dict.fromkeys(r["method"] for r in rows):
        mine = [r for r in rows if r["method"] == name]
        near = [abs(r["z_bias"]) for r in mine if r["z_lo"] >= -300 and r["z_hi"] <= 300]
        every = [abs(r["z_bias"]) for r in mine]
        corner = [r for r in mine if "z_sd" in r and r["place"] != (N_XY * N_XY) // 2 and r["z_lo"] >= -300
                  and r["z_hi"] <= 300]
        ratio_z = np.mean([r["z_sd"] / r["crlb_z"] for r in corner]) if corner else float("nan")
        ratio_x = np.mean([np.hypot(r["x_sd"], r["y_sd"]) / np.hypot(r["crlb_x"], r["crlb_y"]) for r in corner]) \
            if corner else float("nan")
        mad_z = np.mean([r["z_mad"] / r["crlb_z"] for r in corner]) if corner else float("nan")
        mad_x = np.mean([np.hypot(r["x_mad"], r["y_mad"]) / np.hypot(r["crlb_x"], r["crlb_y"]) for r in corner]) \
            if corner else float("nan")
        recall = [r["recall"] for r in mine if "recall" in r]
        print(f"{photons:6.0f} ph  {name:26s} |z bias| near focus max {max(near):6.1f} nm, over ±600 max "
              f"{max(every):6.1f} nm; corners near focus SD/CRLB lateral {ratio_x:.2f}, axial {ratio_z:.2f} "
              f"(robust {mad_x:.2f}, {mad_z:.2f})"
              + (f"; recall {np.mean(recall):.3f}" if recall else ""))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--field", default="synthetic")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--out", default="benchmarks/decode/results/fd")
    p.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("run", help="the prepared run's prefix, e.g. runs/fd/synthetic_x2")
    s.add_argument("--photons", type=float, nargs="+", default=[5000.0, 1500.0])
    s.add_argument("--per-place", type=int, default=600)
    s.add_argument("--device", default="cpu")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out")
    a = ap.parse_args(argv)
    prepare(a) if a.cmd == "prepare" else score(a)


if __name__ == "__main__":
    main()
