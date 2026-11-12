"""The port's DECODE model scored on netfit's fixed test set, as netfit's models are scored on it.

    python benchmarks/decode/suite/port_on_testset.py MODEL.safetensors TESTSET.npz [--keep 0.5]

The test set's frames are photons as the fitter converts them, drawn as an EM camera reads the chip;
the port's predictor takes the frames its model reads (the middle one for a 1-frame model, the window
for a 3-frame one), mirrors them where its calibration was taken without EM gain, and extracts with
its own post-processing at DECODE's raw threshold 0.1; localizations are kept at ``prob`` >= 0.5 and
scored by `smappy.netfit.metrics.evaluate` -- the same matching and numbers as netfit's own models.
"""
import argparse
import json

import numpy as np


def main():
    from decode_smlm.predict import load_predictor
    from smappy.netfit import metrics, testset
    from smappy.training import model_file
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("test_set")
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--out")
    a = ap.parse_args()
    meta = model_file.read_metadata(a.model)
    p = load_predictor(a.model, None, frames_mirrored=not meta.get("calibration_em_on", False), raw_th=0.1)
    w = p.window
    batch, truth, tmeta = testset.load(a.test_set, "cpu", window=2 * w + 1)
    frames = batch.frames.numpy()
    found = {"frame": [], "xyz": [], "err": []}
    px = tuple(tmeta["camera"]["pixel_nm"])
    for i in range(len(frames)):
        cols = p(frames[i], 0)                     # one sample: its window, the middle frame localized
        keep = cols["prob"] >= a.keep
        found["frame"].append(np.full(keep.sum(), i))
        found["xyz"].append(np.c_[cols["x_pix"][keep], cols["y_pix"][keep], cols["z_nm"][keep]])
        found["err"].append(np.c_[cols["x_err_pix"][keep] * px[0], cols["y_err_pix"][keep] * px[1], cols["z_err_nm"][keep]])
    found = {k: np.concatenate(v) for k, v in found.items()}
    r = metrics.evaluate(truth, found, px)
    r.update(model=a.model, test_set=a.test_set, keep=a.keep)
    if a.out:
        open(a.out, "w").write(json.dumps(r, indent=1))
    keys = ("jaccard", "recall", "precision", "rmse_lat_nm", "rmse_ax_nm", "pixel_ks", "crlb_width_lat",
            "crlb_width_ax", "core_rmse_lat_nm", "core_rmse_ax_nm", "core_pixel_ks")
    print({k: round(r[k], 4) for k in keys if r.get(k) is not None})
    for b in r.get("by_abs_z", []):
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items()})


if __name__ == "__main__":
    main()
