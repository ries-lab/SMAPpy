"""Several trainings' curves on one figure: how good, and how soon (`docs/decode/benchmark.md`).

    python benchmarks/decode/suite/curves.py JOB_FOLDER [...] [--out curves.png] [--hours]

Each job folder's ``progress.jsonl`` (a netfit or DECODE-port training): the test set's loss,
Jaccard, lateral RMSE, pixelation (KS) overall and near the focus, error over CRLB -- against epochs,
or with ``--hours`` against the training's own time (seconds summed), so that a slower network's
epochs are not mistaken for faster convergence.  The 11-epoch running median, and a dot where each
quantity came within its tolerance of its final value (`training.progress.convergence`).  Trainings
compared this way should score the same fixed test set (`netfit testset`); the curves of trainings
on their own test sets differ by those sets too.
"""
import argparse
import pathlib

import numpy as np

PANELS = (("loss_test", "test loss"), ("jaccard", "Jaccard"), ("rmse_lat_nm", "RMSE lateral (nm)"),
          ("pixel_ks", "pixelation, KS"), ("core_pixel_ks", "pixelation |z| < 300 nm, KS"),
          ("crlb_width_lat", "error / CRLB lateral"))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from smappy.training import progress
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("curves.png"))
    ap.add_argument("--hours", action="store_true")
    a = ap.parse_args()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for job in a.jobs:
        lines = progress.read(job)
        epochs = progress._epochs(lines)
        if not epochs:
            continue
        x = np.cumsum([float(e.get("seconds") or 0) for e in epochs]) / 3600 if a.hours else \
            np.array([e["epoch"] for e in epochs])
        conv = progress.convergence(lines)
        for ax, (name, label) in zip(axes.flat, PANELS):
            raw = [e.get(name) if name == "loss_test" else (e.get("metrics") or {}).get(name) for e in epochs]
            v = np.asarray(progress._running_median([np.nan if r is None else r for r in raw], 11))
            line, = ax.plot(x, v, label=job.name)
            if name in conv:
                i = next(k for k, e in enumerate(epochs) if e["epoch"] == conv[name]["epoch"])
                ax.plot(x[i], v[i], "o", color=line.get_color())
            ax.set(title=label, xlabel="training hours" if a.hours else "epoch")
            ax.grid(alpha=.3)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=100)
    print("figure:", a.out)


if __name__ == "__main__":
    main()
