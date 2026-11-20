"""A netfit model fitted on one context's blinking C = 2 sets (the study's queue, plex-plan §16):

    python fit_blink_c2.py CONTEXT MODEL OUTDIR

The fits and their records go to OUTDIR as `run_c2` keeps them, to be scored where the sets' truth
and the scoring's dependencies are (`run_c2.run` with the fits copied under ``fits_c2/<RUN>``).
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_c2 as RC

SETS = ("density_1", "density_5", "density_15", "density_25", "density_50", "photons_1000", "photons_3000",
        "photons_10000")


def main():
    ctx, model, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    for s in SETS:
        name = f"{ctx}_blink_{s}"
        info = RC.fit_netfit(name, out / f"{name}.npz", model)
        (out / f"{name}.json").write_text(json.dumps(info))
        print(name, round(info["seconds"], 1), "s", flush=True)


if __name__ == "__main__":
    main()
