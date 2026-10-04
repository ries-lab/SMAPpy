"""The tables of README.md, from results.csv: the bias of each corner counter
against the detected corners of the same pores, in percentage points of ELE.

    python studies/npc_le/summary.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main() -> None:
    d = pd.read_csv(HERE / "results.csv")
    d["bias"] = 100 * (d.ele - d.ele_detected)
    pd.set_option("display.width", 250)
    truth = (d[d.method == "hard"]
             .groupby(["photons", "blinks", "efficiency"])[["ele_labelled", "ele_detected",
                                                            "pores", "junk"]]
             .mean().round(3))
    print("Truth and segmentation, mean of the seeds\n")
    print(truth.to_string(), "\n")
    table = (d.groupby(["method", "photons", "blinks", "efficiency"]).bias.mean()
             .unstack(["photons", "blinks", "efficiency"]).round(1))
    print("Bias, percentage points (method - detected), mean of the seeds\n")
    print(table.to_string(), "\n")
    spread = d.groupby(["method", "photons", "blinks", "efficiency"]).bias.std()
    print("Spread between seeds, median over conditions:",
          round(float(spread.median()), 1), "points\n")
    for photons in sorted(d.photons.unique()):
        part = d[d.photons == photons].groupby(["method", "blinks", "efficiency"]).bias.mean()
        worst = part.abs().groupby("method").max().sort_values()
        print(f"Worst |bias| at {photons} photons, points:")
        print(worst.round(1).to_string(), "\n")


if __name__ == "__main__":
    main()
