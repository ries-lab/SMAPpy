"""Write a v0.10.2 run's per-epoch metrics, from its checkpoint's log dict, as JSON lines.

    GLIBC_TUNABLES=glibc.rtld.execstack=2 python export_log_v0102.py RUN_DIR OUT.jsonl

One line per epoch: {"epoch", "walltime", <metric>: value, ...}.  The checkpoint holds every
scalar the live engine logged (the DictLogger), so TensorBoard's files are not needed.
"""
import json
import pathlib
import sys

import torch

run, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
ckpt = next(run.rglob("ckpt.pt"))
log = torch.load(ckpt, map_location="cpu")["log"]
rows = {}
for key, series in log.items():
    for step, value, wall in zip(series["step"], series["scalar"], series["walltime"]):
        if key == "learning/train_batch":      # per batch, not per epoch
            continue
        row = rows.setdefault(int(step), {"epoch": int(step), "walltime": wall})
        row[key] = float(value)
        row["walltime"] = min(row["walltime"], wall)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for step in sorted(rows):
        f.write(json.dumps(rows[step]) + "\n")
print(f"{len(rows)} epochs from {ckpt} to {out}")
