"""``smappy-decode``: train a job folder, stop it, say where it stands (milestone (f)).

    smappy-decode train FOLDER [--device cuda] [--epochs N] [--workers N]
    smappy-decode stop FOLDER
    smappy-decode status FOLDER
    smappy-decode serve

``train`` runs in the foreground and can be stopped with ``stop`` from another shell (or over
ssh), Ctrl-C or SIGTERM; run again, it resumes from the last epoch.  ``status`` reads only the
folder's files, so it needs neither torch nor the training's machine.  ``serve`` localizes for a
SMAPpy on another computer, over stdin and stdout (`serve`): its DECODE fitter starts it over ssh.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _status(folder: Path) -> int:
    import json

    from smappy.training import progress
    lines = progress.read(folder)
    state = progress.state(lines)
    lock = folder / "lock.json"
    if state == "running" and not lock.exists():
        state = "interrupted"
    print(f"{folder}: {state}")
    if lock.exists():
        print(f"  locked: {lock.read_text().strip()}")
    epochs = [line for line in lines if line["kind"] == "epoch"]
    start = next((line for line in reversed(lines) if line["kind"] == "start"), None)
    if start:
        print(f"  {start.get('epochs')} epochs on {start.get('device')}, seed {start.get('seed')}")
    if epochs:
        e = epochs[-1]
        m = e.get("metrics", {})
        print(f"  epoch {e['epoch']}: test loss {e.get('loss_test')}, jaccard {m.get('jaccard')}, "
              f"rmse lat {m.get('rmse_lat_nm')} nm, {e.get('seconds')} s")
    for line in lines:
        if line["kind"] in ("restart", "error"):
            print(f"  {line['kind']} at epoch {line['epoch']}: {json.dumps({k: v for k, v in line.items() if k not in ('kind', 'epoch')})}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="smappy-decode", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="train the job in FOLDER, resuming where it stopped")
    t.add_argument("folder", type=Path)
    t.add_argument("--device", default=None, help="cuda, cuda:1, mps, cpu or auto (default: the job's)")
    t.add_argument("--epochs", type=int, default=None, help="instead of the job's")
    t.add_argument("--workers", type=int, default=None, help="data-loader workers, instead of the job's")
    t.add_argument("--allow-other-version", action="store_true",
                   help="train a job prepared by another SMAPpy version")
    t.add_argument("--all-epochs", action="store_true",
                   help="train every epoch, not stopping when the training no longer improves")
    t.add_argument("--stop-rule", default=None,
                   help="the rule for stopping, as JSON over SMAPpy's (training.progress.EARLY_STOP): "
                        "window, min_epochs, smooth, loss_tol, jaccard_tol, eff_tol, rmse_tol")
    for name, text in (("stop", "stop the training of FOLDER at its next checkpoint"),
                       ("status", "where the job in FOLDER stands")):
        sub.add_parser(name, help=text).add_argument("folder", type=Path)
    sub.add_parser("serve", help="localize frames sent on stdin with a model sent first (SMAPpy's DECODE fitter)")
    a = ap.parse_args(argv)
    if a.command == "serve":
        from .serve import serve
        return serve()
    if a.command == "status":
        return _status(a.folder)
    if a.command == "stop":
        from .training import request_stop
        print(f"asked to stop: {request_stop(a.folder)}")
        return 0
    from .training import JobError, train
    try:
        result = train(a.folder, device=a.device, epochs=a.epochs, num_workers=a.workers,
                       allow_other_version=a.allow_other_version, all_epochs=a.all_epochs,
                       stop_rule=_stop_rule(a.stop_rule))
    except JobError as e:
        print(f"smappy-decode: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted: run train again to resume from the last epoch", file=sys.stderr)
        return 130
    print(result)
    return 0


def _stop_rule(text):
    """``--stop-rule``'s JSON, or None; a key the rule does not have is refused, not ignored."""
    if not text:
        return None
    import json

    from smappy.training.progress import EARLY_STOP
    rule = json.loads(text)
    unknown = sorted(set(rule) - set(EARLY_STOP))
    if unknown:
        raise SystemExit(f"smappy-decode: --stop-rule: no {', '.join(unknown)} (the rule has {', '.join(EARLY_STOP)})")
    return rule


if __name__ == "__main__":
    sys.exit(main())
