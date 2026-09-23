"""Run a chain of plugins over many files, headless.

    smappy-batch run JOB.batch.yaml [--limit N] [--force]
    smappy-batch validate JOB.batch.yaml | CHAIN.chain.yaml
    smappy-batch plugins [PREFIX]
    smappy-batch describe PLUGIN/PATH | CHAIN.chain.yaml

`docs/batch.md` describes the two files.  Progress goes to stdout as one line
per event, ``SMAPPY_START<tab>3/20<tab>path`` and so on, which is what the
batch window reads; everything else a plugin says goes there too, prefixed
``SMAPPY_PROGRESS``.  The exit code is 0 when every file was processed (or
was already), 1 when any failed, and 2 when the job did not validate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _is_chain(path: str) -> bool:
    return path.endswith((".chain.yaml", ".chain.yml"))


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")          # no window, ever: this runs as a subprocess
    p = argparse.ArgumentParser(prog="smappy-batch", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run a job")
    r.add_argument("job")
    r.add_argument("--limit", type=int, default=None,
                   help="only the first N files: a trial run")
    r.add_argument("--force", action="store_true",
                   help="also re-run files already done with the same settings")
    v = sub.add_parser("validate", help="check a job or a chain, run nothing")
    v.add_argument("file")
    l = sub.add_parser("plugins", help="list the plugins, by path")
    l.add_argument("prefix", nargs="?", default="")
    d = sub.add_parser("describe", help="a plugin's fields, or a chain's steps")
    d.add_argument("what")
    args = p.parse_args(argv)

    from .. import batch, chain, plugins
    if args.command == "plugins":
        for path, ref in plugins.refs(args.prefix).items():
            kind = " (chain)" if ref.kind == "chain" else ""
            print(f"{path}{kind}  --  {ref.description}")
        return 0
    if args.command == "describe":
        if _is_chain(args.what) or Path(args.what).is_file():
            print(batch.describe_chain(chain.read(args.what)))
        else:
            try:
                print(batch.describe_plugin(args.what))
            except KeyError:
                print(f"no plugin {args.what!r}; `smappy-batch plugins` lists them",
                      file=sys.stderr)
                return 2
        return 0
    if args.command == "validate":
        try:
            if _is_chain(args.file):
                _, problems = batch.validate_chain(chain.read(args.file))
            else:
                problems = batch.validate(batch.read_job(args.file))
        except (ValueError, OSError) as error:
            print(f"error: {error}")
            return 2
        for problem in problems:
            print(problem)
        errors = sum(pr.level == "error" for pr in problems)
        print(f"{errors} error(s), {len(problems) - errors} warning(s)")
        return 2 if errors else 0
    emit = batch.emitter()
    try:
        job = batch.read_job(args.job)
        report = batch.run(job, limit=args.limit, force=args.force, emit=emit)
    except (ValueError, OSError) as error:
        for line in str(error).splitlines():
            emit("ERROR", line)
        return 2
    return 1 if report["status"] != "complete" else 0


if __name__ == "__main__":
    sys.exit(main())
