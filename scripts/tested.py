"""Which trees have passed the tests, so that `/land` does not run them twice.

    python scripts/tested.py check --tier slow   # what this tree still needs: none, fast or slow
    python scripts/tested.py list                # the stamps, newest first

The `/land` skill's step 4 (`.claude/skills/land/SKILL.md`).  Several sessions land at once, and
each ran the tests before proposing to land; `/land` then ran them again on the same tree, and a
push rejected because another landing moved main sent it round once more -- with three landings
in flight the last tested three times (2026-11-28).  So every passing full run leaves a stamp
(`tests/conftest.py` calls `record`), and `check` answers from the stamps:

* **none** -- this exact tree passed at this tier or a higher one.  A tree hash covers every
  file's content, committed or not (`worktree_tree`), so this costs no safety at all.
* **fast** -- the tree is new, but only because the branch was rebased: its own changes are the
  ones that passed (the same patch id over the same files), on a base X that main has since moved
  from to Y, and nothing between X and Y touches a file the branch touches or how everything
  builds and runs (`SHARED`), and Y's own tree passed when it landed.  What is left to break is a
  clash between files that do not overlap (a function renamed under a caller the branch adds), so
  the fast tier runs -- about a minute -- and the slow one is not repeated.
* **<tier>** -- anything else: the tier asked for.

The stamps live in the repository's shared git directory (`git rev-parse --git-common-dir`), so
every worktree sees every other's; they are a few hundred bytes each and never committed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

#: a tier covers those below it: --long runs --slow and every package, --release every package
TIERS = {"fast": 0, "slow": 1, "release": 2, "long": 3}

#: a change here between the tested base and the new one means the new base is retested whole:
#: what builds the extensions, how the tests run, what is installed
SHARED = ("csrc", "setup.py", "pyproject.toml", "tests/conftest.py")

MAIN = "origin/main"


def git(root, *args, env=None, input=None) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True,
                          timeout=60, env=env, input=input).stdout.strip()


def store(root) -> pathlib.Path:
    common = pathlib.Path(root, git(root, "rev-parse", "--git-common-dir"))
    path = common / "smappy-tested"
    path.mkdir(exist_ok=True)
    return path


def worktree_tree(root) -> str:
    """The tree of the files as they are: HEAD plus staged, unstaged and untracked changes (not
    ignored ones), written through a copy of the index so the real one is left alone.  Copying it
    keeps git's stat cache, so this is some 50 ms on SMAPpy, not a hash of every file."""
    index = git(root, "rev-parse", "--path-format=absolute", "--git-path", "index")
    with tempfile.TemporaryDirectory() as tmp:
        copy = os.path.join(tmp, "index")
        # copy2 keeps the index's mtime: git trusts an entry's stat data only when the file is
        # older than the index, and a copy made now would hide an edit made in the same instant
        # as the last `git add` (seen under a loaded `-n auto` run: a same-size edit missed)
        shutil.copy2(index, copy)
        env = dict(os.environ, GIT_INDEX_FILE=copy)
        git(root, "add", "-A", env=env)
        return git(root, "write-tree", env=env)


def changed_files(root, a, b) -> list[str]:
    out = git(root, "diff", "--name-only", a, b)
    return out.splitlines() if out else []


def branch_change(root, tree) -> tuple[str, str, list[str]]:
    """The branch's own change, judged against main: its base (the merge base with main), the
    patch id of base..tree and the files it touches.  The patch id is git's hash of a diff with
    line numbers left out, so a clean rebase onto changes elsewhere keeps it."""
    base = git(root, "merge-base", "HEAD", MAIN)
    diff = git(root, "diff", "--binary", base, tree)
    pid = git(root, "patch-id", "--stable", input=diff + "\n").split(" ")[0] if diff else ""
    return base, pid, changed_files(root, base, tree)


def _covers(a, b) -> bool:
    return any(f == p or f.startswith(p + "/") for f in a for p in b)


def record(root, tier: str, passed: int = 0, tree: str | None = None) -> dict:
    """Stamp `tree` (the files as they are) as passed at `tier`."""
    tree = tree or worktree_tree(root)
    base, pid, files = branch_change(root, tree)
    stamp = {"tree": tree, "tier": tier, "base": base, "patch_id": pid, "files": files,
             "head": git(root, "rev-parse", "HEAD"),
             "branch": git(root, "rev-parse", "--abbrev-ref", "HEAD"),
             "passed": passed, "python": sys.executable,
             "date": datetime.datetime.now().isoformat(timespec="seconds")}
    path = store(root) / f"{tree}.json"
    old = _read(path)
    # a lower tier never overwrites a higher one on the same tree
    if old is None or TIERS[tier] >= TIERS.get(old.get("tier"), -1):
        path.write_text(json.dumps(stamp, indent=1))
    return stamp


def _read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def stamps(root) -> list[dict]:
    found = [_read(p) for p in store(root).glob("*.json")]
    return sorted((s for s in found if s), key=lambda s: s.get("date", ""), reverse=True)


def check(root, tier: str) -> tuple[str, str]:
    """What the files as they are still need before landing: ("none" | "fast" | tier, why)."""
    want = TIERS[tier]
    tree = worktree_tree(root)
    all_ = stamps(root)
    for s in all_:
        if s["tree"] == tree and TIERS.get(s["tier"], -1) >= want:
            return "none", f"this tree passed --{s['tier']} on {s['date']} ({s['branch']})"
    if want == TIERS["fast"]:
        return tier, "this tree has not passed the fast tier"
    base, pid, files = branch_change(root, tree)
    main_tree = git(root, "rev-parse", f"{base}^{{tree}}")
    reasons = []
    for s in all_:
        if s["patch_id"] != pid or not pid or TIERS.get(s["tier"], -1) < want:
            continue
        old = s["base"]
        if subprocess.run(["git", "merge-base", "--is-ancestor", old, base], cwd=root,
                          check=False).returncode:
            reasons.append(f"{old[:8]} is not an ancestor of {base[:8]}")
            continue
        moved = changed_files(root, old, base)
        if set(moved) & set(files):
            reasons.append(f"main changed files the branch changes since {old[:8]}")
            continue
        if _covers(moved, SHARED):
            reasons.append(f"main changed {', '.join(SHARED)} since {old[:8]}")
            continue
        if not any(t["tree"] == main_tree for t in all_):
            reasons.append(f"main's tree at {base[:8]} has no passing run stamped")
            continue
        return "fast", (f"the branch's change passed --{s['tier']} on {old[:8]} ({s['date']}); "
                        f"main moved to {base[:8]} in {len(moved)} other files, itself tested")
    why = "; ".join(dict.fromkeys(reasons)) or "this change has not passed this tier on any base"
    return tier, why


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="what this tree still needs: none, fast or the tier")
    c.add_argument("--tier", choices=list(TIERS), default="fast")
    r = sub.add_parser("record", help="stamp this tree as passed (conftest does this itself)")
    r.add_argument("--tier", choices=list(TIERS), required=True)
    sub.add_parser("list", help="the stamps, newest first")
    args = parser.parse_args(argv)
    root = git(".", "rev-parse", "--show-toplevel")
    if args.cmd == "check":
        need, why = check(root, args.tier)
        print(f"need: {need}\n{why}")
    elif args.cmd == "record":
        print(json.dumps(record(root, args.tier), indent=1))
    else:
        for s in stamps(root):
            print(f"{s['date']}  {s['tier']:<7} {s['tree'][:10]}  {s['branch']}  "
                  f"base {s['base'][:8]}  {len(s['files'])} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
