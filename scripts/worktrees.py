"""Which work is not on main yet: every worktree and every local branch, against origin/main.

    python scripts/worktrees.py            # the table
    python scripts/worktrees.py --stale 7  # only what has waited more than 7 days

The `/land` skill's ``status`` (`.claude/skills/land/SKILL.md`), so that finished work in a forgotten
worktree is found and either landed or removed (CLAUDE.md, "Before you edit anything").  Per worktree
or branch: commits not on origin/main, files changed and not committed, the last commit's age, and
what it is -- *land* (commits waiting), *commit* (changes not committed), *merged* (nothing left that
main lacks: removable once no session works in it), *main* (Jonas's checkout).  Reads git only;
changes nothing.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

MAIN_CHECKOUT = pathlib.Path("/Users/ries/git/SMAPpy")


def git(*args, cwd=None) -> str:
    out = subprocess.run(["git", *args], check=False, cwd=cwd, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def worktrees(root) -> list[dict]:
    rows, row = [], {}
    for line in git("worktree", "list", "--porcelain", cwd=root).splitlines() + [""]:
        if not line:
            if row:
                rows.append(row)
            row = {}
        elif line.startswith("worktree "):
            row["path"] = line[9:]
        elif line.startswith("branch "):
            row["branch"] = line[7:].removeprefix("refs/heads/")
        elif line == "detached":
            row["branch"] = None
    return rows


def state(root, ref, path=None) -> dict:
    ahead = int(git("rev-list", "--count", f"origin/main..{ref}", cwd=root) or 0)
    when = int(git("log", "-1", "--format=%ct", ref, cwd=root) or 0)
    dirty = len(git("status", "--porcelain", cwd=path).splitlines()) if path and pathlib.Path(path).exists() else 0
    return {"ahead": ahead, "dirty": dirty, "age_days": (time.time() - when) / 86400 if when else None,
            "subject": git("log", "-1", "--format=%s", ref, cwd=root)[:70]}


def table(root, stale_days=None):
    git("fetch", "-q", "origin", cwd=root)
    rows, seen = [], set()
    for w in worktrees(root):
        ref = w.get("branch") or git("rev-parse", "HEAD", cwd=w["path"]) or "HEAD"
        seen.add(w.get("branch"))
        rows.append({"where": w["path"], "branch": w.get("branch") or "(detached)", **state(root, ref, w["path"])})
    for branch in git("for-each-ref", "--format=%(refname:short)", "refs/heads", cwd=root).splitlines():
        if branch not in seen and branch != "main":
            rows.append({"where": "(no worktree)", "branch": branch, **state(root, branch)})
    for r in rows:
        if pathlib.Path(r["where"]) == MAIN_CHECKOUT:
            r["do"] = "main"
        elif r["dirty"]:
            r["do"] = "commit"
        elif r["ahead"]:
            r["do"] = "land"
        else:
            r["do"] = "merged"
    if stale_days is not None:
        rows = [r for r in rows if r["do"] != "main" and (r["age_days"] or 0) > stale_days]
    order = {"commit": 0, "land": 1, "merged": 2, "main": 3}
    rows.sort(key=lambda r: (order[r["do"]], -(r["age_days"] or 0)))
    print(f"{'do':7s} {'ahead':>5s} {'dirty':>5s} {'days':>5s}  branch / worktree / last commit")
    for r in rows:
        days = f"{r['age_days']:.0f}" if r["age_days"] is not None else "-"
        where = r["where"].replace(str(root) + "/", "")
        print(f"{r['do']:7s} {r['ahead']:5d} {r['dirty']:5d} {days:>5s}  {r['branch']}  [{where}]  {r['subject']}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale", type=float, default=None, help="only what has waited more than this many days")
    a = ap.parse_args()
    root = pathlib.Path(git("rev-parse", "--path-format=absolute", "--git-common-dir")).parent
    table(root, a.stale)


if __name__ == "__main__":
    sys.exit(main())
