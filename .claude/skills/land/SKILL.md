---
name: land
description: Land a session's finished work on SMAPpy's main -- commit its files by name, rebase on origin/main, run the tests, push to main (or open a pull request with "/land pr"), then tidy the worktree. "/land status" lists every worktree and branch with work not yet on main. Use when Jonas says /land, "land it", "merge it" or "push it to main", or asks what is not merged yet.
---

# Landing work on main

Jonas's one command for "this work is done, put it on main".  A session works
in a worktree of its own (CLAUDE.md, "Before you edit anything"); `/land` is
how that work reaches `origin/main` without Jonas doing the git himself.  His
`/land` is the permission to push to `main` -- for this landing, not for later
ones.

## Proposing it

**A session proposes landing itself**, explicitly, whenever a task it did in
its worktree is finished and its tests pass: in the wrap-up, the question
"Land this on main?" with a button that answers `/land` (and, when a review is
worth it, one for `/land pr`).  Work left in a worktree is forgotten work; a
session that ends a task without proposing to land it has left Jonas a stale
branch to find later.  Not proposed: work that is not finished, failing tests,
or a study whose result Jonas has not seen.

## /land

1. **Where.**  `git rev-parse --show-toplevel` must be a worktree, not
   `/Users/ries/git/SMAPpy` (Jonas's checkout).  If the work was done there
   anyway, stop and say so: move it to a branch first, never push from his
   checkout.
2. **What lands.**  `git status --short` and `git log --oneline origin/main..HEAD`.
   Commit what this session changed and has not committed -- staged by name,
   never `git add -u` / `-A`, each commit a change of one kind, the message
   saying why, ending with the attribution lines.  Files this session did not
   touch stay out; name them to Jonas.
3. **Up to date.**  `git fetch origin && git rebase origin/main`.  A conflict in
   a document (`NOTES.md`, the plans under `docs/`) is resolved by keeping both
   sides' content; a conflict in code that is not mechanical stops the landing:
   say which files and what each side did, and ask.  If `csrc/` moved, build
   again: `python setup.py build_ext --inplace`.
4. **Tests.**  The fast tier always: `python -m pytest tests -q -n auto` in
   this worktree (`tests/conftest.py` makes it test this tree).  The slow tier
   (`--slow`) when CLAUDE.md asks for it -- a new plugin, a GUI change, a
   tutorial, a page, what an algorithm computes -- or when Jonas says
   `/land slow`.  A failure the change caused is fixed and step 3 repeated; a
   failure it did not cause (run it alone; compare with `origin/main`) is
   named in the report, not hidden.  Lint what changed (`ruff check`, and
   `src/decode_smlm`'s and uipsf's own configurations for theirs).
5. **Land.**
   - `/land`: `git push origin HEAD:main`.  Rejected because main moved: back
     to step 3.  Never `--force`.
   - `/land pr`: `git push -u origin HEAD`, then `gh pr create --base main`
     with a title, a body that says what and why, the test line ("`-n auto`:
     4644 passed, 1 skipped" or "`--slow -n auto`: ..."), and the attribution
     line.  Report the PR's link.  It is merged when Jonas says so (`/land
     merge`: `gh pr merge --merge`, then as after a push).
6. **Jonas's main follows.**  After a push to main (or a merged PR), bring
   the local checkout up to date, as `git pull` there would:
   `git -C /Users/ries/git/SMAPpy fetch origin` and, when it is on `main`,
   `git -C /Users/ries/git/SMAPpy merge --ff-only origin/main`.  A
   fast-forward only: when it refuses -- his checkout is on another branch,
   has commits main lacks, or has uncommitted changes in files the landing
   touches -- leave it, and say so and why.  If `csrc/` changed in what came
   in, build there too (`python setup.py build_ext --inplace` in
   `/Users/ries/git/SMAPpy`): SMAPpy is installed editable from that
   checkout, so every Python on the machine imports it from there.
7. **Tidy.**  After the work is on main: if the session is done, remove its
   worktree and branch from Jonas's checkout's side --
   `git -C /Users/ries/git/SMAPpy worktree remove <path>` and
   `git -C /Users/ries/git/SMAPpy branch -d <branch>` (and the remote branch
   of a merged PR) -- as the very last step, since the session's own folder
   goes with it.  If the session goes on, keep the worktree and continue from
   `origin/main` (`git rebase origin/main`).
8. **Report** what landed (the commits, `git log --oneline` of them), the
   tests' result, and -- CLAUDE.md, "Remind the user to release" -- what has
   gathered since the last tag when the landing was major work or an
   important fix.  Never tag a release.

## /land status

`python scripts/worktrees.py` (`--stale DAYS` for what has waited longer):
every worktree and local branch with its commits not on `origin/main`, its
uncommitted files, its last commit's age, and *land* / *commit* / *merged* /
*main*.  Show Jonas the *land* and *commit* rows -- forgotten work -- with what
each holds (`git log --oneline origin/main..<branch>`), and offer, per row, to
land it (in its worktree, or a new one for a branch without), to look at it, or
to leave it.  *merged* rows hold nothing main lacks: offer to remove them, all
at once, once no running session works in them (`list_sessions` names their
folders); never remove a worktree with uncommitted files or one a running
session is in.
