"""`scripts/tested.py`: a tree that passed is not tested again, and a branch rebased onto changes
elsewhere needs only the fast tier -- on a throwaway repository with an `origin/main`."""
import importlib.util
import os
import pathlib
import subprocess

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "tested.py"
spec = importlib.util.spec_from_file_location("tested", _SCRIPT)
tested = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tested)

ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
           GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, env=ENV, check=True, capture_output=True,
                          text=True).stdout.strip()


def commit(root, name, text):
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(text)
    git(root, "add", name)
    git(root, "commit", "-qm", name)


def land_on_main(root, name, text, stamp=True):
    """Another session's landing: a commit on origin/main, its tree stamped as tested."""
    git(root, "checkout", "-q", "main")
    git(root, "reset", "-q", "--hard", "origin/main")
    commit(root, name, text)
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    if stamp:
        tested.record(str(root), "fast")
    git(root, "checkout", "-q", "work")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    root = tmp_path / "r"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    commit(root, "a.py", "".join(f"a{i} = {i}\n" for i in range(12)))
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    tested.record(str(root), "fast")
    git(root, "checkout", "-q", "-b", "work")
    commit(root, "b.py", "b = 1\n")
    return root


def test_a_tree_that_passed_needs_nothing_and_a_lower_tier_is_not_enough(repo):
    tested.record(str(repo), "slow")
    assert tested.check(str(repo), "slow")[0] == "none"
    assert tested.check(str(repo), "fast")[0] == "none"
    assert tested.check(str(repo), "release")[0] == "release"


def test_an_uncommitted_edit_is_a_new_tree(repo):
    tested.record(str(repo), "fast")
    (repo / "b.py").write_text("b = 2\n")
    assert tested.check(str(repo), "fast")[0] == "fast"
    (repo / "new.py").write_text("untracked\n")
    assert tested.worktree_tree(str(repo)) != git(repo, "rev-parse", "HEAD^{tree}")


def test_a_rebase_onto_other_files_needs_only_the_fast_tier(repo):
    tested.record(str(repo), "slow")
    land_on_main(repo, "c.py", "c = 1\n")
    git(repo, "rebase", "-q", "origin/main")
    need, why = tested.check(str(repo), "slow")
    assert need == "fast", why


def test_a_rebase_onto_the_same_files_needs_the_tier(repo):
    lines = (repo / "a.py").read_text().splitlines(keepends=True)
    commit(repo, "a.py", "".join(lines[:-1]) + "a11 = 'branch'\n")
    tested.record(str(repo), "slow")
    land_on_main(repo, "a.py", "a0 = 'main'\n" + "".join(lines[1:]))
    git(repo, "rebase", "-q", "origin/main")       # clean: the two edits are far apart
    need, why = tested.check(str(repo), "slow")
    assert need == "slow" and "files the branch changes" in why


def test_a_change_to_how_everything_builds_or_runs_needs_the_tier(repo):
    tested.record(str(repo), "slow")
    land_on_main(repo, "csrc/fit.cpp", "// changed\n")
    git(repo, "rebase", "-q", "origin/main")
    assert tested.check(str(repo), "slow")[0] == "slow"


def test_main_moved_by_an_untested_push_needs_the_tier(repo):
    tested.record(str(repo), "slow")
    land_on_main(repo, "c.py", "c = 1\n", stamp=False)
    git(repo, "rebase", "-q", "origin/main")
    need, why = tested.check(str(repo), "slow")
    assert need == "slow" and "no passing run" in why


def test_a_different_change_on_the_branch_needs_the_tier(repo):
    tested.record(str(repo), "slow")
    land_on_main(repo, "c.py", "c = 1\n")
    git(repo, "rebase", "-q", "origin/main")
    commit(repo, "b.py", "b = 2\n")
    assert tested.check(str(repo), "slow")[0] == "slow"
