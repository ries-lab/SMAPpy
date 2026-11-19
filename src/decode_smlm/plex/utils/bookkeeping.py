import importlib.util
from pathlib import Path

try:
    import git

    _git_available = True
# can cause import errors, not because of package but because of git
except ImportError:
    _git_available = False

import decode_smlm.plex
def decode_state() -> str:
    """Get version tag of decode.

    If decode is imported from a git repository with tags, this returns the
    output of ``git describe``. Otherwise it falls back to the package version.
    """

    p = Path(importlib.util.find_spec("decode").origin).parents[1]

    if _git_available:
        try:
            r = git.Repo(p)
            return r.git.describe(dirty=True)

        except git.exc.InvalidGitRepositoryError:  # not a repo but an installed package
            return decode_smlm.plex.__version__

        except git.exc.GitCommandError:
            return decode_smlm.plex.__version__
    else:
        return decode_smlm.plex.__version__
