"""``python -m smappy``: the ``smappy`` command, for an environment whose
scripts are not on the PATH."""
import sys

from .cli.main import main

sys.exit(main())
