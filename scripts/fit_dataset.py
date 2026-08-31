#!/usr/bin/env python3
"""Run ``smappy-fit`` from a source checkout, installed or not.

The command itself lives in ``smappy.cli.fit``; this is the same thing for
anyone who has not installed the package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smappy.cli.fit import main  # noqa: E402

if __name__ == "__main__":
    main()
