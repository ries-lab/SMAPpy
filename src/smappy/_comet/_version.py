# Vendored from COMET 1.1.0 -- https://github.com/gpufit/Comet
# Copyright (c) 2025 Lenny Reinkensmeier & Mark Bates, MIT licensed: see
# LICENSE.txt beside this file, which travels with every copy.
# Upstream path: comet/_version.py
#
# Changed here: absolute ``comet.`` imports are relative, so this copy is
# self-contained and cannot collide with an installed COMET.

"""Single source of truth for the package version.

Kept in its own dependency-free module so setuptools can read it statically at
build time (see ``[tool.setuptools.dynamic]`` in pyproject.toml) without
importing the rest of the package.
"""

__version__ = "1.1.0"
