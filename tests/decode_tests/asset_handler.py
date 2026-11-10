"""DECODE's test assets, as far as an offline suite has them.

``RMAfterTest`` is DECODE's (decode/test/asset_handler.py, v0.10.2), unchanged.  ``AssetHandler``
downloaded the files listed in DECODE's assets/asset_list.yaml from the web; here a test that
asks for one is skipped, as the conftest skips the tests marked ``web``.
"""
import pathlib
import shutil

import pytest


class RMAfterTest:
    """
    A small helper that provides a context manager for test binaries, i.e. deletes them after leaving the
    with statement.
    """

    def __init__(self, path, recursive: bool = False):
        """

        Args:
            path: path to file that should be deleted after context
            recursive: if true and path is a dir, then the whole dir is deleted. Careful!
        """
        assert isinstance(path, pathlib.Path)
        self.path = path
        self.recursive = recursive

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path.exists():
            if self.path.is_file():
                self.path.unlink()
            elif self.recursive and self.path.is_dir():
                shutil.rmtree(self.path)


class AssetHandler:
    def auto_load(self, file):
        pytest.skip(f"needs DECODE's downloaded test asset {pathlib.Path(file).name}")
