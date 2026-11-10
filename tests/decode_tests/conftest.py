"""decode_smlm's tests: DECODE v0.10.2's own, ported (src/decode_smlm/NOTICE.md), and the port's.

The tiers are SMAPpy's root conftest's: DECODE marked its slower tests ``slow``, which ``--slow``
runs.  Its other markers are registered here: ``plot`` (a test that draws, run as any other
under the Agg backend) and ``web`` / ``webbig`` (a test that downloads), skipped -- the suite runs
offline, and what DECODE fetched from the web its port's tests make or take from the cache.
"""
import os

import pytest
import torch

# One thread per xdist worker: four workers with four threads each are slower than one.
if os.environ.get("PYTEST_XDIST_WORKER"):
    torch.set_num_threads(1)


def pytest_configure(config):
    config.addinivalue_line("markers", "plot: a DECODE test that draws a figure")
    config.addinivalue_line("markers", "web: a DECODE test that needs the web; skipped")
    config.addinivalue_line("markers", "webbig: a DECODE test that downloads large files; skipped")


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="downloads: the suite runs offline")
    for item in items:
        if "web" in item.keywords or "webbig" in item.keywords:
            item.add_marker(skip)
