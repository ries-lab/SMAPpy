"""The compiled modules are the ones the sources describe.

An extension that is out of date still imports, so the only symptom of a
missed rebuild is an `AttributeError` deep inside an unrelated test, which
reads like an environment problem rather than a stale binary.  `setup.py` no
longer places the modules where they cannot be found, and this says plainly
when a rebuild is what is needed.
"""
import re
from pathlib import Path

import pytest

CSRC = Path(__file__).resolve().parents[1] / "csrc"
SOURCES = {"_fit3d": "fit.cpp", "_drift": "drift.cpp",
           "_group": "group.cpp", "_render": "render.cpp"}


def bound_names(source: Path):
    """The names the module's `PYBIND11_MODULE` block defines."""
    return set(re.findall(r'm\.def\(\s*"([^"]+)"', source.read_text()))


@pytest.mark.skipif(not CSRC.is_dir(), reason="installed without the sources")
@pytest.mark.parametrize("module,source", sorted(SOURCES.items()))
def test_every_bound_function_is_in_the_built_module(module, source):
    import importlib
    built = importlib.import_module(f"smappy.{module}")
    expected = bound_names(CSRC / source)
    missing = sorted(name for name in expected if not hasattr(built, name))
    assert not missing, (
        f"smappy.{module} is missing {missing}: the built extension is older "
        f"than csrc/{source}.  Rebuild with "
        f"`python setup.py build_ext --inplace`.")
