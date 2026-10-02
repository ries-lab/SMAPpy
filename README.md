# SMAPpy

SMAPpy is software for single-molecule localization microscopy (SMLM) in
Python. It takes raw camera frames to localizations and super-resolution
images, with a graphical interface, command-line tools and a Python API. It is
a Python port of [SMAP](https://github.com/jries/SMAP).

## Install

    pip install smappy-smlm

The GUI is included. Binary wheels are available for Python 3.9-3.13 on macOS,
Linux and Windows, so nothing needs to be compiled. The package is called
`smappy-smlm` on PyPI; in Python you `import smappy`.

With [uv](https://docs.astral.sh/uv/) you can install the GUI as a standalone
tool, or run it without installing:

    uv tool install smappy-smlm
    uvx --from smappy-smlm smappy-gui

To install from a checkout, which builds the C++ extensions (pip 21.3 or newer):

    git clone https://github.com/ries-lab/SMAPpy.git
    cd SMAPpy
    python -m venv .venv
    .venv/bin/python -m pip install -e .

## Start

    smappy-gui

or, to open a localization file directly:

    smappy-gui FILE.hdf5

From a checkout, activate the venv first or run `.venv/bin/smappy-gui`.

## Tutorials and documentation

**Tutorials.** [ries-lab.github.io/SMAPpy](https://ries-lab.github.io/SMAPpy/)
has short guided tours on simulated data you can follow along with. Start with
[SMAPpy in three minutes](https://ries-lab.github.io/SMAPpy/quickstart/).

**Help pages.** Every plugin and the main parts of the GUI have a help page.
It explains the theoretical background and every control. Open it with the
**?** button on the plugin or panel, with F1, or from *Help → Plugin
documentation*. To write the same pages as a static website:

    python -m smappy.docs -o DIR

**For developers.** [GUI.md](GUI.md) describes the architecture.
[docs/](docs/) holds design notes on the plugin system and individual
algorithms. [NOTES.md](NOTES.md) records design decisions and measurements.

## Tests

    SMAPPY_TEST_CAL=/path/to/_3dcal.mat python -m pytest tests/

## License

SMAPpy is released under the BSD 3-Clause license; see [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
