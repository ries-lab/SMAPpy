# SMAPpy

SMAPpy is software for single-molecule localization microscopy (SMLM) in
Python. It takes raw camera frames to localizations and super-resolution
images, with a graphical interface, command-line tools and a Python API. It is
a Python port of [SMAP](https://github.com/jries/SMAP).

## Install

    pip install smappy-smlm

The GUI is included. Binary wheels are available for Python 3.12-3.14 on macOS,
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

Files can also be dropped onto either window: a localization file opens as
with *File → Open*; a short TIFF (50 frames or fewer) or a PNG is added as an
image layer, as with *File → Open image*; and a longer stack (TIFF, an NDTiff
folder, `*.sim.yaml`) becomes the *file* of the fitter on the Localize tab,
ready to fit.

From a checkout, activate the venv first or run `.venv/bin/smappy-gui`.

## Tutorials and documentation

**Tutorials.** [ries-lab.github.io/SMAPpy](https://ries-lab.github.io/SMAPpy/)
has short guided tours on simulated data you can follow along with. Start with
[SMAPpy in three minutes](https://ries-lab.github.io/SMAPpy/quickstart/).

**Help pages.** Every plugin and the main parts of the GUI have a help page.
It explains the theoretical background and every control. Open it with the
**?** in a plugin's or a Render-tab section's title bar, with F1 in any panel or
window, or from *Help → Plugin documentation*. The same pages are online at
[ries-lab.github.io/SMAPpy/help](https://ries-lab.github.io/SMAPpy/help/), and
`python -m smappy.docs -o DIR` writes them as a static website.

**For developers.** [GUI.md](GUI.md) describes the architecture.
[docs/](docs/) holds design notes on the plugin system and individual
algorithms. [NOTES.md](NOTES.md) records design decisions and measurements.

## Tests

The tests are for developers working from a checkout. Install the test extras
and run them in parallel:

    pip install -e .[test]
    python -m pytest tests -q -n auto            # about a minute
    python -m pytest tests -q -n auto --slow     # everything, a few minutes

They run on simulated data and need no files of their own. A few check that
a 3D bead calibration made by SMAP (a `..._3dcal.mat` file) is read
correctly; they need such a file, so they are skipped unless
`SMAPPY_TEST_CAL` points at one.

## License

SMAPpy is released under the BSD 3-Clause license; see [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
