# Making smappy callable from microclaw

The pipeline itself is in good shape; what is missing is the packaging around it
and two adapters.  Steps are ordered so each one is independently testable and
useful on its own.

## 0. Compatibility baseline

microclaw pins `requires-python >=3.10`, `numpy>=1.26`, `scipy>=1.12`,
`tifffile>=2024.1.1`, `pyyaml>=6`, `ndstorage>=0.1`, `Pillow>=10`, and h5py only
under its `ilastik` extra.  smappy must install into that environment without
moving any of them:

- keep `requires-python >= 3.9` (a superset; it installs fine on 3.10+),
- keep floors *below* microclaw's and put **no upper caps** on numpy, scipy or
  tifffile,
- no numpy build dependency is needed: the extensions include only
  `pybind11/numpy.h`, which carries its own ABI declarations and binds at run
  time, so one wheel serves numpy 1.x and 2.x — worth *testing* in CI (the
  cibuildwheel `test-command` imports all three modules) rather than assuming,
- keep matplotlib in the `viewer` extra: microclaw does not depend on it and
  must not be made to,
- do not add `ndstorage` as a smappy dependency (step 3 makes it unnecessary),
  but stay compatible with it being present.

## 1. Packaging (blocking everything else) — **done**

1. `pyproject.toml`: add `"pybind11>=2.10"` to `build-system.requires`.  Today
   it lists only setuptools while `setup.py` imports `pybind11.setup_helpers`,
   so `pip install .` fails in an isolated build environment.  *Check:*
   `pip install .` into a clean venv, then `python -c "import smappy._fit3d"`.
2. Add the undeclared runtime dependency `pyyaml>=6` — `CameraMetadata.from_yaml`
   imports it and nothing declares it.  It works in microclaw only by accident.
3. Add `[project.scripts]`: `smappy-fit`, `smappy-view`, `smappy-drift`.  Move
   the bodies of `scripts/fit_dataset.py`, `view_locs.py`, `drift_correct.py`
   into `smappy/cli/` so they stop doing `sys.path.insert`; leave the `check_*`
   scripts where they are, they are development tools.
4. Build wheels for macOS arm64, Linux x86_64 and Windows (cibuildwheel,
   `pp*` skipped), so microclaw can declare `smappy` as an optional dependency
   rather than asking a user to compile.  *Check:* wheel installs and fits a
   test dataset on a machine with no compiler.  **Still open** -- see
   "Publishing wheels" below.  The `[tool.cibuildwheel]` configuration is in
   `pyproject.toml` and `setup.py` now picks the optimization flag per compiler
   (`/O2` on MSVC, `-O3` elsewhere), which was the one thing in the build that
   could not have worked on Windows.

## 2. A package-level API — **done**

`src/smappy/__init__.py` is empty, so a caller reassembles the fifteen lines of
`cli/fit.py` by hand.  Export a small facade:

```python
from smappy import fit, view, CameraMetadata

locs = fit(data, out="OUT.h5", camera=dict(conversion=0.49, offset=100.0,
                                           pixelsize_um=0.106),
           calibration="..._3dcal.mat", units="nm")
```

`data` is a path, an `ImageSource`, or an iterable of `(first_frame, block)`;
`camera` is a dict, a `CameraMetadata` or a YAML path; `calibration=None` means
a Gaussian fit.  Returns the table and writes the HDF5 when `out` is given.
Keep `fit_stack`/`LocalizationEngine` exactly as they are — the facade only
assembles them.  Also re-export `load_localizations`, `show`, `render_locs`,
`LiveFit`, `correct_drift`.

*Check:* the whole README "Use" section becomes three lines, and the equivalent
call in `tests/` produces the same table as the long form.  **Done**, with two
additions that came out of the writing: `collect=False`, which streams to `out`
without holding the table (a 46 k-frame acquisition would not fit in memory),
and the stats of the run stored in the file beside the provenance.

## 3. NDTiff reading — **done**

microclaw's `run_timelapse` writes NDTiff (pycro-manager / ndstorage); smappy's
`open_stack` reads only Micro-Manager OME-TIFF series.  Today that forces an
`export_dataset_as_tiff` pass — a full copy of the raw data before fitting can
start, which is unacceptable for a live fit and wasteful offline.

**Port the MATLAB reader already in this repo**, which is standalone and needs
no ndstorage:

- `shared/imageloaders/readNDTiffIndex.m` → `smappy/io/ndtiff.py`: parse
  `NDTiff.index`, a flat little-endian table of `axesLength, axes(JSON),
  filenameLength, filename, pixelOffset, width, height, pixelType,
  pixelCompression, metadataOffset, metadataLength, metadataCompression`,
  stopping at the first `axesLength == 0` (zero padding).  Carry over the parts
  that were learned the hard way: refuse `compression != 0`; derive bytes per
  pixel from `metadataOffset - pixelOffset` rather than trusting `pixelType`;
  sort by the `time` axis; and **drop trailing index entries that point past the
  end of their TIFF**, which is how a truncated acquisition (or the 4 GB 32-bit
  offset limit) presents itself.
- `shared/imageloaders/imageloaderNDTiff.m` → the same module: read pixels
  directly at `pixelOffset` with `np.memmap`, the per-image JSON at
  `metadataOffset`, and the summary JSON from the TIFF header.

Expose it as an `ImageSource` so nothing downstream changes:
`open_stack(path)` detects `NDTiff.index` in a directory and returns an
NDTiff-backed source with the same `frames()`, `watch()`, `shape`, `n_frames`
and `mm_metadata`.  Flatten the per-image JSON to the same flat key space
Micro-Manager uses (`parsejsonflat` in the MATLAB loader), so
`metadata_from_stack` keeps working unchanged.

The index grows during an acquisition, so `watch()` is a re-read from the last
record — the same trick the MATLAB loader uses for online analysis, and simpler
than watching a TIFF for new pages.  This makes the live fit work directly
against what microclaw is writing, with no export step.

*Check:* **done** -- pixels match tifffile byte for byte on a real 100-frame
dataset, a fit through `smappy.fit` and the same fit through `LiveFit`'s watch
path both give 19093 localizations, and `tests/test_ndtiff.py` builds datasets by
hand for the padding, truncation, pixel-size, compression and growing cases.

## 4. Feeding frames from a hook — **done**

`live_view` accepts a path or an object with `.watch()`/`.shape`, so frames
arriving from a pycro-manager `analyze_frame` hook cannot reach it.  Add a
`QueueSource` in `smappy/io/queue_source.py`: an `ImageSource`-shaped object
with `push(frame, index)`, `close()`, and a `watch()` that yields blocks as they
arrive.  `LocalizationEngine.push` already has the right shape; this is the
missing plumbing between it and the live viewer.

*Check:* **done** -- `tests/test_queue_source.py` drives `LiveFit` and the whole
`live_view` window from pushed frames, and covers the block boundaries: a gap in
the numbering ends a block rather than mislabelling it, a slow producer is not
held back waiting for a full one, and a bounded queue makes the producer wait.

## 5. Process boundary for the viewer — **done**

`show` and `live_view` call `plt.show(block=True)` and want the main thread, so
neither can be called inside microclaw's server process.

- add `show(path_or_locs)` — accept an HDF5 path, load it, so the viewer is one
  call from a file;
- `smappy.save_image(locs, path, pixelsize=...)` wraps `FieldOfView.around` +
  `render_locs` + `DisplaySettings.apply` + Pillow (already a microclaw
  dependency), for an image in the session record with no window at all.  It
  takes a table or a saved file, and the same settings and filter the viewer
  takes, so what it writes is what the viewer would show;
- document `LiveFit` as *the* headless handle, and keep the windowed viewer a
  separate process launched via the `smappy-view` entry point from step 1.

## 6. The microclaw side

Only after 1–3 land:

- add `smappy-smlm` to microclaw's `[project.optional-dependencies]` as an
  `analysis` extra, never a hard dependency;
- copy `integration/microclaw/skills/smappy/SKILL.md` to
  `microclaw/skills/smappy/SKILL.md` (its frontmatter already satisfies
  `microclaw/skills.py`), and drop its NDTiff-export workaround once step 3 is
  in;
- cross-reference it from `skills/smlm/SKILL.md`, whose post-processing table
  currently sends the user to MATLAB SMAP;
- keep the fit itself out of the acquisition hook path: microclaw's rule is that
  analysis lives in hooks, but a hook that fits is a hook that must ship a C++
  extension.  The hook pushes frames into a `QueueSource`; the fit runs beside
  the acquisition and the exported standalone pycro-manager script keeps working
  without smappy installed.

## Order and effort

| step | depends on | rough size |
|---|---|---|
| 1 packaging | — | small, unblocks everything |
| 2 facade | 1 | small |
| 3 NDTiff | 1 | medium; the MATLAB port is the bulk |
| 4 QueueSource | 2 | small |
| 5 viewer boundary | 2 | small |
| 6 microclaw wiring | 1–3 | small |


## Publishing wheels — **the repository is split; publishing is not set up**

The blocker for step 1's last item is not the build, it is where the wheels come
from and where they go.  What has to be true: `pip install microclaw[analysis]`
downloads a binary on Windows, macOS and Linux, and nobody needs a compiler.

**Move smappy to its own repository.**  **Done**: `git filter-repo` over a
clone of SMAP, filtering `pysmap` and `SMAPpy` to the root, so all eight commits
of the port kept their history and their messages.  The repository has no remote
and nothing has been pushed.  SMAP still holds its own copy under `SMAPpy/`;
removing it, or carrying it back as a submodule, is a separate decision.

It makes every following step the default rather than a special case:

- cibuildwheel and the `pypa/cibuildwheel` action assume the package is at the
  repository root; from a subdirectory every job needs `package-dir: SMAPpy` and
  a `working-directory`, and `pip install git+https://.../SMAP#subdirectory=SMAPpy`
  is what anyone installing from source has to type;
- a version tag means one thing.  `v0.2.0` in a shared repository is ambiguous
  between MATLAB SMAP and the Python package; smappy needs `smappy-v0.2.0` and
  a tag filter in the workflow, and PyPI's trusted publishing is configured per
  repository and workflow file, so the SMAP repo would be granted the right to
  publish smappy;
- CI cost and noise: every MATLAB commit would otherwise trigger (or have to be
  filtered out of) a fifteen-job wheel matrix, and each job checks out the whole
  SMAP repository to build a package that is a few hundred kilobytes of source.

`git subtree split -P SMAPpy` (or `git filter-repo --subdirectory-filter SMAPpy`)
keeps the history.  SMAP can carry it back as a submodule if the sources should
stay visible from the MATLAB side.

**In the new repository:**

1. **Done**: `.github/workflows/wheels.yml` builds on `ubuntu-latest`,
   `windows-latest`, `macos-14` (arm64) and `macos-13` (x86_64), each for the
   architecture it runs on, plus an `sdist` job that installs the tarball and
   runs the smoke test.  `ci/smoke.py` fits a frame with an emitter at a known
   position rather than merely importing the modules.
2. **Done**: the sdist was missing `csrc/*.hpp` -- setuptools ships the `.cpp`
   files it compiles but not the headers they include -- so it could not have
   built anywhere.  `MANIFEST.in` fixes it and the sdist job is what would have
   caught it.  Verified locally: the tarball builds from scratch in a clean venv
   and passes the smoke test.
3. **Done**: the distribution is named `smappy-smlm`.  `smappy` on PyPI is an
   unrelated package -- a Smappee energy-monitor wrapper, last released in 2018
   -- so it could not have been published under that name, and `pip install
   smappy` would have installed something else entirely.  The import name is
   unchanged.  `provenance()` had to follow: `importlib.metadata.version` takes
   the *distribution* name, so asking it for "smappy" would have stamped every
   output file with the fallback version, or with a stranger's.
4. **Done**: BSD 3-Clause, matching microclaw, with `LICENSE`,
   `THIRD_PARTY_NOTICES.md` (COMET is MIT and vendored but not distributed) and
   the PyPI metadata -- readme, author, URLs, classifiers -- that an otherwise
   blank project page needs.  The version is `0.1.0`, not `0.1.0dev`: a dev
   version is a pre-release, which `pip install` skips without `--pre` and which
   `>=0.1` does not match.
5. **Open**, and all of it needs the repository owner: create the GitHub
   repository, push, register a PyPI trusted publisher for `smappy-smlm` naming
   that repository and `wheels.yml`, create the `pypi` environment, then tag
   `v0.1.0`.  The workflow runs on pull requests and on demand, so the matrix can
   be seen green before any tag.
6. Only then does microclaw's `analysis` extra resolve by name.  Until it does,
   `pip install microclaw[analysis]` fails with "No matching distribution found
   for smappy-smlm" -- clear, but a dead end, which is why the extra carries a
   comment naming the checkout to install from.

**What Windows may still cost.**  The `-O3` problem is fixed (`setup.py` now
chooses `/O2` under MSVC).  The sources are portable in the ways that
usually bite -- `py::ssize_t` rather than POSIX `ssize_t`, no VLAs, no
`unistd.h`, threads through `<thread>` -- and the flag problem is fixed, so the
expected outcome is that it simply builds.  What CI will tell you and a Mac
cannot: MSVC is stricter about two-phase name lookup in templates and about
`M_PI` (which needs `_USE_MATH_DEFINES`; the sources do not use it today).
Budget one round of fixes, not a port.
