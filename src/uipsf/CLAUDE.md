# Working on uipsf

`uipsf` (`src/uipsf/`): uiPSF v2, the PyTorch re-implementation of uiPSF, the PSF-learning tool.  It
lives in SMAPpy's repository and is installed with SMAPpy; SMAPpy's own CLAUDE.md applies on top of this
file, which adds what is particular to uipsf.

**Read first:**
- `docs/uipsf/plan.md`: the plan, its decisions and the milestones.
- `docs/uipsf/conventions.md`: axes, signs, units. SMAPpy's conventions are uipsf's.
- `docs/uipsf/architecture.md`: what is built.

## Where things are

| what | where |
|---|---|
| the package | `src/uipsf/` |
| tests, oracles, fixtures | `tests/uipsf_tests/` |
| benchmarks and their results | `benchmarks/uipsf/` |
| scripts (Zenodo data, the v1 oracles) | `scripts/uipsf/` |
| docs | `docs/uipsf/` |

## Tests

    pip install -e .[test,uipsf]        # torch is an extra: without it these tests are not collected
    python -m pytest tests/uipsf_tests -q -n auto            # fast tier, after every change (~10 s)
    python -m pytest tests/uipsf_tests -q -n auto --slow     # adds benchmarks and end-to-end fits
    python -m pytest tests/uipsf_tests/test_gpu.py --gpu     # by hand, on a Mac and on a CUDA machine
    ruff check src/uipsf tests/uipsf_tests benchmarks/uipsf scripts/uipsf

SMAPpy's whole suite (`python -m pytest tests -q -n auto`) includes these.

- Name a test as the sentence it proves. Test the numbers: simulate with known parameters and recover them.
- The oracles:
  - `tests/uipsf_tests/fixtures/oracle_v1.npz` comes from uiPSF v1 (TensorFlow) via `scripts/uipsf/make_oracles_v1.py`. Regenerate it only in a TF environment, against uiPSF `test_roi` b367f00.
  - `tests/uipsf_tests/oracles/vecfit.py` is the TU Delft model, ported in its own conventions.
  - A change to the optics must keep `tests/uipsf_tests/test_oracles.py` passing. If it departs from v1 on purpose, say so in `docs/uipsf/conventions.md` with a test that pins the departure, as the even-region blur does.
- CI (`.github/workflows/uipsf.yml`) runs the fast tier on Linux, macOS and Windows (CPU) when these paths change.

## Rules

- **uipsf may use smappy, never its GUI.** Loading, the camera database, bead detection and the fitters
  are SMAPpy's, and uipsf calls them rather than keeping its own. It never imports Qt (`PySide6`,
  `pyqtgraph`, `smappy.gui`): the calibration window is SMAPpy's. That keeps a later standalone uipsf
  possible on a headless SMAPpy (`docs/uipsf/plan.md` §7.4).
- **Torch stays inside uipsf.** smappy imports uipsf only where a uipsf model is used, inside functions,
  so that discovery and the rest of the program never load torch.
- **Functional core.**
  - Components hold configuration and precomputed tensors.
  - Parameters arrive as a `dict` of physical values.
  - No `nn.Parameter`, no hidden state.
  - Every parameter is declared as a `ParamSpec`.
- **No variable scaling, no loss weights.** Bounds by transforms, priors in physical units.
- **Lengths in nm, arrays `[..., Y, X]`.** Positions are `(x, y, z)`, with z the emitter's height above focus. See `docs/uipsf/conventions.md`; if your change touches a convention, that file changes in the same commit.
- **Platform-neutral.**
  - Only PyTorch operations that CUDA, MPS and the CPU all have.
  - No float64 on the device (MPS has none); small solves go to the CPU through `device.to_cpu64`.
- **Spline fitting is SMAPpy's.** uipsf never fits or writes cubic splines; it writes its model file, and SMAPpy samples it.
- **No data in git.** `scripts/uipsf/fetch_zenodo.py` puts the paper's example data under `$UIPSF_DATA`; see `docs/uipsf/data.md`.

## House style

- The code is written to be read: small modules.
- Docstrings say *why*: the trade-off, the failure avoided, the number measured.
- Comments mark decisions someone might otherwise undo.
- Cite the paper's supplementary notes or the review (`ries-lab/uiPSF`, `docs/review_2026/`) where an equation or a choice comes from.
- Lines up to 120 characters, ruff's rules in `src/uipsf/ruff.toml`; SMAPpy's own files keep theirs.

## Keeping the docs true

- When a milestone lands, update `docs/uipsf/architecture.md`, and `docs/uipsf/plan.md` where the code changed the plan.
- If a fact here disagrees with the code, trust the code and fix the line.
