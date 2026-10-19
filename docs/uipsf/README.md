# uiPSF v2

A PyTorch re-implementation of [uiPSF](https://github.com/ries-lab/uiPSF) (Liu et al., *Nat. Methods* 2024):
it learns a physical model of a microscope's point spread function -- the pupil, as Zernike polynomials, with a
vectorial high-NA forward model -- from bead stacks or from the single-molecule data themselves.

Status: in development. The forward model is done and validated against uiPSF v1 and the TU Delft vectorial
model; the solver comes next. See `docs/uipsf/plan.md`.

- Runs on NVIDIA GPUs, Apple GPUs or the CPU, chosen automatically.
- Used from [SMAPpy](https://github.com/ries-lab/SMAPpy), whose calibration window will replace its bead calibration.

```
pip install -e ".[dev]"
python -m pytest -q -n auto
```

Documents: `docs/uipsf/plan.md` (plan), `docs/uipsf/conventions.md` (axes, signs, units), `docs/uipsf/architecture.md` (what is
built), `docs/uipsf/data.md` (reference data). BSD-3-Clause.
