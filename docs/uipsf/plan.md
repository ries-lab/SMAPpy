# uiPSF v2: plan for a PyTorch re-implementation

October 2026. Written as a proposal before any code; the code has since followed it (status in `docs/uipsf/roadmap.md`). Revision 4: uiPSF v2 is built around SMAPpy; the decisions from the second and third reviews are in. Revision 5: uipsf moves into SMAPpy's repository (§7.4). Revision 6: the *PSF calibration* plugin, the v1 bridge removed, the microscope database (§7.5). Revision 7: in situ (M5) designed, with a review of v1's (§9, M5).

**Status: see `docs/uipsf/roadmap.md`**, the short page of what is done, what is next and the ideas beyond; this plan keeps the reasoning, the decisions and the milestones' detail.  What exists is in `docs/uipsf/architecture.md`, the conventions as implemented in `docs/uipsf/conventions.md`.

**Basis.**
- `docs/review_2026/REIMPLEMENTATION_BRIEF.md` and `uiPSF_review.html` on `main` of ries-lab/uiPSF (the "brief" and the "review").
- SMAPpy's bead calibration, 3D fitters, calibration format and existing uiPSF bridge, read in ries-lab/SMAPpy at `main` 8aec4d4.
- The Zenodo example data of the uiPSF paper, inspected for this plan (`docs/uipsf/data.md`).

Section numbers like "review §B" refer to the review.

**What changed since revision 1.**

| Topic | Revision 1 | Now |
|---|---|---|
| Reference program | SMAP | SMAPpy. Its conventions are v2's (§4). |
| v1 .h5 output | A compatible writer | Dropped. v1 files are only read, as benchmark references. |
| Output | Native file + v1 file | uipsf writes one model file with every Zernike and setting. SMAPpy samples it, normalises as its bead calibration does and makes the spline (§6). |
| Spline fitting | — | Never in uipsf. Tests of the spline use SMAPpy's fitter (§6, §9). |
| Field dependence, FD-DeepLoc | Milestone M4; FD-DeepLoc export | Later, as a whole. The architecture keeps the slot (§3); FD-DeepLoc dropped. |
| Optics formulation | uiPSF's, implicitly | Decided in §5.2 after comparing uiPSF, vecfit and the EPFL psf-generator. |
| Devices | CUDA, MPS, CPU | The same, now spelled out per platform: Mac, Windows, Linux (§5.1). |

---

## 1. Goals and principles

**Goals.**
1. The uiPSF physics, validated to float32 precision against TF uiPSF and the vecfit port.
2. A fit that needs no hand-set weights, because the solver is scale-invariant (brief decision 4).
3. One architecture for bead stacks, multi-channel, field dependence and in-situ data, which can grow to 4Pi, Jones pupils and excitation PSFs without copying code.
4. Honest quality metrics after every fit: residual atlas, held-out deviance and z-bias, CRLBs.
5. A drop-in replacement for SMAPpy's bead calibration: same inputs, a calibration the 3D fitters read, plus everything a physical model adds.
6. A phase 2 of optimisation: speed first, then physics where the residual atlas asks for it.
7. The same results and reasonable speed on Mac, Windows and Linux, using the GPU wherever that is easy (§5.1).

**Principles.** They decide most of the layout below.

- **Functional core.** Physics and model components are plain Python classes that hold configuration and precomputed tensors. Parameters are passed explicitly as a `dict[str, Tensor]`: no `nn.Parameter`s and no hidden state. That makes `torch.func.jvp`, `vmap` and the Schur-complement solver straightforward, and each component testable alone.
- **One forward model, many uses.** Learning, simulation, localisation (held-out z-bias, in-situ) and CRLBs all call the same `ForwardModel`. Learning and localisation share one solver.
- **Parameters declare themselves.** Each component lists `ParamSpec`s: scope (shared or per emitter), transform (bounds), optional prior and a fixed flag. The solver reads the block structure from them. A new physics term is a new component with its specs and nothing else changes.
- **Composition, not subclasses.** v1 has one class per combination (`PSFZernikeBased`, `…_FD`, `…_vector_smlm`, `PSFMultiChannel_smlm`…). v2 composes a few orthogonal pieces: pupil × coefficient field × geometry × intensity × background × channels × noise.
- **Physical units, SMAPpy's conventions.** Lengths in nm, photons as physical totals, phase in radians, images `[..., Y, X]`. Priors use the same units, and there is no variable scaling anywhere.
- **uipsf uses SMAPpy, never its GUI** (revision 5, §7.4). It calls SMAPpy's loading, camera database, bead detection and fitters rather than keeping its own, and never imports Qt. Until revision 4 uipsf did not depend on SMAPpy, and the boundary was arrays in, model and sampled PSFs out; for the model file it still is (§6).
- **Readable over clever.** Small modules (~100–300 lines). Docstrings say why, equations cite the paper's supplementary notes or the review. Eager by default; speed work comes in phase 2 behind the same APIs.

---

## 2. Repository layout

The layout in ries-lab/uiPSF2. In SMAPpy's repository the same tree is spread over `src/uipsf/`, `tests/uipsf_tests/uipsf_tests/`, `benchmarks/uipsf/uipsf/`, `scripts/uipsf/uipsf/` and `docs/uipsf/uipsf/` (§7.4).

```
uiPSF2/
├── pyproject.toml            # pip install -e .[test]; torch, numpy, scipy, h5py, tifffile, omegaconf
├── README.md, CLAUDE.md
├── docs/
│   ├── plan.md               # this file
│   ├── data.md               # Zenodo example data: contents, fetching, use
│   ├── conventions.md        # (M1) SMAPpy's axes, z sign, units, pixel origin; Zernike sign and indexing
│   └── architecture.md       # (M2) §3–5 as they end up in code
├── src/uipsf/
│   ├── __init__.py           # public API (§5.9)
│   ├── device.py             # CUDA > MPS > CPU, dtypes, float64-on-CPU helper
│   ├── optics/               # pure physics: no parameters, no fitting
│   │   ├── system.py         #   Optics (NA, λ, RIs, pixel size…), automatic pupil sampling
│   │   ├── grid.py           #   PupilGrid: k-space coordinates, aperture, kz, Fresnel, apodisation
│   │   ├── zernike.py        #   Zernike basis, Noll indexing, high-NA modes orthogonal to kz
│   │   ├── vectorial.py      #   dipole field basis (6 pupil fields), polarisation analysers
│   │   ├── mdft.py           #   matrix DFT (two complex matmuls)
│   │   ├── depth.py          #   index mismatch: reduced and full Gibson–Lanni phase
│   │   └── kernels.py        #   bead sphere OTF, Gaussian blur OTF, pixel integration
│   ├── params.py             # ParamSpec, transforms, priors, Params (packing, block layout)
│   ├── model/                # components with parameters, composed into a ForwardModel
│   │   ├── pupil.py          #   ZernikePupil, PixelPupil (unit-energy normalised)
│   │   ├── field.py          #   coefficient fields: Constant, NodalPolynomial, BSplineResidual; FocalSurface
│   │   ├── psf.py            #   VectorPSF / ScalarPSF: pupil + positions -> normalised intensity
│   │   ├── voxel.py          #   VoxelPSF baseline (shared parameters only)
│   │   ├── geometry.py       #   BeadStackGeometry, InSituGeometry: emitter + stage -> xyz per observation
│   │   ├── intensity.py      #   photons, bleaching, per-slice factors
│   │   ├── background.py     #   constant, planar
│   │   ├── channels.py       #   Channel (optics, pupil, transform), projective channel transforms
│   │   └── forward.py        #   ForwardModel: composition -> expected photons mu
│   ├── noise.py              # Camera (ADU -> photons, read noise, EM), Poisson, ShiftedPoisson, EMCCD
│   ├── solve/
│   │   ├── problem.py        #   Problem: model + noise + data + params; objective, normal equations
│   │   ├── lm.py             #   block Levenberg–Marquardt with Schur complement
│   │   ├── lbfgs.py          #   L-BFGS with Fisher-diagonal preconditioning (fallback, comparison)
│   │   └── result.py         #   FitResult: params, history, stop reason, covariance, CRLB
│   ├── data/
│   │   ├── io.py             #   thin standalone readers (tif / ome-tif); SMAPpy passes arrays instead
│   │   ├── rois.py           #   bead and emitter detection, ROI extraction, masks
│   │   ├── pairing.py        #   multi-channel pairing (re-implemented; refines a given transform)
│   │   └── dataset.py        #   Batch / Dataset: ROIs + metadata, chunking, held-out split
│   ├── workflows/            # the recipes callers use; thin
│   │   ├── beads.py          #   learn_beads(stacks, settings) -> Calibration
│   │   ├── insitu.py         #   learn_insitu(frames, settings)
│   │   └── localize.py       #   localize(model, rois): per-emitter MLE on the same solver
│   ├── diagnostics/
│   │   ├── atlas.py          #   residual atlas: deviance per group
│   │   ├── heldout.py        #   held-out deviance and z-bias
│   │   └── figures.py        #   matplotlib figures as functions of results (SMAPpy shows them in its tabs)
│   ├── export/
│   │   ├── model_file.py     #   the model file: every Zernike, setting, parameter, diagnostics
│   │   └── zernike.py        #   conversions to v1 uiPSF / vecfit Zernike conventions
│   ├── simulate.py           # synthetic data from the ForwardModel, plus the mismatch effects of review §B
│   ├── movie.py              # blinking movies imaged with a model: in situ's test data (M5)
│   ├── config/
│   │   ├── schema.py         #   settings dataclasses (also what SMAPpy builds its form from)
│   │   ├── build.py          #   settings -> ForwardModel, Problem
│   │   └── presets/*.yaml    #   beads, beads_two_channel, beads_field, insitu, …
│   └── cli.py                # `uipsf learn settings.yaml stacks/…`, `uipsf report model.h5`
├── tests/
│   ├── fixtures/             # small committed oracle outputs (npz), a few MB in total
│   ├── test_*.py
│   └── conftest.py           # device fixtures; `--slow` and `--gpu` tiers
├── benchmarks/
│   ├── synthetic/            # scenarios (YAML) + runner -> JSON lines; the review's tests as a superset
│   ├── real/manifest.yaml    # Zenodo + lab datasets with expected settings
│   └── speed.py              # forward/backward and per-iteration timing per device
├── scripts/
│   ├── fetch_zenodo.py       # selective download of the paper's example data (works; docs/data.md)
│   └── make_oracles_v1.py    # runs TF uiPSF in its own environment, writes tests/fixtures/*.npz
```

**Why this split.**
- `optics/` has no parameters and no fitting, and is where the oracle tests bite.
- `model/` holds everything with parameters.
- `solve/` knows nothing about optics. It sees a `Problem` with a block structure, so it can be tested on toy problems.
- `workflows/` are short functions over the rest. SMAPpy's window, the CLI and a notebook all call them.

**Dependency direction** (no cycles):
- `optics` ← `model` ← `solve` ← `workflows`. `noise` and `params` are leaves.
- `data`, `diagnostics` and `export` use `model` and `solve` but are never imported by them.
- `uipsf` imports SMAPpy's modules but never its GUI (§7.4); before the move it never imported SMAPpy.

---

## 3. The model: what composes with what

The expected photon count in pixel *j* of observation *k* (a slice or a frame) of emitter *e* in channel *c* is

```
mu[e,c,k,j] = N[e,c,k] · I_c( j ; xyz[e,c,k], field[e] ) + b[e,c,k,j]
```

- `I_c` is the PSF. Its integral over the whole image plane is the collected fraction of one photon (pupil energy, brief decision 5), so `N` is a physical total.
- `xyz` comes from a **geometry** component, `N` from an **intensity** component and `b` from a **background** component.
- Pupil coefficients may depend on the emitter's field position through a **coefficient field**.

Every data type uses the tensor layout `[E, C, K, Y, X]`:
- Bead stacks: K = number of slices.
- In-situ: K = 1 (or a few linked frames).
- Later, 4Pi adds a phase axis inside the channels, and excitation scans reuse K for scan positions.

That shared layout is what lets beads, in-situ data and multi-channel data share the code.

| Component | First implementations | Parameters (scope) | Later |
|---|---|---|---|
| **Pupil** | `ZernikePupil` (phase and magnitude Zernikes; no piston, tilt or defocus in the phase; magnitude normalised), `PixelPupil` | coefficients (shared, or per emitter through a field) | `JonesPupil` (2×2), polychromatic sum |
| **Coefficient field** | `Constant` | none | `NodalPolynomial` (with a fitted axis centre), `BSplineResidual` (second-difference prior), `Sum` of these |
| **PSF** | `VectorPSF` (6 dipole fields, Fresnel, apodisation, optional Ex/Ey analyser), `ScalarPSF`, `VoxelPSF` (baseline) | blur σ (shared); bead radius (fixed by default) | depth-integrated bead, excitation PSF (Richards–Wolf), 4Pi |
| **Geometry** | `BeadStackGeometry`: `xyz = (x_e+δx_k, y_e+δy_k, z_e + s·z_k + δz_k [+ z_f(field)])`; `InSituGeometry`: depth, stage position, `FocalSurface` | x, y, z (emitter); δx, δy, δz per slice and z scale s (shared, optional, with priors) | scan geometry for confocal/MINFLUX |
| **Intensity** | `Bleaching` (default: N_e · exp or low-order polynomial in k), `PerSliceFactor` (shared over beads: laser fluctuations), `PerObservation` (a free photon count per bead and slice: speckled or multi-mode excitation) | per emitter / shared | saturation I/(1+I/I_sat) |
| **Background** | `Constant`, `Planar` (3 per emitter) | per emitter | per-slice background |
| **Channels** | `Channel` = optics + pupil + PSF + projective transform from the main channel + photon share + defocus offset | transform residuals, shares, offsets (shared) | ISM detector array, shared excitation |
| **Noise** | `Poisson`, `ShiftedPoisson` (scalar or per-pixel σ²/g²), `EMCCD` (excess factor 2) | none (the camera is an input) | exact EMCCD |

**Photons per slice.** The default is a smooth bleaching curve per bead. Two options cover real excitation:
- `PerSliceFactor`: one factor per slice shared by all beads in the frame. It absorbs an unstable laser at the cost of K parameters and keeps the z-dependent collection efficiency, which carries information about the pupil.
- `PerObservation`: a free count per bead and slice (v1's `var_photon`). Needed when the brightness changes per bead and frame, as with speckled multi-mode excitation. It costs E·K parameters and throws away that z information, but the Schur solver handles it: the per-bead blocks grow from ~6 to ~K + 5 parameters, still small.

The residual atlas shows which one a dataset needs: a per-slice pattern shared by all beads asks for `PerSliceFactor`, a per-bead per-slice photon scatter beyond the noise asks for `PerObservation`.

**Masks.** Saturated pixels and pixels of neighbouring emitters are carried as a boolean mask on the batch. The likelihood multiplies by it, keeping shapes static. This replaces v1's silent dropping of non-finite pixels with exclusion that is explicit and reported.

**Field dependence comes later, as a whole.** Phase 1 implements only the `Constant` field. The slot stays in the architecture: pupil coefficients pass through a coefficient field, and every batch carries the emitters' chip positions. Adding `NodalPolynomial` and `BSplineResidual` later changes no other component. The same goes for SMAPpy's side (§7.3).

---

## 4. Conventions: SMAPpy's

SMAPpy is the program uipsf will be used with, so its conventions become uipsf's. A model then needs no conversion at the boundary, and a reviewer can check both programs against one document. `docs/uipsf/conventions.md` (M1, task 3) states them with the SMAPpy lines that define them:

- **Image axes.** Arrays are `[..., Y, X]` (row, column). Positions are written `(x, y, z)`, with x the column. In SMAPpy: `psf.py`, and `csrc/models.hpp`: "`ix` is the ROI column and `iy` the row".
- **Pixel origin.** Zero-based pixel centres: pixel *i* covers [i − ½, i + ½]. A ROI of odd size has its centre on a pixel. Spline knots sit on pixel centres.
- **Units.** Lengths in nm. Lateral positions inside ROIs are given in pixels where SMAPpy's fitter tables use pixels, converted with the pixel size (x, y separately: the Zenodo M2 data have 127 × 116 nm pixels).
- **z sign and origin.**
  - The stack is ordered by increasing objective position.
  - Emitter z is the negative of the objective position relative to focus: "a bead is below the focus by as much as the objective was raised" (SMAPpy Spline 3D page; `calibrate/input.py`).
  - z = 0 is the nominal focus of the bead stack (bead model).
  - For in-situ data, z is the emitter's height above the coverslip, with the stage position and index mismatch handled by the depth model.

  The in-situ sign is exactly where SMAPpy's uiPSF bridge had to reverse planes (`uipsf/convert.py`), so v2 defines it once, in `InSituGeometry`, with a test.
- **Field coordinates.** Camera-chip pixel coordinates (zero-based centres), using the camera ROI from the metadata as SMAPpy's dual calibration does. A field-dependent model then transfers between acquisitions with different camera ROIs.
- **Channel transform.** A 3×3 projective map, secondary → main, in chip pixel coordinates, the same object as SMAPpy's `DualColorCalibration.transformation`. "Main" and the split geometry are SMAPpy's.
- **Camera.**
  - Photons are `(ADU − offset) · conversion`, divided by the EM gain with EM.
  - With EM, data and model are halved (excess noise 2).
  - Read noise is one variance per chip, with per-pixel maps optional in uipsf. SMAPpy has no maps yet.

  This matches SMAPpy's `CameraMetadata` and `with_readout`, so a fit in uipsf and in SMAPpy see the same likelihood.
- **Photons.** Inside uipsf they are physical totals (pupil energy), so fitted photon counts and CRLBs do not depend on the ROI size. The spline SMAPpy makes from the model is normalised as its bead calibration is ("maximum plane sum = 1", two channels jointly), so Spline 3D treats both calibrations alike (§6).
- **Zernikes** (decided):
  - SMAPpy uses none, so v2 sets this convention itself: Noll indexing, unit-RMS normalisation over the pupil, and a phase sign anchored on what a user sees.
  - The sign is uiPSF v1's, so v1 coefficients carry over unchanged. Anchored on the image: a positive vertical astigmatism coefficient (Noll 6, cos 2φ) stretches the spot along y for emitters above focus (z > 0) and along x below, in SMAPpy's z sign. A test pins it. (Revision 3 said "Noll 5 along x"; Noll 5 is the oblique astigmatism, and keeping v1's sign gives y.)
  - Converters to v1 uiPSF and vecfit are tested against the review's vecfit port (review §D).

---

## 5. Core APIs

These are sketches to fix the boundaries. The code may refine names and signatures; `docs/uipsf/architecture.md` will record the result.

### 5.1 Devices, platforms and precision (`device.py`)

```python
def default_device() -> torch.device            # cuda, then mps, then cpu; UIPSF_DEVICE overrides
def to_cpu64(t: Tensor) -> Tensor               # small linear algebra (Schur system, covariances)
```

**Platforms.** One code path, three devices. PyTorch picks the backend.

| Platform | GPU used | What it takes |
|---|---|---|
| Mac (Apple silicon) | Apple GPU through Metal (MPS) | Nothing: the standard `pip install torch` includes it. Measured 2.5× over the M1 Pro's own CPU for the matrix DFT (review F3). |
| Linux + NVIDIA | CUDA | Nothing beyond the NVIDIA driver: the standard Linux wheel includes CUDA. |
| Windows + NVIDIA | CUDA | The standard Windows wheel is CPU-only. The CUDA build comes from PyTorch's own package index (`pip install torch --index-url https://download.pytorch.org/whl/cu12x`). An install script or a documented line does that; nothing else changes. |
| Any, no GPU | CPU, multi-threaded | Always works. Bead calibrations are expected to take minutes on a laptop CPU with the new solver. |

**Rules that keep the three the same:**
- No float64 on the device: MPS has none. Small solves go to the CPU in float64.
- No custom CUDA or Metal kernels, only PyTorch operations that all three backends have. The gpu test tier checks complex FFT, matmul and autodiff on each.
- Results must agree across devices within float32 tolerance (gpu tier).

**This GPU is uipsf's alone.** SMAPpy's 3D fitters are C++ on the CPU, multi-threaded, without CUDA; nothing in SMAPpy needs PyTorch. uipsf brings PyTorch for its own calibration, and SMAPpy uses it only through uipsf.

- The forward model runs in float32 (complex64) on the device.
- Sums over many pixels (the objective, the normal-equation blocks) are taken per chunk on the device and added in float64 on the CPU. That keeps the summed likelihood precise without float64 on MPS.
- Gradient checks run in float64 on the CPU.

### 5.2 Optics (`optics/`)

**Which formulation: uiPSF, Rieger (vecfit), or the EPFL psf-generator?**

The physics is the same in all three. The review found uiPSF and the vecfit port agree to float32 precision, including supercritical light (review §D). They differ in how the model is parametrised and computed:

| | uiPSF | vecfit (Stallinga, Rieger) | EPFL psf-generator (Liu et al., *J. Microsc.* 2025) |
|---|---|---|---|
| Field to image | CZT (3 padded FFTs) | CZT | CZT ("Cartesian") or 1-D Bessel integrals ("spherical", axisymmetric pupils only) |
| Photon normalisation | Ideal in-focus PSF summed in the ROI | Pupil energy: physical photons | Not about fitting |
| Zernikes | Noll from Z4, defocus included | (2,±2)…(6,0); no piston, tilt or defocus | Any |
| Index mismatch | Reduced form | Nominal and actual immersion RI, working distance (Gibson–Lanni type) | Gibson–Lanni correction factor |
| Pupil magnitude | Learned | Fixed (aplanatic × Fresnel) | Fixed, optional Gaussian envelope |
| Blur, pixels, bead | Gaussian blur, pixel integration, analytic bead kernel | None | None |
| Derivatives | Autodiff | Analytic | Autodiff |

**The EPFL paper does not change the choice.**
- It proves the Fourier ("Cartesian") and Bessel ("spherical") forms are the same integral.
- The Bessel form is faster for large grids and on GPUs, but only for axisymmetric pupils. Any Zernike aberration beyond spherical breaks that, so for learning aberrations the Fourier form is required, as the paper itself concludes.
- Its Fourier form is a CZT. For our sizes (64–128 px pupil, ~30 px ROIs) the matrix DFT is faster: 3–4× on CPU and 1.7× on MPS (review F3).
- Its library (`psf-generator`, MIT licence, PyPI) is a useful third oracle in the tests. It is not a dependency.

**Decision.** A matrix-DFT Fourier model with autodiff, combining:
- **vecfit's parametrisation**: pupil-energy normalisation (physical photons), no piston, tilt or defocus in the phase, Zernike coefficients bounded, and the fuller index-mismatch model (nominal immersion RI, working distance) for in-situ;
- **uiPSF's imaging chain**: learned pupil magnitude, Gaussian blur, pixel integration and the analytic bead kernel, which vecfit lacks;
- **the review's choices**: matrix DFT, eager PyTorch.

The oracle tests hold it to both uiPSF and vecfit (and psf-generator) in the cases where they overlap.

```python
@dataclass(frozen=True)
class Optics:
    NA: float; wavelength_nm: float
    n_imm: float; n_med: float; n_cov: float
    pixel_size_nm: tuple[float, float]           # (x, y): non-square pixels supported
    pupil_size: int | None = None                # None -> from NA, λ, z range and ROI (review §A)
    n_imm_nominal: float | None = None           # full Gibson–Lanni (in-situ, later)

class PupilGrid:                                 # immutable, device tensors
    @classmethod
    def build(cls, optics, roi_shape, z_range_nm, device) -> "PupilGrid"
    kx, ky, kz_imm, kz_med, aperture, apodisation, fresnel_p, fresnel_s

def zernike_modes(n_max, exclude=("piston", "tilt", "defocus")) -> list[int]   # Noll indices
def zernike_basis(grid, modes, high_na=False) -> Tensor   # [K, P, P]
def dipole_fields(grid, analyser=None) -> Tensor # [F, P, P] complex; F = 6
class MatrixDFT:                                 # out = A_y @ field @ A_xᵀ
    def __init__(self, grid, out_shape, upsample=1)
    def __call__(self, field) -> Tensor          # [..., P, P] -> [..., Y·u, X·u]
def bead_otf(...); def blur_otf(...); def integrate_pixels(img, upsample)
def depth_phase(grid, depth_nm, stage_nm, model="reduced" | "gibson_lanni")
```

- Lateral position is a phase ramp on the pupil, so the DFT matrices stay fixed. Axial position is `exp(i·kz·z)`.
- Blur and the bead kernel act on the intensity in Fourier space and pixel integration bins an upsampled image, as in v1. The maths follows v1 exactly so that the oracle tests can pass; changes come in phase 2.

### 5.3 Parameters (`params.py`)

```python
@dataclass(frozen=True)
class ParamSpec:
    name: str                                    # "pupil.phase", "emitter.xyz", "geometry.dz", …
    shape: tuple[int, ...]                       # without the emitter axis
    scope: Literal["shared", "emitter"]
    init: float | Tensor | Callable[[Batch], Tensor]
    transform: Transform = Identity()            # Softplus(), Exp(), Bounded(lo, hi)
    prior: Prior | None = None                   # Gaussian(mean, sigma), Smoothness(D, sigma), physical units
    fixed: bool = False

class Params:
    """Unconstrained values plus specs; packs the shared and per-emitter blocks."""
    def physical(self) -> dict[str, Tensor]
    def shared_vector(self) -> Tensor            # [P_s]
    def emitter_matrix(self) -> Tensor           # [E, P_e]
    def replace(self, shared=None, emitter=None) -> "Params"
    def fix(self, *names); def free(self, *names)
```

**Priors replace v1's loss weights.**
- Gaussian priors are quadratic, so they add exactly to the Gauss–Newton Hessian and gradient.
- The B-spline smoothness is a second-difference Gaussian prior.
- A drift prior reads "σ = 5 nm per slice".
- With a summed likelihood, a prior's strength does not depend on how many beads or pixels there are (review F2).

**Fixed or fitted is a flag**, which settles the stage-tilt question (§10).

### 5.4 Model components and the forward model (`model/`)

```python
class Component(Protocol):
    def specs(self, batch: Batch) -> list[ParamSpec]

class ForwardModel:
    def __init__(self, channels: list[Channel], geometry, intensity, background)
    def specs(self, batch) -> list[ParamSpec]
    def __call__(self, p: dict[str, Tensor], batch: Batch) -> Tensor        # mu, [E, C, K, Y, X]
    # the PSF itself, as built in M1:
    # VectorPSF(grid, pupil, roi, upsample, kind, analyser, bead_radius_nm)(p, xyz, depth_nm, z_step_nm)
    def psf(self, p, xyz_nm, field_xy=None, channel=0, roi=..., upsample=1) -> Tensor  # sampling for export

class Channel:
    optics: Optics; psf: VectorPSF | ScalarPSF | VoxelPSF
    transform: ProjectiveTransform               # secondary -> main, chip pixels (SMAPpy's)
    name: str
```

**Emitter independence is a contract.** An emitter's pixels depend only on its own per-emitter parameters and the shared ones. The solver's block structure relies on this. A test checks it for every component: a tangent on emitter 0 changes no other emitter's pixels.

### 5.5 Noise and camera (`noise.py`)

```python
@dataclass
class Camera:
    offset: float | Tensor; conversion: float | Tensor   # photons per ADU (SMAPpy's name); maps allowed
    em_gain: float | None = None
    read_noise_var: float | Tensor = 1.0                 # photons², SMAPpy's default without EM
    def to_photons(self, adu) -> tuple[Tensor, Tensor]   # (data, variance offset s)

class NoiseModel(Protocol):
    def deviance(self, mu, data, s, mask) -> Tensor      # per pixel, ≈ 1 when the model is right
    def weight(self, mu, s) -> Tensor                    # Fisher weight 1/(mu + s), for Gauss–Newton
```

**Shifted Poisson.** The deviance is `2·[xlogy(d′, d′/μ′) − (d′ − μ′)]` with `d′ = d + s` and `μ′ = μ + s`.
- This is SMAPpy's read-noise treatment written as a likelihood.
- There is no masking of non-finite pixels, no clipping of the data, and no quantile shift.
- Negative `d′` (possible only below −s) is clamped at 0 and counted in the report.
- `μ` stays positive by construction: softplus photons and background.

### 5.6 Solver (`solve/`)

```python
class Problem:
    def __init__(self, model, noise, params: Params, data: Dataset, chunk_size=None)
    def objective(self, params) -> float                   # ½ deviance + priors, float64 sum
    def normal_equations(self, params) -> Normal           # H_ss, H_se[e], H_ee[e], g_s, g_e[e]
    def value_and_grad(self, params) -> (float, Params)    # for L-BFGS

class LevenbergMarquardt:
    def __init__(self, max_iter=100, tol=1e-6, damping=1e-3)
    def solve(self, problem, progress=None, cancel=None) -> FitResult

@dataclass
class FitResult:
    params: Params; history: list[dict]; stop_reason: str; converged: bool
    covariance_shared: Tensor | None              # [P_s, P_s], from the Schur complement
    crlb_emitter: Tensor | None                   # [E, P_e]
```

**Jacobians, never stored whole.** For each chunk of emitters:
- **Per-emitter block:** P_e forward-mode JVPs, with tangent 1 on parameter *i* of every emitter at once. Because emitters are independent, one pass gives column *i* of every emitter's Jacobian.
- **Shared block:** P_s JVPs, batched with `torch.func.vmap`. That is about 100 for 90 Zernikes plus a few shared terms.
- Each chunk adds `JᵀWJ` and `JᵀW·r` into the blocks.

**The step.**
- The Schur complement is `S = H_ss − Σ_e H_se H_ee⁻¹ H_es`, a P_s × P_s float64 system on the CPU.
- Marquardt scaling `λ·diag(H)` makes the step independent of parameter units. This is what removes `self.weight`.
- Steps are accepted only when the objective decreases, and λ is adapted by the gain ratio.
- Bounds come from the transforms, so there are no penalty ramps.

**Stop reasons:** relative decrease, scaled gradient, step size or maximum iterations. Every one is reported.

**Uncertainty:** `S⁻¹` is the shared covariance; the full block inverse gives the per-emitter CRLBs.

**Large shared blocks.** A pixel pupil (2·64²) and the voxel model are too large for a dense S. They use L-BFGS with Fisher-diagonal preconditioning first. A matrix-free LM (CG on S) is a phase-2 item.

**Localisation on the pupil model is not a second fitter.** It is the learning solver with every shared parameter fixed: blocks of 5–6, batched over thousands of emitters. Learning needs it anyway (in-situ emitters, held-out beads), so it costs no code.

uipsf never fits splines. Anything that tests a spline uses SMAPpy's Spline 3D (§6, §9).

### 5.7 Data (`data/`)

*Revised (§7.5): the standalone readers, finder, pairing and `noise.Camera` are removed; smappy's `calibrate.pupil` makes the `Batch`, and a standalone uipsf is SMAPpy without its GUI.*

SMAPpy already reads Micro-Manager and NDTiff stacks, takes z from the metadata, converts to photons, finds beads, and finds and registers the split layout. uipsf does not duplicate that. Its data layer takes arrays and offers thin readers only for standalone use.

```python
@dataclass
class BeadStack:            # what SMAPpy's calibration window hands over, one per stack
    photons: np.ndarray     # [K, Y, X] (or [C, K, Y, X] after SMAPpy's split)
    z_objective_nm: np.ndarray   # [K], increasing
    read_noise_var: float | np.ndarray
    chip_origin: tuple[int, int] # camera ROI offset, for field coordinates
    saturated: np.ndarray | None # bool mask, from the camera's dtype limit
    source: str

def find_beads(stack, roi_size, min_photons=None) -> Table       # default finder; callers may pass centres
def extract_rois(stacks, centres, roi_size, context_px=2) -> Dataset
def pair_channels(tables, transform0) -> (pairs, ProjectiveTransform)   # refines SMAPpy's transform

@dataclass
class Batch:
    data: Tensor; s: Tensor; mask: Tensor        # [E, C, K, Y, X] photons, variance offset, valid pixels
    z_stage_nm: Tensor                           # [K] or [E, K]
    field_xy: Tensor                             # [E, 2] chip pixels
    roi_origin: Tensor                           # [E, C, 2]
    stack_id: Tensor                             # [E]

class Dataset:
    def chunks(self, size, device) -> Iterator[Batch]
    def split(self, heldout=0.2, seed=0, stratify="field") -> tuple["Dataset", "Dataset"]
```

**Bead finding.** uipsf accepts bead centres from the caller. In SMAPpy those come from its own `detect_beads`, so both calibrations start from the same beads and can be compared bead for bead. uipsf's default finder serves standalone use: a threshold in photons above the local background, set from the noise (review F2), with centres and ROIs filtered together (review F1).

**Division of labour with SMAPpy (decided October 2026): SMAPpy provides it all by default; uipsf keeps a simple loader of its own.**
- **In SMAPpy** (the default), the bead-calibration plugin reads the stacks and identifies the camera from its database (`smappy.camera_db`, with every value's source). It converts to photons, finds the beads (and pairs channels), and hands uipsf arrays: photons, z, read-noise variance, bead centres.
- **Standalone** (notebooks, other UIs), `uipsf.data.io.read_stack(path, camera)` reads a Micro-Manager stack. The camera parameters are user input (`noise.Camera`: conversion, offset, read noise, EM gain); uipsf has no camera database. `uipsf.data.rois.find_beads` finds the beads.
- **One filtering convention for both.** `find_beads` follows SMAPpy's `detect_beads` with the same parameters (σ 2 px, 6 robust σ, padding 5 px, both members of a pair closer than 25 px dropped). On the Zenodo `1ch_40nm_bead` stacks both pick the same 84 beads, and a test holds them equal while SMAPpy is installed. Two-channel pairing follows SMAPpy's convention the same way (M4, `data/pairing.py`). The two copies may diverge later; that is accepted, and a divergence is documented where it is made.
- **Brightness** is not filtered by the finder: uipsf's bead rejection judges each bead by its fit.
- **Why camera numbers matter, measured:** for the Zenodo `1ch_40nm_bead` stacks SMAPpy's database finds the M2 microscope's Evolve 512 in its normal-port mode: 4.7 e⁻/ADU, offset 398.6. v1's settings say gain 0.22, almost exactly 1/4.7, so v1's "photons" are about 21× too few. Its likelihood and CRLBs are off by that factor; the PSF shape is not (§9 M2 result).

**Bead rejection: automatic, explained, and overridable.** v1 rejects outlier beads once and refits (`Fitter.relearn`). After its first fit it localises every bead and drops those with a relative z-curve error > 3, a relative MSE > 3 or a relative photon deviation > 1.5, all ratios to the median bead, or with a non-positive photon count. SMAPpy's bead calibration rejects by brightness, by maximum z/xy shift and by shape: a robust residual over correlation, beyond mean + 1.5 sd (SMAP's rule), in up to three rounds. It keeps the rejected beads aligned for diagnostics, each with a reason. v2 does the same in the bead workflow (`workflows/beads.learn_beads`), as one step between the two Zernike stages of `fit_beads`:
1. **Stage 1** (Zernikes up to n = 4) on all beads not excluded by hand. It is cheap, and the beads' positions are already right after it.
2. **Score every bead** from that fit, with no fit of its own (since M7, October 2026):
   - deviance per pixel, which has an absolute scale (≈ 1 when the model fits); a 300 nm stage jump shows in it;
   - photons against the other beads;
   - lateral position against the ROI centre;
   - saturated or masked pixels, and whether its fit converged.

   Thresholds are robust, median + k·1.4826·MAD per metric, so they adapt to data that sit above 1 for every bead (model mismatch, or a wrong camera conversion). k is a setting with SMAPpy's meaning.
3. **Reject**, recording one reason per bead (the metric and its value against the threshold).
4. **Stage 2** (all Zernikes) on the accepted beads only, starting from stage 1. That saves v1's second full fit.
5. **The rejected beads keep their stage-1 scores and reason**; they are in no final fit. Beads in no fit at all (excluded by hand, held out) are not scored or judged. No beads are held out by default; the held-out report, which localises them, is a diagnostic on request. Per-slice consistency (v1's z bias, a stage jump) is shown by SMAPpy, which fits every bead with the exported model's spline.

**Feedback to the GUI.** The result carries a bead table, `Calibration.beads`, one row per bead:
- id, stack, chip position, photons, background;
- every score and its threshold;
- the automatic state: *used*, *held out* or *rejected (reason)*;
- the manual state: *none*, *included* or *excluded*.

`learn_beads(…, decisions={bead: "include" | "exclude"}, warm_start=previous)` takes the user's decisions, and they always win over the automatic ones. SMAPpy's calibration window (§7.1) shows the table with the scores and reasons, colours the rows by state, and lets Space toggle a bead. *Recalculate* reruns from the previous calibration as a warm start: stage 1 is skipped when no included bead is new, and the scores are recomputed for the new set. The window never re-implements a rule. A bead's state and its reason come from uipsf, so the CLI, a notebook and the window agree.

**Two channels** (built in M4, `docs/uipsf/architecture.md`).
- SMAPpy's dual pipeline supplies the layout, the split, the pairs and a projective transform (RANSAC + soft-L1).
- Standalone, `data.pairing.pair_beads` follows SMAPpy's convention (`collect_dual_beads`: the layout vote, the cut, the offset vote after the mirror, mutual nearest neighbours within 12 px) and starts the transform from a projective fit to the pairs' centroids; on Zenodo `2ch_40nm_bead` it gives SMAPpy's 84 pairs, and a test holds them equal while SMAPpy is installed.
- uipsf starts from that transform and refines it jointly with the PSFs, as uiPSF does.
- Bead rejection, the atlas and the bead table work per pair: one row per bead pair.

### 5.8 Outputs (`export/`)

```python
def save_model(cal: Calibration, path)          # the model file
def load_model(path) -> LearnedPSF              # ready for psf(); a Calibration would need the stacks back
cal.psf(xyz_nm, roi_px, depth_nm=None, include_bead=False, upsample=None) -> Tensor   # sampling, physical units
```

**The model file is uipsf's only output.** It is versioned (`format='uipsf-model'`, `version`) and holds everything needed to recompute the PSF:
- `/model`: settings (YAML), the component list, optics, every Zernike (phase and magnitude) with its CRLB, blur, channel transforms and photon shares, and the conventions version;
- `/fit`: history, stop reason and covariance;
- `/diagnostics`: residual atlas and held-out metrics;
- `/rois` (optional): data and fit.

**SMAPpy makes the spline.** SMAPpy loads the model file and samples it with `cal.psf(...)` on a voxel grid it chooses: z range and step, ROI size, and for in-situ models the focus depth. It then does what its bead calibration does with an averaged bead stack:
- clip and normalise exactly as `_single_model` / `positive_pair_models` do (maximum plane sum = 1, two channels jointly), so the result is a calibration as in SMAP;
- spline with its own `spline_coefficients`;
- save with `save_spline_calibration`, recording the model file and its Zernikes.

So the spline format, its knot convention (`64, z, y, x; 16·pz + 4·py + px`) and its photon normalisation have one owner, SMAPpy. uipsf stays free of SMAPpy, and its own photons stay physical.

**Zernike converters** to v1 uiPSF and vecfit conventions, so coefficients can be compared between labs (review §D).

### 5.9 Workflows, settings and public API

```python
import uipsf
settings = uipsf.BeadSettings(optics=uipsf.Optics(NA=1.43, wavelength_nm=680, ...))
cal = uipsf.learn_beads(stacks, settings, centres=None, progress=cb, cancel=flag)
cal.diagnostics.atlas; cal.figures()            # matplotlib figures
uipsf.save_model(cal, "psf.uipsf.h5")
stack = cal.psf(xyz_nm=grid, roi_px=29)             # what SMAPpy samples to make its spline
locs = uipsf.localize(cal, rois)
data = uipsf.simulate(model, params, scenario)
```

**Settings.** Plain dataclasses with metadata (label, unit, bounds, help), shaped so that SMAPpy can build its form from them the way it builds plugin forms from `param(...)`. *Revised (§7.5): uipsf imports SMAPpy's `param`, which is GUI-free, so the window shows `BeadSettings` as it shows a plugin's settings.* Presets are YAML (omegaconf) for the CLI and notebooks.

**Hooks for a GUI.**
- Workflows take a `progress` callback and a `cancel` flag, which is checked between iterations.
- They return plain data and figures, and touch no GUI or global state.

---

## 6. From uipsf to SMAPpy's 3D fitters

Every calibration takes this path, so it is specified exactly and tested end to end in SMAPpy (M6). Steps 1–4 are SMAPpy code; uipsf only provides `load_model` and `cal.psf`.

1. **Sampling grid** (SMAPpy chooses it):
   - Lateral: pixel centres over `roi_cal = fit ROI + 2 + margin`. SMAPpy refuses a spline whose lateral size is not at least the fit ROI + 2.
   - Axial: the stacks' own step is enough (measured: 10 to 100 nm give the same fit, `docs/uipsf/architecture.md`); for in-situ models, at the focus depth the user gives.
   - Knots sit on pixel centres, with the emitter at the central knot (`x0`), as `models.hpp` expects.
   - Each pixel as the model was learnt: `cal.psf` at the model's own sampling (its centres, by default),
     not integrated again.  *Revised (M6, October 2026):* this said "integrates over pixels exactly"; but
     a model learnt on pixel centres holds the pixel's width in its blur, and integrating it again counted
     the pixel twice (Spline 3D's z slope 0.992, 6 nm rms off the model's own localisation; at its own
     sampling 0.999 and 2.8 nm, `docs/uipsf/architecture.md`, "M6").
2. **z.** z0 is the knot at emitter z = 0, and `z_index_to_nm(k) = −(k − z0)·dz` holds by construction. A test checks the sign against SMAPpy's own bead calibration of the same simulated beads.
3. **Normalisation.** As SMAPpy's bead calibration does it: maximum plane sum = 1 for one channel, two channels normalised jointly with each one's `photon_normalization`. A uipsf calibration and a bead calibration are then interchangeable in Spline 3D.
4. **Two channels.**
   - Both channels are sampled on one z grid, the condition SMAPpy enforces.
   - The transform is SMAPpy's 3×3 projective map, secondary → main, in chip pixels.
   - A biplane offset is in the PSFs, not the grid, so both splines share z, as `dualfit.build_link` assumes.
5. **Check, with SMAPpy's fitter.** SMAPpy's `validation.fit_bead_diagnostics` (*Calculate fit quality*) fits the beads with the real Spline 3D. The same ROIs are also fitted by uipsf's pupil-model localisation (§5.6), which needs no extra code. z and photon ratios must agree within the spline's interpolation error. A disagreement means the sampling or the spline is wrong, not the model.

## 7. SMAPpy integration

### 7.1 The plan, revised (October 2026)

Decided: **SMAPpy provides it all by default, from its bead calibration**; uipsf keeps a simple loader for
standalone use (§5.7), with the same bead-finding convention; a little divergence between the two is
accepted. Both sides stay simple. Checked against SMAPpy `main` (e148090):

- the bead calibration (`calibrate/core.py`, `qt_gui.py`) works in ADU, with no camera conversion; it
  has a *mode* (single channel, dual colour) but no *model* choice; its bead table already takes
  manual exclusions (Space) and *Recalculate* rebuilds without re-reading the files;
- the v1 bridge (`smappy/uipsf/`, plugin `Localize/uiPSF calibration`) already resolves the camera
  (`prepare.resolve_camera`, the fitter's own resolution), converts to photons, carries the optics in
  microscope profiles (`uipsf/microscopes.py`), and turns uiPSF's voxels into a spline
  (`convert.single_calibration`: clip, maximum plane sum = 1, `spline_coefficients`);
- Spline 3D needs knots ≥ fit ROI + 2 laterally, z0 a knot index with z = −(k − z0)·dz, and for two
  channels the same z grid and a `photon_normalization` per channel.

*Superseded by §7.5 where they differ: a new plugin, PSF calibration, rather than a model choice in
the bead calibration; the v1 bridge removed now; uipsf's own loader removed.*

**One window, a model choice.** The bead calibration gets *model*: *bead average* (today's) or
*uiPSF pupil*. Everything else is the same window: the stack list, *Detect + calibrate*, the bead
table with Space and *Recalculate*, *Calculate fit quality*, *Save calibration…*, *Use in the Spline 3D
fitter*. No second window and no new plugin.

**Its settings.** uipsf has many more settings than the bead average; they go in a settings window
opened from the bead calibration (or a part of the form folded by default), with a short default list.
Which ones are shown by default is to be decided (October 2026).

**The figures** stay linked to the bead table as now (a click on a bead's point selects its row, the
table's selection is drawn), whoever draws them.

**Two channels first.** M4 (multi-channel) comes before the SMAPpy work, so that single and dual colour
are integrated in one go (decided October 2026).

**What SMAPpy does in *uiPSF pupil* mode, and what uipsf does:**

| step | owner |
|---|---|
| read the stacks, z from the metadata | SMAPpy, `read_bead_stacks` (as now) |
| the camera: conversion, offset, read noise, EM gain, pixel size | SMAPpy, the fitter's `CameraSettings.resolve` (as the v1 bridge); shown as *auto* hints in the form |
| photons, saturation mask | SMAPpy, `camera.to_photons`; the mask from the dtype's maximum |
| the optics: NA, refractive indices, wavelength | SMAPpy's microscope profile (`uipsf/microscopes.py`), editable in the form |
| bead centres | SMAPpy, `detect_beads` with the window's settings |
| ROIs, the fit, bead rejection, the atlas | uipsf: `extract_rois` from SMAPpy's centres, `learn_beads` |
| the bead table's states and reasons | uipsf (`Calibration.beads`); the window shows them, never decides |
| the spline | SMAPpy: samples `cal.psf`, normalises as `_single_model` (clip, maximum plane sum = 1, no smoothing), `spline_coefficients`, `save_spline_calibration` |
| figures: pupil, Zernikes with CRLBs, residual atlas | uipsf (`diagnostics.figures`, drawing on a figure it is given), so notebooks get the same |
| fit quality | SMAPpy's `validation`, Spline 3D on the exported spline (§6.5) |

SMAPpy's brightness pre-filter (*too dim*, *too bright*) is skipped in this mode: uipsf judges each
bead by its fit, and saturated pixels are masked, not the bead dropped.

**Saved.** *Save calibration…* writes the spline calibration (`_3dcal.h5`, SMAPpy's format, its
`parameters` recording `method = "uipsf2_pupil"`, the Zernikes with CRLBs and the model file's name) and
the model file beside it (`_3dcal.uipsf.h5`, §5.8). Loading the model file again re-samples it on another
grid without refitting.

**Sampling for the spline.** Knots at pixel centres over the window's `roi_size` (27 by default: 26 spline
cells, so fit boxes up to 24, as now); each pixel as the model was learnt (§6.1); z over the beads' range with the
stacks' own step: 10 to 100 nm fit the same, the error is lateral (knots a pixel apart), up to 4 nm at
the pixel edges for a narrow focal line, 0.1–0.2 CRLB at 10⁴ photons (`docs/uipsf/architecture.md`).

**What uipsf adds for it:**
- `BeadSettings`, a dataclass with metadata (label, unit, bounds, help) that SMAPpy's form can show,
  as its plugins' `param(...)`: Zernike order, the first stage's order, the photon model, the rejection's
  k, the held-out fraction; solver options folded away;
- `calibrate_beads(stacks, centres, camera, optics, settings, decisions, warm_start, progress, cancel)`,
  one call from stacks and centres to a `Calibration`;
- `diagnostics.figures`: pupil, Zernike bars with CRLBs, the atlas, the beads over the field.

**Running.** In-process. uipsf comes with SMAPpy, torch as the extra `smappy-smlm[uipsf]` (revised in §7.4: a default install was decided first, then dropped for torch's size on Linux).
Plugin discovery parses without importing, and the window imports uipsf only when the model is
chosen (torch costs nothing until then); were it missing, the choice is greyed with a line saying how to install it. The fit runs on the
window's QThread with uipsf's `progress` and `cancel`; SMAPpy's rules apply (no automatic garbage
collection, no raising `QTimer.singleShot`, canvases from `canvas_class()`).

**Two channels** (uipsf's side built in M4): SMAPpy hands over both channels' ROIs of each pair, with
each ROI's chip position (`Batch.roi_xy`), and its transform as the start (`Channel(transform=…)`);
uipsf refines the transform jointly and learns a pupil per channel.  For the splines, SMAPpy samples
`LearnedPSF.psf(…, channel=c)` for each channel on one z grid (the defocus offset is in the PSFs; each
sample includes the channel's photon share, so the pair holds one photon of the emitter) and
normalises them jointly with each channel's `photon_normalization`, as `positive_pair_models` does;
`LearnedPSF.transform(1)` is its `DualColorCalibration.transformation` and `LearnedPSF.geometry` its
`geometry` (the split as SMAPpy records it, written from the pairing).  **Storage is SMAPpy's dual
calibration** (decided October 2026): SMAPpy saves the result with `save_dual_color_calibration`, so its
global fitter reads a uipsf calibration as one of its own; `tests/uipsf_tests/test_smappy_dual.py` builds one that way
from a two-channel model and recovers x, y (0.005 px), z (3 nm) and the colour ratio with
`DualChannelEngine`.  Bead rejection stays per pair, as SMAPpy's.

**The v1 bridge stays** for what v2 does not do yet (blinking data, in situ, voxel models) and is
retired after M5, its tests becoming v2's acceptance tests. Found in passing: its plugin's `run`
passes an undefined `params` to `learn` (`plugins/uipsf.py`, `uipsf_parameters` imported, never
called); only the slow test gated on `SMAPPY_UIPSF_PYTHON` would catch it.

**Standalone uipsf** (notebooks, other UIs): `read_stack(path, camera)`, `find_beads`,
`calibrate_beads`, `save_model`. No GUI of its own.

**Order of work:**
1. uipsf: `BeadSettings`, `calibrate_beads`, `diagnostics.figures`.
2. SMAPpy, headless: `calibrate/pupil.py`, from the window's inputs to a `Calibration` and from it to a
   `SplineCalibration`; `smappy-calibrate --model pupil`. Tests (skipped without uipsf): simulated beads
   → model → spline → Spline 3D recovers z and photons; the z sign against the bead average on the same
   beads; the normalisation.
3. SMAPpy, the window: the model choice, the uipsf settings form, the tabs, the table's columns, save.
4. Docs: the panel page `Bead calibration.md`, NOTES.md; then the M6 checks of §6 on Zenodo.

### 7.2 How SMAPpy runs uipsf

**In-process, torch as an extra** (§7.1, *Running*; §7.4). SMAPpy's subprocess bridge to TF uiPSF was
needed because TensorFlow fights SMAPpy's packages; with PyTorch that reason goes.

**Python.** uipsf needs Python ≥ 3.10 and SMAPpy ≥ 3.12, so both install into one environment.

**PyTorch is for uipsf only.** SMAPpy's own fitters stay C++ on the CPU. On Mac and Linux the standard
torch wheel uses the GPU; on Windows CUDA takes one pip line with PyTorch's index URL (§5.1), and without
it the calibration runs on the CPU.

### 7.3 What SMAPpy's fitters need later

- **Field dependence** (later, with uipsf's field model). Spline 3D uses one spline for the field. A field-dependent calibration needs either per-tile splines chosen by the ROI position (a small change to `SplineModelSettings` and the engine), or a fitter on the pupil model itself. The first is cheap.
- **In-situ depth.** A model whose shape changes with depth needs the same choice.
- **sCMOS maps.** uipsf supports them in the likelihood. SMAPpy's porting plan lists them for its fitter (#22).

### 7.4 Moving into SMAPpy's repository (decided October 2026)

**Decided:** uipsf moves into ries-lab/SMAPpy before M5. It stays the package `uipsf` (`src/uipsf/`), in
`smappy-smlm`, with torch as the extra `smappy-smlm[uipsf]`. ries-lab/uiPSF2 is archived after the move.

**The move only.** This section moves the code, its tests, docs and CI, nothing more: what the move
copies (M0–M4 and M7, all merged on uiPSF2's `main`) behaves in SMAPpy as it does here. How SMAPpy uses
uipsf, the plugin and its window (§7.1), is a step of its own, to be discussed again before it is
built (below, *To discuss*).

**Why.**
- **One input pipeline in fact, not by convention.** Loading, the camera database, bead detection and the
  dual layout exist once. The tests that hold uipsf to SMAPpy (`test_rois`, `test_camera`, `test_pairing`,
  `test_smappy_dual`, `test_spline`) always run, instead of skipping without SMAPpy.
- **In situ (M5) is a loop over SMAPpy's fitters**: localise, learn the PSF, localise again. In one
  repository that is one call, and its design changes in one place.
- **One change, one commit.** The model file, SMAPpy's sampler and the window change together; there is no
  version of one package to pin against the other.
- **One instrument database** for the fitters and uipsf (below).

**What changes for uipsf.** It may import smappy (loading, the camera database, detection, the fitters) and
stops keeping its own versions where SMAPpy's serve. §1's and §2's "uipsf never imports SMAPpy" is
replaced by two rules, which keep a standalone uipsf a packaging job rather than an untangling one:
- **no Qt in uipsf** (`PySide6`, `pyqtgraph`, `smappy.gui`): the window is SMAPpy's;
- **torch only in uipsf**, which smappy imports inside the functions that use a model, so discovery and the
  rest of the program never load it.

**Torch is an extra, and asked for where it is needed.** Its size, as downloads for Python 3.12 (October
2026, from PyPI and PyPI's dependency resolution):

| platform | torch 2.9.1 | torch 2.14.1 |
|---|---|---|
| Linux x86_64, the default wheel (CUDA) | 4.07 GB in 26 packages, 3.2 GB of them NVIDIA's libraries and triton | 0.56 GB, plus its CUDA 13 libraries (not resolvable through this session's proxy) |
| Linux, PyTorch's CPU index | 0.18 GB | 0.20 GB (0.77 GB installed) |
| Windows (CPU) | 0.11 GB | 0.12 GB |
| macOS arm64 (CPU and MPS) | 0.07 GB | 0.13 GB |

PySide6-Essentials, for comparison, is 0.08 GB. On a Mac or Windows torch would barely count; on Linux it
is gigabytes, often on a fitting machine that never learns a PSF. So `pip install smappy-smlm`
stays as it is, and uipsf is `pip install "smappy-smlm[uipsf]"`. A user who opens it without torch is
told what to type, for this machine, where they are: the plugin's `describe` gives a `Notice`, as
*Find clusters* does for hdbscan (`HDBSCAN_NEEDS`), naming
- on a Mac: `pip install "smappy-smlm[uipsf]"` (MPS included);
- on Windows: the same for the CPU; with an NVIDIA card, PyTorch's CUDA index first
  (`pip install torch --index-url https://download.pytorch.org/whl/cu12x`);
- on Linux: the same with an NVIDIA card (the default wheel is CUDA's), else PyTorch's CPU index, a
  fifth of the size;

with the interpreter's own path (`sys.executable -m pip`), so it lands in the Python that runs SMAPpy.
**For the future (noted October 2026): the plugin installs torch itself.** The notice gets a button that
runs that command for the user: the right wheel for the machine (an NVIDIA card found, a Mac, neither),
into `sys.executable`, on a worker thread with pip's output as progress, cancellable, and the plugin
usable without a restart once it is done. The typed command stays in the notice as the fallback, for a
read-only or managed Python and for a user who wants to choose. Built with the plugin, not before.

**A standalone later** is SMAPpy without its GUI, plus uipsf. SMAPpy's non-GUI modules already import no Qt
(checked October 2026: `smappy.io`, `camera_db`, `detect`, `psf`, `fit_rois` and `pipeline` load neither
PySide6 nor pyqtgraph). What is missing is packaging: a `smappy-core` distribution without PySide6, on
which `smappy-smlm` depends. Until then a standalone install is `smappy-smlm` itself, whose PySide6
(~250 MB) is small beside torch. Copying SMAPpy's fitter (C++) into a standalone uipsf is the fallback,
not the plan.

**The name stays `uipsf`**, not `smappy.something`: the code, its tests and its docs move without renaming,
`smappy.uipsf` (the v1 bridge) keeps its name until it is retired after M5, and a standalone is the folder.

**Layout in SMAPpy** (§2's tree is uiPSF2's):

| uiPSF2 | SMAPpy |
|---|---|
| `src/uipsf/` | `src/uipsf/`, with `LICENSE`, `CLAUDE.md` (this repository's rules, adapted) and `ruff.toml` |
| `tests/uipsf_tests/` | `tests/uipsf_tests/uipsf_tests/`, a package, because its `test_gpu.py` and `test_diagnostics.py` share SMAPpy's names; the helpers are imported from `uipsf_tests.conftest` |
| `benchmarks/uipsf/`, `scripts/uipsf/` | `benchmarks/uipsf/uipsf/`, `scripts/uipsf/uipsf/` |
| `docs/uipsf/`, `README.md` | `docs/uipsf/uipsf/` |
| `.github/workflows/tests.yml` | `.github/workflows/uipsf.yml`: the fast tier on Linux, macOS and Windows when uipsf's paths change, with SMAPpy built from source |
| `pyproject.toml` | SMAPpy's, with the extra `uipsf = ["torch>=2.4"]`. omegaconf is not needed: only `make_oracles_v1.py` uses it, in v1's TensorFlow environment |
| tests without torch | not collected (`collect_ignore` in SMAPpy's root conftest), so a plain SMAPpy checkout runs as before |
| `--slow`, `--gpu` | SMAPpy's root conftest: `--slow` is shared, `--gpu` is added |
| ruff, 120 columns | `src/uipsf/ruff.toml`, which the three other folders extend; SMAPpy's files keep their own style |

**Left behind:** the third-party reference code read for ideas, our notes on it, and the git history,
which holds both. SMAPpy is public, so the move is a snapshot of uiPSF2's HEAD, and the commit message names
that commit. The plan's lines about the reference code are dropped in the copy, and the script refuses to
finish while any copied file still names it.

**How.** `scripts/uipsf/move_to_smappy/move.py PATH/TO/SMAPpy` makes the move in a SMAPpy working tree and commits
nothing: the copy, the paths in text and code, the scrub, and SMAPpy's side (the dependency, the conftest,
the workflow, ruff, `.gitignore`). By hand: SMAPpy's CLAUDE.md (where uipsf is, and its rules file) and a
NOTES.md entry, both test tiers, the commit and the PR.

*Dry run (October 2026, uiPSF2 576d095 into SMAPpy 5251952, 4 CPU cores):*
- 118 files copied; ruff clean on the uipsf paths;
- uipsf's tests in SMAPpy: 138 passed, 23 skipped (slow, GPU, Zenodo data), as in uiPSF2; `--slow`:
  157 passed, 4 skipped (3:43);
- SMAPpy's fast tier with them: 2398 passed, 127 skipped (1:56); without them, 2260 passed.
- of five full runs four were green; one failed in SMAPpy's `test_msr.py`
  (`test_the_open_dialog_queues_what_is_ticked_and_opens_its_images`, a GUI test waiting on a 20 s timer),
  which passes alone and in the other four. Whether uipsf's load on the workers makes it likelier is to be
  watched in the PR's runs.

**The instrument database.** One database with two levels and one editor:
- **camera:** SMAPpy's camera database as it is (`camera_db.py`: conversion, offset, read noise per readout
  state, chosen by the file's metadata);
- **microscope:** the optics and channels of the v1 bridge's profiles (`smappy/uipsf/microscopes.py`: NA,
  refractive indices, emission wavelength, the split, the in-situ start), and its cameras, by their ids in
  the camera database.

They stay two levels because a camera moves between microscopes and one microscope can carry two cameras.
The fitters and uipsf read the same entry. Its design is a step of its own, after the move.

**Order.**
1. Merge or port the open uiPSF2 branches (none open on 19 October 2026: M4 is on `main`); from then on,
   no new work in uiPSF2.
2. Run the script on a branch of SMAPpy, do the steps by hand, run both test tiers; one PR.
3. Archive uiPSF2 (private, read-only), its README pointing to SMAPpy.
4. The plugin, discussed first (below), then built in SMAPpy.

**To discuss: the plugin.** uipsf has two uses, and the plugin may want a switch for each:
- **a calibration for fitting**: beads in, a spline calibration out that Spline 3D reads, with the model
  file beside it. This is §7.1's plan, the bead calibration with a *model* choice;
- **analysing the microscope**: the aberrations themselves, the Zernikes with their CRLBs, the pupil
  magnitude, the residual atlas, the extra blur, the channel transform; compared over time or between
  microscopes, and for in situ, the sample's own aberrations. The result is a report and the model file,
  not necessarily a calibration.

Open with it: one window with two outputs or two plugins over one learning call; what the analysis
compares against (the previous model of the same microscope, from the instrument database below); which
settings each shows by default; where the torch install's button sits (above, *For the future*).  *Decided in §7.5.*

### 7.5 The PSF calibration plugin (decided October 2026)

Revision 6.  Replaces §7.1's *One window, a model choice* and answers §7.4's *To discuss*.

**A new plugin, *PSF calibration*,** named for what comes out, since in situ (M5) learns from blinking
molecules, not beads; it holds the bead average too, as a *method* (revised below, step 5).  It reuses the bead calibration's widgets and code
(the file list, the stack view, the bead table with Space and *Recalculate*, the worker thread, *Calculate
fit quality*), and keeps the shapes that make extensions entries rather than branches: a *source* (bead
stacks now, a blinking movie with M5) and a model whose options are choices, not flags (below).  The
current *Bead calibration* stays as its own plugin, unchanged, until *PSF calibration* does everything it
does; then it is retired.

**The v1 bridge is removed now,** not after M5: `smappy/uipsf/` (runner, worker, prepare, convert,
plots), the plugin `Localize/uiPSF calibration`, its page and `tests/test_uipsf.py`.  In-situ learning is
gone until M5, accepted.  Of `microscopes.py` only the idea survives, in the microscope database (below).

**No calibration / diagnosis switch.**  The diagnostics are cheap beside the fit, so every run shows them
all: the pupil (phase and magnitude), the Zernikes with their CRLBs, the residual atlas and its findings,
the held-out report, and the comparison with the start model when there is one.  One option, *save model*,
on by default: on, the run writes the model file and the spline (or dual) calibration made from it;
off, it only draws, for looking at a microscope without leaving a file behind.

**Start from a previous calibration**, a visible file chooser near the top of the form, used often: its
model is the warm start (`warm_start`), and the comparison page draws the new Zernikes against it.  Empty,
the fit starts from the data as now.  Later it may default to this microscope's latest model from the
database.

**Model options shown, for now.**  They are under investigation (§10, 2 and 3), so they are exposed, as
choices that can grow or be taken out again once the investigations settle them:
- *pupil magnitude*: **free** (1 + Σ c_j Z_j over every mode up to n_max), **symmetric** (the
  rotationally symmetric modes only, m = 0: Noll 4, 11, 22, …, through `magnitude_modes`), **fixed**
  (uniform: the apodisation only, `magnitude_n_max=0`);
- *blur σ*: **free** or **fixed** (at its start value; the machinery is `ParamSet.fix`, the setting is
  new);
- *Zernike order*: auto (6) or a number.

They live in one dataclass of model options in uipsf (`BeadSettings`), so a choice added or removed is one
field, and its defaults are written once.

**The form**, short, the rest folded under *more*:
- *microscope*: from the file's metadata, every instrument value shown with its source, *Edit
  microscope…* beside it;
- *start from*: a previous calibration;
- *bead diameter*;
- the model options above;
- *save model*;
- under *more*: the sample medium's refractive index (from the microscope entry: beads in water, say;
  set here when the sample differs, e.g. index matched, which matters most in situ), rejection's k, the
  held-out fraction.

Solver settings (grid, solver, tolerances, stages, chunking) are not in the form; the device is named in
the status line.

**How smappy feeds uipsf: one adapter, uipsf takes arrays.**  uipsf's own reader (`data/io.py`), bead
finder (`data/rois.find_beads`), pairing (`data/pairing`) and hand-typed camera (`noise.Camera` as an
input) are removed; a standalone uipsf is SMAPpy without its GUI (§7.4).  smappy's
`calibrate/pupil.py`, headless:
1. `read_bead_stacks`, z from the metadata;
2. the camera per file from the camera database, then photons; saturated pixels a mask;
3. beads by `detect_beads`, pairs and the starting transform by `collect_dual_beads`;
4. raw photon ROIs into a `uipsf.Batch` (`z_nm`, `s`, `mask`, `roi_xy`, `field_xy`, `stack_id`);
5. `uipsf.calibrate_beads(batch, optics, settings, decisions, warm_start, progress, cancel)`;
6. sample, normalise as `_single_model` / `positive_pair_models`, save with `save_spline_calibration`
   or `save_dual_color_calibration`, the model file beside it.

This needs `collect_beads` split in two: *find* (positions, records, the brightness and saturation flags,
shared) and *cut* (normalised, background-free volumes for the bead average; raw photon ROIs for the
pupil), so that both calibrations share the bead table and its exclusions.

**The instrument database: microscopes beside cameras.**  Built on `camera_db`'s machinery (`Parameter`:
fixed, a metadata tag, a `map` from what the tag reads; `State`; `Source`), so a value can follow the
acquisition's state: a filter changes the emission wavelength, the cylindrical lens the expected
astigmatism, as the lens already changes the pixel size in the camera database.  An entry:
- *identify*: the tag that names the microscope (to be chosen from real files: which tags the lab's
  microscopes write for the objective, the filters and the lens; pending, the user provides files);
- *cameras*: ids in the camera database;
- *NA*, *immersion index*, *coverslip index*: fixed or by the objective's tag;
- *emission wavelength* per channel: fixed or by a filter's or dichroic's tag;
- *channels*: the layout, split and main channel;
- *sample defaults*: the medium's index, the bead diameter;
- *history*: the previous model files, for the start and the comparison.

Its editor is built as the camera editor: from an open file, showing what the entry makes of it.  **No
silent defaults for an instrument value:** v1's profile fell back to NA 1.43 and 680 nm; here a missing
value is shown in orange and the run refused until it is given.  After the fit, the findings point back
at the instrument values the dictionary of §10, 2 ties to a pattern (a radial magnitude with a z scale
off 1: is the NA or the wavelength right?).  The fitters read the same entry later (pixel size, the
split).

**Settings to test before their defaults are fixed** (on Zenodo and the lab's stacks, by held-out deviance
and z bias): n_max 6 or 8; the magnitude free, symmetric or fixed; σ free or fixed; the intensity model;
upsampling; the first stage's order; the slice thinning (`max_dz_nm`).  Those that settle leave the form.
*Measured on Zenodo 1ch and 2ch and simulated beads (October 2026): §10, 7, a recommendation per choice,
to decide; the table in `docs/uipsf/architecture.md`.*

**Order of work:**
1. uipsf: `BeadSettings` with form metadata (the model options above), `calibrate_beads`, σ fixed,
   the symmetric magnitude, the figures (`diagnostics.figures`, drawing on a figure it is given). *Done
   (October 2026, `docs/uipsf/architecture.md`).*
2. Remove the v1 bridge; remove uipsf's reader, finder, pairing and camera input, its tests moved onto
   smappy's.  *Done (October 2026): `smappy.calibrate.pupil` (`bead_rois`, `single_rois`, `dual_rois`,
   `rois_at`) is the adapter, so step 3's headless part is half built; what is left of it is the split of
   `collect_beads` for the window's shared bead table, and the sampling into a spline.*
3. smappy, headless: split `collect_beads`; `calibrate/pupil.py`; tests from simulated beads to Spline
   3D (z, photons, the z sign against the bead average, the normalisation).  *Done (October 2026,
   `docs/uipsf/architecture.md`).*
4. The microscope database and its editor, once the metadata of real files is in.
5. The *PSF calibration* window and plugin, its page, `default.gui.yaml`.  *Built (October 2026):
   `calibrate/psf_window.py`, `plugins/psf_calibration.py`, the optics as a form until step 4.  **Revised
   (decided October 2026): one window for every PSF calibration**, a *method* choice between the bead
   average and the pupil model (entries in `calibrate/psf_methods.py`), and later a *source* choice (bead
   stacks or single molecules) for in situ, which fixes the method to the pupil model.  *Bead calibration*
   is retired once this window does everything it does.  Since built: the pupil model's *Calculate fit quality* (every slice of every bead fitted with its spline
   by Spline 3D, z against frame per bead, a click selects the bead), the *Beads* map following the
   table's selection (the pupil, Zernikes and atlas describe all beads, so they do not), and *Install
   PyTorch* (`smappy/torch_install.py`).*

---

## 8. Diagnostics (from the first fit on)

**Residual atlas.** Deviance per pixel, averaged in groups:
- slice or frame;
- field position (radial and 2-D bins);
- radius in the ROI;
- z;
- sensor position.

It also gives the signed mean residual per group (the pattern). Each group should be ≈ 1 when the model is adequate.

**Held-out beads.** By default 20 %, stratified by field position. They are excluded from learning and localised afterwards, with uipsf's `localize` and, in SMAPpy, with Spline 3D on the exported spline. The fit reports:
- held-out deviance;
- z-bias per slice;
- lateral bias;
- CRLB against observed spread.

**Report.** Stop reason, iterations, λ history, parameter CRLBs, and the pixels clamped or masked.

These are the benchmark metrics (§9) and the input for any automatic tuning or agent triage (review, "Auto-tuning").

---

## 9. Milestones and how each is tested

The test tiers follow SMAPpy:
- fast CPU tests on every change (`pytest -n auto`, about a minute);
- a `--slow` tier for benchmarks and end-to-end runs;
- a `--gpu` tier run by hand on CUDA and MPS.

GitHub Actions runs the CPU tiers.

**What is tested where.**
- uipsf's own tests need no SMAPpy: optics, forward model, solver, learning on simulated data.
- Tests that need a spline, or SMAPpy's fitter, live in SMAPpy's suite and run against an installed uipsf. They skip when uipsf is not installed, like SMAPpy's other optional tests.
- A few uipsf tests may import SMAPpy for its simulator or bead detection. They skip without it.

### Phase 1: re-implementation

**M0 Scaffold** (½ week)
- pyproject, src layout, CI, ruff, pytest tiers, `device.py`, `CLAUDE.md`.
- `scripts/uipsf/fetch_zenodo.py` (written and working).
- `docs/uipsf/conventions.md` (§4).
- *Test:* the package imports, CI is green, and the fetch script lists the archive.

**M1 Forward model** (2–3 weeks; task list in §11)
- *Oracle tests:*
  - `VectorPSF` against the vecfit NumPy port (from the review's `crosscheck_vecfit.py`) to ≤ 1e-6 of peak, with and without aberrations, including supercritical light.
  - Against TF uiPSF `genpsfmodel`: fixtures generated once in a TF environment and committed as npz. Bin 1 and 2, with blur and bead kernel. Conventions are mapped explicitly, since v1's and SMAPpy's differ.
- *Invariants:*
  - Zernike orthonormality.
  - Matrix DFT equals a padded FFT.
  - The unaberrated PSF is symmetric.
  - A lateral shift equals an image shift.
  - The z sign is SMAPpy's: objective up = emitter down, checked on an astigmatic PSF against SMAPpy's spline of the same simulated beads.
  - Energy normalisation.
  - float32 matches float64.
  - CPU matches MPS/CUDA (gpu tier).
- *Gradients:* `gradcheck` in float64 on a small grid for every parameter; JVPs against finite differences.
- *Speed baseline:* forward + backward for 6 beads × 41 slices, against the review's 0.12 s (CPU prototype).

**M2 Likelihood and solver; synthetic single-channel bead benchmark** (3 weeks)
- *Unit tests:*
  - deviance 0 at μ = d, with analytic gradient and weight;
  - LM solves linear least squares in one step;
  - the Schur solve equals the dense solve;
  - priors add exactly.
- *Scale invariance (the key regression):* rescaling a parameter's units by 10ⁿ leaves the LM iterates unchanged up to round-off.
- *Chunk-size invariance:* the objective, gradient, normal equations and the final fit are the same, up to float32 round-off, for any chunk size and any number of beads per chunk. v1's results depend on `batch_size` because the per-bead gradients and the loss are weighted differently when the beads span several batches (review F1). In v2 a chunk is only a memory unit; sums are exact.
- *Statistics:* the CRLB matches the spread of 200 simulated localisations within 10 %; held-out deviance ≈ 1 on matched data.
- *Benchmark:* the review's bright and dim beads (6 beads, 41 × 50 nm, NA 1.43, λ 680 nm), plus very dim. Dim beads must reach a z error ≤ 5 nm and phase RMS ≤ 0.015 rad (the best tuned v1 run in the review), with no settings changed, in ≤ 50 iterations, for every scaling.

**Result (M2).**
- All of the above hold (`test_noise.py`, `test_solve.py`, `test_forward.py`, `test_benchmark.py`), with one change to the target: **z is held to the CRLB, not to 5 nm.** The z CRLB of a dim bead in this scenario is 5–9 nm, so 5 nm on six beads is the luck of a draw (v1's 5.3 nm was on eight).
  - Dim beads: phase RMS error 0.0085 rad, z spread 3.8 nm (CRLB 0.0084 rad, 5.2 nm), in 30 iterations and 80 s on 4 CPU cores. v1's default run gave 0.059 rad and 61 nm.
  - Bright and very dim beads reach their CRLBs too; matched and held-out deviances are 1.00–1.01.
- **Not in the plan:** the default bead fit runs in two stages, Zernikes up to n = 4 first, then all (`workflows/beads.fit_beads`). In one stage the dim beads stopped in a local minimum: every bead moved by 50 nm in z, with high-order modes bent to match.
- **Cost:** one iteration takes ~8 s for six beads (59 JVPs, each two to three forwards). This is the risk of §12, now measured; speed is M7.
- **The Zenodo `1ch_40nm_bead` ROIs** were fitted as a first look (`benchmarks/uipsf/real/zenodo_1ch_beads.py`); see `docs/uipsf/architecture.md`. The full comparison stays in M3.

**M3 Data, camera, residual atlas; first real data** (3 weeks)

*Progress (October 2026):* the stack reader (camera parameters as input), bead finding by SMAPpy's convention, ROI extraction, and the noise-transfer check are built (`docs/uipsf/architecture.md`). On the Zenodo beads in the camera's own photons, the read noise (21 e⁻, not SMAPpy's default 1 e⁻) and the pixel orientation (116 × 127 nm) were found. **Read noise belongs in the camera database:** SMAPpy's entries carry no read noise, and the default is right only for an sCMOS. Proposed for SMAPpy: `read_noise_e` per readout state, as the conversion is. Bead rejection and the bead table are built (`workflows/learn_beads`, as in §5.7; the warm start always skips stage 1, since the previous pupil already places every bead). The residual atlas, the likelihood-ratio probes and the held-out report are built too. **Not in the plan:** the probes. A 5 % piezo scale is absorbed by the pupil and leaves no residual pattern, so the atlas alone cannot ask for it; refitting with the term free and testing the gain can, and does (`docs/uipsf/architecture.md`). The full Zenodo run end to end is done (`docs/uipsf/architecture.md`). Decided from it: the first slice of a stack is dropped by default. The z scale stays fixed for now (see "To investigate" in §10).
- *Camera:* tests against SMAPpy's photon conversion, read noise and EM handling (same numbers on the same ADU). The real benchmark takes its camera from SMAPpy's `resolve_camera` (§5.7). For Zenodo `1ch_40nm_bead` that is 4.7 e⁻/ADU and an offset of 398.6, not v1's 0.22: rerun the M2 first look in those photons and check that the deviance approaches 1.
- *Bead rejection* (§5.7):
  - on simulated beads with planted outliers (a double bead, a bead with a jump in z, one with photons 3× the rest, a saturated one), each is rejected with the right reason and the fit then reaches its CRLB;
  - matched data lose at most one bead in twenty;
  - manual decisions override the automatic ones, and a warm-started *Recalculate* gives the same result as a cold fit on the same beads;
  - on Zenodo, judged by v2's own measures, not by agreement with v1: a rejected bead must still fit worst on the final pupil, and leaving it out must not worsen the held-out deviance and z bias. v1's and SMAPpy's choices are reported for reference only; v2 does not follow them where its own evidence says otherwise.
- *Atlas:* on simulated mismatch (review §B: background ramp, 5 % piezo scale, polychromatic emission, plus laser fluctuations and speckle) it flags the right group (spatial, per-z, radial, per-slice, per-bead photons), and stays ≈ 1 on matched data.
- *Photon models:* data simulated with laser fluctuations and with speckle are fitted with each intensity option; the right option recovers the pupil, the wrong one shows in the atlas.
- *End-to-end:* Zenodo `1ch_40nm_bead` on the ROIs v1 fitted (`rois/psf_data` in the reference file). Compare Zernikes, pupil, held-out z-bias and deviance with v1's `res/`. Default settings must match or beat it.
- *Bead finder:* compared on the same stacks with SMAPpy's `detect_beads` and v1's finders (`main`, `test_roi`). Every difference must be explained.

**M4 Multi-channel** (3 weeks)
- Channels with their own pupils, a shared emitter, a projective transform and photon shares; biplane and ratiometric.
- *Tests:*
  - synthetic biplane and ratiometric data: transform, shares and both pupils recovered;
  - Zenodo `2ch_40nm_bead` against v1 (`res/T`, per-channel Zernikes) and against SMAPpy's dual calibration of the same stacks (transform residuals);
  - an explicit "unpolarised" default (review F1).

**Result (M4).**  Built as planned (`docs/uipsf/architecture.md`, M4; conventions in `docs/uipsf/conventions.md`,
"Channels"):
- `ForwardModel` takes a list of `Channel`s, layout [E, C, K, Y, X]; each has its own optics, pupil and
  blur, and the secondary a projective transform (SMAPpy's object and direction), a photon share and a
  defocus offset.  The emitter is shared; the backgrounds are per channel.  Unpolarised is the explicit
  default.  The model file is version 2 (channels) and reads version 1; `psf(…, channel=c)` samples
  every channel on one z grid.
- The M7 speed paths work per channel, and the transform, shares and offsets take their columns from the
  emitter linearisation; a channel costs what a single-channel model does (Jacobian 2.2×, fit 3.2× for
  two channels against one).
- Synthetic biplane and mirrored ratiometric beads: transform, share, offset and both pupils within
  3.3 CRLB, from a projective fit to the centroids.  **Not in the plan:** the fit needs the transform's
  start within a pixel or two (a translation alone, ~3 px off at the field's edge, stopped in a wrong
  minimum); a better start is in §10 ("To investigate", 5).
- Zenodo `2ch_40nm_bead`: the same 84 pairs as SMAPpy; the transform within 0.025 px RMS of SMAPpy's dual
  calibration and of v1's `res/T`; the secondary's Zernikes match v1's (correlation 0.9998, 0.0017 rad RMS),
  the main's to 0.007 rad, mostly spherical traded against v1's defocus.  v1's reference channel turned
  out to be SMAPpy's secondary (its arrays transposed, the lower half reversed).  The photon
  shares (0.164 / 0.836) are SMAPpy's `photon_normalization` for the same stacks (0.160 / 0.840).
- `--slow -n auto`: 156 passed, 3 skipped (the gpu tier).

**M5 In-situ, single channel** (3–4 weeks; designed with you, October 2026)
- Detection, a first fit of every candidate, selection stratified in z, then a joint fit of pupil + emitters +
  focus depth, repeated from the whole candidate pool until it is stable.  The design and the review of v1
  it rests on follow.
- *Tests:*
  - simulated blinking frames with known aberrations, depth, density and sign, including the cases
    SMAPpy's bridge failed: aberrations within their CRLB, sign right, z scale within 2 %, from a poor start;
  - Zenodo `1ch_smlm_tubulin` against v1's result.  `1ch_smlm_TP` (tetrapod) waits for the initial-pupil
    work package below.

*v1's in situ, as its code runs it* (`psflearninglib.iterlearn_psf`, `PSFZernikeBased_vector_smlm`,
`PreprocessedImageDataSingleChannel_smlm`, `Fitter.relearn_smlm`):

| step | what v1 does | its settings |
| --- | --- | --- |
| frames | `frame_range` (default the first 3000), flattened to K = 1 per emitter | `frame_range` |
| background | each frame minus the mean of the next 100 | — |
| detection | DoG (0.75σ − σ), 3×3 maximum filter, threshold **relative to the 99.99th percentile of the frame's maxima** | `gauss_sigma`, `max_kernel`, `peak_height` (0.2–0.3), `roi_size` |
| overlaps | `remove_close_rois` drops every ROI with a neighbour closer than the ROI's diagonal (~30 px at 21 × 21) | — |
| initial PSF | a model file; or Zernikes by hand (the demos: Noll 5, 0.5–1 rad); or a search over modes 4–20 at ±0.5 by median LL | `init_pupil_file`, `zernike_index`, `zernike_coeff`, `zkorder_rank` |
| z origin | the stage position sets the defocus; the z range is centred on the aberration-free focus | `stage_pos`, `z_range` (2 µm) |
| pre-selection | localised with the initial PSF (CUDA spline); 1 % tails in x, y, z cut, the lowest 10 % in LL, photons below the `min_photon` quantile (0.4) | `min_photon` |
| stratification | 21 z bins over the histogram, at most 100 each, **ranked by LL against the initial model** | `partition_size` |
| fit | MSE with heuristic weights, L-BFGS-B, 200 iterations; per emitter xyz, bg, I; shared Zernike magnitude and phase, blur, stage position | `var_stagepos`, `var_blur`, `n_max`, `iteration` |
| reject, refit | the worst 20 % by MSE, the xyz tails, negative intensities; refitted | `rej_threshold` |
| repeat | all again from the learnt model, `min_photon` lowered by 0.1 per round, the stage position carried | `repeat` (2) |

Its geometry is `phiz = kz_med·z − kz_imm·stage_pos`; v2's `VectorPSF` has the same split already (`xyz`
through k_z of the immersion, `depth_nm` through the medium's).

*Where v1 works against in situ learning.*  The out-of-focus emitters carry most of what the PSF has to
learn, and v1 loses them three times: the detection threshold is relative to the frame's brightest in-focus
peak, so a defocused spot with as many photons falls out first; the `min_photon` cut and the ranking by LL
against the initial model keep what that model already explains (and the z bins are that model's z); the
overlap rule, by the ROI's size and not the spot's, removes defocused (large) spots most at blinking
densities.  The sign of the even aberrations is set only by the Zernike typed in: for astigmatism and
spherical aberration the twin Φ(k) → −Φ(−k) flips z, so the ±0.5 search cannot tell the two apart and picks
one by noise.  What breaks the symmetry is the depth term (index mismatch) and the coverslip, a hard lower
edge on z.

*The design, decided:*
- **Frames sampled over the whole movie**, in blocks, until every z bin has enough candidates; sparse
  frames (late in the movie) are preferred, not the first, densest ones.
- **Inclusive detection.**  A sliding temporal median as the background; a DoG at several widths
  (σ ≈ 1.2, 2, 3 px) so that defocused spots are found at their own size; a candidate is kept on its
  **absolute significance** (photons in a disc against the background's Poisson noise), never relative to
  the frame.  SMAPpy's `detect.py` is the place for the filters.
- **A first fit of every candidate: an elliptical Gaussian** (SMAPpy's C++ `gauss_xy`, as
  `model/astig.py` uses for beads), giving photons, σx, σy, background, LL and the offset from the ROI
  centre.  The z proxy is D = σx² − σy², linear in z for astigmatism (`astig.py`), or the spot's size, |z|,
  for a 2D PSF.  *A work package of its own, later:* PSFs the Gaussian cannot sort by z (double helix,
  tetrapod, the phase masks of `roadmap.md`) need the first fit to be an initial pupil model instead.
- **Overlaps, by the spot's size, on the fly.**  A neighbour within r_i + r_j (r ≈ 3σ of each fitted spot)
  rejects the candidate; pixels of farther neighbours that reach into the ROI are masked (`Batch`'s mask);
  an emitter with too much of its ROI or its own expected photons masked (~20–30 %) is rejected.  Emitters
  are plenty: be picky, as long as every z bin keeps enough.
- **Blinks linked**, with SMAPpy's grouping (`group.connect`): one emitter seen in K consecutive frames is
  one emitter with a shared xyz and photons per frame (`[E, C, K, Y, X]` with K > 1), which helps the dim,
  defocused ones most.  Only blinks whose shape stays the same across their frames are kept.
- **Selection in z bins, by a brightness window, not the brightest.**  The brightest of a bin are where
  unresolved pairs collect, and with a Gaussian illumination they are the field's centre.  Per bin,
  emitters are drawn at random from a window a factor 3 wide whose top is at the bin's ~90th percentile of
  photons.  Pairs are also caught by: Gaussian widths or deviance far from the bin's typical (σx, σy);
  photons that step between the frames of a linked blink (a neighbour switching); and, from the second
  round, the model's deviance.  Stratified in z only for now, not in the field.
- **How many:** ~2000 emitters to start, a setting.  Choosing the number per bin from the CRLB of the shared
  parameters is a later round, a test first.
- **Held out:** ~10 % of each z bin stays out of the fit; their deviance and z bias against the fitted ones
  are reported, as held-out beads are.
- **The fit is v2's:** Poisson / camera likelihood, priors and no weights, per-emitter blocks (5 parameters,
  or K + 4 for a linked blink with photons per frame) through GN-LBFGS's Schur complement, low orders first
  (`first_stage`), rejection by `rejection.py`'s scores.  Shared: the Zernikes, the blur, the focus depth;
  n_med fixed from the settings, freed with a prior only as a diagnostic.
- **The focus depth** (v1's `stage_pos`, the focal plane above the coverslip): entered by the user, then
  fitted with a loose prior (~0.5 µm); emitters at the coverslip, the z histogram's lower edge, pin depth 0
  (v1's `backgroundROI`, made automatic).
- **Rounds until stable:** localise every candidate with the current model (Spline 3D from the sampled
  model), reselect, learn; stop when the Zernikes move by less than their standard errors and the selection
  barely changes, at most ~4 rounds.  v1's `repeat` reselected only from what the first round kept.
- **The sign of the even aberrations, from four sources; explicit beats the data, the data beat the prior.**
  The start is what the user gave: a bead model of the microscope (the window's *start from*) or Zernikes
  typed in (v1's way, kept for unusual PSFs).  The twin test always runs: the mirrored model refitted as a
  start, ΔLL reported; when the data clearly prefer the other sign it warns, never flips silently.  With
  nothing explicit and the data undecided, the microscope's entry decides: the astigmatism's sign and
  orientation, a field to add to the microscope database (§7.5).
- **Scope:** astigmatic and 2D PSFs, one channel.
- **The window:** the *source* switch (§10) offers a blinking movie; *Detect* shows the frames with the
  candidates ringed by state (accepted, rejected for overlap, masked neighbour), the z proxy's histogram
  with the bins and how many are taken from each, and the learning-time estimate, as for beads.
- **Output:** the model file and a spline sampled at the fitted depth, for data imaged at that depth.
  Splines at other depths later.

*Done, step 1 (October 2026):* `InSituGeometry` (h through k_z,med, the focus through k_z,imm, both on the
linearised Jacobian path) and `workflows.insitu.learn_insitu` on cut regions: localisation from several
heights on the start, the staged fit, the twin test.  On 40 simulated emitters the pupil comes back within
its CRLB and the twin test agrees by 62 nats; started with the wrong sign it learns the twin and warns
(−54 nats).  Focus and heights share an offset to the focus's CRLB (architecture.md, "M5, step 1").

*Done, step 2 (October 2026):* linked blinks, one position over K frames with photons per frame
(`PerObservation`).  Four frames halve the height's CRLB; 300–1000 nm from focus, linking places 1.2–3× as
many emitters within 100 nm; the joint fit's pupil is as precise either way (architecture.md, "M5, step 2").

*Done, step 3 (October 2026):* simulated blinking movies, `uipsf/movie.py`: SMAPpy's blinking model, the
model's own PSF per spot (a spot cut from a frame is the fit's μ), a Gaussian illumination, molecules on
the coverslip, a truth table (architecture.md, "M5, step 3").

*Done, step 4 (October 2026):* detection and selection, `smappy/calibrate/insitu.py`, as decided, with two
findings: detections whose Gaussians meet are one molecule (a defocused spot has several maxima), and a
Gaussian wider than a third of its region is no detection.  **A linked blink is its frames summed** into
one region (Poisson, (ΣN_k)·I + K·b), not K frames with photons per frame: every emitter then has K = 1
and nothing is padded; step 2's per-frame model stays for when per-frame photons are wanted.  From a
simulated movie to a pupil within ~2σ (architecture.md, "M5, step 4").  *Revised:* frames, not summed
blinks, are fitted (``unit="frame"``, v1's way), linking kept for judging blinks and holding molecules
out whole; summing stays an option, compared later.

*Done, step 5 (October 2026):* the rounds and the coverslip edge, `workflows/insitu_rounds.py`.  Each
round localises every kept unit on the model and chooses again by its height, photons and deviance;
stop when the change is within the noise.  The edge of the localised heights (a flat density, a spike if
it pays, κ on the CRLBs, outliers) becomes the next round's focus prior, through a focus per height that
is measured, not the paraxial n_imm/n_med (1.14 against ~1.6–1.85 at NA 1.43).  A focus started 250 nm
off comes back to within a few nanometres with molecules on the coverslip, ~25 nm without
(architecture.md, "M5, step 5").

*Done, step 6 (October 2026):* the PSF calibration window's third method, *in situ (blinking molecules)*
(`smappy.calibrate.psf_methods.InSitu`, `insitu_run`, `insitu_views`), decided with you: movies read in blocks
(*frames per movie*, 4000); the focus required, with *auto* estimating it from the coverslip's molecules;
the start's astigmatism as one field when there is no *start from*; *on the coverslip* and *rounds* in the
form, the detection under *more*; a table of counts, no marks; pages *z bins*, *Rounds and edge*, the
pupil's and *Model vs data* per height; the spline's z above focus or above the coverslip, a choice, over
the used heights.  (*auto* is the focus field's empty state, so the field starts on it; required in the
sense that a value or *auto* is always resolved, never a default.)

*Changed, 2026-10-23 (decided with you):* **molecules, not frames.**  *frames per movie* and *per bin* gave
way to one setting, *molecules* (2000: spots fitted, each frame of a blink one); the movies are read block by
block, coarse to fine over each (`insitu_run.block_order`, the van der Corput order), at least ten blocks of
each, until every z bin has its share passing or one has twenty times it (`enough`); a bin short of its share
gives all it has and the others make up the number (`insitu.quotas`).  The detection's settings are hidden
but *significance*, *region* and *z proxy*.  The Stacks page steps through the frames with molecules only.
*The start's astigmatism* went under *more*, on *auto* the microscope entry's for its 3D lens's state
(`astigmatism_nm`: M2 +62 nm with the iXon, −73 with the Evolve, M5 +65, from the bead models of
`results/lab_beads.jsonl`), shown and asked for only when neither the entry nor *start from* gives it
(`insitu_run.start_astigmatism`): §7.5's "the microscope's entry decides", made.

*Open, optimised once the framework runs:* **depth, n_med and spherical aberration trade against each
other.**  The study needs M5 itself (simulated blinking frames, the joint fit with a depth, the held-out
z bias), so it is not a step before it: build with the defaults below, then on simulations with known
truth (depths, n_med and radial Zernikes varied, the emitters' z range varied) measure which of them a fit
can separate, how the covariance couples them and what each bias does to z, and choose what is fitted,
fixed or given a prior.  The defaults until then: depth fitted, n_med fixed, the radial Zernikes free.

**M6 SMAPpy integration and release** (3–4 weeks)
- In SMAPpy: loading the model file, sampling it, the spline (§6) and the calibration window (§7).
- Tests, in SMAPpy's suite, with SMAPpy's fitter:
  - simulated beads → uipsf → spline → Spline 3D on simulated frames: z slope 1.00 ± 0.01 and photons as with SMAPpy's own bead calibration;
  - Spline 3D and the pupil-model localisation agree on the same ROIs (§6.5).
  *Both done (October 2026, `tests/uipsf_tests/test_pupil_spline.py`): slope 0.999, photons 0.2 % from the
  bead average's; the same ROIs 2.8 nm rms apart in z against a CRLB of 3.4, after the sampling was
  corrected (§6.1).  The dual fit quality fits each pair with Spline 3D 2C's global fit
  (`pupil.fit_quality`, `test_smappy_dual.py`).*
- The panel page in SMAPpy's docs format, and the tutorial storyboard updated.
- The real benchmark (§10, benchmark data): no per-dataset settings, results that match or beat tuned v1.
- Retire the TF bridge. uipsf documentation; release 2.0.

### Phase 2: optimisation

**M7 Speed** (2–3 weeks; can start after M3)

*Decided with you (October 2026):*
- **Target.** A bead calibration (20–50 beads, n_max 8) in about a minute on a laptop GPU. More
  important is what it leaves for later: multi-channel, field dependence and in-situ cost many times
  more, and must run in tens of minutes, not hours.
- **CPU first, in the cloud;** the algorithm comparison (iterations, forward-pass equivalents,
  accuracy) carries over to the GPU. The Mac (MPS) runs the profile by hand early, because the
  per-operation balance there differs. Windows with CUDA is a task of its own, later.
- **v1 is timed** on the same Zenodo ROIs, so "v1 was faster" is measured.
- **A faster solver becomes the default** when it lands within tolerance of LM's optimum (below),
  even if its numbers are not identical.

*Why an LM iteration is expensive.* v1 runs L-BFGS-B: one forward and one backward pass per evaluation,
hundreds of evaluations, and it depends on the units of the parameters (hence `self.weight`). v2's LM
builds the whole Jacobian, one JVP per parameter, every iteration: for six beads, 59 JVPs at ~0.12 s
cost as much as ~90 forward + backward passes (0.09 s). LM converges in 15–50 iterations, L-BFGS in
hundreds; which one is faster in wall time is open, and most of LM's cost is in how the Jacobian is
computed.

*Candidates, in the order they are tried* (all compared on the same matrix):
- **A. The same LM, a cheaper Jacobian** (numbers unchanged):
  - the pupil is linear in its Zernike coefficients: the column of mode j is one matrix DFT of the stored
    field times Z_j, without recomputing the forward model;
  - one forward pass shared by all tangents, larger tangent batches on the GPU;
  - a shared fine z grid of pupil fields, interpolated per bead;
  - complex maths split into real and imaginary parts so `torch.compile` can fuse it;
  - cached DFT matrices and bases; chunk sizes per device.

  *Progress on the Mac (October 2026):* the linearised columns, a real and wide matrix DFT, the blur as
  two small matrices, tangent batches per device and `torch.compile` on MPS took one LM iteration from
  1.6 s to 0.4 s (6 beads) and 5.7 s to 1.2 s (20 beads) on an M1 Pro's GPU, and 2.9× on its CPU
  (`docs/uipsf/architecture.md`).  The shared z grid was dropped: the defocus waves are 4 % of a forward pass.
  Cached DFT matrices too: a few ms per Jacobian, at the price of hidden state.
- **B. A lazy Jacobian:** the emitter block (~6 JVPs) every iteration, the shared block every k-th, or
  updated by rank-one (Broyden) corrections.
- **C. Matrix-free Gauss–Newton (Newton–CG):** each CG step one JVP and one VJP, preconditioned exactly by
  the emitter blocks; inexact steps early. Needed anyway for pixel pupils and voxel models.
- **D. Preconditioned L-BFGS, then LM to finish:** the Gauss–Newton matrix of the start point (refreshed
  rarely) removes v1's dependence on units; summed loss, bounds by transforms, no ramps. One or two LM
  iterations at the end confirm convergence and give the CRLBs.
- **E. Emitter parameters eliminated:** for each trial pupil the beads' parameters are solved exactly
  (variable projection), so the outer solver sees only the ~90 shared ones.
- **F. A cheaper first stage:** fewer slices, upsampling 1, fewer beads for the n ≤ 4 stage. Synthetic
  data answer whether it costs accuracy.
- Not tried: Adam and other first-order methods (thousands of steps and a learning rate: the settings
  v2 removed).

*Decided after the comparison:* GN-L-BFGS is the default solver of the bead workflows, and slices are
thinned to at most 50 nm (`workflows.beads.MAX_DZ_NM`).

*Sampling in z, for every fit (not only F).* Bead stacks come at 10, 20 or 50 nm. A fit uses slices at
most a set distance apart: the largest spacing the acquisition offers that is not above it (a multiple
of the stack's step), not "every other slice".

*Test matrix.*
- Scenarios: the synthetic bright, dim and very dim beads (6, 20 and 40 beads), n_max 6 and 8; Zenodo
  `1ch_40nm_bead` on v1's 19 ROIs (raw, 21 e⁻); the full 72-bead run.
- Measures: wall time per stage and in all, forward + backward equivalents, the final objective against
  LM's (Δ nats), phase error and z spread against the CRLB, held-out deviance; success over ~10 seeds
  and the unit rescalings (v1's failure).
- Pass: the M1–M3 tests unchanged; the objective within ~0.5 nats of LM's and every parameter within
  0.1 CRLB; no seed or scaling fails.
- Tools: `benchmarks/uipsf/profile_lm.py` (where an iteration's time goes), `benchmarks/uipsf/speed.py`.

**Result (M7).**  The full Zenodo `1ch_40nm_bead` calibration (84 beads found, n_max 8, `learn_beads`
defaults, rejection included) went from 90 min (M3) to **45 s on Apple's GPU (M1 Pro), 113 s on its
CPU and 7.5 min on 4 cloud x86 cores**; v1 took 7.6 min for 17 beads.  What did it, in order of effect:
- *Scoring without a fit*: 55 of the 64 minutes of a calibration were bead scoring, which localised
  every bead and every slice again; beads are now scored from the learning fit's residuals (§5.7).
- *Gauss–Newton L-BFGS* as the default solver (`solve/lbfgs.py`): the Gauss–Newton matrix rebuilt only
  every 10 steps, L-BFGS in between; unit-free, LM's CRLBs; 1.7–12× faster than LM and the better optimum
  in every synthetic case (LM fell into a worse one in 2 of 9).
- *Linearised Jacobians*: the pupil's columns and the beads' (x, y, z as pupil factors) from one DFT
  each instead of JVPs through the model: 2.5× and 4×.
- *On the GPU* (the MPS session): a real, wide matrix DFT, blur and binning as small matrices, tangent
  batches per device, `torch.compile` on MPS.
- *Smaller*: slices thinned to 50 nm, stage 1 stopping at 1 nat.
Tried and kept as options, not defaults: matrix-free Gauss–Newton (`GaussNewtonCG`), localising every
bead before stage 1, the spline localiser (`SplineFrames`, SMAPpy's fitter) and the per-emitter LM
(`BlockLM`).  Tried and dropped: a smaller pupil grid.  Not done: candidate E (variable projection, not
needed for beads); Windows and CUDA (a task of its own).  Left for later (§10, "To investigate"): local
minima (a reproducible case), the blur σ, the aperture edge, how a fit starts, rejection by a fit.

**M8 Physics where the atlas asks for it** (ongoing)
- The aperture edge: on a pixel grid it does not converge in P (64 against 128 samples differs by
  χ² ≈ 60 per bright bead stack, M7); an anti-aliased edge, or the edge as a fitted soft cut-off (also a
  candidate for the pupil magnitude, §10 "To investigate").
- In the order the real benchmark's atlas ranks them:
  - per-slice stage terms and z scale;
  - planar background;
  - saturation mask;
  - polychromatic sum;
  - depth-integrated bead;
  - full Gibson–Lanni and coverslip offset;
  - n_max by held-out deviance;
  - Jones pupil.
- Each is a component with specs and a synthetic test that it recovers its own parameter and removes its atlas signature.

The cheap stage terms are defined as components in M2–M3 already, because the review found them most damaging. M8 decides whether they are on by default; the z scale is the first case (§10, "To investigate").

**Later:** the ideas beyond the milestones -- in situ with several channels, field dependence, 4Pi, excitation PSFs for confocal, ISM and MINFLUX (review §C), SMAPpy fitters on the pupil model, learning without Zernikes -- are collected in `docs/uipsf/roadmap.md`, "Ideas".

---

## 10. Open questions: decisions and what is still open

**Decided:**
- **Repository and name.** The package `uipsf` (the PyPI name is free), moving from ries-lab/uiPSF2 into ries-lab/SMAPpy (§7.4). The move is a snapshot, so uiPSF2's history, which holds the reference code read for ideas and our notes on it, never becomes public; uiPSF2 is archived after it.
- **Output.** One uipsf model file with every Zernike and setting. SMAPpy samples it and makes the spline, normalised as its bead calibration does (§6). No v1 .h5 compatibility; FD-DeepLoc not needed.
- **Spline fitting** stays in SMAPpy. Tests that need it run in SMAPpy's suite.
- **GUI.** In SMAPpy, a calibration window built like its bead calibration and replacing it (§7).
- **Zernike phase sign.** As proposed (§4).
- **Field dependence.** Later, as a whole; the architecture keeps the slot (§3).
- **Optics formulation.** Matrix-DFT Fourier model with vecfit's parametrisation and uiPSF's imaging chain (§5.2).
- **Platforms.** Mac, Windows and Linux, with the GPU where PyTorch gives it easily (§5.1).
- **Stage tilt.** Measured or fitted with no code change: a `FocalSurface` that can be fitted, fixed or given a prior. For beads it is not needed. It arrives with in-situ (M5); the field-dependent focal surface comes later with field dependence.

- **Licence.** BSD-3, as SMAPpy.
- **Benchmark data.** First goal: robust results on the Zenodo example data with default settings, including results that do not depend on chunk or batch size (M2 test). More datasets, including ones that needed tuning in v1, will follow from you.
- **Milestone order.** Multi-channel (M4) before in-situ (M5).
- **In-situ in SMAPpy.** The same calibration window, with a *source* choice: bead stacks or blinking movie.
- **Photons per slice.** Bleaching by default; a shared per-slice factor and a free count per bead and slice as options (§3).

**Still open:**
1. **Coordination with collaborators.** Share the oracle fixtures, synthetic scenarios and `conventions.md` as language-neutral files, so Julia or other PyTorch implementations test against the same numbers. Whether to offer the Li lab a shared core is yours to decide; with the GUI in SMAPpy, sharing test data may be enough.
2. **Licence holder** for the BSD-3 notice: you personally, the Ries lab, the University of Vienna, or "uiPSF contributors" as SMAPpy does.

**To investigate (robustness optimisation, after M3):**
1. **The z scale.** The Zenodo `1ch_40nm_bead` stacks ask for s = 0.968 (2193 nats on 16 beads), with no trace in the residuals, because the pupil absorbs it. A piezo error is unlikely. What acts as a z scale for beads on the coverslip is anything that changes k_z,imm(ρ) proportionally:
   - the effective NA: 0.013 below the nominal 1.43 gives 0.968;
   - the emission wavelength: 3 % redder (~620 instead of 600 nm) gives 0.971, a pure scale;
   - not the immersion oil: n is specified to ±0.0005 at 23 °C, and dn/dT ≈ −3.5…−3.9·10⁻⁴/K, so 10 K changes the scale by only ~0.6 %;
   - not the coverslip thickness or the sample medium, which give spherical aberration or a focus offset only.

   To decide: refits with s fixed at 1, scanning λ, then NA; the one that recovers the likelihood of a free s is the cause. Until then s is fixed (`BeadStackGeometry(z_scale=False)`), fitted on request.

   *Measured further (October 2026, §10, 7): the magnitude makes most of it.*  Simulated beads with s = 1 and a radial magnitude, fitted with the magnitude fixed, ask for s = 0.929 (1.000 with it free).  On Zenodo 1ch / 2ch s is 0.920 / 0.897 with the magnitude fixed, 0.962 / 0.948 free, 0.977 / 0.966 free at order 8, where 1ch gains only 1000 nats from it.  A 4 % piezo error is not needed; what 2ch still asks for (10 600 nats at order 8) is the next thing to explain, with the magnitude we do not yet understand (item 2) first in line.

   *Measured first (October 2026, §10, 7):* with v2's current defaults (beads in water, 1.33) a free s is 0.962 and gains 5300 nats on 43 beads, 3300 on 11 held out.  Neither scan recovers that: NA −0.007…−0.020 at most 2700 (at −0.013), λ +10…+30 nm at most 400; on the held-out beads both ~1000.  So neither is the cause alone.  2ch asks for 0.957 too, but its fit with s free ends in a worse basin than with s fixed.  Simulated beads with s = 1 give 1.000: the free s does not invent a scale.
2. **The pupil magnitude: physical, not only better.** v2 fits it as 1 + Σ c_j Z_j over every Zernike up to n_max, including the non-rotationally-symmetric modes. In v1, fitted magnitudes were often strong and asymmetric, though little in the beam path absorbs asymmetrically, and a flat magnitude (apodisation only) never worked well. The worry is that an incomplete model hands part of the real phase aberrations to the magnitude. The goal is a magnitude explained by physics, with a free remainder that is small and tested.

   **Why might a flat magnitude fail?** Candidates for what the fixed amplitude model gets wrong, each one a radial or a cos 2φ pattern:
   - **the sample medium for beads:** Fresnel transmission at the coverslip and supercritical-angle light depend strongly on n_med, which the defocus barely feels. Beads dried in air (n = 1), in water (1.33) or in mounting medium (~1.4–1.5) give very different amplitudes above NA = n_med. v1's 1.35 may be wrong for these beads. *The Zenodo `1ch_40nm_bead` beads were in water (confirmed): n_med = 1.33–1.335, close to v1's 1.35, so the medium is not the main suspect there;*
   - **the effective NA:** a nominal 1.43 that is lower in practice, or a soft edge (vignetting, coatings losing transmission at high angles) instead of a sharp cut-off;
   - **the apodisation:** √(cos θ_imm)/cos θ_med is the aplanatic factor for a dipole in the medium. Its form and the dipole model (freely rotating, or fixed for beads) are worth checking against vecfit and psf-generator for the bead case;
   - **polarisation-dependent transmission:** a dichroic at 45° transmits s and p differently. For dipole emission that is a real cos 2φ magnitude pattern, so some asymmetry can be physical. It belongs in the model as two transmissions (t_s, t_p), not as free Zernikes;
   - **field dependence:** vignetting off-axis and field-varying aberrations, averaged over beads across the field, can leave an asymmetric "mean" magnitude. On Zenodo the field group was one of the largest residuals (chi² 345).

   **How to find out:**
   - **physical terms as probes:** n_med, an NA edge (position and width), t_s/t_p, a radial transmission polynomial. Each is a likelihood-ratio probe (`diagnostics/probes.py`) against the flat magnitude. Free Zernike magnitudes are allowed only for what the physical terms leave, and only if they win by likelihood ratio on the held-out beads;
   - **trade-off with the phase:** fit with flat, physical and free magnitudes and compare the phase coefficients. A phase change beyond its CRLBs when the magnitude is freed means the magnitude is absorbing aberrations. The covariance between magnitude and phase blocks (the Schur complement) shows which modes trade;
   - **simulations that make asymmetric magnitudes:** simulate each candidate error (wrong n_med, wrong NA, a dichroic's t_s ≠ t_p, a field-dependent aberration, a phase mode the model lacks), fit with a free magnitude, and record which magnitude pattern and which phase bias each one produces. That gives a dictionary for reading a real fit's magnitude;
   - **then real data:** Zenodo beads with the physical amplitude model, compared by held-out deviance and z bias, and with the phase stable across the magnitude options.

   *The dictionary (October 2026, three seeds, n = 6 and 8; `docs/uipsf/architecture.md`):* amplitude errors (NA, a soft edge, the aplanatic factor) give a radial magnitude, and freed it keeps the phase unbiased; only phase modes above n_max gave an asymmetric one; a wrong n_med is the costliest error and shows mostly in σ (83 nm for dried beads fitted as water, (24, 15) for 1.40) and in the phase; a dichroic shows as σx ≠ σy (43 vs 24 nm from 35), not in the magnitude; field-dependent astigmatism shows only in the deviance. Next: t_s/t_p and n_med as probes, then Zenodo.

3. **The extra blur σ: physical, or a sink for mismatch?** The model keeps v1's Gaussian image blur
   (`psf.sigma`, σx and σy, shared, fitted); the bead's size and pixel integration are modelled apart.
   On Zenodo it fits to (67, 75) nm, large for any single cause.
   - **What it could stand for:** motion during the exposure (drift, vibration), charge spread between
     pixels, polychromatic emission (not modelled yet), aberrations above n_max, or mismatch in
     general.
   - **Crosstalk with the magnitude:** a magnitude falling towards the edge widens the PSF as a blur
     does, so σ trades with the rotationally symmetric magnitude modes, and σx ≠ σy with astigmatic ones.
     The Schur covariance between the σ and magnitude blocks shows how much (no new fit needed).
   - **Model tests:** simulate each known cause (a Gaussian jitter, pixel crosstalk, a bandwidth, a soft
     aperture edge, a missing high mode) and fit with σ free and fixed: which σ and which magnitude
     pattern each produces, and whether the phase moves.  Then on real data: σ fixed at a physical
     value (camera and drift measured) against free, by held-out deviance and z bias.
   - *Measured (October 2026, §10, 7):* no fixed value is close (0–40 nm cost 59 000–660 000 nats held out).
     Drift between frames is not it: a free drift per bead and slice (5–9 nm) leaves σ unchanged, and the
     shared per-slice drift (10–17 nm) takes 2–4 nm from it.  Left: motion within the exposure, or mismatch.

   *The lab's beads (October 2026, `docs/uipsf/architecture.md`, "Is the blur the microscope's?"):* with the
   pixel's share taken out, M2 blurs by 40–83 nm and M5 by 0–32, never overlapping, while one microscope's
   acquisitions spread by up to 30 nm; σ is unmoved by a free z scale, larger along x on both, and on M2
   changed with July's camera swap and realignment.  The radial magnitude (twice as large on M5) partly
   absorbs the z scale, which every stack asks for (0.89–0.98).  Next: an NA scan, M5's NA 1.50 first.
   - **The pixel's share** is item 8: at upsample 1 σ also holds a²/12 of the camera pixel.

4. **Local minima.** Both solvers can end in a minimum that is not the best one, and which one they reach
   depends on the path (M7). On synthetic dim beads (seed 2) LM stopped 100 nats up with a phase error of
   0.10 rad, where Gauss–Newton L-BFGS reached the CRLB; M2's two stages exist because one stage stopped
   50 nats up with every bead 50 nm off in z. Not a local minimum but a trap of the parametrisation: a
   softplus background stepped into its flat tail cannot come back (Zenodo, M7; guarded in the line
   search now). To find out:
   - how often: many seeds and scenarios, each fit compared with the best of several starts;
   - what distinguishes the minima: per-bead (z, photons, background) or shared (pupil modes trading);
   - a reproducible case: synthetic very dim beads, seed 1 (`benchmarks/uipsf/synthetic/solvers.py "very dim"
     --seeds 1`).  GN-L-BFGS reached 54114.9 (phase error 0.023 rad, CRLB 0.026) before the MPS merge
     and 54125.7 (0.080 rad) after it: the faster DFT computes the same model with different float32
     rounding, and that alone picks the basin.  Stage 1's tolerance does not (0.001 to 1 nat, all
     54125);
   - on real data (October 2026, §10, 7): on Zenodo 2ch three models that contain the default ended worse
     than it -- order 8 with bleaching of order 2 (+170 000 nats, though each alone gains over 120 000), a
     free z scale (+624 000) and the per-slice photon factor (+27 M, z off by 80 nm) -- and a first stage
     to order 2 put every bead 430 nm off in z.  Every 1ch and simulated fit was consistent;
   - started from the simpler model's solution (`start_values`), every one of those 2ch fits reached the
     expected gain (free s: −43 100 instead of +624 000; order 8 with bleaching 2: −243 500 instead of
     +170 000): freeing terms in stages is the cheapest remedy, and works;
   - remedies by cost: per-bead restarts (cheap: the emitter blocks are independent, a localisation per
     bead from a z grid), a check after each stage that every bead's deviance is in line, better starts
     (a per-bead z from the data), continuation over n_max (the two stages, generalised); only then
     global methods.

5. **The start of a bead fit** (measured October 2026: `benchmarks/uipsf/real/starts.py`, the tables in
   `docs/uipsf/architecture.md`, "How a bead fit starts").  Today: the pupil at zero, each bead at its
   brightest slice, the transform from the pairing's centroids.
   - one channel never needed better: the cold fit reached the optimum on every simulated and real
     dataset, and neither the astigmatism nor each bead's z from elliptical Gaussians (`model/astig.py`)
     made it faster; stage 1 still places beads and low modes together;
   - a previous model of the same microscope: **into stage 1** (`learn_beads(start_stage1=True)`), 0.5–0.8
     of the cold fit's time on M2 and always at the optimum, also from a model across a camera change;
     replacing stage 1, as *start from* does now, was slower from a model 0.26 rad off and once ended in a
     wrong minimum.  Decided and built: *start from* takes it into stage 1.  A 665LP model started the 676/37
     acquisitions as well as their own: no wavelength rescaling needed;
   - two channels: the transform matters.  A start 2.5 px off fails on mirrored pairs, 4.5 px off always;
     the Gaussians' positions start 2–3 × closer than the centroids (0.11–0.25 px) and every fit from
     them converged; on Zenodo 2ch, where the cold fit stalls in stage 1 (roadmap 4), they reached the
     optimum to ~100 nats.  Decided and built: the pairing's transform refitted from the Gaussians' positions
     (`calibrate.gaussian_transforms`, the pairing's kept when the beads do not fix it);
   - an angle for SMAPpy's elliptical Gaussian is not needed: the astigmatism of the lab's and Zenodo's
     data is Noll 6 to within 0.02 rad;
   - the stage-1 stall on Zenodo 2ch was not the transform's: the start's backgrounds at softplus' floor,
     a saturation guard that shortened every step for them and a wrong gradient from `torch.polar`
     (roadmap 4, fixed); after the fix every start reaches one optimum, 52 000 nats below the one above.

6. **Rejection by a fit?**  Since M7 bead rejection uses the learning fit's residuals only (deviance,
   photons, xy offset, saturation); the z bias (each slice localised alone, v1's ``msez``) is gone: on
   Zenodo it rejected 2 of 27 beads on its own.  When looking at the model, test whether a
   localisation-based score rejects better.  Per-slice consistency (a stage jump) shows in SMAPpy,
   which fits every bead with the exported model's spline.

7. **The form's choices and the hidden defaults: measured, to decide** (October 2026,
   `benchmarks/uipsf/real/model_choices.py`, the table in `docs/uipsf/architecture.md`).  One calibration
   per choice on the same ROIs and the same beads, judged on 20 % held-out beads (deviance, and the pupil's
   z bias) and by Spline 3D on the exported spline; Zenodo 1ch, 2ch, and simulated beads with a known pupil.
   Nothing is changed yet; per choice, a recommendation:
   - *Zernike order*: **fix the default (auto) at 8.**  Held out, −10 000 nats (1ch) and −69 000 (2ch), z
     bias 5.4 → 3.4 and 3.3 → 2.6 nm, the same time; on simulated beads whose truth stops at 6 it costs
     4–6 nats.  Keep the field, under *more*, for a microscope that needs more.  This replaces "n_max = 6
     with a held-out check of 8" below.
   - *pupil magnitude*: **keep free as the default; drop *fixed*** (z bias 21–30 nm on every dataset, the
     worst choice measured).  **Drop *symmetric* from the form too**: it loses 29 000 / 216 000 nats to
     the free magnitude even at order 8, on held-out beads as well, and gains nothing on the simulation.
     `magnitude_modes` keeps it for studies such as §10, 2.  With one choice left, the field goes.
   - *blur*: **drop the choice; σ stays free.**  Fixing it at 0, 20 or 40 nm costs 59 000–660 000 nats
     held out and raises the z bias to 12–20 nm; free, it recovers a simulated 30 nm.  *blur σ* then is only
     a start and can go under *more* or out.  What the 65–75 nm are stays §10, 3.
   - *z scale*: **keep fixed at 1, not in the form**, as now; s stays a probe.  It is mostly the pupil's
     amplitude, not the stage (item 1): a missing magnitude asks for s = 0.93 on simulated beads with
     s = 1, and on Zenodo s goes 0.92 → 0.96 → 0.98 as the magnitude is fixed, free, free at order 8.
   - *photons per slice* (not in the form): **bleaching of order 2 as the default, learnt after the
     order-6 bleaching-1 fit.**  −27 000 / −29 000 nats held out, z bias 3.9 / 2.2 nm, no cost on simulated
     beads.  With order 8, fresh, 2ch fell into a worse basin; started from the base fit it is the best
     fit measured (−91 600 held out, z bias 1.9 nm).  So: a third stage, or richer terms freed after
     stage 2.  Do not offer the per-slice factor (overfits 1ch, wrecks 2ch); `PerObservation` cannot run
     in `learn_beads` (`score_beads` reads `emitter.photons`): fix or drop it.
   - *stage drift* (not in the form): **the shared drift, freed after stage 2, is worth having**: from the
     base fit it gains 4 800 / 53 300 nats held out and gives z bias 2.9 / 1.9 nm, with 10–17 nm per
     slice.  Fresh it ended in worse basins.  It takes only 2–4 nm from σ, and a drift per bead none:
     drift between frames is not the 65–75 nm blur (§10, 3).  Per file, below.
   - *first stage* (not in the form): **keep order 4.**  No stage 1 gives the same fit to the nat on all four
     datasets, order 2 is a trap on 2ch; the stage costs little and guards dim beads (item 4).
   - *slice thinning* (not in the form): **`MAX_DZ_NM` 100 nm.**  Held out −7 / +340 nats of 0.23 / 0.92 M,
     z bias unchanged, 1.6× faster on 2ch; 200 nm costs 120 / 1550 nats and is 2.6× faster, worth trying on
     the lab's 10–20 nm stacks before going that far.
   - **to do (asked October 2026): the stage drift per slice *and per file*.**  `BeadStackGeometry`'s drift is
     one shift per slice shared by every bead (`geometry.drift`, [K, 3]), but beads from different files
     are different acquisitions with their own drift (Zenodo 1ch: ~18 files).  Index it by
     `Batch.stack_id`: [files, K, 3], the 5 nm prior per file.  As shared parameters that is 1440 for
     Zenodo 1ch, more than MPS holds in one Jacobian (`model_choices.py` failed so); a block per file,
     solved as the emitter blocks are, avoids it.  The study's per-bead drift (`BeadDrift`) is the stand-in.
   - not measured: rejection's k, the held-out fraction, the bead diameter, upsampling.  To be repeated on
     the lab's own stacks: `model_choices.py "<glob>" --NA … --wavelength …` (one argument per dataset).

8. **The camera pixel: sampled or integrated?** (planned October 2026, after M6; nothing run yet beyond
   the noise-free checks below.)  Bead learning uses `VectorPSF(upsample=1)`, as v1 does for beads (its
   `config_base.yaml` says `bin: 2`, but `channeltype/1ch.yaml`, `2ch.yaml`, `4pi.yaml`, `MF.yaml` set
   `bin: 1`; v1 blurs the upsampled grid and sums `bin × bin` blocks by a strided convolution, the same as
   v2's `bin_pixels`).  On integrated pixels the learnt σ takes the pixel's width (35 → 47.3 nm,
   √(35² + 110²/12) = 47.4; M6).  The questions: what upsample computes, whether that explains σ, and what
   it does to the magnitude, the phase, the localisation and the cost.

   **What the code computes (read, and checked numerically).**
   - The matrix DFT evaluates the image field at chosen points, the emitter placed by the matrices'
     phase (`MatrixDFT.real_matrices`), so it is exact at any sub-pixel position.  *upsample 1*: the
     intensity at pixel centres, then the Gaussian blur as a circulant on those samples
     (`blur_bin_matrices`).  No pixel average of any kind.
   - *upsample u*: u × u samples per pixel, centred on it, placed by the same DFT, blurred on the fine
     grid, summed (`bin_pixels`): the midpoint rule for the pixel's integral.  Its kernel is u points
     a/u apart, of variance a²(u² − 1)/(12u²); the true box's a²/12 is missed by a²/(12u²).
   - *The proposal* (the PSF on a 3× grid, shifted to the emitter, binned) **is upsample 3** when the
     shift is exact (a phase, as here, or an FFT shift on an alias-free grid).  Shifting a precomputed
     grid by interpolation is an approximation that upsample avoids.
   - *A Fourier pixel*: the box is a sinc in the image's frequencies, sinc(π q a), and folds into the
     blur's transfer function as the bead kernel does, exact and at no cost **if the samples do not
     alias**.  The intensity reaches |q| = 2NA/λ, so the spacing must be below λ/(4NA): 119 nm at 680 nm,
     105 nm at 600 nm (Zenodo's 127 × 116 nm pixels alias at upsample 1).  Where it aliases: upsample u
     with the sub-box sinc(π q a/u) and then the u-sum, which is exactly the full box (sinc(2t) =
     sinc(t) cos t).  So upsample 2 is exact for every case here.

   **Noise-free numbers** (λ 680 / 110 nm and Zenodo's 600 nm / 127 × 116 nm, astigmatic, σ 35 nm, 30
   positions ±½ px, ±600 nm; the scratch script becomes E0 below):

   | model of the pixel | max error / peak, 110 nm \| Zenodo | nats vs exact at 5000 photons, bg 20 |
   |---|---|---|
   | centres, σ 35 (pixel missing) | 4.1 % \| 6.4 % | 25 \| 64 |
   | centres, σ free (best: 47.5 \| (51, 49)) | 0.08 % \| 0.12 % | 0.03 \| 0.12 |
   | u = 2 / 3 / 4 summed, σ 35 | 1.0 / 0.4 / 0.2 % \| 1.5 / 0.6 / 0.3 % | 1.4 / 0.3 / 0.1 \| 3.4 / 0.7 / 0.2 |
   | u = 1, Fourier sinc | 0.07 % \| 0.06 % | 0.02 \| 0.09 (aliasing) |
   | u = 2, sub-box sinc, summed | ≤ 0.04 % (the reference's own error) | 0.00 |

   The pixel's box and a Gaussian of the same variance differ by ≤ 0.1 nats per 30 images: on beads a
   free σ absorbs the pixel almost exactly, so learning cannot tell them apart.  CPU forward (20 beads ×
   41 slices, 25 px, n_max 8): 79, 132, 214, 303 ms for u = 1–4.

   **Hypotheses.**
   - H1. With σ free, the pixel model changes σ only: σ² = σ_optics² + a²/(12u²) for u-summing, σ_optics
     with an exact pixel; phase, magnitude, z and the deviance move by less than their CRLBs and a few nats.
   - H2. With σ fixed (*blur fixed*, 30 nm by default), a missing pixel costs 25–64 nats per 30 images
     and goes into the magnitude as a radial falloff (narrower OTF) and into the spherical phase.  The
     magnitude dictionary (*What a free pupil magnitude absorbs*) is unaffected: its truth and its fit
     both sampled pixel centres.
   - H3. Zenodo 1ch's (68, 75) nm are (57, 67) nm of optics and other blur (127 and 116 nm pixels): the
     pixel explains about a third of σ², not the size, nor σx ≠ σy (the pixel widens x more, σ is wider in y).
   - H4. The z slope of 1.011 against the stage does not depend on the pixel model (the pixel is lateral).
   - H5. The localisation follows the model: the model file stores `upsample` and the export samples at
     it (`pupil.UPSAMPLE = None`), so Spline 3D and `slice_fits` keep agreeing whichever pixel model is learnt.

   **Experiments** (proposed; which ones to run is to be agreed, see below):
   - E0 `benchmarks/uipsf/synthetic/pixels.py --check`: the table above, kept, plus `pupil_linearisation`
     time and peak memory for u = 1–4 and the Fourier pixel, on the CPU and MPS.
   - E1 the same script, learning: the magnitude benchmark's scenario (12 beads, 31 slices ±750 nm, 17 × 17
     px of 127 × 116 nm, λ 680, 5000 photons, bg 20, Noll 6, 7, 11 = 0.5, 0.15, 0.2 rad), the truth's
     pixels integrated exactly (u = 2 with the sub-box sinc), σ 35, magnitude flat and Noll 4 = −0.15;
     learnt with u = 1, 2, 4 and the exact pixel, σ free; seeds 0, 1, 2.  Records σ, phase and magnitude
     bias / CRLB, the magnitude's radial share, deviance against the truth's, z bias per bead, time,
     memory.  Decides H1: σ against the formula to 0.5 nm, everything else within 1 CRLB.
   - E2 the same, σ fixed at 35 and at the default 30, magnitude free / symmetric / fixed, u = 1 against
     exact: a *pixels missing* row for the magnitude dictionary.  Decides H2 and whether *blur fixed* can
     be offered without an exact pixel.
   - E3 `benchmarks/uipsf/real/zenodo_pixels.py`: Zenodo 1ch and 2ch, `calibrate_beads` defaults, u = 1
     against the exact pixel (u = 2 + sinc); u = 3, 4 only if those two disagree beyond H1.  Records σ,
     Zernikes against u = 1 in CRLBs, the radial share, held-out deviance, the atlas's groups, the z slope
     against the stage, wall time and memory on MPS.  Decides H3, H4 and the default.
   - E4 the M6 checks on each E3 model: Spline 3D against `slice_fits` (rms, slope, photon ratio), and the
     z of the u = 1 and exact models' splines on the same ROIs.  Decides H5;
     `tests/uipsf_tests/test_pupil_spline.py` gains a case learnt with the exact pixel.

   **E1, first run** (`benchmarks/uipsf/synthetic/pixels.py`, seed 0; `results/pixels.jsonl`): a point
   source with no blur, its pixels integrated 16 × 16 (127 × 116 nm, λ 680 nm), 12 beads × 31 slices, σ free.

   | learnt with | σ at 10⁵ photons (predicted) | nats over the best, 10⁵ \| 10⁴ \| 10³ | Spline 3D on pixelated data, rms x, y, z (nm) |
   |---|---|---|---|
   | pixel centres (u1) | (36.9, 33.3) (36.7, 33.5) | 311 \| 11 \| 0 | 0.18, 0.15, 0.63 |
   | u2 summed | (17.9, 16.1) (18.3, 16.7) | 30 \| 1 \| 1 | 0.18, 0.14, 0.56 |
   | u4 summed | (8.0, 7.2) (9.2, 8.4) | 0 \| 0 \| 1 | 0.18, 0.14, 0.54 |
   | u2 + sub-pixel sinc | (0.1, 0.0) (0, 0) | 35 \| 1 \| 2 | 0.18, 0.14, 0.55 |
   | the truth's own spline | | | 0.16, 0.16, 0.58 |

   - σ is the pixel's share and nothing else; with no blur in the truth there is no other blur to find.
   - Phase and magnitude do not move: phase rms error 0.0004–0.0005 rad and magnitude rms ≤ 0.0004 in
     every model at 10⁵ photons.  The Gaussian surrogate is detectable only by its deviance, from about
     10⁴ photons per slice: 311 nats at 10⁵.
   - The sub-pixel sinc is 35 nats behind u4 at 10⁵, probably because its long tails wrap around the
     region (circular).  u1 plus the sinc is no better than u1, because 127 nm aliases at 680 nm.
   - **Spline 3D does not care.** Each model's spline, sampled at its own setting, fits noise-free
     pixelated images (400 positions, ±½ px, ±600 nm) as well as the truth's own spline does: z slope
     0.9995–0.9998 for models learnt at 10⁵ photons.  A pixelated PSF that the spline shifts fits
     pixelated data to well under 1 nm.
   - With brightness, σ (u1) is (36.9, 33.3) at 10⁵, (38.2, 33.6) ± 0.6 at 10⁴ and (39.2, 29.4) ± 3 at
     10³.  There is no clear trend, and this is one seed.

   **Field dependence as a source of σ** (`pixels.py --field NOLL SD`, `results/pixels_field.jsonl`; seed
   0, 2·10⁴ photons, no blur in the truth, pixels integrated): one mode varying from bead to bead (normal,
   sd in rad), learnt with u4.  The pixel's share alone is (7, 6) nm.

   | varying mode, sd | σ learnt (nm) | deviance per pixel | magnitude rms, radial share |
   |---|---|---|---|
   | none | (7, 6) | 1.00 | 0.0007, — |
   | astigmatism (Noll 6), 0.1 \| 0.2 | (14, 11) \| (19, 20) | 1.24 \| 1.90 | 0.001 \| 0.003, 0.68 |
   | coma (Noll 7), 0.1 \| 0.2 | (11, 18) \| (19, 31) | 1.37 \| 2.37 | 0.002 \| 0.009, 0.53 |
   | spherical (Noll 11), 0.1 \| 0.2 | (10, 11) \| (21, 18) | 1.18 \| 1.67 | 0.001 \| 0.004, 0.57 |

   Field dependence does go into σ, and coma does so along its own axis.  It barely goes into the
   magnitude, and the mean phase stays within 0.003 rad.  But it costs far more deviance than blur: by the
   time it adds 15–30 nm of σ, the deviance per pixel is 1.7–2.4.  Zenodo 1ch has (57, 67) nm of
   non-pixel blur at a deviance of 1.38.  On its own, field dependence is not enough there, unless
   Zenodo's photon counts make the deviance far less sensitive (to check).

   **σ across the Zenodo sets** (`benchmarks/uipsf/real/zenodo_sigma.py`, `results/zenodo_sigma.jsonl`;
   `calibrate_beads` defaults, n_max 6, pixel centres; "σ − pixel" is √(σ² − a²/12)):

   | set | camera, pixels | beads used | magnitude free: σ − pixel (nm) | magnitude rms (radial share) | deviance, free \| fixed |
   |---|---|---|---|---|---|
   | 1ch 40 nm, 600 nm | Evolve EMCCD, 116 × 127 | 52 | (56, 63) | 0.025 (0.92) | 1.45 \| 1.87 |
   | 2ch 40 nm, 680 nm, main | the same chip | 71 | (68, 49) | 0.031 (0.88) | 1.73 \| 3.23 |
   | 2ch, secondary | | | (48, 65) | 0.031 (0.95) | |
   | FD, centre (r ≤ 400 px), NA 1.5 | Prime 95B sCMOS, 110 | 15 | (61, 85) | 0.055 (0.93) | 1.76 |
   | FD, whole field (30 brightest) | | 28 | (77, 105) | 0.102 (0.89) | 5.70 |

   - 2ch used v1's indices (1.516 / 1.35): with 1.518 / 1.33, the window's defaults, its fit stalls at
     the start (deviance 20.8).  Not the indices' doing: roadmap 4, fixed (architecture.md, "Why Zenodo
     2ch stalled"), after which 1.33 fits better than 1.35.
   - FD: one stack of two, ±600 nm (slices 41–160, every 5th), since its 128-sample pupil at ±1 µm needed
     more than 9 GB.  The FD beads are fitted as points (v1's bead radius 0; their size is not recorded).

   The blur is large on both microscopes, both cameras and both wavelengths.  The whole FD field adds
   15–20 nm of σ, doubles the magnitude and raises the deviance from 1.8 to 5.7: field dependence, as the
   synthetic test predicts, but on top of a central σ already at 60–85 nm.  The two 2ch channels
   share a chip but are widened along different axes (x in the main, y in the secondary): that part of σ
   comes from each channel's own path after the splitter, not from the camera.  Fixing the magnitude
   lowers σ by up to 11 nm and costs 0.4–1.5 in deviance per pixel.

   **Not stage wander** (`zenodo_sigma.py --wander`, `results/zenodo_wander.jsonl`): every slice of every
   used bead was localised alone on the learnt pupil (`scoring.slice_fits`) and compared with where the
   stack fit puts it.
   - The common lateral shift per slice (the median over a stack's beads, its mean removed, less the
     median's noise) is 4.6, 4.4 nm (1ch, 19 stacks), 4.0, 3.9 (2ch, 14) and 3.6, 2.0 (FD centre, one stack).
   - Common and per-bead together, the residuals exceed the CRLB by 6–7 nm (1ch, 2ch) and 11–12 nm (FD).
   - Against a σ of 50–85 nm beyond the pixel, wander between z steps is excluded.  Vibration within
     one exposure is not measured this way.

   **Decisions.**
   - D1 (after E1, E3): if H1 holds, the pixel model is a convention on σ.  Then the exact pixel by
     default, so that σ means the optics, motion and the camera's crosstalk and can be compared across
     cameras, if its cost on Zenodo is below ~1.5× (u = 1 + sinc where λ/(4NA) allows, u = 2 + sinc
     otherwise); else keep u = 1 and report σ_optics = √(σ² − a²/12) beside σ.
   - D2 (after E2): *blur fixed* is physical only with the pixel integrated; it is offered with the exact
     pixel only, or removed.
   - D3: the model file records the pixel model; files written before keep meaning pixel centres.  The
     export needs no change while it samples at the model's own setting.  In-situ (M5) and `localise`
     use the same `VectorPSF`, so they follow.

**Small technical defaults, unless you object:**
- Negative shifted data (`d + s < 0`) clamped at 0 and counted, rather than the slower exact Poisson–Gaussian likelihood.
- n_max = 6 for beads with a held-out check of 8; pupil size chosen automatically (§5.2).
- No GPU in CI; the gpu tier runs by hand on a Mac and a CUDA machine before releases.

## 11. First milestone (M0 + M1): task list

Goal: a validated, differentiable forward model on all devices, in SMAPpy's conventions, and the parameter system the solver will need.

1. **Scaffold.** BSD-3 `LICENSE`, `pyproject.toml` (torch ≥ 2.4, numpy, scipy, h5py, tifffile, omegaconf; extras `test`, `dev`), `src/uipsf/`, ruff, pytest with `--slow` and `--gpu`, GitHub Actions on CPU, `CLAUDE.md` with the working rules (pull first, test tiers, page and conventions upkeep).
2. **`device.py`:** device choice, dtypes, `to_cpu64`; a test that `UIPSF_DEVICE` overrides.
3. **`docs/uipsf/conventions.md`:** §4 written out, each point with the SMAPpy line that defines it, and the mapping to v1 uiPSF and vecfit (axis swap, conjugation, 180° rotation; review §D). Written before the code that depends on it.
4. **`optics/system.py`:** `Optics`, with the automatic pupil size (defocus phase step ≤ π per pupil pixel over the z range; the ROI fits the DFT output). Test it on the review's example (NA 1.43, |z| ≈ 2.5–3 µm).
5. **`optics/grid.py`:** `PupilGrid` (kx, ky; kz for immersion and medium with a stated evanescent branch; aperture; Fresnel T_p, T_s; apodisation √cos θ_imm / cos θ_med). Tests: evanescent branch, T at normal incidence, non-square pixels.
6. **`optics/zernike.py`:** Noll modes, basis, exclusion of piston/tilt/defocus, `high_na` orthogonalisation against kz (Booth 1998). Tests: orthonormality; v1's `zernike_polynomial` from the Zenodo result file reproduced after the convention mapping.
7. **`optics/mdft.py`:** `MatrixDFT` with upsampling. Tests: equals a zero-padded FFT; zero-based pixel-centre output; gradients.
8. **`optics/vectorial.py`:** 6 dipole pupil fields, Ex/Ey analysers. Test against the vecfit port's pupil matrix.
9. **`optics/kernels.py`:** bead OTF (J₃/₂ sphere), Gaussian blur (σ in pixels, x and y pixel sizes kept apart), pixel integration. Tests: normalisation, σ meaning, non-square pixels.
10. **`params.py`:** `ParamSpec`, `Identity`/`Softplus`/`Exp`/`Bounded`, `Gaussian` prior, `Params` packing into shared and emitter blocks. Tests: round trip, invertible transforms, block layout.
11. **`model/pupil.py`:** `ZernikePupil` (phase, magnitude, unit-energy normalisation), `PixelPupil`.
12. **`model/psf.py`:** `VectorPSF.__call__(p, xyz_nm[E,K,3]) -> I[E,K,Y,X]`: phase ramps, defocus, MDFT, |·|² over fields, blur, bead kernel, binning.
13. **Oracles.**
    - The vecfit port goes into `tests/uipsf_tests/oracles/vecfit.py`.
    - `scripts/uipsf/make_oracles_v1.py` runs in a TF environment with uiPSF at `test_roi` b367f00, as the review did, over a fixed set of cases (aberrations, bin, blur, bead radius, non-square pixels).
    - The npz fixtures are committed (< 5 MB).
    - psf-generator (EPFL, MIT licence) as an optional third oracle for the unaberrated and Gibson–Lanni cases; it skips when not installed.
14. **SMAPpy z-sign and Zernike-sign check.** Done without SMAPpy:
    - the v1 oracles with emitters off focus prove `z_v2 = −z_v1`, and SMAPpy's uiPSF bridge takes v1's stack order as its own;
    - the Zernike sign is pinned by a test on the image (positive Noll 6 stretches along y above focus).

    A SMAPpy simulator would only test its own arbitrary choice of astigmatism.
15. **Tests.** Oracle agreement (≤ 1e-6 of peak vs vecfit, ≤ 1e-5 vs TF after the convention mapping), invariants (M1 above), `gradcheck`, the emitter-independence contract, and the gpu tier.
16. **Speed baseline.** `benchmarks/uipsf/speed.py`: forward + backward for 6/20/40 beads × 41 slices, on CPU/MPS/CUDA, written as JSON lines.
17. **Write-up.** `docs/uipsf/architecture.md` with the APIs as built, and this plan updated where the code changed it.

**Result (M1).**
- **Against v1:** 8 of 11 cases agree to ≤ 3·10⁻⁷ of the peak. The other three differ only where v2 departs on purpose (the even-region blur, and bead wrap-around at the stack ends), and match exactly once v1's behaviour is reproduced in the test.
- **Against vecfit:** agreement to 5·10⁻¹⁰ (float64).
- **Speed:** 0.09 s forward + backward for 6 beads × 41 slices on 4 CPU cores.
- **GPU:** the gpu tier passes on an Apple-silicon Mac (MPS is 3.5× its CPU). CUDA is still to run.
- **Not added:** psf-generator as a third oracle.

**Done when:**
- all oracle tests pass on the CPU;
- the gpu tier passes on a Mac (MPS) and on a CUDA machine, Windows if one is at hand, and CPU results agree on all three systems;
- the speed baseline is recorded;
- a reader can follow `optics/` and `model/psf.py` against Suppl. Note 3 of the paper without a debugger.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| The shared-block JVPs (~100 per chunk) make an LM iteration slow | Measured in M2: ~0.12 s per JVP for 6 beads × 41 slices on 4 CPU cores, ~8 s per iteration, 15–30 iterations per fit. M7 targets it |
| The Schur system is ill-conditioned (remaining near-degeneracies) | Degenerate modes removed by construction; condition number reported; small physical-unit priors as a last resort |
| Oracle disagreements turn out to be conventions, not bugs | `conventions.md` first; fixtures record v1's settings exactly |
| The exported spline loses what the pupil model gains (interpolation, z step) | The §6.5 check, with Spline 3D and `localize` on the same ROIs; z step chosen from it |
| Real data still need tuning | Then it is model mismatch: the atlas shows it, and M8 adds the physics |
