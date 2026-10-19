# Reference data: the uiPSF paper's example data on Zenodo

**Record.** "Example data for uiPSF", doi:[10.5281/zenodo.8267520](https://doi.org/10.5281/zenodo.8267520). This is the concept DOI; the latest version is record 10602740, from 2024-02-01. It is the data used by the demo notebooks of Liu et al., *Nat. Methods* 2024.

**Format.** The record is one zip of 9.35 GB. Members are stored or compressed with Deflate64. There are no separate per-dataset downloads.

**Fetching.** `scripts/uipsf/fetch_zenodo.py` reads single folders out of the zip by HTTP range requests:
- it needs `pip install zipfile-deflate64`;
- it writes to `$UIPSF_DATA` (default `~/.cache/uipsf/zenodo`);
- it skips files already complete, so an interrupted run can be repeated.

This was tested in October 2026 from a cloud container: `1ch_40nm_bead`, `2ch_40nm_bead` and their result files (about 400 MB) came down in a few minutes. Transfers were occasionally cut off; the script retries.

On macOS with a recent SDK, `zipfile-deflate64` builds only with `CFLAGS="-Dfdopen=fdopen" pip install zipfile-deflate64` (its bundled zlib redefines `fdopen`).

**Never commit the data.** Tests that need them skip when `$UIPSF_DATA` lacks them.

## Contents

| Folder | Size | Files | What it is | Use in v2 |
|---|---|---|---|---|
| `1ch_40nm_bead/` | 0.18 GB | 31 stacks, 41 × 214 × 214 (uint16), 50 nm steps | Single-channel astigmatic beads, M2 microscope (pixels 127 × 116 nm) | M3 end-to-end, ROI-finder regression, M2-style benchmark on real data |
| `2ch_40nm_bead/` | 0.16 GB | 14 stacks, 41 × 512 × 236, two channels up–down on one camera | Dual-channel (ratiometric) TetraSpeck beads | M4 multi-channel and pairing |
| `1ch_FD_bead/` | 2.1 GB | 2 stacks, 1608 × 1608 FOV, 10 nm steps over 2 µm | Beads over a large FOV (Prime 95B) | M4 field dependence |
| `1ch_LLS_bead/` | 1.7 GB | 3 tiffs | Lattice light-sheet beads (voxel model) | Voxel baseline |
| `4pi_40nm_bead/` | 0.18 GB | 26 beads | 4Pi bead stacks | Later (4Pi) |
| `1ch_smlm_TP/` | 0.23 GB | 10 × .mat | In-situ Tetrapod (pupil model) | M5, smallest in-situ set |
| `1ch_smlm_tubulin/` | 1.8 GB | 1 ome.tif | In-situ, astigmatism at 1.5 µm depth | M5 (index mismatch) |
| `1ch_FD_smlm_nup96/` | 4.3 GB | 1 ome.tif + recommended parameters (png) | In-situ field-dependent, large FOV | Later (in-situ FD) |
| `2ch_smlm_nup96_WGA/` | 4.3 GB | 1 ome.tif | In-situ, two channels | Later (in-situ multi-channel) |
| `4pi_smlm_nup96/` | 0.47 GB | | In-situ 4Pi | Later |
| `results/<set>/` | 0.84 GB | one or more `psfmodel_*.h5` per set | v1 outputs for every set above | Regression references and the v1 export format |

## The v1 result files

Checked on `results/1ch_40nm_bead/psfmodel_zernike_vector_single.h5` and `results/2ch_40nm_bead/psfmodel_zernike_vector_multi.h5`. They hold everything a v2 regression needs:

- **`rois/psf_data`** (19 beads × 39 slices × 25 × 25 for the single channel) and **`rois/psf_fit`**. These are the exact ROIs v1 fitted, after its bead selection. v2 can fit the same ROIs, separating solver comparisons from ROI-finding differences. `rois/cor`, `rois/fileID` and `rois/image_size` say where the ROIs came from.
- **`res/`**: the learned model.
  - `zernike_coeff` (2 × 45: magnitude and phase, Noll up to n_max = 8), `zernike_polynomial` (45 × 64 × 64), `pupil` and `apodization` (64 × 64 complex).
  - `sigma` (blur), `I_model` (43 × 25 × 25), `I_model_bead`, `I_model_reverse`.
  - Per bead: `pos`, `intensity` (per slice), `bg`, `drift_rate`.
  - For multi-channel: `channel<i>/…`, plus `T` (3 × 3 affine), `xyshift` and `imgcenter`.
- **`locres/`**: `coeff` (cspline, 64 × 42 × 24 × 24), the localisations of the training beads with this model (`loc/x,y,z`), `P`, `CRLB`, `LL`.
- **The `params` attribute**: the full v1 configuration as JSON. For `1ch_40nm_bead` it gives:
  - NA 1.43, λ 0.6 µm, RI immersion/medium/coverslip 1.516/1.35/1.516;
  - gain 0.22, offset 400;
  - pixel size x 0.127 µm, y 0.116 µm, z 0.05 µm;
  - pupil size 64, n_max 8, blur σ 0.5 (fitted), `var_photon: true`, bead radius 0.025 µm;
  - ROI 25 × 25.

  The 2-channel set uses λ 0.68 µm and gain 1.

v2 does not write this format (plan §10.2); the files are read only, as benchmark references.

## Limitations

- These sets were chosen because v1 handles them. They will not show the tuning problem the review describes. The real benchmark needs datasets that currently need tuning (plan §9, question 5).
- The camera is described by one gain and offset; there are no per-pixel maps.
- **v1's gain for `1ch_40nm_bead` (0.22) is about the inverse of the camera's conversion.** SMAPpy's camera database identifies the stacks' camera from their Micro-Manager tags as the M2 microscope's Evolve 512 in its normal-port mode, with 4.7 e⁻/ADU and an offset of 398.6. Use SMAPpy's values (`smappy.io.tiff.resolve_camera`; standalone: `benchmarks/uipsf/real/zenodo_camera.py`); v1's ROIs in `rois/psf_data` are in v1's units, ~21× too few photons.
- The file's own pixel size is 0.127 µm in one number; the 127 × 116 nm come from v1's settings, in v1's transposed frame. **In the camera's frame the pixels are 116 nm in x and 127 nm in y**: fitted both ways, the deviance prefers it by ~6000 nats over 19 beads (docs/uipsf/architecture.md, M3). SMAPpy's entry should read `[0.116, 0.127]` (x, y).
- **Sample medium:** the beads were in water (n ≈ 1.33; v1's settings use 1.35).
- **Read noise:** about 4.5 ADU, 21 e⁻ rms at 4.7 e⁻/ADU (the residual variance at zero signal). Fit with `read_noise_e=21`; SMAPpy's default of 1 e⁻ gives a mean deviance of 12.7 instead of 1.4.
- **How v1's ROIs relate to the stacks, exactly** (to 10⁻⁴, all 19 beads): `rois/psf_data[e]` = transpose((ADU[1:40, y−12:y+13, x−12:x+13] − 389) · 0.22) with (x, y) = `rois/cor[e]` and `rois/fileID` indexing the stacks sorted by name. So:
  - v1's arrays are the camera's transposed (`swapxy: true`): v1's x is the camera's y. Its Zernikes are in that frame;
  - the offset is 400 minus v1's shift of the data by their 0.1 % quantile (11 ADU), so no pixel is negative; v2 does not shift (plan §5.5);
  - slices 1–39 of 41.
  `benchmarks/uipsf/real/zenodo_1ch_beads.py --source raw` cuts the same beads from the stacks in the camera's frame and SMAPpy's photons.
- The 1ch beads have non-square pixels (127 × 116 nm). That is useful as a test, because v1's bead kernel swaps x and y for non-square pixels (review F1, minor).
