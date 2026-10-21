# Conventions

Axes, signs, units and normalisation, as the code implements them. SMAPpy's conventions are uipsf's (plan §4). Every statement here is checked by a test, named in brackets.

## Units and axes

- **Lengths** are nanometres everywhere: positions, pixel sizes, wavelength, blur, bead radius, depth.
- **Arrays** are `[..., Y, X]`: rows are y, columns are x. Positions are written `(x, y, z)`, with x the column, as in SMAPpy (`csrc/models.hpp`: "`ix` is the ROI column and `iy` the row").
- **Pixel sizes** are given as `(x, y)` and may differ: the paper's M2 data have 127 × 116 nm.

## Pixels and regions

- Pixel *i* covers [i − ½, i + ½]: coordinates are zero-based pixel centres, as in SMAPpy.
- **Image samples.** A region of X samples has them at `x_m = (m − (X−1)/2)·Δx` about its centre. An odd region has a sample on its centre, so an emitter at x = 0 sits on the centre pixel. [`test_mdft.py::test_an_odd_region_has_its_centre_sample_on_the_emitter`]
- **Pixel integration.** With `upsample = u`, each camera pixel is sampled u × u times at `Δx/u` spacing, on the same centred grid, and the samples are summed: uiPSF's `bin`.

## Positions and the sign of z

- **x, y** are relative to the region's centre, in nm.
- **z is the emitter's height above the focal plane.** Positive z is further from the objective, deeper into the sample.
- **In a bead stack, z = −(objective position − focus).** Stacks are ordered by increasing objective position, so z decreases along the stack. This is SMAPpy's convention:
  - `SplineCalibration.z_index_to_nm(k) = −(k − z0)·dz`;
  - "a bead is below the focus by as much as the objective was raised" (Spline 3D page).
- **In-situ depth** (`depth_nm`) is the emitter's depth in the sample above the coverslip. It adds the medium's axial phase, `k_z,med · depth`, to the stage defocus, `k_z,imm · z`. Light above the critical angle decays with depth. [`test_psf.py::test_depth_in_the_sample_loses_supercritical_light`]
- **In situ, an emitter's z is its height h above the coverslip, and the focus f is the shared `geometry.focus`:** how far the objective moved its nominal focal plane up from the coverslip's focus (uiPSF v1's `stage_pos`). `InSituGeometry` hands the PSF z = −f (through `k_z,imm`) and depth = h (through `k_z,med`), so the defocus wave is `exp(2πi (k_z,med h − k_z,imm f))`. With matched indices this is a bead at z = h − f. [`test_insitu.py::test_with_matched_indices_an_emitter_in_the_sample_is_a_bead_at_its_height_above_the_focus`]
- **The twin** of a pupil, `A*(−k)`, images the mirrored z: phase Zernikes `c_j → −(−1)^m c_j`, magnitude `c_j → (−1)^m c_j` (astigmatism and spherical flip, coma does not). Exact for scalar light with matched indices; the index mismatch is what tells the two apart in situ. [`test_insitu.py::test_with_matched_indices_the_twin_images_the_mirrored_height_exactly`]

## Phase convention

A plane wave in the pupil carries `exp(+2πi(k_x x + k_y y + k_z z))`, as for an outgoing wave `exp(+i k·r)`. The field in the image is

```
E(x, y) = Σ_k  P(k) · h_f(k) · exp(iW(k)) · exp(2πi k_z,imm z) · exp(2πi k_z,med d) · exp(2πi[k_x(x − x0) + k_y(y − y0)]) Δk²
```

- `P(k)` is the aperture × apodisation × the normalised magnitude.
- `h_f(k)` is one of the six dipole fields.
- `W(k)` is the aberration phase.
- `(x0, y0, z)` is the emitter's position and `d` its depth.

Above the critical angle, the axial wave numbers take the `+i·|·|` branch of the square root, so `exp(2πi k_z d)` decays. [`test_optics.py::test_above_the_critical_angle_kz_in_the_medium_is_positive_imaginary_so_depth_decays`]

**The pupil grid.** P × P samples at `u, v = (i − (P−1)/2)·2/P`:
- u runs along x (columns) and v along y (rows);
- `k_x = u·NA/λ` and `k_y = v·NA/λ`, in cycles per nm;
- the spacing is `Δk = 2NA/(λP)`.

[`test_optics.py::test_pupil_samples_are_centred_and_spaced_by_two_NA_over_lambda_P`]

## Zernike polynomials

- **Indexing:** Noll's single index j:
  - 1 piston; 2, 3 tilt x, y; 4 defocus;
  - 5 oblique astigmatism (sin 2φ); 6 vertical astigmatism (cos 2φ);
  - 7, 8 coma y, x; 9, 10 trefoil; 11 spherical.

  [`test_zernike.py::test_noll_indices_follow_nolls_table`]
- **Normalisation:** unit RMS over the unit disc, so a coefficient is the RMS wavefront in radians. φ = atan2(v, u) is measured from +x towards +y.
- **Left out of the phase by default:** piston, tilt and defocus. They are the emitter's invisible offset, lateral position and z.
- **The sign, anchored on the image:** a positive coefficient of Noll 6 (cos 2φ) stretches the spot **along y for emitters above focus** (z > 0) and along x below focus. [`test_psf.py::test_positive_vertical_astigmatism_stretches_the_spot_along_y_above_focus`]
- **Magnitude:** `1 + Σ c_j Z_j` over Noll 2 and above. It is rescaled so the pupil's energy is that of the aberration-free pupil (next section).

## Photons and normalisation

- **The reference:** the image of an emitter in focus at the coverslip, through an aberration-free pupil, integrates to 1 over the whole image plane. Photon counts are therefore physical totals, as in vecfit, and do not depend on the region's size. [`test_psf.py::test_the_image_integrates_to_one_photon_and_aberrations_keep_that`]
- **Aberrations** keep that total: phase aberrations by Parseval's theorem, the magnitude because it is rescaled.
- **Less than 1:** light lost to depth (supercritical light) and light outside the region make the sum over a region smaller.
- **Splines for SMAPpy** are normalised by SMAPpy, as its bead calibration does (plan §6).

## Data, camera and likelihood

- **Data are photons**, `[E, C, K, Y, X]` (`Batch`): emitters, channels, observations (slices), pixels.
- **The slices' z** (`Batch.z_nm`) is the nominal emitter z of each slice, already in the sign above.
  `Batch.z_from_objective(objective_nm)` makes it from objective positions: z = −objective.
  [`test_forward.py::test_a_bead_moves_through_the_stack_in_smappys_z_sign`]
- **Camera**, as SMAPpy's `CameraMetadata`: photons = (ADU − offset) · conversion / EM gain, with
  `conversion` in electrons per ADU. [`test_noise.py::test_the_camera_converts_as_smappy_does`]
- **Read noise** is a variance s added to data and model alike (shifted Poisson), in the noise model's
  units: electrons divided by the excess factor, squared.  Unset: 1 e⁻ without EM, 0 with EM, as SMAPpy.
- **EM gain** halves data and model before the Poisson likelihood (excess noise 2), inside the noise
  model; fitted photons stay physical.
- **The objective** is ½ Σ deviance + priors, summed over every pixel, never averaged.
- **The first slice of a stack is dropped by default** (`smappy.calibrate.pupil.KEEP`, the cut's default `slices`): on the Zenodo beads the first frame is 22 % brighter than the rest. The nominal focus is the middle of the slices kept.
- **The stage's z scale** (`geometry.z_scale`): emitter z in slice k is z_e + s·z_k, s = 1 and fixed by default; fitted on request with a prior N(1, 0.1). [`test_atlas.py::test_a_fitted_z_scale_recovers_a_piezo_scale`]
- **Mirroring: none.** An EMCCD's EM port reads out mirrored relative to its normal port; beads and data must be taken on the same port. uipsf's own reader could flip the beads (`mirror="x"`); it went with that reader (plan §7.5), and SMAPpy's fitters warn when a calibration's EM state differs from the data's. v1's `swapxy` transposed instead; its ROIs are the camera's transposed (`docs/uipsf/data.md`).

## Blur and bead

- **Blur:** a Gaussian with standard deviation `(σx, σy)` in nm (parameter `psf.sigma`), applied to the upsampled intensity in Fourier space.
- **Bead:** a uniform sphere of radius R. Its transform `3(sin t − t cos t)/t³`, with `t = 2π|q|R`, is applied in 3-D along a z stack with an explicit step `z_step_nm`. The stack is padded by R // Δz + 2 planes at each end, as uiPSF pads it.

## Relation to uiPSF v1

v2 reproduces v1's PSFs to float32 precision [`test_oracles.py`]. The mapping, with v1's positions in (upsampled) pixels and slices:

| Quantity | v1 | v2 |
|---|---|---|
| lateral position | the image sits at −pos | `x = −pos_x · pixel_x / bin`, `y = −pos_y · pixel_y / bin` |
| z | `pos_z + Zrange`, increasing along the stack | `z = −(pos_z + Zrange) · Δz` |
| depth (with_IMM) | `pos_depth` | `depth = pos_depth · Δz` |
| Zernike coefficient | index i | Noll j = i + 1, same value and sign |
| magnitude | `Σ c Z` with `c_1` = piston | `1 + Σ (c_j / c_1) Z_j` |
| blur | σ_v1 = π × sd in upsampled pixels | `σ = σ_v1 / π · pixel / bin` (nm) |
| photons | the in-focus ideal PSF summed over the region | the whole image plane (above) |

**Where v2 departs from v1 on purpose:**
- **Blur on even regions.** v1 builds the blur on the frequency grid `(i − L/2 + ½)/L`. That equals the FFT's frequencies for odd L, but is half a sample off for even L (its `bin = 2` regions). v2 uses the FFT's frequencies. [`test_oracles.py::test_with_an_even_region_v1s_blur_is_off_by_half_a_frequency_sample_and_nothing_else`]
- **Bead kernel at the ends of a stack.** v1 convolves its padded stack circularly. v2 pads beyond the planes it returns, so only the end planes differ. [`test_oracles.py::test_the_bead_kernel_is_v1s_and_only_v1s_wrap_around_differs`]
- **Magnitude** is not clipped at 0, so its gradient never vanishes. Its scale is fixed by the energy normalisation, which removes v1's exact degeneracy between pupil scale and photons.
- **Non-square pixels** are handled on both axes. v1's bead kernel swaps x and y and its CZT uses the x size for both (review F1, minor).

## Relation to vecfit (TU Delft)

**Intensities agree.** vecfit uses the complex-conjugate convention: waves `exp(−i k·r)`, phases with a minus sign, and the `−i` branch above the critical angle. Its fields are the complex conjugates of v2's, so the intensities are identical. The test port is written in vecfit's own convention and agrees with v2 to 10⁻⁸ in float64. [`test_oracles.py::test_v2_reproduces_the_tu_delft_vectorial_model`]

**Coefficients do not compare directly.** In vecfit's MATLAB arrays x and y are swapped (column-major `meshgrid`) and the image is rotated by 180° relative to uiPSF (review §D), which changes the signs of some modes. The per-mode conversion table comes with the Zernike converters (`export/zernike.py`, plan §5.8). It will be checked against vecfit's own output when MATLAB results are at hand.

## Relation to SMAPpy

The axes, the pixel origin and the z sign above are SMAPpy's, so a uipsf model needs no conversion at the boundary:
- **Axes and pixel centres:** the same.
- **z:** `z = −(objective position − focus)`, as SMAPpy's `z_index_to_nm`.
- **The chain to v1 is checked:** the v1 oracles confirm `z_v2 = −z_v1`, and SMAPpy's uiPSF bridge (`smappy/uipsf/convert.py`) takes v1's stack order as SMAPpy's own.
- **The camera:** offset, conversion, read noise and EM gain as in SMAPpy (above); the check against SMAPpy's own conversion on the same ADU comes with M3.
- **Channels:** the transform, the split and "main" are SMAPpy's, and the photon shares are its `photon_normalization` (next section).

## Channels (M4)

- **Layout:** `[E, C, K, Y, X]`; channel 0 is the main channel, SMAPpy's "main" (the left or upper half of a split chip). Every channel's ROI has the same shape. `Batch.roi_xy[e, c]` is the chip pixel (x, y) of ROI c's centre.
- **The transform** of channel c is a 3×3 projective map from its chip pixels to the main channel's, zero-based pixel centres on the camera chip: SMAPpy's `DualColorCalibration.transformation`, the same object and direction (secondary → main). It is stored with H₂₂ = 1 (SMAPpy scales to unit norm: the same map). A mirrored split is in the transform (a negative scale across the split, SMAPpy's "up-down mirrored" / "right-left mirrored"), so every channel's PSF is in the camera's own orientation, as SMAPpy's secondary spline is. [`test_channels.py::test_the_secondary_channel_sees_the_emitter_where_the_transform_maps_it_on_the_chip`]
- **Positions:** the emitter's (x, y) are the main channel's, in nm from its ROI's centre. Channel c sees it at H⁻¹ of its chip position, relative to ROI c's centre, in that channel's pixel size; applied exactly, not linearised.
- **Photon shares:** f_c, the fraction of the emitter's photons in channel c, Σ f_c = 1, so the emitter's photon count is its total over the channels: SMAPpy's convention for a shared photon number (`positive_pair_models`, `photon_normalization`). Each channel's PSF holds one photon in focus, as a single channel's does; a polarising channel too, its polariser's fraction being in its share. [`test_channels.py::test_the_photons_are_the_emitters_total_split_by_the_shares`]
- **Defocus offset:** the emitter's height above channel c's focal plane is z + δz_c, z its height above the main channel's. Sampled for SMAPpy (`LearnedPSF.psf(..., channel=c)`), every channel takes the main channel's z, so both share one z grid and the offset is in the PSFs (plan §6.4); the sample includes the channel's share, so a pair together holds the emitter's photon.
- **Polarisation:** a channel is unpolarised unless its PSF says `analyser="x"` or `"y"` (review F1). [`test_channels.py::test_a_channel_is_unpolarised_unless_it_says_otherwise_and_x_plus_y_is_unpolarised`]
- **Parameter names:** the main channel keeps the single-channel names (`pupil.phase`, `psf.sigma`); channel c ≥ 1 prefixes its own with `ch<c>.` and has `ch<c>.transform`, `ch<c>.share`, `ch<c>.dz`. The emitter's background is one per channel, `emitter.background` [E, C].
- **uiPSF v1's two-channel frame** (Zenodo `2ch_40nm_bead`, from its 74 ROI centres): arrays transposed (`swapxy`), and v1's channel 0 — its reference — is the camera's *lower* half with y reversed, its channel 1 the upper half. So v1's reference is SMAPpy's secondary, and its `T` maps the secondary to the main channel (`benchmarks/uipsf/real/zenodo_2ch_beads.py`, `v1_to_chip`).
