"""Single-channel experimental PSF calibration, independent of the GUI.

Axes are always z,y,x. Lateral distances are pixels; z distances are objective
nanometres. No camera pixel size, gain conversion, or mirroring is applied.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage, optimize, signal
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from ..io.calibration import SplineCalibration
from .input import BeadStack, discover_acquisitions, read_bead_stacks


@dataclass
class CalibrationSettings:
    roi_size: int = 27
    padding: int = 5                 # extra pixels for registration, then discarded
    dz_nm: Optional[float] = None    # None: use acquisition metadata
    detection_sigma_px: float = 2.
    detection_threshold_sigma: float = 6.
    min_distance_px: float = 25.
    max_xy_shift_px: float = 3.
    max_z_shift_nm: float = 250.
    alignment_range_nm: float = 500.
    registration_iterations: int = 2
    rejection_mad: float = 1.5
    smooth_z_nm: float = 20.         # Gaussian sigma; deliberately not MATLAB lambda
    smooth_xy_px: float = 0.
    min_beads: int = 1
    brightness_range: Optional[float] = 6.  # full accepted max/min range; None disables
    saturation_adu: Optional[float] = None  # None: integer dtype maximum when available

    def validate(self):
        for name in ('roi_size', 'padding', 'registration_iterations', 'min_beads'):
            v = getattr(self, name)
            if not isinstance(v, (int, np.integer)) or v < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.roi_size < 7 or self.roi_size % 2 != 1:
            raise ValueError('roi_size must be odd and at least 7')
        for name in ('detection_sigma_px', 'detection_threshold_sigma',
                     'min_distance_px', 'alignment_range_nm', 'rejection_mad'):
            v = getattr(self, name)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ('max_xy_shift_px', 'max_z_shift_nm', 'smooth_z_nm', 'smooth_xy_px'):
            v = getattr(self, name)
            if not np.isfinite(v) or v < 0:
                raise ValueError(f'{name} must be nonnegative and finite')
        if self.max_xy_shift_px + 1 >= self.padding:
            raise ValueError('padding must exceed max_xy_shift_px by more than one pixel')
        if self.dz_nm is not None and (not np.isfinite(self.dz_nm) or self.dz_nm <= 0):
            raise ValueError('dz_nm must be positive and finite')
        if self.brightness_range is not None and (
                not np.isfinite(self.brightness_range) or self.brightness_range < 1):
            raise ValueError('brightness_range must be at least one and finite, or None')
        if self.saturation_adu is not None and (
                not np.isfinite(self.saturation_adu) or self.saturation_adu <= 0):
            raise ValueError('saturation_adu must be positive and finite, or None')


@dataclass
class BeadCollection:
    volumes: np.ndarray             # bead,z,y,x; border background subtracted
    records: list                   # stable bead id, source, coordinates, brightness
    projections: list              # one per original stack, for review
    sources: list
    dz_nm: float
    settings: CalibrationSettings
    original_volumes: np.ndarray | None = None
    channel_offsets: np.ndarray | None = None


@dataclass
class CalibrationResult:
    calibration: SplineCalibration
    beads: BeadCollection
    accepted: np.ndarray
    shifts: np.ndarray              # applied shifts, z planes,y pixels,x pixels
    residuals: np.ndarray
    correlations: np.ndarray
    coarse_shifts: np.ndarray
    coarse_correlations: np.ndarray
    reasons: list
    raw_psf: np.ndarray
    z_crop_start: int
    messages: list = field(default_factory=list)
    refits: dict = field(default_factory=dict)
    channel_calibrations: list = field(default_factory=list)
    channel_raw_psfs: list = field(default_factory=list)

    def save(self, path, overwrite=False):
        from ..io.calibration import save_spline_calibration
        diagnostics = {'accepted': self.accepted, 'shifts_zyx': self.shifts,
                       'coarse_shifts_zyx': self.coarse_shifts,
                       'coarse_correlations': self.coarse_correlations,
                       'residuals': self.residuals, 'correlations': self.correlations,
                       'raw_psf': self.raw_psf,
                       'bead_records_json': self.beads.records,
                       'reasons_json': self.reasons, 'sources_json': self.beads.sources,
                       'messages_json': self.messages, 'z_crop_start': self.z_crop_start}
        diagnostics.update({'refit_'+k: v for k, v in self.refits.items()})
        save_spline_calibration(path, self.calibration, diagnostics, overwrite=overwrite)


def detect_beads(images, settings):
    projection = np.max(images, axis=0)
    filtered = ndimage.gaussian_filter(projection.astype(float), settings.detection_sigma_px)
    background = np.median(filtered)
    noise = 1.4826 * np.median(np.abs(filtered - background))
    threshold = background + settings.detection_threshold_sigma * max(noise, np.finfo(float).eps)
    peaks = (filtered == ndimage.maximum_filter(filtered, size=3)) & (filtered > threshold)
    labels, n = ndimage.label(peaks)
    centers = ndimage.center_of_mass(filtered, labels, np.arange(1, n + 1)) if n else []
    yx = np.rint(centers).astype(int).reshape(-1, 2)
    radius = settings.roi_size // 2 + settings.padding
    good = ((yx >= radius) & (yx < np.array(images.shape[1:]) - radius)).all(axis=1)
    # Reject both members of a close pair, including a neighbor outside the usable border.
    for a, b in cKDTree(yx).query_pairs(settings.min_distance_px):
        good[a] = good[b] = False
    return yx[good], projection


def collect_beads(inputs, settings=None, progress=None):
    """Extract only bead volumes; full camera stacks are released acquisition by acquisition.

    Inputs may be paths or an iterable of BeadStack objects for in-memory use.
    Multiple channels are rejected rather than averaged into an invalid PSF.
    """
    settings = settings or CalibrationSettings()
    settings.validate()
    report = progress or (lambda message: None)
    if isinstance(inputs, BeadStack):
        inputs = [inputs]
    elif isinstance(inputs, (str, bytes)) or hasattr(inputs, '__fspath__'):
        inputs = [inputs]
    inputs = list(inputs)
    if not inputs:
        raise ValueError('select at least one acquisition')
    if isinstance(inputs[0], BeadStack):
        batches = ([s] for s in inputs)
    else:
        paths = discover_acquisitions(inputs)
        batches = (read_bead_stacks(p, settings.dz_nm) for p in paths)
    volumes, records, projections, sources = [], [], [], []
    dz = None
    channels = set()
    radius = settings.roi_size // 2 + settings.padding
    for batch in batches:
        for stack in batch:
            report(f'Reading beads: {stack.source} {stack.axes}')
            camera_images = np.asarray(stack.images)
            images = np.asarray(camera_images, dtype=np.float32)
            z = np.asarray(stack.z_nm, dtype=float)
            if (images.ndim != 3 or z.shape != (images.shape[0],) or len(z) < 4
                    or not np.isfinite(images).all() or not np.isfinite(z).all()
                    or np.any(np.diff(z) <= 0)
                    or not np.allclose(np.diff(z), stack.dz, rtol=.02, atol=.1)):
                raise ValueError(f'{stack.source}: invalid images or nonuniform increasing z grid')
            channels.add(str(stack.axes.get('channel', 0)))
            if len(channels) > 1:
                raise ValueError('selected acquisitions contain multiple channels; '
                                 'single-channel calibration cannot pool them')
            if dz is None:
                dz = stack.dz
            if not np.isclose(stack.dz, dz, rtol=.02, atol=.1):
                raise ValueError('selected stacks have different z spacing; resample explicitly')
            yx, projection = detect_beads(images, settings)
            saturation = settings.saturation_adu
            if saturation is None and np.issubdtype(camera_images.dtype, np.integer):
                saturation = float(np.iinfo(camera_images.dtype).max)
            si = len(sources)
            sources.append({'path': stack.source, 'axes': stack.axes,
                            'shape': list(images.shape), 'z_nm': z.tolist()})
            projections.append(projection)
            for y, x in yx:
                volume = images[:, y-radius:y+radius+1, x-radius:x+radius+1].copy()
                peak_adu = float(np.max(volume))
                saturated = saturation is not None and peak_adu >= saturation
                border = np.concatenate((volume[:, 0], volume[:, -1],
                                         volume[:, 1:-1, 0], volume[:, 1:-1, -1]), axis=1)
                # One scalar for the whole bead z-stack: preserve axial changes
                # in the background/PSF tails instead of removing them plane-wise.
                background = float(np.median(border))
                volume -= background
                brightness = float(np.max(volume.sum(axis=(1, 2))))
                if brightness <= 0:
                    continue
                records.append({'id': len(records), 'stack': si, 'x_px': int(x),
                                'y_px': int(y), 'brightness_adu': brightness,
                                'background_adu': background, 'peak_adu': peak_adu,
                                'saturation_adu': saturation, 'saturated': bool(saturated)})
                volumes.append(volume / brightness)
    brightness = np.asarray([r['brightness_adu'] for r in records])
    unsaturated = np.asarray([not r['saturated'] for r in records])
    if np.any(unsaturated) and settings.brightness_range is not None:
        center = float(np.median(brightness[unsaturated]))
        half_range = np.sqrt(settings.brightness_range)
        lower, upper = center/half_range, center*half_range
    else:
        lower, upper = None, None
    for rec in records:
        if rec['saturated']:
            reason = 'saturated'
        elif lower is not None and rec['brightness_adu'] < lower:
            reason = 'too dim'
        elif upper is not None and rec['brightness_adu'] > upper:
            reason = 'too bright'
        else:
            reason = 'accepted'
        rec['brightness_reason'] = reason
        rec['brightness_accepted'] = reason == 'accepted'
        rec['brightness_lower_adu'] = lower
        rec['brightness_upper_adu'] = upper
    usable = sum(r['brightness_accepted'] for r in records)
    if usable < settings.min_beads:
        raise ValueError(f'only {usable} brightness-qualified isolated beads found; '
                         f'need {settings.min_beads}. '
                         'Review detection, brightness range, and saturation settings.')
    # Center crop different stack lengths without stretching z or extrapolating.
    nz = min(v.shape[0] for v in volumes)
    nz -= 1 - nz % 2
    for rec, vol in zip(records, volumes):
        rec['start_plane'] = (vol.shape[0] - nz) // 2
    volumes = np.stack([v[r['start_plane']:r['start_plane']+nz]
                        for v, r in zip(volumes, records)]).astype(np.float32)
    return BeadCollection(volumes, records, projections, sources, dz, settings)


def estimate_shift(reference, moving, limits=None, z_window=None, lateral_window=None,
                   return_quality=False):
    """Linear normalized cross-correlation plus local subpixel refinement.

    ``limits`` constrains only the correlation search and is independent of the
    acceptance limits in :class:`CalibrationSettings`. A central ``z_window`` is
    useful after a full-stack coarse pass. Correlation is linear, never circular.
    Returned shifts are applied to ``moving`` in z,y,x order.
    """
    reference = np.asarray(reference, float)
    moving = np.asarray(moving, float)
    if reference.ndim == 4 and reference.shape == moving.shape:
        return estimate_pair_shift(reference, moving, limits, z_window, lateral_window,
                                   return_quality)
    if reference.shape != moving.shape or reference.ndim != 3:
        raise ValueError('reference and moving volumes must have the same 3D shape')
    shape = np.asarray(reference.shape)
    search_slices = [slice(0, n) for n in shape]
    if z_window is not None:
        width = min(shape[0], max(7, int(round(z_window))))
        start = (shape[0]-width)//2
        search_slices[0] = slice(start, start+width)
    if lateral_window is not None:
        for axis in (1, 2):
            width = min(shape[axis], max(5, int(round(lateral_window))))
            start = (shape[axis]-width)//2
            search_slices[axis] = slice(start, start+width)
    search_slices = tuple(search_slices)
    ref_search = reference[search_slices].copy()
    mov_search = moving[search_slices].copy()
    ref_search -= ref_search.mean()
    mov_search -= mov_search.mean()
    if limits is None:
        limits = (np.asarray(ref_search.shape)-3)//2
    limits = np.minimum(np.asarray(limits, float), (np.asarray(ref_search.shape)-3)//2)
    limits = np.maximum(limits, 0)

    # Normalize every linear-correlation lag by the energy in its actual overlap.
    # This retains SMAP's broad peak search without its circular wraparound.
    reverse = (slice(None, None, -1),)*3
    numerator = signal.fftconvolve(ref_search, mov_search[reverse], mode='full')
    ref_energy = signal.fftconvolve(ref_search**2, np.ones_like(mov_search), mode='full')
    mov_energy = signal.fftconvolve(np.ones_like(ref_search), (mov_search**2)[reverse],
                                    mode='full')
    correlation = numerator/np.sqrt(np.maximum(ref_energy*mov_energy, 1e-30))
    center = np.asarray(mov_search.shape)-1
    ranges = [np.arange(-int(np.floor(limit)), int(np.floor(limit))+1)
              for limit in limits]
    block = correlation[np.ix_(*[c+r for c, r in zip(center, ranges)])]
    peak = np.unravel_index(np.nanargmax(block), block.shape)
    initial = np.asarray([r[i] for r, i in zip(ranges, peak)], float)

    # Refine only one pixel around the coarse peak on a fixed, valid overlap.
    lo = np.asarray([s.start for s in search_slices])
    hi = np.asarray([s.stop for s in search_slices])
    lo = np.maximum(lo, np.ceil(initial+2).astype(int))
    hi = np.minimum(hi, np.floor(shape+initial-2).astype(int))
    if np.any(hi-lo < 3):
        quality = float(block[peak])
        return (initial, quality) if return_quality else initial
    grid = np.mgrid[tuple(slice(a, b) for a, b in zip(lo, hi))].astype(float)
    ref = reference[tuple(slice(a, b) for a, b in zip(lo, hi))].ravel()
    ref -= ref.mean()
    ref /= max(np.linalg.norm(ref), 1e-15)

    def objective(delta):
        shift = initial+delta
        sample = ndimage.map_coordinates(moving, grid-shift[:, None, None, None],
                                         order=1, prefilter=False).ravel()
        sample -= sample.mean()
        return 1-np.dot(sample, ref)/max(np.linalg.norm(sample), 1e-15)

    refined = optimize.minimize(objective, np.zeros(3), method='Powell',
                                bounds=[(-1., 1.)]*3,
                                options={'xtol': .005, 'ftol': 1e-7, 'maxiter': 30})
    shift = initial+refined.x
    quality = float(1-objective(refined.x))
    return (shift, quality) if return_quality else shift


def robust_shape_error(sample, reference):
    """Amplitude-invariant normalized error against a leave-one-out 3D PSF.

    A robust amplitude ratio is measured on the reference's brightest quartile;
    the full-volume RMS and correlation then retain sensitivity to broad shape
    disagreement. Background is not refitted because extraction already removes
    one independently measured border background from each bead. The caller
    supplies a leave-one-out reference where possible, so a bead cannot improve
    its own template. Returns shape error and diagnostic correlation.
    """
    sample = np.asarray(sample, float)
    reference = np.asarray(reference, float)
    if sample.shape != reference.shape or sample.ndim not in (3, 4):
        raise ValueError('sample and reference must be matching 3D volumes')
    informative = reference > np.quantile(reference, .75)
    valid = informative & np.isfinite(sample) & np.isfinite(reference) & (
        np.abs(reference) > 1e-12)
    if np.count_nonzero(valid) < 8:
        return np.nan, np.nan
    amplitude = float(np.median(sample[valid]/reference[valid]))
    if not np.isfinite(amplitude) or amplitude <= 0:
        return np.inf, -1.
    matched = sample/amplitude
    signal_power = max(float(np.mean(reference**2)), 1e-20)
    error = float(np.sqrt(np.mean((matched-reference)**2)/signal_power))
    x_centered = reference.ravel()-reference.mean()
    y_centered = matched.ravel()-matched.mean()
    correlation = float(np.dot(x_centered, y_centered) / max(
        np.linalg.norm(x_centered) * np.linalg.norm(y_centered), 1e-15))
    return error, correlation


def spline_coefficients(psf):
    """Tensor-product not-a-knot cubic coefficients in fitter order.

    Three batched 1D transforms replace one 64x64 solve per voxel. Coefficient
    index is 16*pz+4*py+px, and the spatial layout is z,y,x (C contiguous).
    """
    psf = np.asarray(psf, float)
    if psf.ndim != 3 or min(psf.shape) < 4 or not np.isfinite(psf).all():
        raise ValueError('spline input must be finite, 3D, with at least four samples per axis')
    nz, ny, nx = psf.shape
    cx = CubicSpline(np.arange(nx), psf, axis=2).c[::-1]  # px,x,z,y
    cy = CubicSpline(np.arange(ny), cx, axis=3).c[::-1]   # py,y,px,x,z
    cz = CubicSpline(np.arange(nz), cy, axis=4).c[::-1]   # pz,z,py,y,px,x
    return np.ascontiguousarray(cz.transpose(0, 2, 4, 1, 3, 5).reshape(64, nz-1, ny-1, nx-1),
                                dtype=np.float32)


def build_calibration(beads, excluded=(), progress=None):
    """Align and average a collection; excluded contains stable integer bead IDs.

    Alignment follows SMAP's two-stage structure: a broad full-stack correlation,
    rejection before template construction, then central correlation refinements.
    Unlike SMAP, correlations are linear rather than circular and maximum shifts
    are post-search acceptance criteria rather than optimizer bounds.
    """
    s = beads.settings
    s.validate()
    report = progress or (lambda message: None)
    volumes = beads.volumes
    paired = volumes.ndim == 5
    n = len(volumes)
    nz, ny, nx = volumes.shape[-3:]
    excluded = set(excluded)
    if not excluded.issubset(range(n)):
        raise ValueError('excluded contains unknown bead IDs')
    manual = np.array([i not in excluded for i in range(n)])
    brightness_ok = np.array([r.get('brightness_accepted', True) for r in beads.records])
    candidates = manual & brightness_ok
    if candidates.sum() < s.min_beads:
        raise ValueError(f'keep at least {s.min_beads} brightness-qualified beads')
    candidate_ids = np.flatnonzero(candidates)
    p = s.padding

    # SMAP seeds registration with the least-deviant half of the unaligned beads,
    # not one possibly atypical bead. The input volumes are already brightness
    # normalized, so squared shape deviation is sufficient for this ranking.
    preliminary = np.mean(volumes[candidates], axis=0)
    deviation = np.mean((volumes[candidate_ids]-preliminary)**2,
                        axis=tuple(range(1, volumes.ndim)))
    reference_count = max(int(np.ceil(len(candidate_ids)/2)), min(5, len(candidate_ids)))
    reference_ids = candidate_ids[np.argsort(deviation)[:reference_count]]
    reference = np.mean(volumes[reference_ids], axis=0)

    lateral_window = min(13, ny, nx)
    if lateral_window % 2 == 0:
        lateral_window -= 1
    full_limits = np.array([(nz-7)//2, (lateral_window-3)//2,
                            (lateral_window-3)//2], float)
    central_planes = min(nz, max(7, int(round(s.alignment_range_nm/beads.dz_nm))))
    refine_limits = np.array([(central_planes-3)//2, full_limits[1], full_limits[2]], float)
    shifts = np.full((n, 3), np.nan)
    coarse_shifts = np.full((n, 3), np.nan)
    coarse_quality = np.full(n, np.nan)
    shifted = np.zeros_like(volumes)

    for i in candidate_ids:
        report(f'Coarse full-stack alignment: bead {i+1}/{n}')
        coarse_shifts[i], coarse_quality[i] = estimate_shift(
            reference, volumes[i], full_limits, lateral_window=lateral_window,
            return_quality=True)
    # Correlation defines shifts relative to an arbitrary seed-reference origin.
    # A robust common translation makes maximum-shift rejection relative to the
    # bead consensus instead of whichever subset happened to seed the reference.
    origin = np.median(coarse_shifts[candidates], axis=0)
    coarse_shifts[candidates] -= origin
    shifts[candidates] = coarse_shifts[candidates]

    def shift_is_acceptable(values):
        z_ok = np.abs(values[:, 0])*beads.dz_nm <= s.max_z_shift_nm
        xy_ok = np.hypot(values[:, 1], values[:, 2]) <= s.max_xy_shift_px
        if paired and beads.channel_offsets is not None:
            total_xy = values[:, None, 1:]+beads.channel_offsets[:, :, 1:]
            xy_ok &= np.all(np.abs(total_xy) <= p-1, axis=(1, 2))
        return z_ok, xy_ok

    def apply_shifts(ids):
        for bead_id in ids:
            if paired and beads.original_volumes is not None:
                for ch in range(volumes.shape[1]):
                    shifted[bead_id, ch] = ndimage.shift(beads.original_volumes[bead_id, ch],
                            shifts[bead_id]+beads.channel_offsets[bead_id, ch], order=3,
                            mode='constant', cval=0.)
                continue
            delta = np.r_[0., shifts[bead_id]] if paired else shifts[bead_id]
            shifted[bead_id] = ndimage.shift(volumes[bead_id], delta, order=3,
                                             mode='constant', cval=0.)

    def overlap_region(mask):
        crop = int(np.ceil(np.max(np.abs(shifts[mask, 0]))))+2
        if nz-2*crop < 7:
            raise ValueError('too little common z overlap after registration')
        region = (slice(crop, nz-crop), slice(p, ny-p), slice(p, nx-p))
        return crop, ((slice(None),)+region if paired else region)

    def normalize_and_score(base_mask, accepted_mask, region):
        """SMAP-like amplitude matching plus leave-one-out robust shape errors."""
        ids = np.flatnonzero(base_mask)
        scaled = shifted.copy()
        factors = np.ones(n)
        for _ in range(4):
            template = np.mean(scaled[accepted_mask], axis=0)
            ref = template[region]
            bright = ref > np.quantile(ref, .75)
            for bead_id in ids:
                sample = shifted[bead_id][region]
                valid = bright & np.isfinite(sample) & (np.abs(ref) > 1e-12)
                factor = np.median(sample[valid]/ref[valid]) if np.any(valid) else np.nan
                if np.isfinite(factor) and factor > 0:
                    factors[bead_id] = factor
                    scaled[bead_id] = shifted[bead_id]/factor
        accepted_count = int(accepted_mask.sum())
        accepted_sum = np.sum(scaled[accepted_mask], axis=0)
        template = accepted_sum / accepted_count
        residual = np.full(n, np.nan)
        correlation = np.full(n, np.nan)
        for bead_id in ids:
            if accepted_mask[bead_id] and accepted_count > 1:
                bead_reference = ((accepted_sum-scaled[bead_id]) /
                                  (accepted_count-1))
            else:
                bead_reference = template
            residual[bead_id], correlation[bead_id] = robust_shape_error(
                scaled[bead_id][region], bead_reference[region])
        return scaled, template, residual, correlation

    def reject_shapes(base_mask, region, rounds):
        accepted_mask = base_mask.copy()
        for _ in range(rounds):
            scaled, template, residual, correlation = normalize_and_score(
                base_mask, accepted_mask, region)
            # The robust error measures local mismatch; the independent
            # correlation prevents a broadly wrong but locally smooth bead from
            # receiving an artificially benign Huber score.
            score = residual/np.maximum(correlation, 1e-6)
            values = score[accepted_mask & np.isfinite(score) & (correlation > 0)]
            if not len(values):
                raise ValueError('no positively correlated beads survive alignment')
            median = np.median(values)
            mad = 1.4826*np.median(np.abs(values-median))
            # Match SMAP's robust mean + N*sample-std behavior. With very small
            # bead counts, pre-trimming by MAD is unstable and can reject one of
            # three nearly identical beads merely because its residual ranks last.
            robust = values
            if len(values) >= 6 and mad > 0:
                robust = values[np.abs(values-median) <= 4*mad]
            center = float(np.mean(robust))
            spread = float(np.std(robust, ddof=1)) if len(robust) > 1 else 0.
            threshold = center+s.rejection_mad*max(spread, 1e-6)
            updated = base_mask & np.isfinite(score) & (correlation > 0) & (score <= threshold)
            if updated.sum() < s.min_beads:
                raise ValueError('too few beads survive alignment/shape rejection')
            accepted_mask = updated
        return accepted_mask, scaled, template, residual, correlation

    apply_shifts(candidate_ids)
    z_ok, xy_ok = shift_is_acceptable(shifts)
    shift_ok = candidates & z_ok & xy_ok
    if shift_ok.sum() < s.min_beads:
        raise ValueError('too few beads remain within maximum z/xy shifts')
    crop_z, score_region = overlap_region(shift_ok)
    accepted, scaled, reference, residuals, correlations = reject_shapes(
        shift_ok, score_region, 1 if s.registration_iterations > 1 else 3)

    # Refine already aligned stacks on the central window, accumulating residual
    # shifts. Each final volume is resampled from its original exactly once.
    for iteration in range(1, s.registration_iterations):
        refine_ids = np.flatnonzero(accepted)
        deltas = np.zeros((len(refine_ids), 3))
        for row, i in enumerate(refine_ids):
            report(f'Central alignment refinement {iteration}/{s.registration_iterations-1}: '
                   f'bead {i+1}/{n}')
            deltas[row], _ = estimate_shift(reference, shifted[i], refine_limits,
                                            z_window=central_planes,
                                            lateral_window=lateral_window,
                                            return_quality=True)
        # A common residual translation is arbitrary; remove it so acceptance
        # limits stay referenced to the bead consensus across all passes.
        deltas -= np.median(deltas, axis=0)
        shifts[refine_ids] += deltas
        apply_shifts(refine_ids)
        z_ok, xy_ok = shift_is_acceptable(shifts)
        shift_ok = accepted & z_ok & xy_ok
        if shift_ok.sum() < s.min_beads:
            raise ValueError('too few beads remain within maximum z/xy shifts')
        crop_z, score_region = overlap_region(shift_ok)
        rounds = 3 if iteration == s.registration_iterations-1 else 1
        accepted, scaled, reference, residuals, correlations = reject_shapes(
            shift_ok, score_region, rounds)

    # Register pre-excluded beads for display/held-out diagnostics only.
    for i in np.flatnonzero(~candidates):
        report(f'Aligning rejected bead {i+1} for independent diagnostics')
        shifts[i], coarse_quality[i] = estimate_shift(
            reference, volumes[i], full_limits, lateral_window=lateral_window,
            return_quality=True)
        coarse_shifts[i] = shifts[i]

    z_ok, xy_ok = shift_is_acceptable(shifts)
    crop_z, region = overlap_region(accepted)
    diagnostic_mask = candidates
    scaled, reference, residuals, correlations = normalize_and_score(
        diagnostic_mask, accepted, region)
    report('Averaging PSF and calculating spline coefficients')
    raw = reference[region].copy()
    if paired:
        models, raws = positive_pair_models(raw, beads.dz_nm, s)
        cal, psf = models[0], models[0].psf
        raw = raws[0]
    else:
        models, raws = [], []
        cal, raw = _single_model(raw, beads.dz_nm, s)
        psf = cal.psf
    reasons = []
    for i in range(n):
        if not manual[i]:
            reason = 'manual exclusion'
        elif not brightness_ok[i]:
            reason = beads.records[i].get('brightness_reason', 'brightness outlier')
        elif not z_ok[i] and not xy_ok[i]:
            reason = 'z and xy shift outlier'
        elif not z_ok[i]:
            reason = 'z shift outlier'
        elif not xy_ok[i]:
            reason = 'xy shift outlier'
        elif not accepted[i]:
            reason = 'shape/correlation outlier'
        else:
            reason = 'accepted'
        reasons.append(reason)
    messages = []
    if accepted.sum() < 10:
        messages.append('Fewer than 10 beads: inspect robustness on independent stacks.')
    rejected_shift = candidates & (~z_ok | ~xy_ok)
    if np.any(rejected_shift):
        messages.append(f'{rejected_shift.sum()} beads excluded by maximum shift limits.')
    return CalibrationResult(cal, beads, accepted, shifts, residuals, correlations,
                             coarse_shifts, coarse_quality, reasons, raw, crop_z, messages,
                             channel_calibrations=models, channel_raw_psfs=raws)


def _single_model(raw, dz_nm, s):
    psf = ndimage.gaussian_filter(raw, (s.smooth_z_nm/dz_nm, s.smooth_xy_px,
                                        s.smooth_xy_px), mode='reflect')
    psf = np.maximum(psf, 0)
    norm = float(np.max(psf.sum(axis=(1, 2))))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('averaged PSF has no positive signal')
    psf /= norm
    raw /= norm
    cal = SplineCalibration(spline_coefficients(psf), dz_nm,
                            (psf.shape[0]-1)/2, x0=(s.roi_size-1)/2,
                            psf=psf, em_mirror=False,
                            parameters={'method': 'smappy_bead_spline_v4',
                                        'settings': asdict(s),
                                        'background': 'one median of lateral border pixels across all z planes per bead stack',
                                        'normalization': 'maximum plane sum = 1',
                                        'z_reference': 'aligned stack center',
                                        'z_convention': 'emitter_nm = -(index-z0)*dz',
                                        'lateral_unit': 'pixel'})
    return cal, raw


def calibrate(inputs, settings=None, excluded=(), progress=None):
    """Build one spline calibration from all selected acquisitions or bead stacks."""
    return build_calibration(collect_beads(inputs, settings, progress), excluded, progress)


def estimate_pair_shift(reference, moving, limits=None, z_window=None,
                        lateral_window=None, return_quality=False):
    """Sum channel correlation numerators/energies; never correlate across channels."""
    shape = np.array(reference.shape[1:])
    widths = shape.copy()
    if z_window is not None:
        widths[0] = min(shape[0], max(7, int(round(z_window))))
    if lateral_window is not None:
        widths[1:] = np.minimum(shape[1:], max(5, int(lateral_window)))
    starts = (shape-widths)//2
    region = tuple(slice(a, a+w) for a, w in zip(starts, widths))
    ref, mov = reference[(slice(None),)+region].copy(), moving[(slice(None),)+region].copy()
    # Subtract channel means without changing either channel's amplitude scale.
    ref -= ref.mean(axis=(1, 2, 3), keepdims=True)
    mov -= mov.mean(axis=(1, 2, 3), keepdims=True)
    numerator = energy_r = energy_m = 0
    for r, m in zip(ref, mov):
        reverse = (slice(None, None, -1),)*3
        numerator = numerator+signal.fftconvolve(r, m[reverse], mode='full')
        energy_r = energy_r+signal.fftconvolve(r*r, np.ones_like(m), mode='full')
        energy_m = energy_m+signal.fftconvolve(np.ones_like(r), (m*m)[reverse], mode='full')
    corr = numerator/np.sqrt(np.maximum(energy_r*energy_m, 1e-30))
    limits = np.minimum((widths-3)//2, (widths-3)//2 if limits is None else limits).astype(int)
    ranges = [np.arange(-v, v+1) for v in limits]
    block = corr[np.ix_(*[w-1+r for w, r in zip(widths, ranges)])]
    peak = np.unravel_index(np.argmax(block), block.shape)
    initial = np.array([r[i] for r, i in zip(ranges, peak)], float)
    lo = np.maximum(starts, np.ceil(initial+2).astype(int))
    hi = np.minimum(starts+widths, np.floor(shape+initial-2).astype(int))
    if np.any(hi-lo < 3):
        return (initial, float(block[peak])) if return_quality else initial
    grid = np.mgrid[tuple(slice(a, b) for a, b in zip(lo, hi))].astype(float)
    r = reference[(slice(None),)+tuple(slice(a, b) for a, b in zip(lo, hi))].copy()
    r -= r.mean(axis=(1, 2, 3), keepdims=True)
    r = r.ravel()/max(np.linalg.norm(r), 1e-15)
    def objective(delta):
        sample = np.stack([ndimage.map_coordinates(ch, grid-(initial+delta)[:, None, None, None],
                            order=1, prefilter=False) for ch in moving])
        sample -= sample.mean(axis=(1, 2, 3), keepdims=True)
        return 1-np.dot(sample.ravel(), r)/max(np.linalg.norm(sample), 1e-15)
    result = optimize.minimize(objective, np.zeros(3), method='Powell',
            bounds=[(-1., 1.)]*3, options={'xtol': .005, 'ftol': 1e-7, 'maxiter': 30})
    shift = initial+result.x
    return (shift, float(1-objective(result.x))) if return_quality else shift


def positive_pair_models(raw, dz_nm, settings):
    """Constant channel offsets certify positivity throughout every cubic cell.

    A polynomial lies within its Bernstein coefficient bounds on [0,1]^3.
    Bounding the actual float32 coefficients also covers cubic undershoot.
    """
    from math import comb
    s = settings
    psfs = ndimage.gaussian_filter(raw, (0, s.smooth_z_nm/dz_nm, s.smooth_xy_px,
                                        s.smooth_xy_px), mode='reflect')
    transform = np.array([[comb(k, i)/comb(3, i) if i <= k else 0
                           for i in range(4)] for k in range(4)])
    def lower_bound(coeff):
        powers = coeff.reshape(4, 4, 4, -1).astype(float)
        return float(np.einsum('ai,bj,ck,ijkn->abcn', transform, transform, transform,
                              powers, optimize=True).min())
    offsets = []
    for psf in psfs:
        coeff = spline_coefficients(psf)
        epsilon = max(float(np.max(np.abs(psf)))*1e-6, 1e-12)
        offset = max(0., epsilon-lower_bound(coeff))
        psf += offset
        offsets.append(offset)
    norm = float(psfs[0].sum(axis=(1, 2)).max())
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('invalid paired PSF normalization')
    models = [SplineCalibration(spline_coefficients(psf/norm), dz_nm,
              (psf.shape[0]-1)/2, x0=(s.roi_size-1)/2, psf=psf/norm, em_mirror=False,
              parameters={'method': 'smappy_dual_bead_v1', 'settings': asdict(s),
                          'positivity_offset_before_normalization': offset,
                          'common_normalization': norm,
                          'positivity': 'Bernstein lower bound; relative floor 1e-6',
                          'z_reference': 'joint aligned stack center'})
              for psf, offset in zip(psfs, offsets)]
    for model in models:
        bound = lower_bound(model.coeff)
        if not np.isfinite(bound) or bound <= 0:
            raise ValueError('could not certify strict positivity of the final float32 spline')
        model.parameters['continuous_positive_lower_bound'] = bound
        model.parameters['peak_plane_integral'] = float(model.psf.sum(axis=(1, 2)).max())
    return models, list(raw/norm)
