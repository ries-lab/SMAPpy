"""Split-frame paired calibration. Coordinates are x,y pixels, arrays channel,z,y,x."""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from pathlib import Path
import json
import os
import tempfile
import warnings

from typing import Optional

import numpy as np
from scipy import ndimage, optimize
from scipy.spatial import cKDTree, ConvexHull, QhullError

from .core import (CalibrationSettings, BeadCollection, collect_beads,
                   build_calibration, spline_coefficients, robust_shape_error)
from .input import BeadStack, discover_acquisitions, read_bead_stacks
from ..io.calibration import SplineCalibration, _validate_native_calibration

LAYOUTS = ('right-left', 'right-left mirrored', 'up-down', 'up-down mirrored')
# Smallest soft-L1 transition used in the geometric refinement, in main-channel
# pixels.  Far below any achievable registration accuracy, so it never binds on
# a sane axis limit; it only keeps a pathological one numerically sane.
MIN_LOSS_SCALE_PX = 1e-3


@dataclass
class DualColorSettings(CalibrationSettings):
    layout: str = 'up-down mirrored'
    main_channel: str = 'upper'
    split_position: Optional[int] = None
    min_pairs: int = 8
    reprojection_threshold_px: float = 2.
    transform_axis_limit_px: float = .15

    def validate(self):
        super().validate()
        if self.layout not in LAYOUTS:
            raise ValueError(f'layout must be one of {LAYOUTS}')
        if self.main_channel not in (('left', 'right') if 'right-left' in self.layout else ('upper', 'lower')):
            raise ValueError('main channel must match the split layout')
        if self.split_position is not None and (not isinstance(self.split_position, (int, np.integer)) or self.split_position < 1):
            raise ValueError('split_position must be a positive integer or None')
        if not isinstance(self.min_pairs, (int, np.integer)) or self.min_pairs < 4:
            raise ValueError('min_pairs must be at least four (default eight)')
        if not np.isfinite(self.reprojection_threshold_px) or self.reprojection_threshold_px <= 0:
            raise ValueError('reprojection threshold must be positive and finite')
        if not np.isfinite(self.transform_axis_limit_px) or self.transform_axis_limit_px <= 0:
            raise ValueError('transformation dx/dy limit must be positive and finite')


def map_points(matrix, points):
    points = np.asarray(points, float)
    out = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
    if np.any(np.abs(out[:, 2]) < 1e-12):
        raise ValueError('projective mapping crosses infinity')
    return out[:, :2]/out[:, 2:]


def fit_projective(source, target):
    """Normalized DLT; source and target must span a two-dimensional region."""
    def normalize(points):
        points = np.asarray(points, float)
        center = points.mean(axis=0)
        scale = np.sqrt(2)/max(np.sqrt(np.mean(np.sum((points-center)**2, axis=1))), 1e-15)
        if np.linalg.matrix_rank(points-center, tol=1e-7) < 2:
            raise ValueError('bead pairs are collinear or coincident')
        matrix = np.array([[scale, 0, -scale*center[0]], [0, scale, -scale*center[1]], [0, 0, 1]])
        return map_points(matrix, points), matrix
    if len(source) < 4:
        raise ValueError('projective fit needs at least four pairs')
    a, na = normalize(source)
    b, nb = normalize(target)
    x, y = a.T
    u, v = b.T
    z, o = np.zeros(len(a)), np.ones(len(a))
    design = np.stack((np.c_[-x, -y, -o, z, z, z, u*x, u*y, u],
                       np.c_[z, z, z, -x, -y, -o, v*x, v*y, v]), axis=1).reshape(-1, 9)
    _, singular, vectors = np.linalg.svd(design, full_matrices=True)
    if singular[7] < singular[0]*1e-8:
        raise ValueError('ill-conditioned projective point coverage')
    h = np.linalg.inv(nb) @ vectors[-1].reshape(3, 3) @ na
    h /= np.linalg.norm(h)
    if abs(np.linalg.det(h)) < 1e-15:
        raise ValueError('singular projective transformation')
    return h


def robust_projective(source, target, threshold, minimum):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError('projective coordinates must be matching N by 2 arrays')
    if len(source) < max(4, minimum) or not np.isfinite([source, target]).all():
        raise ValueError('too few or nonfinite projective coordinates')
    rng = np.random.default_rng(1729)
    best = np.zeros(len(source), bool)
    best_error = np.inf
    for _ in range(500):
        ids = rng.choice(len(source), 4, replace=False)
        try:
            h = fit_projective(source[ids], target[ids])
            residual = np.linalg.norm(map_points(h, source)-target, axis=1)
        except (ValueError, np.linalg.LinAlgError):
            continue
        good = residual <= threshold
        error = np.sum(np.minimum(residual, threshold)**2)
        if (good.sum(), -error) > (best.sum(), -best_error):
            best, best_error = good, error
    if best.sum() < minimum:
        raise ValueError(f'only {best.sum()} projective inliers; require {minimum}')
    # Re-fit on the inliers until the inlier set stops moving.  Exact set
    # equality is the right test for the tens of pairs a bead calibration has
    # and the wrong one for the thousands a localization registration has: a
    # handful of pairs always sit within noise of the threshold and flip every
    # iteration, so the set never repeats however still the transformation is.
    # Settle instead for a set that is no longer changing meaningfully, and if
    # even that is not reached, hand back the last fit -- by then it has
    # stopped moving, and the caller's own dx/dy screen is what decides
    # whether the result is usable.
    settled = max(1, int(0.002*len(source)))
    for _ in range(10):
        h = fit_projective(source[best], target[best])
        residual = np.linalg.norm(map_points(h, source)-target, axis=1)
        good = residual <= threshold
        if good.sum() < minimum:
            raise ValueError('too few pairs after projective refinement')
        churn = int(np.count_nonzero(good != best))
        best = good
        if churn <= settled:
            return h, best, residual
    return h, best, residual


def refine_projective(source, target, initial, loss_scale, weights=None):
    """Minimize soft-L1 x/y reprojection errors in main-channel pixels.

    Optimize in centered/scaled coordinates for numerical stability, including
    chip ROIs far from the origin.

    ``weights`` are per-pair and multiply the residual, so passing the inverse
    localization precision makes this an inverse-variance fit.  For beads they
    are left out and the soft-L1 loss expresses geometric robustness alone;
    for a registration built from localizations they are the difference
    between a fit and a fit that believes its dimmest pairs.
    """
    def normalization(points):
        center = np.mean(points, axis=0)
        scale = np.sqrt(np.mean(np.sum((points-center)**2, axis=1)))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('degenerate projective coordinates')
        return np.array([[1/scale, 0, -center[0]/scale],
                         [0, 1/scale, -center[1]/scale], [0, 0, 1.]])
    # Validate spatial coverage independently of optimizer convergence.
    fit_projective(source, target)
    ns, nt = normalization(source), normalization(target)
    src, dst = map_points(ns, source), map_points(nt, target)
    h = nt @ initial @ np.linalg.inv(ns)
    if abs(h[2, 2]) < 1e-12:
        raise ValueError('projective transform crosses the coordinate center')
    h /= h[2, 2]
    def unpack(parameters):
        return np.r_[parameters, 1.].reshape(3, 3)
    w = None if weights is None else np.sqrt(np.asarray(weights, float))[:, None]
    def residual(parameters):
        error = (map_points(unpack(parameters), src)-dst)/nt[0, 0]
        return (error if w is None else error*w).ravel()
    fitted = optimize.least_squares(residual, h.ravel()[:8], loss='soft_l1',
                 f_scale=loss_scale, x_scale='jac', max_nfev=500,
                 ftol=1e-10, xtol=1e-10, gtol=1e-10)
    if not fitted.success or not np.isfinite(fitted.fun).all():
        raise ValueError('robust geometric projective refinement failed')
    h = np.linalg.inv(nt) @ unpack(fitted.x) @ ns
    h /= np.linalg.norm(h)
    if abs(np.linalg.det(h)) < 1e-15:
        raise ValueError('singular refined projective transformation')
    return h


@dataclass
class ProjectiveFit:
    transformation: np.ndarray
    accepted: np.ndarray
    round1_transformation: np.ndarray
    round1_inliers: np.ndarray
    round1_dxdy: np.ndarray
    dxdy: np.ndarray
    weights_xy: np.ndarray


def fit_dual_transform(source, target, settings, precision=None):
    """Two rounds: robust initialization, componentwise dx/dy screen and refit.

    The round-two training mask is determined by round-one residuals and stays
    fixed during refinement. No PSF shape measurements participate in either fit.
    """
    settings.validate()
    source, target = np.asarray(source, float), np.asarray(target, float)
    precision = None if precision is None else np.asarray(precision, float)
    # RANSAC stays unweighted: it is looking for the inlier set, and a precise
    # outlier is still an outlier
    initial, coarse, _ = robust_projective(source, target,
                    settings.reprojection_threshold_px, settings.min_pairs)
    # Half the axis limit, as documented, but never below MIN_LOSS_SCALE_PX:
    # with a smaller soft-L1 scale every residual sits deep in the linear
    # regime and the refinement stops converging, which would hide the real
    # cause -- an axis limit no pair can meet -- behind an optimizer failure.
    scale = max(settings.transform_axis_limit_px/2, MIN_LOSS_SCALE_PX)
    initial = refine_projective(source[coarse], target[coarse], initial, scale,
                                None if precision is None else precision[coarse])
    delta1 = map_points(initial, source)-target
    good = coarse & np.all(np.abs(delta1) <= settings.transform_axis_limit_px, axis=1)
    if good.sum() < settings.min_pairs:
        raise ValueError(f'only {good.sum()} pairs pass the dx/dy limit '
                         f'({settings.transform_axis_limit_px:g} px); require {settings.min_pairs}. '
                         'Inspect pairing/coverage or increase the transformation dx/dy limit.')
    h = refine_projective(source[good], target[good],
                          fit_projective(source[good], target[good]), scale,
                          None if precision is None else precision[good])
    delta = map_points(h, source)-target
    weights = np.zeros_like(delta)
    weights[good] = 1/np.sqrt(1+(delta[good]/scale)**2)
    return ProjectiveFit(h, good, initial, coarse, delta1, delta, weights)


def point_coverage_area(points):
    """Convex-hull area (px²), not a guarantee of interpolation accuracy."""
    if len(points) < 3:
        return 0.
    try:
        return float(ConvexHull(points).volume)
    except QhullError:
        return 0.


def check_geometry(geometry, shape, roi=None):
    """Check a later split frame against the geometry a transformation was found in.

    Shared by every kind of channel registration: the question -- is this the
    camera the transformation was measured on? -- does not depend on whether a
    PSF model came with it.
    """
    shape = tuple(shape[-2:])
    if geometry['coordinate_system'] == 'roi-local':
        if shape != tuple(geometry['image_shape']):
            raise ValueError('ROI-local calibration requires the original bead image dimensions')
        if roi is not None:
            known = [s['roi'] for s in geometry.get('sources', []) if s['roi'] is not None]
            if known and any(tuple(r) != tuple(roi) for r in known):
                raise ValueError('ROI differs from the ROI-local bead calibration')
        warnings.warn('ROI-local calibration: identical shape cannot verify identical camera location.', stacklevel=3)
    elif roi is None:
        warnings.warn('Camera ROI is required to apply the full-chip transformation.', stacklevel=3)
        raise ValueError('supply the target camera ROI before applying this transformation')
    elif len(roi) != 4 or tuple(roi[2:]) != (shape[1], shape[0]):
        raise ValueError('target ROI dimensions disagree with image dimensions')


def polynomial_terms(points, order, centre, scale):
    """Monomials of (x, y) up to `order`, in normalised coordinates.

    Normalised because the raw ones are chip pixels in the hundreds, and a
    cube of that is 10^8 against a constant term of 1 -- a design matrix no
    least-squares solve should be handed.
    """
    u, v = ((np.asarray(points, float) - centre)/scale).T
    return np.stack([u**i * v**(k-i) for k in range(order+1) for i in range(k+1)],
                    axis=1)


def n_polynomial_terms(order):
    return (order+1)*(order+2)//2


@dataclass
class Polynomial:
    """A bivariate polynomial map between the channels, fitted both ways.

    A projective map cannot represent radial distortion.  That distortion is
    by definition ``r' = r(1 + k1 r^2 + ...)``, which in components is

        u' = u(1 + k1(u^2 + v^2)) = u + k1(u^3 + u v^2)

    -- the leading term is *cubic*.  A quadratic polynomial spans no cubic
    monomial at all and so cannot describe it to any degree, which is why the
    order here is three and there is no option for two: measured on a
    distorted field, order two is indistinguishable from projective and order
    three is thirty times better.

    Both directions are fitted and stored, because a polynomial has no
    closed-form inverse.  They are therefore not exact inverses of each other;
    over the region they were fitted on the disagreement is far below the
    registration accuracy, and outside it neither is to be trusted anyway.
    """
    order: int
    forward_centre: np.ndarray       # secondary -> reference
    forward_scale: float
    forward_coef: np.ndarray         # (terms, 2)
    inverse_centre: np.ndarray       # reference -> secondary
    inverse_scale: float
    inverse_coef: np.ndarray

    def forward(self, points):
        return polynomial_terms(points, self.order, self.forward_centre,
                                self.forward_scale) @ self.forward_coef

    def _seed(self, points):
        return polynomial_terms(points, self.order, self.inverse_centre,
                                self.inverse_scale) @ self.inverse_coef

    def inverse(self, points, refine=2):
        """The reverse map, seeded by the fitted coefficients and then solved.

        Fitting a cubic to the inverse of a cubic is not exact -- on a field
        distorted by five per cent it is half a pixel out -- and half a pixel
        is not affordable here: this map places the partner ROI, and whatever
        it gets wrong goes into the fitter's link as if it were real.  So the
        fitted inverse is only a starting point, and a couple of Newton steps
        against the forward map take it to machine precision.

        The steps need the forward map to be locally invertible, which it is
        wherever it means anything.  Where it is not -- far outside the pairs,
        where a cubic folds over -- the step is dropped and the fitted seed
        stands, since nothing out there is trustworthy either way.
        """
        points = np.asarray(points, float)
        q = self._seed(points)
        step = 1e-3
        for _ in range(max(int(refine), 0)):
            f0 = self.forward(q)
            jx = (self.forward(q + [step, 0]) - f0)/step
            jy = (self.forward(q + [0, step]) - f0)/step
            det = jx[:, 0]*jy[:, 1] - jy[:, 0]*jx[:, 1]
            ok = np.abs(det) > 1e-12
            if not ok.any():
                break
            r = f0 - points
            dq = np.zeros_like(q)
            dq[ok, 0] = ( jy[ok, 1]*r[ok, 0] - jy[ok, 0]*r[ok, 1])/det[ok]
            dq[ok, 1] = (-jx[ok, 1]*r[ok, 0] + jx[ok, 0]*r[ok, 1])/det[ok]
            q = q - dq
        return q


class _Mapping:
    """Both directions between the channels, whatever describes them.

    `dualfit` asks only these two questions, so that adding a model here costs
    nothing there: `combine_peaks` used to invert a 3x3 itself, which quietly
    made the projective map the only one possible.
    """

    def to_reference(self, secondary_xy):
        if getattr(self, 'polynomial', None) is not None:
            return self.polynomial.forward(secondary_xy)
        return map_points(self.transformation, secondary_xy)

    def to_secondary(self, reference_xy):
        if getattr(self, 'polynomial', None) is not None:
            return self.polynomial.inverse(reference_xy)
        return map_points(np.linalg.inv(self.transformation), reference_xy)

    def transform(self, secondary_xy):
        return self.to_reference(secondary_xy)


@dataclass
class ChannelTransform(_Mapping):
    """Where the second channel is, and nothing else.

    A projective map from secondary to reference chip coordinates, with the
    split-frame geometry it applies to.  `DualColorCalibration` is this plus a
    PSF model per channel; a 2D two-colour fit needs only this much, which is
    why it can be measured from localizations (`smappy.calibrate.transform`)
    instead of from beads.
    """
    transformation: np.ndarray       # always present: the projective fit, and
    geometry: dict                   # the starting point of any better one
    parameters: dict
    polynomial: Optional[Polynomial] = None

    @property
    def model(self) -> str:
        return 'polynomial' if self.polynomial is not None else 'projective'

    def validate_image(self, shape, roi=None):
        check_geometry(self.geometry, shape, roi)


@dataclass
class DualColorCalibration(_Mapping):
    main: SplineCalibration
    secondary: SplineCalibration
    transformation: np.ndarray
    geometry: dict
    parameters: dict

    @property
    def psf(self):
        return self.main.psf

    @property
    def dz(self):
        return self.main.dz

    def z_index_to_nm(self, index):
        return self.main.z_index_to_nm(index)

    def as_transform(self):
        """Just the registration, for a fit that brings its own PSF model."""
        return ChannelTransform(self.transformation, self.geometry, self.parameters)

    def validate_image(self, shape, roi=None):
        """Check a later split frame before applying this calibration."""
        check_geometry(self.geometry, shape, roi)


@dataclass
class DualBeads:
    channels: list
    records: list
    projections: list
    sources: list
    settings: DualColorSettings
    geometry: dict
    main_points: np.ndarray
    secondary_points: np.ndarray
    pair_indices: np.ndarray
    unmatched: list
    initial_transformation: np.ndarray | None = None
    initial_residuals: np.ndarray | None = None


def collect_dual_beads(inputs, settings=None, progress=None):
    s = settings or DualColorSettings()
    s.validate()
    report = progress or (lambda message: None)
    if isinstance(inputs, BeadStack):
        inputs = [inputs]
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    inputs = list(inputs)
    if not inputs:
        raise ValueError('select bead acquisitions')
    stacks = inputs if isinstance(inputs[0], BeadStack) else [st for p in discover_acquisitions(inputs)
                    for st in read_bead_stacks(p, s.dz_nm)]
    shape = stacks[0].images.shape[-2:]
    axis = 1 if 'right-left' in s.layout else 0  # y,x array axes
    split = s.split_position if s.split_position is not None else shape[axis]//2
    if not 0 < split < shape[axis]:
        raise ValueError('split must lie inside the image')
    if any(st.images.shape[-2:] != shape for st in stacks):
        raise ValueError('bead acquisitions must share split-frame image dimensions')
    full_chip = all(st.roi is not None for st in stacks)
    if not full_chip:
        warnings.warn('Missing camera ROI: using internal image coordinates; saved calibration is ROI-local.', stacklevel=2)
        known = {st.roi for st in stacks if st.roi is not None}
        if len(known) > 1:
            raise ValueError('cannot pool differing known ROIs with missing ROI metadata')
    first_main = s.main_channel in ('left', 'upper')
    slices = [slice(0, split), slice(split, shape[axis])]
    order = [0, 1] if first_main else [1, 0]
    batches, projections, sources = [[], []], [], []
    origins = []
    for st in stacks:
        origin = np.array(st.roi[:2] if full_chip else (0, 0), float)
        origins.append(origin)
        projections.append(np.max(st.images, axis=0))
        sources.append({'path': st.source, 'axes': st.axes, 'roi': st.roi,
                        'shape': list(st.images.shape), 'z_nm': st.z_nm.tolist()})
        for ch, half in enumerate(order):
            region = [slice(None)]*3
            region[axis+1] = slices[half]
            batches[ch].append(BeadStack(st.images[tuple(region)], st.z_nm, st.source, st.axes))
    base = CalibrationSettings(**{k: v for k, v in asdict(s).items() if k in CalibrationSettings.__dataclass_fields__})
    base.min_beads = 1
    channels = [collect_beads(batch, base, report) for batch in batches]
    points = []
    from ..psf import GaussianPSF
    for ch, collection in enumerate(channels):
        positions = []
        half = order[ch]
        for rec, volume in zip(collection.records, collection.volumes):
            # Same objective planes in both halves; fit x/y only, never a channel z correction.
            mid = volume.shape[0]//2
            center = volume.shape[-1]//2
            roi = volume[max(0, mid-5):mid+6, center-6:center+7, center-6:center+7].mean(axis=0)
            roi = roi*rec['brightness_adu']+rec['background_adu']
            fit = GaussianPSF(sigma=2, elliptical=True).fit(np.array([roi], np.float32), iterations=75)
            xy = fit.theta[0, :2]
            if not np.isfinite(xy).all() or np.any(np.abs(xy-6) > 4):
                rec['brightness_accepted'] = False
                rec['brightness_reason'] = 'unreliable lateral fit'
                xy = np.array([6., 6.])
            local = np.array([rec['x_px'], rec['y_px']], float)
            local[1-axis] += split*half
            rec['full_image_xy'] = local.tolist()
            positions.append(local+origins[rec['stack']]+xy-6)
        points.append(np.asarray(positions))
    # Translation votes are collected within acquisitions after the declared reflection.
    approx = points[1].copy()
    coord_axis = 1-axis
    if 'mirrored' in s.layout:
        for i, rec in enumerate(channels[1].records):
            approx[i, coord_axis] = 2*(origins[rec['stack']][coord_axis]+split)-1-approx[i, coord_axis]
    else:
        approx[:, coord_axis] += split*(1 if not first_main else -1)
    diffs = []
    for si in range(len(stacks)):
        ia = [i for i, r in enumerate(channels[0].records) if r['stack'] == si and r['brightness_accepted']]
        ib = [i for i, r in enumerate(channels[1].records) if r['stack'] == si and r['brightness_accepted']]
        if ia and ib:
            diffs.extend((points[0][ia, None]-approx[None, ib]).reshape(-1, 2))
    if not diffs:
        raise ValueError('no brightness-qualified correspondence candidates')
    diffs = np.asarray(diffs)
    low = np.floor(diffs.min(axis=0)/2).astype(int)-4
    bins = np.rint(diffs/2).astype(int)-low
    hist = np.zeros(tuple(bins.max(axis=0)+5))
    np.add.at(hist, tuple(bins.T), 1)
    peak = np.array(np.unravel_index(np.argmax(ndimage.gaussian_filter(hist, 1)), hist.shape))
    approx += (peak+low)*2
    pairs = []
    for si in range(len(stacks)):
        ia = np.array([i for i, r in enumerate(channels[0].records) if r['stack'] == si], int)
        ib = np.array([i for i, r in enumerate(channels[1].records) if r['stack'] == si], int)
        if not len(ia) or not len(ib):
            continue
        distance, index = cKDTree(approx[ib]).query(points[0][ia])
        _, reverse = cKDTree(points[0][ia]).query(approx[ib])
        for k, (d, j) in enumerate(zip(distance, index)):
            if d <= 12 and reverse[j] == k:
                pairs.append((ia[k], ib[j]))
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    if len(pairs) < s.min_pairs:
        raise ValueError(f'only {len(pairs)} automatic pairs; require {s.min_pairs}')
    records = []
    for i, (a, b) in enumerate(pairs):
        ra, rb = channels[0].records[a], channels[1].records[b]
        record = dict(ra, id=i, channel_records=[ra, rb])
        record['x_px'], record['y_px'] = ra['full_image_xy']
        record['brightness_accepted'] = ra['brightness_accepted'] and rb['brightness_accepted']
        record['brightness_reason'] = '; '.join(f'{name}: {r["brightness_reason"]}' for name, r in
                    zip(('main', 'secondary'), (ra, rb)) if not r['brightness_accepted']) or 'accepted'
        records.append(record)
    geometry = {'layout': s.layout, 'main_channel': s.main_channel, 'split_position': split,
                'image_shape': list(shape), 'coordinate_system': 'camera-chip' if full_chip else 'roi-local',
                'sources': sources, 'mirror_axis_xy': coord_axis if 'mirrored' in s.layout else None,
                'convention': 'zero-based pixel centers; split is first index of second half'}
    unmatched = [[i for i in range(len(c.records)) if i not in set(pairs[:, ch])]
                 for ch, c in enumerate(channels)]
    return DualBeads(channels, records, projections, sources, s, geometry,
                     points[0][pairs[:, 0]], points[1][pairs[:, 1]], pairs, unmatched)


@dataclass
class DualColorResult:
    calibration: DualColorCalibration
    beads: DualBeads
    registration: object
    transform_residuals: np.ndarray
    initial_transformation: np.ndarray
    initial_residuals: np.ndarray
    transform_accepted: np.ndarray
    transform_fit: ProjectiveFit

    def __getattr__(self, name):
        if name in ('shifts', 'residuals', 'correlations', 'reasons', 'messages', 'raw_psf'):
            return getattr(self.registration, name)
        raise AttributeError(name)

    @property
    def accepted(self):
        return self.registration.accepted

    def save(self, path, overwrite=False):
        save_dual_color_calibration(path, self.calibration, self, overwrite)


def build_dual_calibration(beads, excluded=(), progress=None):
    s = beads.settings
    s.validate()
    report = progress or (lambda message: None)
    n = len(beads.records)
    excluded = set(excluded)
    if not excluded.issubset(range(n)):
        raise ValueError('unknown excluded pair ID')
    mask = np.array([i not in excluded and r['brightness_accepted'] for i, r in enumerate(beads.records)])
    if mask.sum() < s.min_pairs:
        raise ValueError(f'keep at least {s.min_pairs} brightness-qualified pairs')
    eligible = mask.copy()
    ids = np.flatnonzero(eligible)
    report(f'Transformation: two-round robust fit, {len(ids)} eligible pairs')
    fitted = fit_dual_transform(beads.secondary_points[ids], beads.main_points[ids], s)
    h, initial = fitted.transformation, fitted.round1_transformation
    mask[ids] = fitted.accepted
    coarse = np.zeros(n, bool)
    coarse[ids] = fitted.round1_inliers
    delta1 = map_points(initial, beads.secondary_points)-beads.main_points
    delta = map_points(h, beads.secondary_points)-beads.main_points
    weights = np.zeros((n, 2))
    weights[ids] = fitted.weights_xy
    fitted = ProjectiveFit(h, mask.copy(), initial, coarse, delta1, delta, weights)
    initial_residuals, errors = np.linalg.norm(delta1, axis=1), np.linalg.norm(delta, axis=1)
    report(f'Transformation: {mask.sum()}/{len(ids)} pairs pass dx/dy screening; '
           'joint PSF registration will not change this selection')
    # Check forward and inverse denominators across the images, not just at beads.
    height, width = beads.geometry['image_shape']
    inverse = np.linalg.inv(h)
    for source in beads.sources:
        origin = source['roi'][:2] if beads.geometry['coordinate_system'] == 'camera-chip' else (0, 0)
        corners = np.array([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]])+origin
        for matrix in (h, inverse):
            denominators = np.c_[corners, np.ones(4)] @ matrix[2]
            if np.min(denominators)*np.max(denominators) <= 0:
                raise ValueError('projective transform crosses infinity inside the camera image')
    pair_volumes, records, original_volumes, offsets = [], [], [], []
    mirror = beads.geometry['mirror_axis_xy']
    for i, (a, b) in enumerate(beads.pair_indices):
        ra, rb = beads.channels[0].records[a], beads.channels[1].records[b]
        va = beads.channels[0].volumes[a]*ra['brightness_adu']
        vb = beads.channels[1].volumes[b]*rb['brightness_adu']
        origin = np.array(beads.sources[ra['stack']]['roi'][:2] if beads.geometry['coordinate_system'] == 'camera-chip' else (0, 0))
        main_center = np.array(ra['full_image_xy'])+origin
        secondary_center = np.array(rb['full_image_xy'])+origin
        predicted = map_points(inverse, main_center[None])[0]
        # Match SMAP: subpixel extraction correction, no projective warp of the PSF.
        offset = predicted-secondary_center
        within_padding = np.max(np.abs(offset)) <= s.padding-1
        correction = np.array([0., -offset[1], -offset[0]])
        if mirror is not None:
            vb = np.flip(vb, axis=2-mirror)
            correction[2-mirror] *= -1
        nz = min(len(va), len(vb))
        va = va[(len(va)-nz)//2:(len(va)-nz)//2+nz]
        vb = vb[(len(vb)-nz)//2:(len(vb)-nz)//2+nz]
        scale = ra['brightness_adu']+rb['brightness_adu']
        original_volumes.append(np.stack((va, vb))/scale)
        offsets.append(np.stack((np.zeros(3), correction)))
        vb = ndimage.shift(vb, correction, order=3, mode='constant')
        pair_volumes.append(np.stack((va, vb))/scale)
        reason = ('projective initialization outlier' if not coarse[i] else 'transformation dx/dy outlier')
        if not eligible[i]:
            reason = beads.records[i]['brightness_reason']
        elif mask[i]:
            reason = 'accepted' if within_padding else 'transform-derived ROI correction exceeds padding'
        records.append(dict(beads.records[i], brightness_adu=scale,
                            brightness_accepted=bool(mask[i] and within_padding), brightness_reason=reason))
    # PSF support has its own minimum; projective identifiability was checked above.
    collection = BeadCollection(np.asarray(pair_volumes, np.float32), records,
                  beads.projections, beads.sources, beads.channels[0].dz_nm, s,
                  np.asarray(original_volumes, np.float32), np.asarray(offsets))
    result = build_calibration(collection, excluded, report)
    models = result.channel_calibrations
    # Rejected pairs are aligned only for diagnostics. Score them against the
    # completed reference as well, so exclusion does not erase their plot points.
    reference = np.asarray(result.channel_raw_psfs)
    crop, padding = result.z_crop_start, s.padding
    for i in np.flatnonzero(~result.accepted):
        sample = np.stack([ndimage.shift(result.beads.original_volumes[i, ch],
                    result.shifts[i]+result.beads.channel_offsets[i, ch], order=3,
                    mode='constant', cval=0.) for ch in range(2)])
        sample = sample[:, crop:sample.shape[1]-crop, padding:-padding, padding:-padding]
        result.residuals[i], result.correlations[i] = robust_shape_error(sample, reference)
    secondary = models[1]
    if mirror is not None:
        native = np.flip(secondary.psf, axis=2-mirror).copy()
        secondary = replace(secondary, psf=native, coeff=spline_coefficients(native))
    ratios = [beads.records[i]['channel_records'][1]['brightness_adu']/
              beads.records[i]['channel_records'][0]['brightness_adu'] for i in np.flatnonzero(result.accepted)]
    coverage = {name: point_coverage_area(beads.main_points[selected]) for name, selected in
                (('eligible', eligible), ('transformation', mask), ('psf', result.accepted))}
    result.messages.append(f'{mask.sum()} transformation pairs; {result.accepted.sum()} PSF pairs. '
                           'Shape rejection does not alter the transformation.')
    cal = DualColorCalibration(models[0], secondary, h, beads.geometry,
          {'settings': asdict(s), 'secondary_main_brightness_ratio': float(np.median(ratios)),
           'transform_method': 'two-round RANSAC / soft-L1 geometric refinement; dx/dy screening',
           'transform_selection': 'round-one abs(dx) and abs(dy); independent of PSF shape',
           'transform_loss_scale_px': s.transform_axis_limit_px/2,
           'transform_weight_meaning': 'per-axis soft-L1 influence weights; not localization precision',
           'transform_pair_count': int(mask.sum()), 'psf_pair_count': int(result.accepted.sum()),
           'coverage_area_px2': coverage,
           'intensity_unit': 'camera ADU; no photon conversion',
           'transform_direction': 'secondary to main, native camera coordinates',
           'registration': 'one shared z,y,x shift and amplitude per pair',
           'joint_psf_orientation': 'secondary mirrored to main orientation; no projective PSF warp'})
    if beads.initial_transformation is None and not excluded:
        beads.initial_transformation = initial.copy()
        beads.initial_residuals = initial_residuals.copy()
    return DualColorResult(cal, beads, result, errors,
            beads.initial_transformation.copy() if beads.initial_transformation is not None else initial,
            beads.initial_residuals.copy() if beads.initial_residuals is not None else initial_residuals,
            mask, fitted)


def calibrate_dual(inputs, settings=None, excluded=(), progress=None):
    return build_dual_calibration(collect_dual_beads(inputs, settings, progress), excluded, progress)


def save_dual_color_calibration(path, calibration, result=None, overwrite=False):
    import h5py
    path = Path(path)
    for model in (calibration.main, calibration.secondary):
        _validate_native_calibration(model)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, 'w') as f:
            f.attrs.update(format='smappy-dual-color-calibration', version=1)
            f['geometry_json'] = json.dumps(calibration.geometry)
            f['parameters_json'] = json.dumps(calibration.parameters)
            f['transformation'] = calibration.transformation
            for name, model in zip(('main', 'secondary'), (calibration.main, calibration.secondary)):
                g = f.create_group(name)
                g.attrs.update(dz=model.dz, z0=model.z0, x0=model.x0)
                g['parameters_json'] = json.dumps(model.parameters)
                g.create_dataset('coeff', data=model.coeff, compression='gzip')
                g.create_dataset('psf', data=model.psf, compression='gzip')
            if result is not None:
                g = f.create_group('diagnostics')
                r = result.registration
                for name, value in {'accepted': r.accepted, 'shifts_zyx': r.shifts,
                    'transform_accepted': result.transform_accepted,
                    'transform_weights_xy': result.transform_fit.weights_xy,
                    'transform_dxdy_px': result.transform_fit.dxdy,
                    'round1_transformation': result.transform_fit.round1_transformation,
                    'round1_transform_inliers': result.transform_fit.round1_inliers,
                    'round1_dxdy_px': result.transform_fit.round1_dxdy,
                    'coarse_shifts_zyx': r.coarse_shifts, 'residuals': r.residuals,
                    'channel_extraction_offsets_zyx': r.beads.channel_offsets,
                    'correlations': r.correlations, 'transform_residuals_px': result.transform_residuals,
                    'initial_transformation': result.initial_transformation,
                    'initial_residuals_px': result.initial_residuals,
                    'main_points': result.beads.main_points, 'secondary_points': result.beads.secondary_points,
                    'raw_joint_psfs': np.asarray(r.channel_raw_psfs),
                    'pair_indices': result.beads.pair_indices}.items():
                    g.create_dataset(name, data=value, compression='gzip')
                g['records_json'] = json.dumps(result.beads.records)
                g['reasons_json'] = json.dumps(r.reasons)
                g['unmatched_json'] = json.dumps(result.beads.unmatched)
                g['all_detections_json'] = json.dumps([c.records for c in result.beads.channels])
                if r.refits:
                    for ch, d in r.refits.items():
                        cg = g.create_group('refits_'+str(ch))
                        for name, value in d.items():
                            cg[name] = value
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_dual_color_calibration(path):
    import h5py
    with h5py.File(path, 'r') as f:
        if f.attrs.get('format') != 'smappy-dual-color-calibration' or f.attrs.get('version') != 1:
            raise ValueError('unsupported dual-color calibration format/version')
        models = []
        for name in ('main', 'secondary'):
            g = f[name]
            model = SplineCalibration(np.ascontiguousarray(g['coeff'][...]), float(g.attrs['dz']),
                    float(g.attrs['z0']), x0=float(g.attrs['x0']), psf=g['psf'][...],
                    em_mirror=False, source=Path(path), parameters=json.loads(g['parameters_json'][()]))
            _validate_native_calibration(model)
            models.append(model)
        if models[0].z0 != models[1].z0 or models[0].dz != models[1].dz or models[0].shape != models[1].shape:
            raise ValueError('dual PSFs must share exactly the same grid and z origin')
        h = f['transformation'][...]
        if h.shape != (3, 3) or not np.isfinite(h).all() or np.linalg.matrix_rank(h) != 3:
            raise ValueError('invalid projective transformation')
        return DualColorCalibration(*models, h, json.loads(f['geometry_json'][()]),
                                    json.loads(f['parameters_json'][()]))


def channel_result(result, channel):
    """Adapt a joint result to the existing independent channel refit diagnostics."""
    r = result.registration
    records = [dict(rec['channel_records'][channel], id=i) for i, rec in enumerate(result.beads.records)]
    # Use unresampled camera bead volumes for refitting in native channel orientation.
    original = result.beads.channels[channel]
    volumes = original.volumes[result.beads.pair_indices[:, channel]]
    collection = replace(r.beads, volumes=volumes, records=records)
    raw = r.channel_raw_psfs[channel]
    if channel == 1 and result.beads.geometry['mirror_axis_xy'] is not None:
        raw = np.flip(raw, 2-result.beads.geometry['mirror_axis_xy'])
    shifts = r.shifts+r.beads.channel_offsets[:, channel]
    if channel == 1 and result.beads.geometry['mirror_axis_xy'] is not None:
        shifts = shifts.copy()
        shifts[:, 2-result.beads.geometry['mirror_axis_xy']] *= -1
    return replace(r, calibration=(result.calibration.main if channel == 0 else result.calibration.secondary),
                   beads=collection, raw_psf=raw, shifts=shifts)
