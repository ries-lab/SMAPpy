"""Diagnostic refitting of bead planes with the production spline fitter."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def aligned_midline_profiles(result, bead_ids=None, normalize=True):
    """Return lateral-x and axial-z midlines from aligned beads and their average."""
    beads = result.beads
    ids = list(np.flatnonzero(result.accepted) if bead_ids is None else bead_ids)
    if not ids or not set(ids).issubset(range(len(beads.records))):
        raise ValueError('select at least one known bead ID')
    p = beads.settings.padding
    crop_z = result.z_crop_start
    x_profiles, z_profiles = [], []
    for i in ids:
        volume = ndimage.shift(beads.volumes[i], result.shifts[i], order=3,
                               mode='constant', cval=0.)
        volume = volume[crop_z:volume.shape[0]-crop_z, p:volume.shape[1]-p,
                        p:volume.shape[2]-p]
        zc, yc, xc = (n//2 for n in volume.shape)
        x_profiles.append(volume[zc, yc, :])
        z_profiles.append(volume[:, yc, xc])
    average = result.raw_psf
    zc, yc, xc = (n//2 for n in average.shape)
    average_x = average[zc, yc, :].copy()
    average_z = average[:, yc, xc].copy()
    x_profiles = np.asarray(x_profiles)
    z_profiles = np.asarray(z_profiles)
    if normalize:
        def normalized(values):
            scale = np.max(values, axis=-1, keepdims=True)
            return values/np.maximum(scale, np.finfo(float).eps)
        x_profiles = normalized(x_profiles)
        z_profiles = normalized(z_profiles)
        average_x = normalized(average_x[None, :])[0]
        average_z = normalized(average_z[None, :])[0]
    return {'bead_id': np.asarray(ids),
            'x_px': np.arange(average.shape[2])-(average.shape[2]-1)/2,
            'z_nm': result.calibration.z_index_to_nm(np.arange(average.shape[0])),
            'x_profiles': x_profiles, 'z_profiles': z_profiles,
            'average_x': average_x, 'average_z': average_z}


def fit_bead_diagnostics(result, bead_ids=None, plane_stride=5):
    """Return fitted-vs-expected z, xy, and residuals in camera ADU units.

    Uses several z starts and the best likelihood to expose ambiguous PSFs.
    Fits on calibration beads are in-sample checks, not accuracy estimates.
    For held-out validation pass IDs excluded during build_calibration.
    CRLBs are deliberately not reported: ADU noise is not a photon noise model.
    """
    from ..psf import SplinePSF
    cal, beads = result.calibration, result.beads
    if not hasattr(beads, 'volumes'):
        raise TypeError('a dual-colour result has two channels per bead: use '
                        'fit_paired_bead_diagnostics for the two-channel fit, '
                        'or calibrate.dual.channel_result(result, channel) to '
                        'look at one channel on its own')
    if not isinstance(plane_stride, (int, np.integer)) or plane_stride < 1:
        raise ValueError('plane_stride must be a positive integer')
    ids = list(np.flatnonzero(result.accepted) if bead_ids is None else bead_ids)
    if not set(ids).issubset(range(len(beads.records))):
        raise ValueError('unknown bead ID')
    size = beads.settings.roi_size - 4  # leave two pixels for fitted lateral shifts
    radius = size//2
    c = beads.volumes.shape[-1]//2
    raw, expected, identity = [], [], []
    for i in ids:
        rec = beads.records[i]
        for plane in range(2, beads.volumes.shape[1]-2, plane_stride):
            index = plane+result.shifts[i, 0]-result.z_crop_start
            if not 2 <= index <= cal.shape[0]-2:
                continue  # only compare planes inside the model's supported z interval
            # Restore the original camera values, including the single scalar
            # background removed during extraction. Do not subtract plane minima.
            roi = beads.volumes[i, plane, c-radius:c+radius+1, c-radius:c+radius+1]
            roi = roi*rec['brightness_adu']+rec['background_adu']
            raw.append(roi)
            expected.append(cal.z_index_to_nm(index))
            identity.append((i, plane))
    # Fit the unsmoothed aligned average as its own diagnostic curve. Restore a
    # representative camera scale because the production fitter uses a Poisson
    # likelihood while the stored average has unit peak-plane integral.
    average_brightness = float(np.median([
        beads.records[i]['brightness_adu'] for i in ids]))
    average_background = float(np.median([
        beads.records[i]['background_adu'] for i in ids]))
    average = result.raw_psf
    ac = average.shape[-1]//2
    for plane in range(2, average.shape[0]-2, plane_stride):
        roi = average[plane, ac-radius:ac+radius+1, ac-radius:ac+radius+1]
        raw.append(np.maximum(roi*average_brightness+average_background, 0))
        expected.append(cal.z_index_to_nm(plane))
        identity.append((-1, plane))
    if not raw:
        raise ValueError('no bead planes available for validation')
    raw = np.ascontiguousarray(raw, dtype=np.float32)
    best = None
    zmax = cal.shape[0]-1
    for z in np.linspace(1, zmax, 5):
        fit = SplinePSF(cal, z_start_nm=float(cal.z_index_to_nm(z))).fit(raw, iterations=75)
        if best is None:
            best = fit
        else:
            keep = np.isfinite(fit.logl) & (~np.isfinite(best.logl) | (fit.logl > best.logl))
            for key in ('theta', 'crlb', 'logl', 'iterations'):
                getattr(best, key)[keep] = getattr(fit, key)[keep]
    fitted = cal.z_index_to_nm(best.theta[:, 4])
    valid = np.isfinite(best.logl) & np.isfinite(best.theta).all(axis=1)
    fitted[~valid] = np.nan
    expected = np.asarray(expected)
    identity = np.asarray(identity)
    # Global z shift is not identifiable on held-out beads. Report both the raw
    # reference and the per-bead centered error, which measures curve distortion.
    error = fitted-expected
    centered = error.copy()
    for i in list(ids)+[-1]:
        use = identity[:, 0] == i
        if np.any(use & valid):
            centered[use] -= np.median(error[use & valid])
    return {'bead_id': identity[:, 0], 'plane': identity[:, 1],
            'expected_z_nm': expected, 'fitted_z_nm': fitted,
            'centered_error_nm': centered, 'x_px': best.theta[:, 0],
            'y_px': best.theta[:, 1], 'log_likelihood': best.logl,
            'fit_valid': valid,
            'at_z_boundary': (best.theta[:, 4] < 1) | (best.theta[:, 4] > cal.shape[0]-1)}


def paired_bead_rois(result, bead_ids=None, plane_stride=5):
    """Both channels of every bead pair, as camera ADU, plane by plane.

    The paired collection holds the two channels already brought into one
    frame -- the secondary de-mirrored and shifted onto the main by the
    transformation -- which is the frame its channel models were built in, so
    the pair can go straight into the global fitter with an identity link.
    """
    r = result.registration
    beads = r.beads
    if beads.volumes.ndim != 5:
        raise ValueError('paired diagnostics need a dual-colour result')
    if not isinstance(plane_stride, (int, np.integer)) or plane_stride < 1:
        raise ValueError('plane_stride must be a positive integer')
    ids = list(np.flatnonzero(r.accepted) if bead_ids is None else bead_ids)
    if not set(ids).issubset(range(len(beads.records))):
        raise ValueError('unknown bead pair ID')
    cal = r.calibration
    size = beads.settings.roi_size-4    # leave two pixels for fitted lateral shifts
    radius = size//2
    c = beads.volumes.shape[-1]//2
    cut = (slice(None), slice(c-radius, c+radius+1), slice(c-radius, c+radius+1))
    raw, expected, identity = [], [], []
    for i in ids:
        rec = beads.records[i]
        # brightness_adu of a pair is the sum of both channels', which is what
        # the paired volumes were divided by; the background was removed per
        # channel and has to go back per channel.
        background = np.array([[[part['background_adu']]]
                               for part in rec['channel_records']], float)
        for plane in range(2, beads.volumes.shape[2]-2, plane_stride):
            index = plane+r.shifts[i, 0]-r.z_crop_start
            if not 2 <= index <= cal.shape[0]-2:
                continue
            pair = beads.volumes[i][(slice(None), plane)+cut[1:]]
            raw.append(pair*rec['brightness_adu']+background)
            expected.append(cal.z_index_to_nm(index))
            identity.append((i, plane))
    # The aligned average pair as its own curve, at a representative scale.
    average = np.asarray(r.channel_raw_psfs, float)
    brightness = float(np.median([beads.records[i]['brightness_adu'] for i in ids]))
    background = np.array([[[np.median([beads.records[i]['channel_records'][ch]
                                        ['background_adu'] for i in ids])]]
                           for ch in range(average.shape[0])], float)
    ac = average.shape[-1]//2
    acut = (slice(None), slice(ac-radius, ac+radius+1), slice(ac-radius, ac+radius+1))
    for plane in range(2, average.shape[1]-2, plane_stride):
        pair = average[(slice(None), plane)+acut[1:]]
        raw.append(np.maximum(pair*brightness+background, 0))
        expected.append(cal.z_index_to_nm(plane))
        identity.append((-1, plane))
    if not raw:
        raise ValueError('no bead planes available for validation')
    return (np.ascontiguousarray(raw, np.float32), np.asarray(expected),
            np.asarray(identity))


def fit_paired_bead_diagnostics(result, bead_ids=None, plane_stride=5):
    """Refit the calibration beads with the *two-channel* fitter.

    This is the workflow a dual-colour dataset is actually fitted with --
    `smappy.dualfit`'s global spline with x, y and z shared and the photons
    free per channel -- run backwards on the beads that built the calibration.
    There is one z per bead *pair*, not one per channel, which is the whole
    point of the global fit, so the result is a single set of curves.

    In-sample, like the single-channel version: for a real accuracy estimate
    pass IDs that were excluded when the calibration was built.
    """
    from ..dualfit import LINK_XYZ
    from ..psf import GlobalSplinePSF

    r = result.registration
    models = list(r.channel_calibrations)
    if len(models) != 2:
        raise ValueError('paired diagnostics need two channel models')
    raw, expected, identity = paired_bead_rois(result, bead_ids, plane_stride)
    cal = r.calibration
    model = GlobalSplinePSF(tuple(models), shared=LINK_XYZ)
    # The production link, on ROIs that are already aligned: no sub-pixel
    # offset and no local scale left to correct, only the splitter's photon
    # ratio, so that channel 1's photons are read in channel 0's units.
    link = np.zeros((len(raw), 2, 2, 5), np.float32)
    link[:, 1] = 1.0
    link[:, 1, 1, 2] = float(result.calibration.parameters.get(
        'secondary_main_brightness_ratio', 1.0)) or 1.0

    best = None
    zmax = cal.shape[0]-1
    for z in np.linspace(1, zmax, 5):
        model.z_start_nm = float(cal.z_index_to_nm(z))
        fit = model.fit(raw, link=link, iterations=75)
        if best is None:
            best = fit
        else:
            keep = np.isfinite(fit.logl) & (~np.isfinite(best.logl) | (fit.logl > best.logl))
            for key in ('theta', 'crlb', 'logl', 'iterations'):
                getattr(best, key)[keep] = getattr(fit, key)[keep]
    values = model.unpack(best)
    fitted = values['z_nm']
    valid = np.isfinite(best.logl) & np.isfinite(best.theta).all(axis=1)
    fitted = np.where(valid, fitted, np.nan)
    error = fitted-expected
    centered = error.copy()
    for i in list(np.unique(identity[:, 0])):
        use = identity[:, 0] == i
        if np.any(use & valid):
            centered[use] -= np.median(error[use & valid])
    z_index = best.theta[:, model.slot(4)]
    return {'bead_id': identity[:, 0], 'plane': identity[:, 1],
            'expected_z_nm': expected, 'fitted_z_nm': fitted,
            'centered_error_nm': centered,
            'x_px': values['x_roi'], 'y_px': values['y_roi'],
            'ratio': np.where(valid, values['ratio'], np.nan),
            'photons': values['photons'],
            'log_likelihood': best.logl, 'fit_valid': valid,
            'at_z_boundary': (z_index < 1) | (z_index > cal.shape[0]-1)}
