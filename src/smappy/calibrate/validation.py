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
