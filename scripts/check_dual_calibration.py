"""Validate split-frame calibration and an acquisition holdout without editing raw data."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from smappy.calibrate.dual import (DualColorSettings, collect_dual_beads, build_dual_calibration,
                                   channel_result, load_dual_color_calibration, map_points,
                                   fit_dual_transform)
from smappy.calibrate.validation import fit_bead_diagnostics


def transformation_holdouts(beads):
    """Refit both geometric rounds without the held-out acquisition/spatial tile.

    Detection and correspondences remain fixed; this tests geometric prediction,
    not an independent end-to-end detection/PSF pipeline or absolute accuracy.
    """
    eligible = np.array([r['brightness_accepted'] for r in beads.records])
    acquisitions = np.array([r['stack'] for r in beads.records])
    center = np.median(beads.main_points[eligible], axis=0)
    tiles = ((beads.main_points[:, 0] >= center[0]).astype(int) +
             2*(beads.main_points[:, 1] >= center[1]).astype(int))
    output = {}
    for kind, labels in (('acquisition', acquisitions), ('spatial_quadrant', tiles)):
        folds, pooled = [], []
        for label in np.unique(labels[eligible]):
            test = eligible & (labels == label)
            train = eligible & ~test
            fold = {'label': int(label), 'heldout_pair_ids': np.flatnonzero(test).tolist()}
            try:
                fitted = fit_dual_transform(beads.secondary_points[train], beads.main_points[train], beads.settings)
                delta = map_points(fitted.transformation, beads.secondary_points[test])-beads.main_points[test]
                errors = np.linalg.norm(delta, axis=1)
                fold.update(training_pairs=int(fitted.accepted.sum()), dxdy_px=delta.tolist(),
                            errors_px=errors.tolist())
                pooled.extend(errors.tolist())
            except ValueError as exc:
                fold['unavailable'] = str(exc)
            folds.append(fold)
        output[kind] = {'folds': folds, 'evaluated_pairs': len(pooled),
                       'unavailable_folds': sum('unavailable' in f for f in folds)}
        if pooled:
            output[kind].update(median_px=float(np.median(pooled)),
                               p90_px=float(np.percentile(pooled, 90)), max_px=float(np.max(pooled)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--layout', default='up-down mirrored')
    parser.add_argument('--main-channel', default='lower')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    beads = collect_dual_beads(args.path, DualColorSettings(layout=args.layout, main_channel=args.main_channel))
    def progress(message):
        if 'pass' in message or 'Transformation:' in message:
            print(message, flush=True)
    result = build_dual_calibration(beads, progress=progress)
    checks = [fit_bead_diagnostics(channel_result(result, ch)) for ch in range(2)]
    result.registration.refits = {str(ch): d for ch, d in enumerate(checks)}
    result.save(args.out/'dual_calibration.h5')
    loaded = load_dual_color_calibration(args.out/'dual_calibration.h5')
    np.testing.assert_array_equal(loaded.secondary.coeff, result.calibration.secondary.coeff)
    held_ids = [i for i, r in enumerate(beads.records) if r['stack'] == len(beads.sources)-1]
    held = build_dual_calibration(beads, excluded=held_ids, progress=progress)
    held_checks = [fit_bead_diagnostics(channel_result(held, ch), bead_ids=held_ids) for ch in range(2)]
    held.registration.refits = {str(ch): d for ch, d in enumerate(held_checks)}
    held.save(args.out/'holdout_calibration.h5')
    held_errors = np.linalg.norm(held.calibration.transform(beads.secondary_points[held_ids])-
                                beads.main_points[held_ids], axis=1)
    def metrics(d):
        use = d['bead_id'] >= 0
        values = np.abs(d['centered_error_nm'][use])
        return {'planes': int(use.sum()), 'failed': int((~d['fit_valid'][use]).sum()),
                'boundaries': int(d['at_z_boundary'][use].sum()),
                'median_abs_centered_z_nm': float(np.nanmedian(values)),
                'p90_abs_centered_z_nm': float(np.nanpercentile(values, 90))}
    report = {'pairs': len(beads.records), 'accepted': int(result.accepted.sum()),
              'transformation_pairs': int(result.transform_accepted.sum()),
              'coverage_area_px2': result.calibration.parameters['coverage_area_px2'],
              'transformation_holdouts': transformation_holdouts(beads),
              'psf_knots': list(loaded.main.psf.shape), 'dz_nm': loaded.main.dz,
              'geometry': loaded.geometry, 'seconds': time.monotonic()-started,
              'median_projective_residual_px': float(np.median(result.transform_residuals[result.transform_accepted])),
              'max_projective_residual_px': float(np.max(result.transform_residuals[result.transform_accepted])),
              'secondary_main_ratio': loaded.parameters['secondary_main_brightness_ratio'],
              'in_sample': [metrics(d) for d in checks], 'heldout_pairs': held_ids,
              'heldout_training_pairs': int(held.accepted.sum()),
              'heldout_transform_training_pairs': int(held.transform_accepted.sum()),
              'heldout_projective_residuals_px': held_errors.tolist(),
              'heldout': [metrics(d) for d in held_checks],
              'rejection_reasons': result.reasons}
    (args.out/'metrics.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    colors = np.where(result.accepted, 'green', np.where(result.transform_accepted, 'darkorange', 'red'))
    delta = result.calibration.transform(beads.secondary_points)-beads.main_points
    vectors = axes[0, 0].quiver(*beads.main_points.T, *delta.T, color=colors, angles='xy')
    axes[0, 0].quiverkey(vectors, .85, .07, .1, '0.1 px', coordinates='axes', labelpos='N')
    axes[0, 0].margins(.15)
    axes[0, 0].set(title='Projective residual vectors', xlabel='Main x (camera pixels)', ylabel='Main y (camera pixels)')
    axes[1, 0].scatter(result.shifts[:, 0]*loaded.main.dz, result.residuals, c=colors)
    axes[1, 0].set(title='Pair rejection', xlabel='Shared z shift (nm)', ylabel='Shape residual')
    for ch, name in enumerate(('Main', 'Secondary')):
        model = loaded.main if ch == 0 else loaded.secondary
        axes[0, ch+1].imshow(model.psf[:, model.psf.shape[1]//2], aspect='auto', cmap='inferno',
                  extent=(0, model.psf.shape[2]-1, model.z_index_to_nm(model.psf.shape[0]-1), model.z_index_to_nm(0)))
        axes[0, ch+1].set(title=name+' PSF, native xz', xlabel='x (pixels)', ylabel='Emitter z (nm)')
        d = held_checks[ch]
        for i in held_ids:
            use = d['bead_id'] == i
            axes[1, ch+1].plot(d['expected_z_nm'][use], d['centered_error_nm'][use], '.-', lw=.8)
        axes[1, ch+1].set(title=name+' held-out z refits', xlabel='Expected emitter z (nm)', ylabel='Centered error (nm)')
    fig.savefig(args.out/'review.png', dpi=150)
    print(json.dumps({k: v for k, v in report.items() if k != 'geometry'}, indent=2))


if __name__ == '__main__':
    main()
