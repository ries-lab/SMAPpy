"""Build a calibration and refit an independently held-out acquisition.

Example:
    python scripts/check_bead_calibration.py /data/beads --out /tmp/bead_check

The last discovered acquisition is held out by default. Output contains a native
calibration, JSON metrics, and a PNG review. Raw acquisitions are never modified.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
from smappy.calibrate import collect_beads, build_calibration
from smappy.calibrate.validation import fit_bead_diagnostics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('paths', nargs='+')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--holdout-stack', type=int, default=-1, help='zero-based stack index; default last')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    beads = collect_beads(args.paths, progress=print)
    result = build_calibration(beads, progress=print)
    elapsed = time.monotonic()-start
    result.refits = fit_bead_diagnostics(result)
    result.save(args.out/'beads_calibration.h5')
    if not -len(beads.sources) <= args.holdout_stack < len(beads.sources):
        raise ValueError('holdout stack index is outside the discovered stacks')
    stack = args.holdout_stack % len(beads.sources)
    held_ids = [r['id'] for r in beads.records if r['stack'] == stack]
    if not held_ids:
        raise ValueError('the held-out stack has no detected beads; choose another stack')
    held = build_calibration(beads, excluded=held_ids, progress=print)
    check = fit_bead_diagnostics(held, held_ids)
    held.refits = check
    held.save(args.out/'heldout_calibration.h5')
    def metrics(d):
        error = np.abs(d['centered_error_nm'])
        return {'planes': len(error), 'failed': int((~d['fit_valid']).sum()),
                'boundary': int(d['at_z_boundary'].sum()),
                'median_absolute_centered_error_nm': float(np.nanmedian(error)),
                'p90_absolute_centered_error_nm': float(np.nanpercentile(error, 90))}
    report = {'detected': len(beads.records), 'accepted': int(result.accepted.sum()),
              'shape': list(result.calibration.psf.shape), 'dz_nm': beads.dz_nm,
              'seconds_including_io': elapsed, 'in_sample': metrics(result.refits),
              'heldout_source': beads.sources[stack], 'heldout_beads': held_ids,
              'heldout_training_accepted': int(held.accepted.sum()), 'heldout': metrics(check)}
    (args.out/'metrics.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    psf = result.calibration.psf
    ax[0, 0].imshow(psf[len(psf)//2], cmap='inferno')
    ax[0, 0].set(title='Average PSF, center plane', xlabel='x (pixels)', ylabel='y (pixels)')
    zextent = result.calibration.z_index_to_nm(np.array([len(psf)-1, 0]))
    ax[0, 1].imshow(psf[:, psf.shape[1]//2], aspect='auto', cmap='inferno',
                    extent=(0, psf.shape[2]-1, *zextent))
    ax[0, 1].set(title='Average PSF, xz', xlabel='x (pixels)', ylabel='Emitter z (nm)')
    for axes, d, title in [(ax[1, 0], result.refits, 'In-sample refits'),
                           (ax[1, 1], check, 'Held-out acquisition')]:
        for i in np.unique(d['bead_id']):
            use = d['bead_id'] == i
            axes.plot(d['expected_z_nm'][use], d['centered_error_nm'][use], '.-', ms=3, lw=.6)
        axes.axhline(0, color='black', lw=.5)
        axes.set(title=title, xlabel='Expected emitter z (nm)', ylabel='Centered z error (nm)')
    fig.savefig(args.out/'review.png', dpi=150)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
