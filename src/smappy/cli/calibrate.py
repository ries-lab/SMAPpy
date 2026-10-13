"""Create a single-channel or split-frame dual-color spline PSF calibration."""
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('paths', nargs='*', help='acquisitions or directories to pool')
    p.add_argument('--out', help='headless run: save a native calibration HDF5')
    p.add_argument('--dz', type=float, help='override objective z spacing in nm')
    p.add_argument('--roi-size', type=int, default=27)
    p.add_argument('--smooth-z', type=float, default=20., help='smoothing sigma in nm')
    p.add_argument('--layout', choices=('auto', 'right-left', 'right-left mirrored', 'up-down',
                                        'up-down mirrored'),
                   help='a split camera: dual colour; auto detects the layout from the beads')
    p.add_argument('--main-channel', choices=('left', 'right', 'upper', 'lower'),
                   help='default: the left or upper half')
    p.add_argument('--split-position', type=int, help='first pixel index of second half; default midpoint')
    p.add_argument('--min-pairs', type=int, help='default 8')
    p.add_argument('--reprojection-threshold', type=float, help='initial RANSAC radius in pixels; default 2')
    p.add_argument('--transform-axis-limit', type=float,
                   help='round-two screening: maximum absolute dx and dy in pixels; default 0.15')
    args = p.parse_args()
    from ..calibrate import calibrate, CalibrationSettings
    settings = CalibrationSettings(dz_nm=args.dz, roi_size=args.roi_size,
                                   smooth_z_nm=args.smooth_z)
    if args.layout:
        from dataclasses import replace
        from ..calibrate.dual import DualColorSettings, bead_registration, calibrate_dual
        given = {'layout': args.layout, 'main_channel': args.main_channel,
                 'split_position': args.split_position, 'min_pairs': args.min_pairs,
                 'reprojection_threshold_px': args.reprojection_threshold,
                 'transform_axis_limit_px': args.transform_axis_limit}
        registration = replace(bead_registration(),
                               **{k: v for k, v in given.items() if v is not None})
        settings = DualColorSettings(dz_nm=args.dz, roi_size=args.roi_size, smooth_z_nm=args.smooth_z,
                                     registration=registration)
        calibrate = calibrate_dual
    elif args.main_channel or args.split_position is not None:
        p.error('--main-channel and --split-position require --layout')
    if args.out:
        result = calibrate(args.paths, settings, progress=print)
        result.save(args.out)
        print(f'Saved {args.out}: {result.accepted.sum()} accepted beads')
    else:
        from ..calibrate.qt_gui import show_calibration_qt
        show_calibration_qt(args.paths, settings)


if __name__ == '__main__':
    main()
