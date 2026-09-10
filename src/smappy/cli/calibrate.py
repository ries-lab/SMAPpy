"""Create a single-channel or split-frame dual-color spline PSF calibration."""
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('paths', nargs='*', help='acquisitions or directories to pool')
    p.add_argument('--out', help='headless run: save a native calibration HDF5')
    p.add_argument('--dz', type=float, help='override objective z spacing in nm')
    p.add_argument('--roi-size', type=int, default=27)
    p.add_argument('--smooth-z', type=float, default=20., help='smoothing sigma in nm')
    p.add_argument('--layout', choices=('right-left', 'right-left mirrored', 'up-down', 'up-down mirrored'))
    p.add_argument('--main-channel', choices=('left', 'right', 'upper', 'lower'))
    p.add_argument('--split-position', type=int, help='first pixel index of second half; default midpoint')
    p.add_argument('--min-pairs', type=int, default=8)
    p.add_argument('--reprojection-threshold', type=float, default=2., help='initial RANSAC radius in pixels')
    p.add_argument('--transform-axis-limit', type=float, default=.15,
                   help='round-two screening: maximum absolute dx and dy in pixels')
    p.add_argument('--tk', action='store_true',
                   help='the old Tk interface; needs Tk 8.6 (macOS ships a broken 8.5)')
    args = p.parse_args()
    from ..calibrate import calibrate, CalibrationSettings
    settings = CalibrationSettings(dz_nm=args.dz, roi_size=args.roi_size,
                                   smooth_z_nm=args.smooth_z)
    if args.layout:
        from ..calibrate.dual import DualColorSettings, calibrate_dual
        settings = DualColorSettings(dz_nm=args.dz, roi_size=args.roi_size, smooth_z_nm=args.smooth_z,
                    layout=args.layout, main_channel=args.main_channel or ('left' if 'right-left' in args.layout else 'upper'),
                    split_position=args.split_position, min_pairs=args.min_pairs,
                    reprojection_threshold_px=args.reprojection_threshold,
                    transform_axis_limit_px=args.transform_axis_limit)
        calibrate = calibrate_dual
    elif args.main_channel or args.split_position is not None:
        p.error('--main-channel and --split-position require --layout')
    if args.out:
        result = calibrate(args.paths, settings, progress=print)
        result.save(args.out)
        print(f'Saved {args.out}: {result.accepted.sum()} accepted beads')
    else:
        if not args.tk:
            try:
                from ..calibrate.qt_gui import show_calibration_qt
            except ImportError:
                pass                      # no PySide6: fall through to Tk
            else:
                show_calibration_qt(args.paths, settings)
                return
        if args.layout:
            from ..calibrate.dual_gui import show_dual_calibration
            show_dual_calibration(args.paths, settings)
            return
        from ..calibrate.gui import show_calibration
        show_calibration(args.paths, settings)


if __name__ == '__main__':
    main()
