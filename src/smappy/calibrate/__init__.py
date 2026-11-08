"""Experimental spline PSF calibration from bead z-stacks.

The names below are loaded on first use (PEP 562), not when the package is
imported.  Any `smappy.calibrate.<module>` import runs this file first, and
eager imports of `core`, `dual` and `transform` cost about half a second
(scipy.signal, which pulls in scipy.stats, and scipy.interpolate).  That cost
landed on light lookups too: Simulate's form imports `microscope_psf` to list
the microscopes' pupils, and opening it took a second after a restart.
"""
import importlib

_WHERE = {
    'input': ['BeadStack', 'discover_acquisitions', 'read_bead_stacks'],
    'core': ['CalibrationSettings', 'CalibrationResult', 'collect_beads',
             'build_calibration', 'calibrate'],
    'dual': ['DualColorSettings', 'DualColorCalibration', 'DualColorResult',
             'ChannelTransform', 'collect_dual_beads', 'build_dual_calibration',
             'calibrate_dual', 'load_dual_color_calibration',
             'save_dual_color_calibration'],
    'transform': ['ArrangementSettings', 'RegisterSettings', 'RegistrationResult',
                  'register_channels', 'load_channel_transform', 'load_transform',
                  'save_channel_transform'],
}
_MODULE_OF = {name: module for module, names in _WHERE.items() for name in names}

__all__ = list(_MODULE_OF)


def __getattr__(name):
    module = _MODULE_OF.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value          # once found, a plain attribute
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
