"""Single-molecule localization fitting: detection, MLE fitting, rendering.

The short way in::

    import smappy

    locs = smappy.fit(data, out="OUT.h5",
                      camera={"conversion": 6.7, "offset": 400,
                              "pixelsize_um": 0.127},
                      calibration="..._3dcal.mat")
    smappy.show("OUT.h5")       # the GUI, in a process of its own

`data` is a path, an image source, or frames already in memory.  Every stage is
also available on its own -- `smappy.fit` only assembles them, and the names
below are re-exports of the modules that define them.

They are resolved on first use rather than at import, so ``import smappy`` costs
nothing but this file: showing a table does not drag in Qt, and a fit does
not drag in scipy until a calibration is actually read.
"""

from __future__ import annotations

__all__ = [
    # the short way
    "fit", "view", "load",
    # settings and tables
    "CameraMetadata", "FitSettings", "Localizations",
    # the stages, for holding them yourself
    "open_stack", "open_ndtiff", "camera_metadata", "PeakFinder", "DoGFilter", "GaussFilter",
    "DynamicCutoff", "AbsoluteCutoff", "SplinePSF", "GaussianPSF",
    "load_spline_calibration", "LocalizationEngine", "fit_stack", "provenance",
    # results
    "LocalizationWriter", "save_localizations", "load_localizations",
    "LocFilter", "group", "GroupSettings",
    "FieldOfView", "RenderSettings", "RenderAxes", "DisplaySettings", "render_locs",
    "save_image", "show",
    "correct_drift", "DriftSettings",
    "LiveFit", "LiveSettings", "QueueSource", "queue_source",
    "CalibrationSettings", "save_spline_calibration", "DualColorSettings",
    "DualColorCalibration", "calibrate_dual", "load_dual_color_calibration", "__version__",
]

_EXPORTS = {
    "fit": "api", "view": "api", "load": "api",
    "CameraMetadata": "metadata",
    "FitSettings": "pipeline", "LocalizationEngine": "pipeline",
    "fit_stack": "pipeline", "provenance": "pipeline",
    "Localizations": "locs",
    "open_stack": "io.tiff", "camera_metadata": "io.tiff",
    "open_ndtiff": "io.ndtiff",
    "PeakFinder": "detect", "DoGFilter": "detect", "GaussFilter": "detect",
    "DynamicCutoff": "detect", "AbsoluteCutoff": "detect",
    "SplinePSF": "psf", "GaussianPSF": "psf",
    "load_spline_calibration": "io.calibration",
    "save_spline_calibration": "io.calibration",
    "CalibrationSettings": "calibrate",
    "DualColorSettings": "calibrate",
    "DualColorCalibration": "calibrate",
    "calibrate_dual": "calibrate",
    "load_dual_color_calibration": "calibrate",
    "LocalizationWriter": "io.hdf5", "save_localizations": "io.hdf5",
    "load_localizations": "io.hdf5",
    "LocFilter": "filter",
    "group": "group", "GroupSettings": "group",
    "FieldOfView": "render", "RenderSettings": "render", "RenderAxes": "render",
    "DisplaySettings": "render", "render_locs": "render",
    "save_image": "render",
    "show": "api",
    "correct_drift": "drift", "DriftSettings": "drift",
    "LiveFit": "live", "LiveSettings": "live",
    "QueueSource": "io.queue_source", "queue_source": "io.queue_source",
}


# ---- tifffile's complaints about Micro-Manager's metadata, kept out of the log
# Micro-Manager writes OME-TIFFs whose private tags tifffile cannot parse
# (tag 50839 with a value offset of 0, a display header of 0, ASCII tags in
# UTF-16), and tifffile logs an error for each one, on every page it reads, in
# every thread.  The images read correctly.  With no handler of its own, the
# ``tifffile`` logger printed these lines to the console, which `diagnostics`
# copies into ``smappy.log``.  In the first usage logs (2026-11) they were
# more than 600 lines across two bundles, most of each log, burying the errors
# that mattered.  So they are dropped at the logger, in every process that
# imports smappy, and only these: anything else tifffile says -- a series
# missing files, a page it cannot read -- still reaches the log.
#
# Here, standard library only, because every process that imports smappy
# needs it and `import smappy` loads no other module (`test_assistant`).
import logging as _logging
import re as _re

#: what tifffile says about metadata it could not parse while the image itself reads fine
TIFF_NOISE = _re.compile(r"<TiffTag\.fromfile> raised"
                         r"|coercing invalid ASCII to bytes"
                         r"|<tifffile\.read_micromanager_metadata> failed to read display settings")


class _TiffMetadataNoise(_logging.Filter):
    def filter(self, record):
        try:
            return TIFF_NOISE.search(record.getMessage()) is None
        except Exception:  # noqa: BLE001 -- a record that cannot be formatted is not ours to drop
            return True


if not any(isinstance(f, _TiffMetadataNoise) for f in _logging.getLogger("tifffile").filters):
    _logging.getLogger("tifffile").addFilter(_TiffMetadataNoise())


def _installed_version() -> str:
    """The installed version, ``smappy-smlm``'s.  Here, not in `pipeline` as it was, so that asking
    for it loads nothing: a training machine with SMAPpy's sources and no fitter asks too
    (`decode_smlm.training`)."""
    try:
        from importlib.metadata import version
        return version("smappy-smlm")
    except Exception:
        return "0.1.0.dev0+unknown"


def __getattr__(name: str):
    if name == "__version__":
        return _installed_version()
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'smappy' has no attribute '{name}'")
    from importlib import import_module
    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
