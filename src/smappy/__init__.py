"""Single-molecule localization fitting: detection, MLE fitting, rendering.

The short way in::

    import smappy

    locs = smappy.fit(data, out="OUT.h5",
                      camera={"conversion": 6.7, "offset": 400,
                              "pixelsize_um": 0.127},
                      calibration="..._3dcal.mat")
    smappy.view("OUT.h5")

`data` is a path, an image source, or frames already in memory.  Every stage is
also available on its own -- `smappy.fit` only assembles them, and the names
below are re-exports of the modules that define them.

They are resolved on first use rather than at import, so ``import smappy`` costs
nothing but this file: the viewer does not drag in matplotlib, and a fit does
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
    "FieldOfView", "RenderSettings", "DisplaySettings", "render_locs",
    "save_image", "show",
    "correct_drift", "DriftSettings",
    "LiveFit", "LiveSettings", "live_view", "QueueSource", "queue_source",
    "__version__",
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
    "LocalizationWriter": "io.hdf5", "save_localizations": "io.hdf5",
    "load_localizations": "io.hdf5",
    "LocFilter": "filter",
    "group": "group", "GroupSettings": "group",
    "FieldOfView": "render", "RenderSettings": "render",
    "DisplaySettings": "render", "render_locs": "render",
    "save_image": "render",
    "show": "viewer",
    "correct_drift": "drift", "DriftSettings": "drift",
    "LiveFit": "live", "LiveSettings": "live", "live_view": "live",
    "QueueSource": "io.queue_source", "queue_source": "io.queue_source",
}


def __getattr__(name: str):
    if name == "__version__":
        from .pipeline import _version
        return _version()
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'smappy' has no attribute '{name}'")
    from importlib import import_module
    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
