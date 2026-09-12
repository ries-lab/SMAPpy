"""Analysis ROI projects: the model, the plugins and their storage.

The interface lives in the GUI (`smappy.gui.roi_window` and the control
window's ROI tab), which drives this model over the session's files.  A
project can also be built and evaluated from a script, with no window::

    from smappy.roi_manager import ROIProject

    project = ROIProject()
    source = project.add_file("localizations.h5")
    project.set_geometry(300, "circle")
    project.find(source.id)
    project.evaluate()
    rows = project.results()

The plugins themselves live in `smappy.plugins.roi`, with everything else, so
that the folder tree stays the plugin tree; they are re-exported here because
this is where one goes looking for them.
"""
from .core import ROI, ROIProject
from ..plugins.roi import (DensityPeaks, Histograms, Statistics, density_peaks,
                           histograms, site_statistics)
from .pipeline import Step

__all__ = ['ROI', 'ROIProject', 'Step', 'DensityPeaks', 'Statistics', 'Histograms',
           'density_peaks', 'site_statistics', 'histograms']
