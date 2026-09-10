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
"""
from .core import ROI, ROIProject
from .plugins import DensityPeaks, Histograms, Statistics

__all__ = ['ROI', 'ROIProject', 'DensityPeaks', 'Statistics', 'Histograms']
