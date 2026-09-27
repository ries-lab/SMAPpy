"""Simulated SMLM data: localizations, or the camera frames to fit.

One model with two outputs, so what is simulated as localizations can be
fitted from its frames by changing nothing but the output:

1. **structure** -- label positions, from a YAML of points, lines, circles,
   polygons and images at a density, and copies of them (`structure`);
2. **labelling** -- a label carries a fluorophore with some efficiency,
   one by default, optionally a Poisson number, off by a linkage error;
3. **blinking** -- on, off, bleached, in continuous time; the off-time set
   from the number of blinks in the measurement, the activation spread evenly
   or decaying; photons per blink from a distribution (`kinetics`);
4. **drift**, optionally;
5. either **localizations** -- Poisson photons, a detection limit, the
   Mortensen precision, close emitters removed or averaged
   (`localizations`) -- or **camera frames** -- pixel-integrated Gaussian
   spots, astigmatic if asked, on a flat background with shot and read noise,
   nothing removed (`camera`), saved as a recipe a fitter opens (`source`).

In the package rather than in ``scripts/`` because it is also a plugin
(``File/Simulate/Blinking Structure``): something to try the whole pipeline
on without a microscope.
"""
from .camera import (ASTIGMATISM, ISOLATED_NM, astigmatic_sigmas, bead_stacks,
                     camera_frames, camera_truth, dual_bead_stacks, dual_camera_frames,
                     dual_transformation, render)
from .kinetics import blink, emission, expected_blinks, label, off_time_for
from .localizations import (GroundTruth, ground_truth, localizations, mortensen,
                            simulate, unresolvable)
from .settings import (BlinkingSettings, CameraOutputSettings, LabellingSettings,
                       LocalizationOutputSettings, OpticsSettings, SimulationSettings,
                       StructureSettings)
from .structure import Labels, Structure, load_structure, presets

__all__ = [
    "ASTIGMATISM", "ISOLATED_NM", "astigmatic_sigmas", "bead_stacks", "camera_frames",
    "camera_truth", "dual_bead_stacks", "dual_camera_frames", "dual_transformation",
    "render", "blink", "emission", "expected_blinks", "label", "off_time_for",
    "GroundTruth", "ground_truth", "localizations", "mortensen", "simulate",
    "unresolvable", "BlinkingSettings", "CameraOutputSettings", "LabellingSettings",
    "LocalizationOutputSettings", "OpticsSettings", "SimulationSettings",
    "StructureSettings", "Labels", "Structure", "load_structure", "presets",
]
