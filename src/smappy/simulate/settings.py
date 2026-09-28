"""The settings of a simulation, one dataclass per stage.

Shared by the two outputs -- localizations and camera frames -- so a
simulation that was looked at as localizations can be fitted from its frames
with nothing changed but the output, and by the plugin, whose form is built
from these fields.  A camera-frame simulation is also saved as exactly this
(`smappy.simulate.source`), which is what lets a fitter open one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..plugins import param
from .structure import preset_choices

STRUCTURE_FILTER = "Structures (*.yaml *.yml);;All files (*)"
SIMULATION_FILTER = "Simulations (*.sim.yaml);;All files (*)"


@dataclass
class StructureSettings:
    """What is labelled (`smappy.simulate.structure`)."""
    preset: str = param("demo", label="structure", choices=lambda: preset_choices(),
                        help="a built-in structure; its YAML in smappy/data/structures "
                             "is an example of the syntax")
    file: str = param("", label="structure file", kind="open_file",
                      file_filter=STRUCTURE_FILTER,
                      help="a structure YAML: label positions, or lines, circles, "
                           "polygons and images at a density, and copies of them; "
                           "wins over the built-in when set")

    def source(self) -> str:
        return self.file or self.preset


@dataclass
class LabellingSettings:
    """Labels -> fluorophores."""
    efficiency: float = param(1.0, label="labelling efficiency", min=0, max=1,
                              help="the probability that a label carries a fluorophore")
    fluorophores: float = param(1.0, label="fluorophores per label", min=0,
                                help="exactly this many on a label that is labelled "
                                     "(rounded), or its mean with Poisson ticked")
    poisson: bool = param(False, label="Poisson number",
                          help="the number of fluorophores per label is Poisson")
    linkage_nm: float = param(0.0, label="linkage error, fixed", unit="nm", min=0,
                              help="each fluorophore is off its label by a Gaussian "
                                   "of this width per axis, the same for all its "
                                   "blinks: a rigid linker or antibody")
    linkage_free_nm: float = param(0.0, label="linkage error, free", unit="nm", min=0,
                                   help="a Gaussian offset of this width per axis "
                                        "drawn anew for every blink: a dye that "
                                        "turns freely on a flexible linker")


@dataclass
class BlinkingSettings:
    """On, off, bleached: one model for STORM, PALM and PAINT."""
    on_time: float = param(1.5, label="on time", unit="frames", min=0.01,
                           help="mean of an exponential on-time; switching happens "
                                "at any time within a frame")
    blinks: float = param(3.0, label="blinks", min=0.01,
                          help="mean number of blinks of a fluorophore within the "
                               "measurement, which sets the off time; with "
                               "activation 'every blink', the mean number before "
                               "it bleaches")
    off_time: Optional[float] = param(None, label="off time", unit="frames",
                                      min=0.01, advanced=True,
                                      help="mean of an exponential off-time; auto: "
                                           "from the number of blinks")
    bleaching: float = param(0.1, label="bleaching probability", min=0, max=1,
                             help="the probability to bleach after each blink: at "
                                  "most 1/p blinks on average, however long one "
                                  "measures (not used with 'every blink', where it "
                                  "is 1 / blinks)")
    activation: str = param("constant", label="activation",
                            choices=(("constant", "constant: ramped against bleaching"),
                                     ("decay", "decaying: all dark at the start"),
                                     ("all", "every blink until bleached, spread evenly")),
                            advanced=True,
                            help="constant: the blinks are spread evenly over the "
                                 "measurement, as an activation raised while the "
                                 "fluorophores bleach keeps them; decaying: a fixed "
                                 "rate, so most blinks come early; every blink: each "
                                 "fluorophore shows all its blinks before it bleaches "
                                 "(1 / blinks per blink), whatever the number of "
                                 "frames, spread evenly -- SMAP's 'Dye' model")
    photons: float = param(5000.0, label="photons per blink", min=0,
                           help="mean over a whole blink; a frame gets its share "
                                "by the time the fluorophore was on in it")
    photons_std: float = param(2500.0, label="photons std", min=0,
                               help="spread between blinks (a gamma distribution); "
                                    "0: every blink the same")


@dataclass
class OpticsSettings:
    """The PSF and the pixel, for the precision and for the frames."""
    pixelsize_nm: float = param(100.0, label="pixel size", unit="nm", min=1,
                                help="the camera pixel in the sample: enters the "
                                     "precision and sets the frames' scale")
    sigma_nm: float = param(130.0, label="PSF sigma", unit="nm", min=1,
                            help="the width (standard deviation) of the Gaussian "
                                 "spot in focus")
    astigmatism: bool = param(False, label="astigmatic",
                              help="a cylindrical lens: the spot's widths follow z")
    focal_offset_nm: float = param(300.0, label="focal offset", unit="nm",
                                   advanced=True,
                                   help="astigmatic: x is in focus at +this z, "
                                        "y at -this")
    depth_nm: float = param(400.0, label="focal depth", unit="nm", min=1, advanced=True,
                            help="astigmatic: how far from its focus a width has "
                                 "grown by sqrt(2)")
    calibration: str = param("", label="PSF calibration", kind="open_file",
                             file_filter="Calibrations (*_3dcal.mat *_3Dcal.mat *.h5);;"
                                         "All files (*)",
                             help="camera frames drawn with a measured spline PSF "
                                  "(a bead calibration) instead of a Gaussian: a fit "
                                  "then meets a PSF it did not make itself")


@dataclass
class LocalizationOutputSettings:
    """What a fit would have made of the frames."""
    min_photons: float = param(10.0, label="detection limit", unit="photons", min=0,
                               help="fewer photons than this in a frame and there is "
                                    "no localization; the dim ones that remain are "
                                    "kept, as SMAP keeps them, and Ground Truth leaves "
                                    "them out of its score")
    close: str = param("remove", label="close emitters",
                       choices=(("remove", "remove both"),
                                ("average", "one localization, averaged")),
                       help="emitters on in one frame closer than the separation "
                            "below are one spot: dropped, or fitted as one at "
                            "their photon-weighted mean")
    min_separation_nm: float = param(250.0, label="separation", unit="nm", min=0,
                                     help="closer than this in one frame, two "
                                          "emitters are one spot; 0: never")
    z_factor: float = param(3.0, label="z precision", unit="x lateral", min=0,
                            advanced=True,
                            help="the z precision as a multiple of the lateral one")
    emccd: bool = param(False, label="EMCCD", advanced=True,
                        help="the excess noise of EM gain: the precision is "
                             "worse by sqrt(2)")


@dataclass
class CameraOutputSettings:
    """Camera frames, for a fitter to fit."""
    path: str = param("", label="simulation file", kind="save_file",
                      file_filter=SIMULATION_FILTER,
                      help="written as the recipe (*.sim.yaml); open it in a "
                           "fitter as its file, and the frames are made as they "
                           "are read")
    tiff: bool = param(False, label="also write a TIFF",
                       help="the frames themselves, beside the recipe, for other "
                            "software")
    size_px: int = param(100, label="size", unit="pixels", min=8,
                         help="the side of the square frame, which covers x and y "
                              "from 0 to this times the pixel size")
    conversion: float = param(0.5, unit="e-/ADU", min=1e-6,
                              help="electrons per camera count")
    offset: float = param(100.0, unit="ADU",
                          help="the camera count of a dark pixel")
    read_noise: float = param(1.5, label="read noise", unit="ADU", min=0,
                              help="Gaussian noise per pixel and frame, in counts")
    em_gain: float = param(0.0, label="EM gain", min=0,
                           help="0: an sCMOS; otherwise an EMCCD with this gain, its "
                                "multiplication noise included (the excess factor "
                                "of 2)")
    load_truth: bool = param(False, label="load the truth",
                             help="the true positions of every spot drawn, as a "
                                  "localization table")


@dataclass
class SimulationSettings:
    output: str = param("localizations", label="simulate",
                        choices=(("localizations", "localizations"),
                                 ("camera", "camera frames")),
                        help="a localization table directly, or camera frames "
                             "for a fitter to fit")
    n_frames: int = param(20000, label="frames", min=1,
                          help="the length of the measurement")
    background: float = param(20.0, label="background", unit="photons/pixel", min=0,
                              help="per pixel and frame: in the precision, or "
                                   "drawn into the frames")
    background_std: float = param(0.0, label="background std", min=0,
                                  help="spread between localizations, or between "
                                       "frames for the camera; 0: always the same")
    drift: bool = param(False, label="add drift",
                        help="a smooth random walk plus a slow creep of ~100 nm; "
                             "the truth is kept in the metadata as drift_truth")
    seed: int = param(0, label="seed", min=0,
                      help="the same seed and settings give the same simulation")
    structure: StructureSettings = param(default_factory=StructureSettings,
                                         label="structure")
    labelling: LabellingSettings = param(default_factory=LabellingSettings,
                                         label="labelling", collapsed=True)
    blinking: BlinkingSettings = param(default_factory=BlinkingSettings,
                                       label="blinking")
    optics: OpticsSettings = param(default_factory=OpticsSettings, label="optics",
                                   collapsed=True)
    localizations: LocalizationOutputSettings = param(
        default_factory=LocalizationOutputSettings, label="localizations",
        collapsed=True)
    camera: CameraOutputSettings = param(default_factory=CameraOutputSettings,
                                         label="camera frames", collapsed=True)
