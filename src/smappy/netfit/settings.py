"""What a training is: the simulation, the camera, and the parts, each by name.

A model file records all of it, so that a benchmark report says exactly what was trained
(`docs/decode/benchmark.md`, "A run"); the benchmark's run files name the same parts.  The defaults
are version 1: DECODE's method as published, on the frames DECODE trains on (40 x 40, 3 frames),
with the example package's ranges -- what the port's `decode_B` was trained with.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class SimulationSettings:
    """How training frames are drawn: a window of frames, emitters that blink over it, a background
    per frame.  Fresh for every batch (decided 2026-11-12): no pool of frames to overfit."""
    frame_px: int = 40                        # square frames, DECODE's size
    window: int = 3                           # frames a sample holds; the middle one is localized
    margin_px: float = 0.0                    # emitters drawn this far beyond the frame (DECODE: 0)
    emitters_per_frame: float = 50.0          # mean number on in a frame (DECODE's emitter_av; the example package's 50)
    lifetime_frames: float = 1.0              # mean on-time, exponential (DECODE's lifetime_avg)
    photons_mean: float = 7000.0              # flux while on, a full frame's photons
    photons_sd: float = 3000.0
    photons_min: float = 0.0                  # the flux drawn no lower (DECODE's intensity_th, unset there)
    background: tuple = (20.0, 200.0)         # photons a pixel, uniform per frame
    z_nm: tuple = (-800.0, 800.0)
    target_photons_min: float = 0.0           # a target is a molecule with at least this many photons in the frame


@dataclass
class CameraModel:
    """The camera the frames are drawn for, and back to photons as the fitter converts them
    (`CameraMetadata.adu_to_photons`), so the network sees what it will be given."""
    conversion: float = 5.0                   # e- per ADU
    offset: float = 398.6                     # ADU
    em_gain: float = 100.0                    # 0: no EM register
    read_noise_e: float = 58.8
    qe: float = 1.0

    @property
    def adu_to_photons(self) -> float:
        return self.conversion / self.em_gain if self.em_gain else self.conversion


@dataclass
class Part:
    """One part of the method, by name, with its settings."""
    name: str
    settings: dict = field(default_factory=dict)


@dataclass
class Parts:
    """The swappable parts (`docs/decode/benchmark.md`): version 1's are DECODE's."""
    backbone: Part = field(default_factory=lambda: Part("unet2", {"features": 48, "depth_frame": 2, "depth_union": 2}))
    time_model: Part = field(default_factory=lambda: Part("frame_stack"))
    output: Part = field(default_factory=lambda: Part("offsets"))
    extraction: Part = field(default_factory=lambda: Part("lookahead"))
    loss: Part = field(default_factory=lambda: Part("mixture", {"terms": ["count", "mixture", "background"]}))
    schedule: Part = field(default_factory=lambda: Part("adamw_step"))


@dataclass
class TrainingSettings:
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    camera: CameraModel = field(default_factory=CameraModel)
    parts: Parts = field(default_factory=Parts)
    calibration: str | None = None
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
