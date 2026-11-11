"""*Train DECODE*: a DECODE network trained for one's own data, in its own window.

Only a launcher (`Plugin.window`), as *PSF calibration* is: a training is prepared by looking at the
data -- the prefit, the ranges -- and followed for hours, and that is a window (`gui/train_window.py`).
The work is `smappy.training` (the prefit, the job folder, the progress, the model file) and
``smappy-decode train FOLDER`` (`decode_smlm.training`), which a script calls the same.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import Context, Plugin, Result, register


@dataclass
class TrainDecodeSettings:
    """Nothing: everything is chosen in the window."""


@register("Localize/Train DECODE")
class TrainDecode(Plugin):
    """A DECODE network for this microscope and these data, trained here or on a workstation."""
    description = ("train a DECODE network for the DECODE fitter: a prefit of the movie sets what the training "
                   "simulates, the job trains here or on a workstation, and the window follows it; opens its own window")
    changes_locs = False
    Settings = TrainDecodeSettings
    version = "1"
    window = "decode training"

    def run(self, ctx: Context, settings: TrainDecodeSettings) -> Result:
        raise ValueError("training is prepared and followed in a window; in a script, prepare a job folder "
                         "(smappy.training.job.write) and train it with smappy-decode train FOLDER")
