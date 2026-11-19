"""
DECODE
This software package implements a DeepLearning based framework for high-density fitting in SMLM.

"""

__version__ = "1.0.0dev0"  # do not modify by hand set and sync with bumpversion
__author__ = "Lucas-Raphael Mueller, Artur Speiser"
__repo__ = "https://github.com/TuragaLab/DECODE/master/gateway.yaml"  # main repo
__gateway__ = (
    "https://raw.githubusercontent.com/TuragaLab/DECODE/master/gateway.yaml"  # gateway
)

import warnings

import decode_smlm.plex.evaluation
import decode_smlm.plex.generic
import decode_smlm.plex.io
import decode_smlm.plex.neuralfitter
import decode_smlm.plex.plot
import decode_smlm.plex.renderer
import decode_smlm.plex.simulation

from . import emitter
from . import io
from . import validation
from .emitter.emitter import EmitterSet
from .emitter.emitter import factory as emitter_factory

# check device capability
import torch
import decode_smlm.plex.utils.hardware

if torch.cuda.is_available():
    device_capa = decode_smlm.plex.utils.hardware.get_device_capability()
    if float(device_capa) < 3.7:
        warnings.warn(
            f"Your GPU {torch.cuda.get_device_name()} has cuda capability {device_capa} and is no longer supported (minimum is 3.7)."
            f"\nIf you have multiple devices make sure to select the index of the most modern one."
            f"\nOtherwise you can use your CPU to run DECODE or switch to Google Colab.",
            category=UserWarning,
        )
