"""uiPSF v2: learns a physical model of a microscope's point spread function.

The forward model (`optics`, `model`), the likelihood (`noise`) and the
solver (`solve`) are pure PyTorch and runs on NVIDIA
GPUs, Apple GPUs or the CPU.  See docs/uipsf/plan.md for the architecture and
docs/uipsf/conventions.md for axes, signs and units.
"""
from .data import Batch
from .device import default_device
from .export import load_model, save_model
from .model import Channel, ForwardModel, PixelPupil, VectorPSF, ZernikePupil
from .noise import EMCCD, Poisson, ShiftedPoisson
from .optics import MatrixDFT, Optics, PupilGrid, choose_pupil_size
from .params import Params, ParamSpec
from .solve import FitResult, LevenbergMarquardt, Problem

__version__ = "2.0.0.dev0"
__all__ = ["Optics", "PupilGrid", "MatrixDFT", "choose_pupil_size", "ZernikePupil", "PixelPupil",
           "VectorPSF", "Channel", "ForwardModel", "ParamSpec", "Params", "Batch", "Poisson",
           "ShiftedPoisson", "EMCCD", "Problem", "LevenbergMarquardt", "FitResult", "default_device", "save_model",
           "load_model"]
