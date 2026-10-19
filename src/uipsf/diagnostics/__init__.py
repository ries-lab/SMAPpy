"""What a fit says about itself and the data: the camera check and the residual atlas (plan §8)."""
from .atlas import Atlas, Finding, Group, residual_atlas
from .camera import NoiseTransfer, noise_transfer
from .heldout import HeldOutReport, heldout_report
from .probes import Probe, probe_terms, select_terms

__all__ = ["NoiseTransfer", "noise_transfer", "Atlas", "Finding", "Group", "residual_atlas", "Probe", "probe_terms",
           "select_terms", "HeldOutReport", "heldout_report"]
