"""What a fit says about itself and the data: the camera check and the residual atlas (plan §8)."""
from .atlas import Atlas, Finding, Group, residual_atlas
from .camera import NoiseTransfer, noise_transfer
from .figures import coverslip_tilt, draw_atlas, draw_beads, draw_pupil, draw_zernikes, figures, pupil_maps, zernikes
from .halves import MIN_BEADS, HalfSplits, half_splits, half_spread, split_beads
from .heldout import HeldOutReport, heldout_report
from .probes import Probe, probe_terms, select_terms

__all__ = ["NoiseTransfer", "noise_transfer", "Atlas", "Finding", "Group", "residual_atlas", "Probe", "probe_terms",
           "select_terms", "HeldOutReport", "heldout_report", "figures", "coverslip_tilt", "draw_pupil",
           "draw_zernikes", "draw_atlas", "draw_beads", "pupil_maps", "zernikes", "HalfSplits", "half_splits",
           "half_spread", "split_beads", "MIN_BEADS"]
