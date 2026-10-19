"""A batch of emitters: the data, its camera noise, and where each observation was taken.

Every data type uses one layout, ``[E, C, K, Y, X]`` (emitters, channels,
observations, pixels; plan §3).  For bead stacks K is the number of slices.
The model and the solver only read attributes of a batch and never import
this module, so the dependency direction of plan §2 holds.

With several channels, ``roi_xy`` places each channel's ROI on the chip,
where the channel transform maps between them; ``field_xy`` is the main
channel's.

``z_nm`` is the nominal emitter z of each slice, already in SMAPpy's sign:
z = −(objective position − focus) (`docs/uipsf/conventions.md`).
`from_objective` makes it from objective positions, so that minus sign is
written once.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass
class Batch:
    data: torch.Tensor                       # [E, C, K, Y, X] photons
    z_nm: torch.Tensor                       # [K] or [E, K]: nominal emitter z of each observation
    s: torch.Tensor | float = 0.0            # read-noise variance in the noise model's units; broadcasts
    mask: torch.Tensor | None = None         # bool, [E, C, K, Y, X] or broadcastable; False = excluded
    field_xy: torch.Tensor | None = None     # [E, 2] ROI centres in chip pixels (x, y): field coordinates
    stack_id: torch.Tensor | None = None     # [E] which stack each emitter came from
    roi_xy: torch.Tensor | None = None       # [E, C, 2] each channel's ROI centre in chip pixels (x, y);
                                             # needed with several channels (`model.channels`)

    @staticmethod
    def z_from_objective(objective_nm) -> torch.Tensor:
        """SMAPpy's z sign: a bead is below the focus by as much as the objective was raised."""
        return -torch.as_tensor(objective_nm)

    @property
    def n_emitters(self) -> int:
        return self.data.shape[0]

    @property
    def z_step_nm(self) -> float:
        """The stack's step, |Δz| between neighbouring slices (for the bead kernel)."""
        z = self.z_nm if self.z_nm.dim() == 1 else self.z_nm[0]
        return float((z[1:] - z[:-1]).abs().mean()) if z.numel() > 1 else 0.0

    def subset(self, index) -> Batch:
        """Some emitters only (a slice or an index tensor), the rest of the batch unchanged."""
        def pick(t):
            if isinstance(t, torch.Tensor) and t.dim() == self.data.dim() and t.shape[0] == self.n_emitters:
                return t[index]
            return t
        def rows(t):
            return t[index] if isinstance(t, torch.Tensor) else t
        z = self.z_nm[index] if self.z_nm.dim() == 2 else self.z_nm
        return replace(self, data=self.data[index], z_nm=z, s=pick(self.s), mask=pick(self.mask),
                       field_xy=rows(self.field_xy), stack_id=rows(self.stack_id), roi_xy=rows(self.roi_xy))

    def thinned(self, max_dz_nm: float | None) -> Batch:
        """Every n-th slice, n the largest whole number with n·Δz ≤ ``max_dz_nm`` (None: every slice).

        Stacks come at 10, 20 or 50 nm; a fit needs a z sampling, not a slice
        count, so the rule is a maximum spacing and the step is the largest
        multiple of the acquisition's that does not exceed it (plan §9 M7).
        The middle slice, the nominal focus, is always kept, so the beads'
        fitted z keep their origin.  The bead kernel then convolves along the
        coarser stack (`VectorPSF`, ``z_step_nm``).
        """
        step = self.z_step_nm
        if max_dz_nm is None or step <= 0:
            return self
        n = max(1, int(float(max_dz_nm) / step + 1e-6))
        K = self.data.shape[2]
        if n == 1:
            return self
        idx = torch.arange((K - 1) // 2 % n, K, n, device=self.data.device)

        def along_k(t):
            if isinstance(t, torch.Tensor) and t.dim() == self.data.dim() and t.shape[2] == K:
                return t.index_select(2, idx)
            return t
        return replace(self, data=self.data.index_select(2, idx), z_nm=self.z_nm.index_select(-1, idx.to(
            self.z_nm.device)), s=along_k(self.s), mask=along_k(self.mask))

    def to(self, device=None, dtype=None) -> Batch:
        def move(t, keep_dtype=False):
            if not isinstance(t, torch.Tensor):
                return t
            return t.to(device=device) if keep_dtype or dtype is None else t.to(device=device, dtype=dtype)
        data = move(self.data)
        # ROI centres are whole pixels, exact in float32; in the data's dtype, so never float64 on MPS
        roi = None if self.roi_xy is None else self.roi_xy.to(device=data.device, dtype=data.dtype)
        return replace(self, data=data, z_nm=move(self.z_nm), s=move(self.s),
                       mask=move(self.mask, keep_dtype=True), roi_xy=roi)
