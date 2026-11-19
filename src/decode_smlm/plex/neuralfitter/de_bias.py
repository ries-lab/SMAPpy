import numpy as np
import torch

from deprecated import deprecated


@deprecated(version="0.11", reason="use coordinate based solution instead")
class UniformizeOffset:
    ...


class DebiasLateral:
    def __init__(
        self,
        lateral: "UniformizeOffsetCoordinateBased",
        bins_z: int | torch.Tensor | None = None,
    ):
        self._lateral = lateral
        self._bins_z = bins_z

    def forward(
        self, offsets: torch.Tensor, uncertainties: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor | torch.LongTensor:
        """
        Returns debiased offsets and inverse indices due to z-binning

        Args:
            offsets: x/y offset coordinates of shape (N, 2).
            uncertainties: x/y sigma coordinates of shape (N, 2).
            z: z coordinate of shape (N,)
        """
        if self._bins_z is None:
            return self._lateral.forward(offsets, uncertainties), None
        if isinstance(self._bins_z, int):
            z_bins = torch.linspace(z.min(), z.max(), self._bins_z)
        else:
            z_bins = self._bins_z

        z_bin_indices = torch.bucketize(z, z_bins)
        ix_lin = torch.arange(len(z))

        out = []
        ix_out = []
        for z_ix in range(z_bin_indices.max() + 1):
            ix = z_bin_indices == z_ix
            if ix.sum() == 0:
                continue
            out.append(self._lateral.forward(offsets[ix], uncertainties[ix]))
            ix_out.append(ix_lin[ix])

        out = torch.cat(out, 0)
        ix_out = torch.cat(ix_out, 0)

        return out, ix_out


class UniformizeOffsetCoordinateBased:
    def __init__(self, n_bins: int):
        """
        Rescales x and y offsets so that they are distributed uniformly
        within [-0.5, 0.5] to correct for biased outputs.

        Args:
            n_bins: The bias scales with the uncertainty of the localization.
             Therefore all detections are binned according to their predicted uncertainty.
             Detections within different bins are then rescaled separately.
             This specifies the number of bins.
        """
        self.n_bins = n_bins

    def histedges_equal_n(self, x):
        npt = len(x)
        return torch.tensor(
            np.interp(
                np.linspace(0, npt, self.n_bins + 1), np.arange(npt), np.sort(x.numpy())
            )
        )

    def uniformize(self, x):
        x = torch.clamp(x, -0.99, 0.99)
        x_cdf = torch.histc(x, bins=200, min=-1, max=1)
        x_cumsum = torch.cumsum(x_cdf, dim=0) / torch.sum(x_cdf)
        x_re = self.cdf_get(x_cumsum, x)
        return (x_re - 0.5).float()

    def cdf_get(self, cdf, val):
        ind = ((val + 1) / 2 * 200 - 1.0).clamp(0, 199)
        ind_floor = torch.floor(ind).long()
        ind_ceil = torch.ceil(ind).long()
        dec = ind - ind_floor
        return dec * cdf[ind_ceil] + (1 - dec) * cdf[ind_floor]

    def forward(
        self, offsets: torch.Tensor, uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """Rescales x and y offsets (inplace) so that they are distributed uniformly within [-0.5, 0.5] to correct for biased outputs.

        Args:
            offsets (torch.Tensor): x/y offset coordinates of shape (N, 2).
            uncertainties (torch.Tensor): x/y sigma coordinates of shape (N, 2).
        """
        offsets = offsets.clone()
        x_sigma = uncertainties[:, 0]
        y_sigma = uncertainties[:, 1]
        x_sigma_var = torch.var(x_sigma)
        y_sigma_var = torch.var(y_sigma)
        weighted_sig = (
            x_sigma**2 + (torch.sqrt(x_sigma_var / y_sigma_var) * y_sigma) ** 2
        )

        bins = self.histedges_equal_n(weighted_sig)
        for i in range(self.n_bins):
            inds = torch.where((weighted_sig > bins[i]) & (weighted_sig < bins[i + 1]))[
                0
            ]
            offsets[inds, 0] = self.uniformize(offsets[inds, 0]) + torch.mean(
                offsets[inds, 0]
            )
            offsets[inds, 1] = self.uniformize(offsets[inds, 1]) + torch.mean(
                offsets[inds, 1]
            )

        return offsets
