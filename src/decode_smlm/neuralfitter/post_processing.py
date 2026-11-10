from abc import ABC, abstractmethod  # abstract class
from collections.abc import Callable

import torch

from decode_smlm.generic.emitter import EmitterSet, EmptyEmitterSet


class PostProcessing(ABC):
    _return_types = ('batch-set', 'frame-set')

    def __init__(self, xy_unit, px_size, return_format: str):
        """

        Args:
            return_format (str): return format of forward function. Must be 'batch-set', 'frame-set'. If 'batch-set'
            one instance of EmitterSet will be returned per forward call, if 'frame-set' a tuple of EmitterSet one
            per frame will be returned
            sanity_check (bool): perform sanity check
        """

        super().__init__()
        self.xy_unit = xy_unit
        self.px_size = px_size
        self.return_format = return_format

    def sanity_check(self):
        """
        Sanity checks
        """
        if self.return_format not in self._return_types:
            raise ValueError("Not supported return type.")

    def skip_if(self, x):
        """
        Skip post-processing when a certain condition is met and implementation would fail, i.e. to many
        bright pixels in the detection channel. Default implementation returns False always.

        Args:
            x: network output

        Returns:
            bool: returns true when post-processing should be skipped
        """
        return False

    def _return_as_type(self, em, ix_low, ix_high):
        """
        Returns in the type specified in constructor

        Args:
            em (EmitterSet): emitters
            ix_low (int): lower frame_ix
            ix_high (int): upper frame_ix

        Returns:
            EmitterSet or list: Returns as EmitterSet or as list of EmitterSets

        """

        if self.return_format == 'batch-set':
            return em
        elif self.return_format == 'frame-set':
            return em.split_in_frames(ix_low=ix_low, ix_up=ix_high)
        else:
            raise ValueError

    @abstractmethod
    def forward(self, x: torch.Tensor) -> (EmitterSet, list):
        """
        Forward anything through the post-processing and return an EmitterSet

        Args:
            x:

        Returns:
            EmitterSet or list: Returns as EmitterSet or as list of EmitterSets

        """
        raise NotImplementedError


class NoPostProcessing(PostProcessing):
    """
    The 'No' Post-Processing post-processing. Will always return an empty EmitterSet.

    """

    def __init__(self, xy_unit=None, px_size=None, return_format='batch-set'):
        super().__init__(xy_unit=xy_unit, px_size=px_size, return_format=return_format)

    def forward(self, x: torch.Tensor):
        """

        Args:
            x (torch.Tensor): any input tensor where the first dim is the batch-dim.

        Returns:
            EmptyEmitterSet: An empty EmitterSet

        """

        return EmptyEmitterSet(xy_unit=self.xy_unit, px_size=self.px_size)


class LookUpPostProcessing(PostProcessing):
    """
    Simple post-processing in which we threshold the probability output (raw threshold) and then look-up the features
    in the respective channels.

    """

    def __init__(self, raw_th: float, xy_unit: str, px_size=None,
                 pphotxyzbg_mapping: list | tuple = (0, 1, 2, 3, 4, -1),
                 photxyz_sigma_mapping: list | tuple | None = (5, 6, 7, 8)):
        """

        Args:
            raw_th: initial raw threshold
            xy_unit: xy unit unit
            px_size: pixel size
            pphotxyzbg_mapping: channel index mapping of detection (p), photon, x, y, z, bg
        """
        super().__init__(xy_unit=xy_unit, px_size=px_size, return_format='batch-set')

        self.raw_th = raw_th
        self.pphotxyzbg_mapping = pphotxyzbg_mapping
        self.photxyz_sigma_mapping = photxyz_sigma_mapping

        assert len(self.pphotxyzbg_mapping) == 6, "Wrong length of mapping."
        if self.photxyz_sigma_mapping is not None:
            assert len(self.photxyz_sigma_mapping) == 4, "Wrong length of sigma mapping."

    def _filter(self, detection) -> torch.BoolTensor:
        """

        Args:
            detection: any tensor that should be thresholded

        Returns:
            boolean with active px

        """

        return detection >= self.raw_th

    @staticmethod
    def _lookup_features(features: torch.Tensor, active_px: torch.Tensor) -> tuple:
        """

        Args:
            features: size :math:`(N, C, H, W)`
            active_px: size :math:`(N, H, W)`

        Returns:
            torch.Tensor: batch-ix, size :math: `M`
            torch.Tensor: extracted features size :math:`(C, M)`

        """

        assert features.dim() == 4
        assert active_px.dim() == features.dim() - 1

        batch_ix = active_px.nonzero(as_tuple=False)[:, 0]
        features_active = features.permute(1, 0, 2, 3)[:, active_px]

        return batch_ix, features_active

    def forward(self, x: torch.Tensor) -> EmitterSet:
        """
        Forward model output tensor through post-processing and return EmitterSet. Will include sigma values in
        EmitterSet if mapping was provided initially.

        Args:
            x: model output

        Returns:
            EmitterSet

        """
        """Reorder features channel-wise."""
        x_mapped = x[:, self.pphotxyzbg_mapping]

        """Filter"""
        active_px = self._filter(x_mapped[:, 0])  # 0th ch. is detection channel
        prob = x_mapped[:, 0][active_px]

        """Look-Up in channels"""
        frame_ix, features = self._lookup_features(x_mapped[:, 1:], active_px)

        """Return EmitterSet"""
        xyz = features[1:4].transpose(0, 1)

        """If sigma mapping is present, get those values as well."""
        if self.photxyz_sigma_mapping is not None:
            sigma = x[:, self.photxyz_sigma_mapping]
            _, features_sigma = self._lookup_features(sigma, active_px)

            xyz_sigma = features_sigma[1:4].transpose(0, 1).cpu()
            phot_sigma = features_sigma[0].cpu()
        else:
            xyz_sigma = None
            phot_sigma = None

        return EmitterSet(xyz=xyz.cpu(), frame_ix=frame_ix.cpu(), phot=features[0, :].cpu(),
                          xyz_sig=xyz_sigma, phot_sig=phot_sigma, bg_sig=None,
                          bg=features[4, :].cpu() if features.size(0) == 5 else None,
                          prob=prob.cpu(), xy_unit=self.xy_unit, px_size=self.px_size)


class SpatialIntegration(LookUpPostProcessing):
    """
    Spatial Integration post processing.
    """

    _p_aggregations = ('sum', 'norm_sum')  # , 'max', 'pbinom_cdf', 'pbinom_pdf')
    _split_th = 0.6

    def __init__(self, raw_th: float, xy_unit: str, px_size=None,
                 pphotxyzbg_mapping: list | tuple = (0, 1, 2, 3, 4, -1),
                 photxyz_sigma_mapping: list | tuple | None = (5, 6, 7, 8),
                 p_aggregation: str | Callable = 'norm_sum'):
        """

        Args:
            raw_th: probability threshold from where detections are considered
            xy_unit: unit of the xy coordinates
            px_size: pixel size
            pphotxyzbg_mapping: channel index mapping
            photxyz_sigma_mapping: channel index mapping of sigma channels
            p_aggregation: aggreation method to aggregate probabilities. can be 'sum', 'max', 'norm_sum'
        """
        super().__init__(raw_th=raw_th, xy_unit=xy_unit, px_size=px_size,
                         pphotxyzbg_mapping=pphotxyzbg_mapping,
                         photxyz_sigma_mapping=photxyz_sigma_mapping)

        self.p_aggregation = self.set_p_aggregation(p_aggregation)

    def forward(self, x: torch.Tensor) -> EmitterSet:
        x[:, 0] = self._nms(x[:, 0], self.p_aggregation, self.raw_th, self._split_th)

        return super().forward(x)

    @staticmethod
    def _nms(p: torch.Tensor, p_aggregation, raw_th, split_th) -> torch.Tensor:
        """
        Non-Maximum Suppresion

        Args:
            p:

        """

        with torch.no_grad():
            p_copy = p.clone()

            """Probability values > 0.3 are regarded as possible locations"""
            p_clip = torch.where(p > raw_th, p, torch.zeros_like(p))[:, None]

            """localize maximum values within a 3x3 patch"""
            pool = torch.nn.functional.max_pool2d(p_clip, 3, 1, padding=1)
            max_mask1 = torch.eq(p[:, None], pool).float()

            """Add probability values from the 4 adjacent pixels"""
            diag = 0.  # 1/np.sqrt(2)
            filt = torch.tensor([[diag, 1., diag], [1, 1, 1], [diag, 1, diag]]).unsqueeze(0).unsqueeze(0).to(p.device)
            conv = torch.nn.functional.conv2d(p[:, None], filt, padding=1)
            p_ps1 = max_mask1 * conv

            """
            In order do be able to identify two fluorophores in adjacent pixels we look for
            probablity values > 0.6 that are not part of the first mask
            """
            p_copy *= (1 - max_mask1[:, 0])
            # p_clip = torch.where(p_copy > split_th, p_copy, torch.zeros_like(p_copy))[:, None]
            max_mask2 = torch.where(p_copy > split_th, torch.ones_like(p_copy), torch.zeros_like(p_copy))[:, None]
            p_ps2 = max_mask2 * conv

            """This is our final clustered probablity which we then threshold (normally > 0.7)
            to get our final discrete locations"""
            p_ps = p_aggregation(p_ps1, p_ps2)
            assert p_ps.size(1) == 1

            return p_ps.squeeze(1)

    @classmethod
    def set_p_aggregation(cls, p_aggr: str | Callable) -> Callable:
        """
        Sets the p_aggregation by string or callable. Return s Callable

        Args:
            p_aggr: probability aggregation

        """

        if isinstance(p_aggr, str):

            if p_aggr == 'sum':
                return torch.add
            elif p_aggr == 'max':
                return torch.max
            elif p_aggr == 'norm_sum':
                def norm_sum(*args):
                    return torch.clamp(torch.add(*args), 0., 1.)

                return norm_sum
            else:
                raise ValueError

        else:
            return p_aggr
