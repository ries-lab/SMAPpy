from typing import Optional, Union

import torch
from pydantic import BaseModel, validator

from .emitter import EmitterSet
from ..generic import models
from ..generic import utils as gen_utils


class FluorophoreData(models.MixinEqualDevice, models.MixinEqualLength, BaseModel):
    """
    Helper class which holds and validates the data fields of the FluorophoreSet.
    Usually this class not used directly.
    """

    xyz: models.Coordinate
    flux: models.Tensor
    t0: models.Tensor
    ontime: models.Tensor
    id: models.LongTensor
    code: Optional[models.Tensor]
    prob: Optional[models.Tensor]
    bg: Optional[models.Tensor]
    xyz_cr: Optional[models.Tensor]
    phot_cr: Optional[models.Tensor]
    bg_cr: Optional[models.Tensor]
    xyz_sig: Optional[models.Tensor]
    phot_sig: Optional[models.Tensor]
    bg_sig: Optional[models.Tensor]

    @validator("flux")
    def flux_non_negative(cls, v):
        if (v < 0).any():
            raise ValueError("Negative flux values encountered.")
        return v

    @validator("id")
    def id_unique(cls, v):
        if v.unique().numel() != v.numel():
            raise ValueError("IDs are not unique.")

        return v

    @validator("ontime")
    def ontime_non_neg(cls, v):
        if (v < 0).any():
            raise ValueError("Negative ontime encountered.")

        return v


class FluorophoreSet:
    _data_holder = {
        "xyz",
        "flux",
        "t0",
        "ontime",
        "id",
        "code",
        "prob",
        "bg",
        "xyz_cr",
        "phot_cr",
        "bg_cr",
        "xyz_sig",
        "phot_sig",
        "bg_sig",
    }

    def __init__(
        self,
        xyz: torch.Tensor,
        flux: torch.Tensor,
        t0: torch.Tensor,
        ontime: torch.Tensor,
        xy_unit: str,
        px_size: Union[tuple, torch.Tensor] = None,
        id: Optional[torch.LongTensor] = None,
        code: Optional[torch.Tensor] = None,
        prob: Optional[torch.Tensor] = None,
        bg: Optional[torch.Tensor] = None,
        xyz_cr: Optional[torch.Tensor] = None,
        phot_cr: Optional[torch.Tensor] = None,
        bg_cr: Optional[torch.Tensor] = None,
        xyz_sig: Optional[torch.Tensor] = None,
        phot_sig: Optional[torch.Tensor] = None,
        bg_sig: Optional[torch.Tensor] = None,
        sanity_check=True,
    ):
        """
        FluorophoreSet is a collection of fluorophores.
        Something that starts to emit light at time `t0` and is on for a specific
        ontime. Related to the standard EmitterSet. However, here we do not specify a
        frame_ix but rather a (non-integer) initial point in time where the emitter
        starts to blink and an on-time.

        Args:
            xyz: coordinates. Dimension: N x 3
            flux: flux, i.e. photon flux per time unit. Dimension N
            t0: initial blink event. Dimension: N
            ontime: duration in frame-time units how long the emitter blinks.
                Dimension N
            id: identity of the emitter. Dimension: N
            xy_unit: unit of the coordinates
            px_size: Pixel size for unit conversion. If not specified, derived attributes cannot be accessed
            id: id of the emitter
            sanity_check: performs a sanity check if true
        """
        self._data_container = (
            FluorophoreData if sanity_check else FluorophoreData.construct
        )
        self._data = self._data_container(
            xyz=xyz,
            flux=flux,
            t0=t0,
            ontime=ontime,
            id=id if id is not None else torch.arange(len(flux)),
            code=code,
            prob=prob,
            bg=bg,
            xyz_cr=xyz_cr,
            phot_cr=phot_cr,
            bg_cr=bg_cr,
            xyz_sig=xyz_sig,
            phot_sig=phot_sig,
            bg_sig=bg_sig,
        )
        self.xy_unit = xy_unit
        self.px_size = px_size
        self.sanity_check = sanity_check

    def __len__(self) -> int:
        return len(self.xyz)

    @property
    def te(self):
        # end time
        return self.t0 + self.ontime

    @staticmethod
    def _compute_time_distribution(
        t_start: torch.FloatTensor, t_end: torch.FloatTensor
    ) -> (torch.LongTensor, torch.Tensor):
        """
        Compute time distribution, i.e. on how many frames an emitter is visible
        and what the ontime per emitter per frame is

        Args:
            t_start: start time
            t_end: end time
        """
        # compute total number of frames per emitter
        ix_start = torch.floor(t_start).long()
        ix_end = torch.floor(t_end).long()

        n_frames = (ix_end - ix_start + 1).long()

        # compute ontime per frame
        # ontime = torch.repeat_interleave(torch.ones_like(t_start), n_frames, 0)
        pseudo_id = torch.repeat_interleave(torch.arange(len(t_start)), n_frames, 0)
        n_frame_per_emitter = gen_utils.cum_count_per_group(pseudo_id)

        # ontime since start, to end
        t_since_start = (
            torch.repeat_interleave(t_start.ceil(), n_frames)
            + n_frame_per_emitter
            - torch.repeat_interleave(t_start, n_frames)
        )

        t_to_end = torch.repeat_interleave(t_end, n_frames) - (
            n_frame_per_emitter + torch.repeat_interleave(t_start.floor(), n_frames)
        )

        t_total_diff = torch.repeat_interleave(
            t_end, n_frames
        ) - torch.repeat_interleave(t_start, n_frames)

        ontime = t_since_start.minimum(t_to_end).minimum(t_total_diff).clamp(max=1)

        return n_frames, ontime

    def frame_bucketize(self) -> EmitterSet:
        """
        Returns EmitterSet with distributed emitters.
        The emitters ID is preserved such that localisations coming from the same
        fluorophore will have the same ID.

        Returns:
            EmitterSet
        """
        n_frames, ontime = self._compute_time_distribution(self.t0, self.te)

        em = EmitterSet(
            xyz=self.xyz,
            frame_ix=self.t0.floor().long(),
            phot=self.flux,
            id=self.id,
            code=self.code,
            prob=self.prob,
            bg=self.bg,
            xyz_cr=self.xyz_cr,
            phot_cr=self.phot_cr,
            bg_cr=self.bg_cr,
            xyz_sig=self.xyz_sig,
            phot_sig=self.phot_sig,
            bg_sig=self.bg_sig,
            xy_unit=self.xy_unit,
            px_size=self.px_size,
            sanity_check=self.sanity_check
        ).repeat(n_frames, step_frames=True)

        # adjust photons by ontime (i.e. flux * ontime)
        em.phot *= ontime if em.phot.dim() == 1 else ontime.view(-1, 1)

        return em

    def __getattr__(self, item):
        # refer to data holder
        if item in self._data_holder:
            return getattr(self._data, item)

        raise AttributeError(f"Attribute {item} not found.")
