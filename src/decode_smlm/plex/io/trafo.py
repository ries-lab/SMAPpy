from pathlib import Path
from typing import Literal

import scipy.io
import torch

from decode.simulation.trafo.pos import trafo
from decode.utils import dev

from decode.io.psf import load_h5


def _coord_permute_smap(xyz: torch.Tensor) -> torch.Tensor:
    return xyz[..., [1, 0, 2]]


def load_trafo_smap(
    path: Path,
    *,
    shift: tuple[float, ...] | None = None,
    scale: float | None = None,
    switch_xy: bool = False,
    reference: Literal["trafo_raw", "trafo_inv_raw"] = "trafo_raw",
    device: str | torch.device = "cpu",
) -> trafo.XYZTransformation:
    """
    Load a transformation from a trafo file along with some convenience options
    to wrap the transformation in.

    The order of execution is:
    shift -> scale -> switch_xy -> trafo -> unswitch_xy -> unscale -> unshift

    Args:
        path: path to trafo file
        shift: shift; from SMAP we usually expect 1 because of 1-based indexing
        scale: scale factor; from SMAP we usually expect 1/1000.
        switch_xy: permute x/y because of different matrix definition
        reference: reference to use for trafo, either "trafo_raw" or "trafo_inv_raw"
         which is necessary to load the correct direction of the trafo
        device: device to load trafo to

    Returns:
        transformation object
    """
    # currently assumes: reference is element 0, tar is element 1
    # ToDo: Change: Load transformation from bead cal directly

    data = scipy.io.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    data = data[reference]
    trafo_mat = torch.stack(
        [
            torch.as_tensor(data.ref, device=device).float(),
            torch.as_tensor(data.tar, device=device).float(),
        ],
        0,
    )

    t = trafo.XYZTransformationMatrix(trafo_mat)

    if scale is not None:
        scale_in = trafo.XYZScaleTransformation(scale, device=device)
        scale_out = trafo.XYZScaleTransformation(1 / scale, device=device)

    if switch_xy:
        permute_in = trafo.XYZTransformationGeneric(
            _coord_permute_smap,
            device=device,
            device_arbitrary=True,
        )
        permute_out = permute_in

    if scale is not None or switch_xy:
        t = [t]
        if switch_xy:
            t.insert(0, permute_in)
            t.append(permute_out)
        if scale is not None:
            t.insert(0, scale_in)
            t.append(scale_out)

        t = trafo.XYZCompositTransformation(t)

    if shift is not None:
        t = trafo.offset_trafo(t, offset=shift, rebound=True)
    return t

def load_trafo_h5(
    path: Path,
    *,
    shift: tuple[float, ...] | None = None,
    scale: float | None = None,
    switch_xy: bool = False,
    device: str | torch.device = "cpu",
) -> trafo.XYZTransformation:
    """
    Load a transformation from a trafo file along with some convenience options
    to wrap the transformation in.

    The order of execution is:
    shift -> scale -> switch_xy -> trafo -> unswitch_xy -> unscale -> unshift

    Args:
        path: path to trafo file
        shift: shift; from SMAP we usually expect 1 because of 1-based indexing
        scale: scale factor; from SMAP we usually expect 1/1000.
        switch_xy: permute x/y because of different matrix definition
        reference: reference to use for trafo, either "trafo_raw" or "trafo_inv_raw"
         which is necessary to load the correct direction of the trafo
        device: device to load trafo to

    Returns:
        transformation object
    """
    # currently assumes: reference is element 0, tar is element 1
    # ToDo: Change: Load transformation from bead cal directly
    data, p = load_h5(path)
    trafo_mat = torch.tensor(data.res.T)
    trafo_mat = trafo_mat.unsqueeze(0) if trafo_mat.dim()==2 else trafo_mat
    trafo_mat = torch.cat([torch.eye(3).unsqueeze(0),trafo_mat]) # save the first channel 
    
    if shift is None and data.res.imgcenter is not None:
        shift = (-data.res.imgcenter[1], -data.res.imgcenter[0], 0) if switch_xy else  (-data.res.imgcenter[0], -data.res.imgcenter[1], 0)

    t = trafo.XYZTransformationMatrix(trafo_mat)
 
    if scale is not None:
        scale_in = trafo.XYZScaleTransformation(scale, device=device)
        scale_out = trafo.XYZScaleTransformation(1 / scale, device=device)

    if switch_xy:
        permute_in = trafo.XYZTransformationGeneric(
            _coord_permute_smap,
            device=device,
            device_arbitrary=True,
        )
        permute_out = permute_in
    
    if scale is not None or switch_xy:
        t = [t]
        if switch_xy:
            t.insert(0, permute_in)
            t.append(permute_out)
        if scale is not None:
            t.insert(0, scale_in)
            t.append(scale_out)

        t = trafo.XYZCompositTransformation(t)
    
    if shift is not None:
        t = trafo.offset_trafo(t, offset=shift, rebound=True)

    return t

@dev.experimental(tested=True, level="info")
def load_xyz_trafo(
    path: Path,
    *,
    shift: tuple[float, ...] | None = None,
    scale: float | None = None,
    switch_xy: bool = False,
    reference: Literal["trafo_raw", "trafo_inv_raw"] = "trafo_raw",
    device: str | torch.device = "cpu",
    reverse_trafo: bool = True,
) -> trafo.XYZTransformation:
    """
    Load a transformation from a trafo file along with some convenience options
    to wrap the transformation in.

    The order of execution is:
    shift -> scale -> switch_xy -> trafo -> unswitch_xy -> unscale -> unshift

    Args:
        path: path to trafo file
        shift: shift; from SMAP we usually expect 1 because of 1-based indexing
        scale: scale factor; from SMAP we usually expect 1/1000.
        switch_xy: permute x/y because of different matrix definition
        reference: reference to use for trafo, either "trafo_raw" or "trafo_inv_raw"
         which is necessary to load the correct direction of the trafo
        device: device to load trafo to

    Returns:
        transformation object
    """
    # currently assumes: reference is element 0, tar is element 1
    # ToDo: Change: Load transformation from bead cal directly

    path = path if isinstance(path, Path) else Path(path)
    
    match path.suffix:
        case ".mat":
            return load_trafo_smap(path, 
                                   shift=shift,
                                   scale=scale,
                                   switch_xy=switch_xy,
                                   reference=reference,
                                   device=device)
        case ".h5":
            return load_trafo_h5(path,
                                 shift=None,
                                 scale=None,
                                 switch_xy=switch_xy,
                                 device=device,
                                 )
