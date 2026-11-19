# here we define pydantic custom types and models

from typing import Sequence, Union

import pydantic
import torch

from decode.generic import dtype


class Tensor(torch.Tensor):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Union[torch.Tensor, Sequence]) -> torch.Tensor:
        v = torch.as_tensor(v)
        return v


class LongTensor(Tensor):
    @classmethod
    def validate(cls, v: Union[torch.Tensor, Sequence[int]]) -> torch.Tensor:
        v = super().validate(v)
        v_long = torch.as_tensor(v, dtype=torch.long)

        if not dtype.is_integer(v) and torch.any((v_long - v).abs() > 1e-6):
            raise TypeError("Tensor is not integer and cannot be coerced to integer")

        return v_long


class Coordinate(Tensor):
    @classmethod
    def validate(cls, v: torch.Tensor):
        v = super().validate(v)

        if v.dim() == 1 or v.dim() > 3:
            raise ValueError("Not supported shape.")

        if v.size(-1) == 2:
            v = torch.cat((v, torch.zeros_like(v[..., [0]])), -1)

        return v


class MixinEqualDevice(pydantic.BaseModel):
    @pydantic.root_validator
    def equal_device(cls, v: dict) -> dict:
        x = next(vv for vv in v.values() if vv is not None)
        device = x.device
        if x is not None and not all(
            device == vv.device for vv in v.values() if vv is not None
        ):
            raise ValueError("Not all attributes are on the same device")
        return v


class MixinEqualLength(pydantic.BaseModel):
    @pydantic.root_validator
    def equal_length(cls, v: dict) -> dict:
        n = {len(vv) for vv in v.values() if vv is not None}
        if len(n) >= 2:
            raise ValueError(f"Unequal length of fields. Got {n}")

        return v
