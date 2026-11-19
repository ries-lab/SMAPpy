from abc import ABC
from typing import Any

from pydantic import BaseModel


class ROIMixin(BaseModel):
    roi_pos: tuple[int, int]  # x, y roi position


class GainMixin(BaseModel):
    gain: int


class MirrorDimMixin(BaseModel):
    mirror_dim: int | None


class GlobMetaData(BaseModel):
    frame_size: tuple[int, int]  # width, height


class MetaData(BaseModel):
    glob: GlobMetaData
    raw: dict[str, Any] | str | None = None


class MetaDataLoader(ABC):
    def loads(self, content: str | Any) -> MetaData:
        raise NotImplementedError
