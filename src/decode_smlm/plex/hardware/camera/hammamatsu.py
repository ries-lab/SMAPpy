import copy
import json
from pathlib import Path
from typing import Any

import torch

from . import base


class GlobmetaData(base.ROIMixin, base.GainMixin, base.GlobMetaData):
    # omit mirror mixin as it's derived from gain directly
    @property
    def gain_mirror_dim(self) -> int | None:
        return -1 if self.gain is not None else None

    def get_roi_shift(self, gain_corrected: bool) -> list[int, int]:
        """
        ROI Shift is not necessarily the same as the ROI position.

        gain_corrected: if True, the shift is corrected for the gain mirror
        """
        if not gain_corrected:
            return list(self.roi_pos[1::-1])
        shift = torch.as_tensor(self.roi_pos)
        shift[0] = 512 - self.roi_pos[0] - self.frame_size[0] + 2  # due to mirroring
        shift = shift[[1, 0]]
        return shift.tolist()


class MetaData(base.MetaData):
    glob: GlobmetaData
    parsed: dict[str, Any] | None


class MetaDataLoader(base.MetaDataLoader):
    def __init__(
        self,
        roi_key: str = "ROI",
        gain_key: str = "Evolve512-MultiplierGain",
        src_roi: str = "FrameKey-0-0-0",
        src_gain: str = "FrameKey-0-0-0",
    ):
        super().__init__()

        self.roi_key = roi_key
        self.gain_key = gain_key
        self.src_roi = src_roi
        self.src_gain = src_gain

    @property
    def parseable_keys(self) -> set[str]:
        return {self.src_roi, self.src_gain}

    def loads(self, content: dict) -> MetaData:
        parsed = {k: convert_value(content[k]) for k in self.parseable_keys}
        glob = GlobmetaData(
            roi_pos=parsed[self.src_roi][self.roi_key][:2],
            frame_size=parsed[self.src_roi][self.roi_key][2:],
            gain=parsed[self.src_gain][self.gain_key],
        )
        m = MetaData(glob=glob, raw=content, parsed=parsed)
        return m

    def load(self, path: Path, encoding="ISO-8859-1") -> dict[str, str]:
        with open(path, "r", encoding=encoding) as f:
            content = json.load(f)
        return content


def load(
    metadata: Path | dict[str, str],
    roi_key: str = "ROI",
    gain_key: str = "Evolve512-MultiplierGain",
    src_roi: str = "FrameKey-0-0-0",
    src_gain: str = "FrameKey-0-0-0",
) -> MetaData:
    """
    Functional interface to load hammamatsu metadata.

    Args:
        metadata: path to metadata file or already loaded metadata
        roi_key: key for roi in metadata
        gain_key: key for gain in metadata
        src_roi: src level for roi
        src_gain: src level for gain

    Returns:
        MetaData object

    """
    loader = MetaDataLoader(
        roi_key=roi_key, gain_key=gain_key, src_roi=src_roi, src_gain=src_gain
    )
    if isinstance(metadata, Path):
        metadata = loader.load(metadata)
    return loader.loads(metadata)


def convert_value(value: dict | str) -> dict | int | float | tuple | str:
    if isinstance(value, dict):
        return {k: convert_value(v) for k, v in value.items()}
    if not isinstance(value, str):
        return value
    if value.isdigit():
        return int(value)
    if value.replace(".", "", 1).isdigit():
        return float(value)
    if "-" in value:
        value_tuple = tuple([convert_value(v) for v in value.split("-")])
        # only return tuple if all were actually numbers
        if all(isinstance(v, (int, float)) for v in value_tuple):
            return value_tuple
    return value
