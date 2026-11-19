# this file specifies config / param schema for parsing using pydantic standard

from pydantic.dataclasses import dataclass
from pydantic.types import DirectoryPath, FilePath


@dataclass
class InOutSchemaTraining:
    calibration_file: FilePath
    experiment_out: DirectoryPath
    checkpoint_init: FilePath | None = None
    model_init: FilePath | None = None
