from pathlib import Path

import pydantic

from decode.hardware import camera as cam


@pydantic.validate_arguments
def load_metadata(path: Path, model: str = "hammamatsu", **kwargs) -> cam.base.MetaData:
    """
    Load metadata from a metadata file.

    Args:
        path:
        model:
        **kwargs: passed to the loader

    """
    match model:
        case "hammamatsu":
            return cam.hammamatsu.load(path, **kwargs)
        case _:
            raise ValueError(f"Unknown camera model: {model}")
