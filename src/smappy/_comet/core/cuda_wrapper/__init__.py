# Vendored from COMET 1.1.0 -- https://github.com/gpufit/Comet
# Copyright (c) 2025 Lenny Reinkensmeier & Mark Bates, MIT licensed: see
# LICENSE.txt beside this file, which travels with every copy.
# Upstream path: comet/core/cuda_wrapper/__init__.py
#
# Changed here: absolute ``comet.`` imports are relative, so this copy is
# self-contained and cannot collide with an installed COMET.

# core/cuda_wrapper/__init__.py


_USE_EXPERIMENTAL = False # default is False

if _USE_EXPERIMENTAL:
    # import everything from the experimental version
    from .cuda_wrapper_experimental import cuda_wrapper_chunked, cost_function_full_3d_chunked
else:
    # import everything from the “normal” version
    from .cuda_wrapper import cuda_wrapper_chunked, cost_function_full_3d_chunked

# Optionally expose a user‐readable flag (so code can check if it's experimental):
__all__ = ["cuda_wrapper_chunked", "cost_function_full_3d_chunked"]
