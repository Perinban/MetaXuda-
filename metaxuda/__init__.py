from .env import setup_environment

# First run will exit after relaunch, second run continues
if setup_environment():
    import sys
    sys.exit(0)

# Only in second (real) run:
from .patch import patch_libdevice, get_libcudart_path
from .buffer import GPUMemoryBuffer
from .stream import GPUStream, DEFAULT_STREAM
from .pipeline import run_pipeline
from .streampool import StreamPool

__version__ = "0.1.6"

# Patch libdevice path resolution so Numba finds libdevice.bc
patch_libdevice()

__all__ = [
    "GPUMemoryBuffer",
    "GPUStream",
    "DEFAULT_STREAM",
    "StreamPool",
    "run_pipeline",
    "get_libcudart_path",
]