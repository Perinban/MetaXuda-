import sys
from .env import setup_environment

if setup_environment():
    sys.exit(0)

from .buffers import GPUMemoryBuffer, TieredBuffer, ManagedGPUBuffer
from .streams import GPUStream, ManagedStream
from .execution import run_pipeline, Pipeline, PipelinePool
from .patch import patch_libdevice

patch_libdevice()

__version__ = "2.0.0"
__all__ = [
    "GPUMemoryBuffer",
    "TieredBuffer",
    "ManagedGPUBuffer",
    "GPUStream",
    "ManagedStream",
    "run_pipeline",
    "Pipeline",
    "PipelinePool",
    "patch_libdevice",
]