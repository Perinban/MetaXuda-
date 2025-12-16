"""Environment setup for MetaXuda CUDA shim."""

import os
import sys
import ctypes
from pathlib import Path

__ulimit_set = False
__env_patched = False

NATIVE_DIR = Path(__file__).resolve().parent / "native"

CUDA_DRIVER_PATH = (NATIVE_DIR / "libcuda.dylib").resolve()
CUDART_PATH = (NATIVE_DIR / "libcudart.dylib").resolve()
NVVM_PATH = (NATIVE_DIR / "libnvvm.dylib").resolve()


def _raise_ulimit_once():
    """Raise ulimit -n to 65536 (optional performance optimization)."""
    global __ulimit_set
    if __ulimit_set:
        return
    __ulimit_set = True
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < 65536:
            resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard))
    except Exception:
        pass


def _preload_library(path: Path):
    """Load a shared library globally."""
    try:
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


def setup_environment():
    """
    Setup CUDA shim environment.

    The shim reads configuration from environment variables:
    - MX_ENABLE_DATASTORE_COMPRESSION (default: 1)
    - MX_DATASTORE_COMPRESSION_TYPE (default: lz4)
    - MX_DATASTORE_COMPRESSION_LEVEL (default: 3)
    - MX_DISK_PARALLELISM_LEVEL (default: auto)
    - MX_DISK_SPILL_ENABLED (default: 0)
    - MX_TIER3_STRATEGY (default: prefer_external)
    - MX_TIER3_INTERNAL_PATH (default: block_store)
    - MX_TIER3_EXTERNAL_DEVICES (format: "id:path,id:path")
    - MX_DEBUG (options: memory)
    """
    global __env_patched
    if __env_patched:
        return
    __env_patched = True

    _raise_ulimit_once()

    # Load CUDA shim libraries
    for lib in (CUDA_DRIVER_PATH, CUDART_PATH, NVVM_PATH):
        _preload_library(lib)

    # Setup environment for child processes
    dyld_path = str(NATIVE_DIR)
    existing = os.environ.get('DYLD_LIBRARY_PATH', '')
    os.environ["DYLD_LIBRARY_PATH"] = f"{dyld_path}:{existing}" if existing else dyld_path
    os.environ["NUMBA_CUDA_DRIVER"] = str(CUDA_DRIVER_PATH)

    # Add native/ to Python path
    if str(NATIVE_DIR) not in sys.path:
        sys.path.insert(0, str(NATIVE_DIR))