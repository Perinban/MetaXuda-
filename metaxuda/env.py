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
    """Raise ulimit -n to 65536 only once per interpreter session."""
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
    """Try to load a shared library globally, warn if it fails."""
    try:
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError as e:
        print(f"[MetaXuda] Warning: could not load {path}: {e}", file=sys.stderr)
        return False


def setup_environment():
    """
    Ensure CUDA shim is injected before Numba initializes.

    - Raises ulimit (for many file handles).
    - Force-loads our shim dylibs globally.
    - Sets env vars (DYLD_LIBRARY_PATH, NUMBA_CUDA_DRIVER) for children.
    """
    global __env_patched
    if __env_patched:
        return False
    __env_patched = True

    _raise_ulimit_once()

    for lib in (CUDA_DRIVER_PATH, CUDART_PATH, NVVM_PATH):
        _preload_library(lib)

    # Make sure child processes inherit correct search path
    dyld_path = str(NATIVE_DIR)
    os.environ["DYLD_LIBRARY_PATH"] = f"{dyld_path}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
    os.environ["NUMBA_CUDA_DRIVER"] = str(CUDA_DRIVER_PATH)

    return False