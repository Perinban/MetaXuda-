import os
import sys
import ctypes
from pathlib import Path

# One-time init flags
__ulimit_set = False
__env_patched = False

# Paths to bundled native libs
NATIVE_DIR = Path(__file__).resolve().parent / "native"

CUDA_DRIVER_PATH = (NATIVE_DIR / "libcuda.dylib").resolve()
CUDART_PATH      = (NATIVE_DIR / "libcudart.dylib").resolve()
NVVM_PATH        = (NATIVE_DIR / "libnvvm.dylib").resolve()


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


def _maybe_relaunch():
    """Bootstrap run: relaunch with DYLD_LIBRARY_PATH set, then exit fast."""
    if os.environ.get("_METAXUDA_RELAUNCHED") == "1":
        return False

    env = os.environ.copy()
    dyld_path = str(NATIVE_DIR)
    env["DYLD_LIBRARY_PATH"] = f"{dyld_path}:{env.get('DYLD_LIBRARY_PATH', '')}"
    env["NUMBA_CUDA_DRIVER"] = str(CUDA_DRIVER_PATH)
    env["_METAXUDA_RELAUNCHED"] = "1"

    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


def setup_environment():
    """
    Ensure CUDA shim is injected before Numba initializes.
    Bootstrap run: only relaunches, no heavy work.
    Real run: preload libs + raise ulimit.
    """
    global __env_patched
    if __env_patched:
        return False
    __env_patched = True

    if os.environ.get("_METAXUDA_RELAUNCHED") != "1":
        _maybe_relaunch()

    # Real run only
    _raise_ulimit_once()

    # Preload shims once so symbols are global
    for lib in (CUDA_DRIVER_PATH, CUDART_PATH, NVVM_PATH):
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    return False