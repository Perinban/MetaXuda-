# env.py
import os
import sys
import ctypes
from pathlib import Path

# ---- one-time init flags ----
__ulimit_set = False
__env_patched = False

# Paths to bundled native libs
NATIVE_DIR   = Path(__file__).resolve().parent / "native"
CUDA_DRIVER  = (NATIVE_DIR / "libcuda.dylib").resolve()
CUDART       = (NATIVE_DIR / "libcudart.dylib").resolve()
NVVM         = (NATIVE_DIR / "libnvvm.dylib").resolve()

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
        return False  # already relaunched

    env = os.environ.copy()
    dyld_path = str(NATIVE_DIR)
    env["DYLD_LIBRARY_PATH"] = f"{dyld_path}:{env.get('DYLD_LIBRARY_PATH', '')}"
    env["NUMBA_CUDA_DRIVER"] = str(CUDA_DRIVER)
    env["_METAXUDA_RELAUNCHED"] = "1"

    # Relaunch and replace current process (no return)
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

def setup_environment():
    """
    Ensure CUDA shim is injected before Numba initializes.
    Bootstrap run: only relaunches, no heavy work.
    Real run: preload libs + raise ulimit.
    """
    global __env_patched
    if __env_patched:
        return
    __env_patched = True

    if os.environ.get("_METAXUDA_RELAUNCHED") != "1":
        _maybe_relaunch()  # exits before doing heavy work

    # --- real run only ---
    _raise_ulimit_once()

    # Preload shims once so symbols are global
    for lib in (CUDA_DRIVER, CUDART, NVVM):
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass