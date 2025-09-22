import os
import numba.cuda.cuda_paths

def setup_native_libs():
    """
    Ensure native libs exist:
    - libdevice.bc (empty file)
    - libnvvm.dylib (empty stub)
    - libcuda.dylib (symlink -> libcudart.dylib)
    """
    base_dir = os.path.dirname(__file__)
    native_dir = os.path.join(base_dir, "native")
    os.makedirs(native_dir, exist_ok=True)

    # libdevice.bc (always empty)
    libdevice_path = os.path.join(native_dir, "libdevice.bc")
    if not os.path.exists(libdevice_path):
        with open(libdevice_path, "wb"):
            pass

    # libnvvm.dylib (empty stub)
    libnvvm_path = os.path.join(native_dir, "libnvvm.dylib")
    if not os.path.exists(libnvvm_path):
        with open(libnvvm_path, "wb"):
            pass

    # libcudart.dylib (shim must be built and placed manually)
    libcudart_path = os.path.join(native_dir, "libcudart.dylib")
    if not os.path.exists(libcudart_path):
        raise FileNotFoundError(
            f"libcudart shim not found at {libcudart_path}. "
            "Please place your built Rust dylib here."
        )

    # libcuda.dylib (symlink → libcudart.dylib)
    libcuda_path = os.path.join(native_dir, "libcuda.dylib")
    if not os.path.exists(libcuda_path):
        os.symlink("libcudart.dylib", libcuda_path)

    return native_dir, libdevice_path, libcudart_path

def patch_libdevice():
    """
    Monkeypatch Numba to point libdevice.bc to shim folder.
    """
    _, libdevice_path, _ = setup_native_libs()

    from collections import namedtuple
    _env_path_tuple = namedtuple("_env_path_tuple", ["by", "info"])
    numba.cuda.cuda_paths._get_libdevice_paths = (
        lambda: _env_path_tuple("CUSTOM_MONKEYPATCH", libdevice_path)
    )

def get_libcudart_path():
    """
    Return the absolute path to libcudart shim (Rust-built dylib).
    """
    _, _, libcudart_path = setup_native_libs()
    return libcudart_path