import numpy as np
from numba import cuda


class GPUMemoryBuffer:
    """
    GPU buffer wrapper backed by Numba DeviceNDArray.
    Supports 1D and multi-dimensional allocations.
    """

    def __init__(self, arr: np.ndarray = None, length: int = None,
                 dtype=np.float32, shape=None):
        if arr is not None:
            if not isinstance(arr, np.ndarray):
                raise TypeError("GPUMemoryBuffer requires a NumPy array")
            self.dtype = arr.dtype
            self.shape = arr.shape
            self.length = arr.size
            self.size = arr.nbytes
            # Allocate on device and upload
            self.dev_array = cuda.to_device(arr)
        elif length is not None:
            self.dtype = np.dtype(dtype)
            if shape is None:
                self.shape = (length,)
            else:
                if np.prod(shape) != length:
                    raise ValueError(
                        f"Shape {shape} does not match length {length}"
                    )
                self.shape = shape
            self.length = length
            self.size = self.length * self.dtype.itemsize
            self.dev_array = cuda.device_array(self.shape, dtype=self.dtype)
        else:
            raise ValueError(
                "Must provide either arr (NumPy array) or length+dtype"
            )

    @classmethod
    def from_dev_array(cls, dev_array):
        """Wrap an existing Numba DeviceNDArray inside GPUMemoryBuffer."""
        buf = cls.__new__(cls)
        buf.dev_array = dev_array
        buf.dtype = dev_array.dtype
        buf.shape = dev_array.shape
        buf.length = dev_array.size
        buf.size = dev_array.nbytes
        return buf

    def upload(self, arr: np.ndarray, stream=None):
        """Copy host array → device."""
        if arr.shape != self.shape:
            raise ValueError(
                f"Array shape {arr.shape} must match buffer shape {self.shape}"
            )
        if stream:
            self.dev_array.copy_to_device(arr, stream=stream.numba)
        else:
            self.dev_array.copy_to_device(arr)
        cuda.synchronize()

    def download(self, stream=None) -> np.ndarray:
        """Copy device array → host numpy array."""
        if stream:
            return self.dev_array.copy_to_host(stream=stream.numba)
        else:
            return self.dev_array.copy_to_host()

    def __getitem__(self, key):
        """Allow slicing into GPU buffer (downloads requested slice)."""
        full_array = self.dev_array.copy_to_host()
        return full_array[key]

    def free(self):
        """Explicitly release device array."""
        if hasattr(self, "dev_array"):
            del self.dev_array

    def __del__(self):
        self.free()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.free()