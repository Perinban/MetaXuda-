import numpy as np
from numba import cuda

class GPUMemoryBuffer:
    """
    Simple GPU buffer wrapper for host ↔ device transfers.
    """

    def __init__(self, arr: np.ndarray):
        if not isinstance(arr, np.ndarray):
            raise TypeError("GPUMemoryBuffer requires a NumPy array")
        self.host = arr
        self.device = cuda.to_device(arr)

    def download(self):
        """Copy GPU data back to a new NumPy array."""
        out = np.empty_like(self.host)
        self.device.copy_to_host(out)
        return out

    def upload(self, arr: np.ndarray):
        """Copy new NumPy array into this buffer."""
        if arr.shape != self.host.shape or arr.dtype != self.host.dtype:
            raise ValueError("Shape and dtype must match the original array")
        self.device.copy_to_device(arr)
        self.host = arr
