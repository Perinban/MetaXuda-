from numba import cuda

class GPUStream:
    """
    Thin wrapper for cuda.stream.
    """

    def __init__(self):
        self._stream = cuda.stream()

    def sync(self):
        """Synchronize the stream."""
        self._stream.synchronize()

# Default global stream
DEFAULT_STREAM = GPUStream()
