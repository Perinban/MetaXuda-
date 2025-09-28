from numba import cuda


class GPUStream:
    """
    Simple GPU stream wrapper using Numba streams.
    """

    def __init__(self):
        self._numba_stream = cuda.stream()

    @property
    def numba(self):
        """Return Numba stream object for kernel launches."""
        return self._numba_stream

    def sync(self):
        """Synchronize the stream."""
        self._numba_stream.synchronize()

    def close(self):
        """Close is a no-op since Numba cleans up automatically."""
        self._numba_stream = None

    def __del__(self):
        self.close()


# Global default stream
DEFAULT_STREAM = GPUStream()