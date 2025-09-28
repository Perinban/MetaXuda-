from .stream import GPUStream


class StreamPool:
    """Round-robin stream pool."""

    def __init__(self, num_streams: int = 8):
        if num_streams <= 0:
            raise ValueError("StreamPool must be initialized with ≥1 stream")
        self.streams = [GPUStream() for _ in range(num_streams)]
        self._index = 0

    def next(self) -> GPUStream:
        stream = self.streams[self._index]
        self._index = (self._index + 1) % len(self.streams)
        return stream

    def all(self):
        """Return all streams for manual management."""
        return list(self.streams)

    def sync_all(self):
        for s in self.streams:
            s.sync()