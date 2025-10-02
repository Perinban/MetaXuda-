import numpy as np, math
from numba import cuda
from metaxuda import GPUMemoryBuffer, StreamPool

def run():
    print("\n=== Large Single Buffer Test (≈6 GB) ===")

    pool = StreamPool(10)
    stream = pool.next()

    total_bytes = 6 * 1024**3
    length = total_bytes // 4  # float32
    print(f"Allocating buffer: {total_bytes / (1024**3):.1f} GB "
          f"({length:,} float32 elements)")

    pattern = 3.14159
    host_arr = np.full(length, pattern, dtype=np.float32)

    buf = GPUMemoryBuffer(length=length, dtype=np.float32)
    buf.upload(host_arr, stream)
    stream.sync()

    out = buf.download(stream)
    stream.sync()

    print("Verification:", float(out[0]), "expected", pattern)
    print("First 10 values:", out[:10])

if __name__ == "__main__":
    run()