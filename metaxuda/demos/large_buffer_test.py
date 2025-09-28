import numpy as np
import math
import random

from numba import cuda
from metaxuda import GPUMemoryBuffer, run_pipeline, StreamPool


# ================= NUMBA KERNEL ====================
@cuda.jit
def sin_kernel(a, out, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = math.sin(a[i])


# ================= LARGE BUFFER TEST ===========================
def large_buffer_test():
    print("\n=== Large Buffer Test (GPUMemoryBuffer + StreamPool) ===")
    pool = StreamPool(8)
    streams = pool.all()

    num_blocks = 64  # total buffers
    block_size = 128 * 1024 * 1024  # 128 MB each
    total_gb = (num_blocks * block_size) / (1024 ** 3)
    print(f"Allocating {num_blocks} × {block_size // (1024 * 1024)} MB "
          f"(≈{total_gb:.1f} GB total)")

    buffers = []

    # Create and upload buffers one by one with explicit synchronization
    for i in range(num_blocks):
        pattern = 1.0 + i * 0.1
        arr = np.full(block_size // 4, pattern, dtype=np.float32)
        stream = streams[i % len(streams)]

        print(f"Creating buffer {i} with pattern {pattern:.2f}")

        # Create buffer without data first, then upload explicitly
        buf = GPUMemoryBuffer(length=block_size // 4, dtype=np.float32)
        buf.upload(arr, stream)
        stream.sync()  # Ensure this upload completes before moving on

        # Store buffer with its expected pattern for verification
        buffers.append((buf, pattern, stream, i))

    print("✅ All buffers uploaded")

    # Test all buffers, not just random sample, to see pattern of corruption
    print("Verifying all buffers:")
    for idx, (buf, expected_pattern, stream, orig_idx) in enumerate(buffers):
        out = buf.download(stream)
        stream.sync()
        got = float(out[0])
        status = "✅" if abs(got - expected_pattern) < 1e-3 else "❌"
        print(f"  {status} Buffer {orig_idx}: expected={expected_pattern:.2f}, got={got:.2f}")
        if abs(got - expected_pattern) >= 1e-3:
            # Show a few more values to see if there's a pattern
            print(f"    First 10 values: {out[:10]}")
            return  # Exit early on first failure to see the pattern

    # Free buffers
    for i, (buf, _, _, _) in enumerate(buffers, 1):
        buf.free()
        if i % 4 == 0 or i == len(buffers):
            print(f"  Freed {i}/{len(buffers)}")

    print("✅ Large buffer test passed")


# ================= ENTRY ==========================
if __name__ == "__main__":
    large_buffer_test()
    print("\n🎉 Large buffer test complete.")