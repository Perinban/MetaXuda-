import metaxuda
import numpy as np

def run():
    x = np.linspace(0, 1, 10, dtype=np.float32)
    buf = metaxuda.GPUMemoryBuffer(x)

    print("Buffer allocated. Syncing stream…")
    metaxuda.DEFAULT_STREAM.sync()
    print("Stream sync complete.")

if __name__ == "__main__":
    run()
