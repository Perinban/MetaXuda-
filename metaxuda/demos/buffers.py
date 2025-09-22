import metaxuda
import numpy as np

def run():
    x = np.arange(5, dtype=np.float32)
    buf = metaxuda.GPUMemoryBuffer(x)
    result = buf.download()

    print("Input: ", x)
    print("Output:", result)

if __name__ == "__main__":
    run()
