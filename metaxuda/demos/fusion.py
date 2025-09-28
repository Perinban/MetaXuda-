import numpy as np, math
from numba import cuda
import metaxuda

@cuda.jit
def sin_sqrt(a, out):
    i = cuda.grid(1)
    if i < a.size:
        out[i] = math.sqrt(math.sin(a[i]))

@cuda.jit
def exp_log(a, out):
    i = cuda.grid(1)
    if i < a.size:
        out[i] = math.exp(math.log(a[i]))

def run():
    x = np.linspace(1, 10, 8, dtype=np.float32)
    d_x = cuda.to_device(x)
    d_y = cuda.device_array_like(x)

    exp_log[1, x.size](d_x, d_y)
    cuda.synchronize()

    print("Input: ", x)
    print("Output:", d_y.copy_to_host())

if __name__ == "__main__":
    run()
