import metaxuda
import numpy as np

def run():
    x = np.linspace(0, np.pi, 8, dtype=np.float32)

    # Run fused ops (CPU fallback here, GPU fusion in shim backend)
    out = metaxuda.run_pipeline([np.sin, np.sqrt], x)

    print("Input: ", x)
    print("Output:", out)

if __name__ == "__main__":
    run()
