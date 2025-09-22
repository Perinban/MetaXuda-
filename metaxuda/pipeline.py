import numpy as np

def run_pipeline(ops, arr: np.ndarray):
    """
    Run a sequence of operations on an array.
    Currently CPU fallback — in the shim backend this will be GPU fused.
    Example:
        run_pipeline([np.sin, np.sqrt], x)
    """
    out = arr
    for op in ops:
        out = op(out)
    return out