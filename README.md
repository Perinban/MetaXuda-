# MetaXuda

MetaXuda is an **experimental CUDA-compatible runtime shim for Apple Silicon**, written in **Rust**, that allows **Numba CUDA kernels** to run *unmodified* by transparently mapping CUDA runtime calls to **Apple Metal**.

It is designed as a drop-in replacement for core CUDA runtime libraries, enabling GPU-accelerated Python workflows on macOS **without requiring the NVIDIA CUDA Toolkit or NVIDIA hardware**.

---

## ✨ Features

* Drop-in replacement for `libcudart.dylib` and `libcuda.dylib`
* Run **Numba CUDA kernels** (`@cuda.jit`) directly on Apple Metal
* Metal-backed implementations of core CUDA APIs:

  * `cudaMalloc` / `cudaFree`
  * `cudaMemcpy` / `cudaMemcpyAsync`
  * `cudaLaunchKernel`
* Asynchronous execution with **stream-style overlap** (copy / compute / copy)
* Tier-aware memory management (GPU-first execution)
* Ships with:

  * Stubbed `libdevice.bc` for Numba compatibility
  * Precompiled Metal `.metallib` shaders for fused math operations
  * **`cuda_pipeline.so`**, exposing a low-level execution API that allows Numba and other callers to bypass the CUDA runtime shim and dispatch operations directly
* No CUDA Toolkit, NVIDIA drivers, or NVIDIA GPU required

---

## ⚠️ Project Status

**Alpha / Research Prototype**

MetaXuda is under active development and currently targets:

* Numba CUDA kernels
* Single-GPU execution on Apple Silicon

Not all CUDA APIs are implemented, and behavior may differ from NVIDIA CUDA in edge cases.

---

## ⚙️ Installation

### Requirements

* macOS 13+
* Python >= 3.10
* NumPy >= 1.23
* Numba >= 0.59

### Install (Editable / Dev)

```bash
# Clone the repository
git clone https://github.com/perinban/MetaXuda.git
cd MetaXuda

# Install in editable mode
pip install -e .
```

The installation places the required shim libraries (`libcudart.dylib`, `libcuda.dylib`, and `libdevice.bc`) inside the package so they can be discovered by Numba at runtime.

---

## 📂 Package Layout

MetaXuda ships demos and helper modules **inside the Python package** so they are available in editable and installed modes:

```
metaxuda/
├── buffers/        # GPU, managed, and tiered buffer abstractions
├── execution/      # Direct and pooled execution backends
├── streams/        # Stream and async execution helpers (Numba-compatible)
├── demos/          # End-to-end demos and debug examples
├── native/         # Native shims and pipelines
│   ├── libcudart.dylib
│   ├── libcuda.dylib
│   ├── libnvvm.dylib
│   ├── libdevice.bc
│   └── cuda_pipeline.so
├── env.py          # Environment detection and setup
├── patch.py        # Numba / runtime patching hooks
└── __init__.py
```

The **`demos/`** directory contains runnable examples covering kernel execution, buffers, streams, disk tiering, and the direct math pipeline.

You can run them directly once the package is installed:

```bash
python -m metaxuda.demos.add
python -m metaxuda.demos.pipeline
```

---

## 🚀 Usage

Once installed, existing Numba CUDA code should run **without modification**:

```python
from numba import cuda
import numpy as np

@cuda.jit
def add(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a[i] + b[i]

n = 1024
a = np.arange(n, dtype=np.float32)
b = np.arange(n, dtype=np.float32)
out = np.zeros_like(a)

add[32, 32](a, b, out)
print(out[:5])
```

Execution is transparently dispatched to Metal via the MetaXuda runtime.

---

## 🗜️ Quantization, Compression, and Disk Tiering

MetaXuda supports **quantized and compressed data storage** for non-resident buffers and intermediate results. These behaviors are controlled via environment variables and handled by the runtime initialization logic in `env.py`.

This is primarily used for **Tier‑3 (disk-backed) storage**, allowing large workloads to exceed GPU memory limits while minimizing I/O and storage overhead.

### Environment Configuration

The shim reads the following environment variables at startup:

* `MX_ENABLE_DATASTORE_COMPRESSION` *(default: `1`)*
  Enable or disable compression for spilled data blocks.

* `MX_DATASTORE_COMPRESSION_TYPE` *(default: `lz4`)*
  Compression algorithm to use (e.g. `lz4`).

* `MX_DATASTORE_COMPRESSION_LEVEL` *(default: `3`)*
  Compression level passed to the backend compressor.

* `MX_DISK_PARALLELISM_LEVEL` *(default: `auto`)*
  Controls parallel read/write behavior for disk operations.

* `MX_DISK_SPILL_ENABLED` *(default: `0`)*
  Enable spilling GPU buffers to disk when memory pressure occurs.

* `MX_TIER3_STRATEGY` *(default: `prefer_external`)*
  Strategy for selecting Tier‑3 storage locations.

* `MX_TIER3_INTERNAL_PATH` *(default: `block_store`)*
  Directory used for internal Tier‑3 storage.

* `MX_TIER3_EXTERNAL_DEVICES` *(format: `id:path,id:path`)*
  Comma‑separated list of external devices or paths for Tier‑3 storage.

* `MX_DEBUG` *(options: `memory`)*
  Enable debug logging for specific subsystems.

These settings allow fine‑grained control over **compression, quantization, disk spill behavior, and debugging** without changing application code.

---

## 🧮 Operation Coverage

MetaXuda includes a **precompiled Metal math pipeline** (`cuda_pipeline.so`) implementing a broad set of scalar and elementwise operations that can be invoked directly by Numba or higher-level tooling.

* **230+ operations** covering:

  * Arithmetic, comparison, and logical ops
  * Trigonometric and hyperbolic functions
  * Exponentials, logarithms, and powers
  * Reductions and distance metrics
  * Activation functions (ReLU, GELU, SiLU, Mish, etc.)
  * Probability distributions and loss functions
  * Signal, interpolation, and utility math
* Each operation is mapped to a corresponding **Metal expression**
* Selected ops support **fast-math variants** where numerically safe
* **Full operation list**: See `config/operations.json` for all supported operations and their signatures

This allows many Numba-generated kernels to execute without requiring full PTX → Metal translation, significantly reducing overhead.

---

## 🧠 Architecture Overview

* **Rust-based CUDA shim** implementing core CUDA runtime APIs
* **Metal compute pipelines** for kernel execution
* **Stubbed NVVM / libdevice layer** for Numba compilation compatibility
* Python package acts as a loader and distribution mechanism for native libraries

---

## License

MetaXuda is free for students and personal use. Commercial use requires a license.

- 🎓 **Students**: Free with valid educational email
- 👤 **Personal**: Free for non-commercial projects
- 🏢 **Commercial**: Contact p.perinban@gmail.com

See [LICENSE](LICENSE) for full terms.

---

## 🙏 Disclaimer

MetaXuda is **not affiliated with NVIDIA**. CUDA is a trademark of NVIDIA Corporation. This project is an independent compatibility layer intended for research and development purposes.