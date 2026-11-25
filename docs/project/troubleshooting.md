# Troubleshooting

## Confirm the interpreter and version

```bash
python -m pip show neurogebra
python -c "import sys, neurogebra; print(sys.executable); print(neurogebra.__version__); print(neurogebra.__file__)"
```

Use `python -m pip` from the environment that runs your notebook or script. Restart the notebook kernel after upgrading. If you are working from source, install that checkout with `python -m pip install -e .`.

## Match the error to its cause

| Symptom | Check | Next action |
|---|---|---|
| `ModuleNotFoundError` | The interpreter printed above | Install into that environment; select the same notebook kernel |
| No C/C++ compiler found | GCC or Clang is on PATH | Install a toolchain or evaluate with `backend="numpy"` |
| Permission denied when compiling | The JIT cache directory is writable | Set `NEUROGEBRA_CACHE_DIR` to a writable directory; C artifacts use its `c_jit` subdirectory |
| CUDA unavailable | Driver, NVIDIA device, NVRTC shared libraries and `cuda-python` | Install the complete CUDA runtime/toolkit stack; the Python extra alone does not supply every native component |
| Model training uses the CPU | Which API is running | `Model.fit()` is NumPy based; CUDA JIT applies to supported expression kernels |
| Dataset unpacking fails | The specific loader's return convention | Some return `(X, y)`, others return two train/test pairs; check the [catalog](../reference/datasets.md) |
| Very high accuracy on a familiar dataset name | Data provenance | Confirm whether the loader generated synthetic data before interpreting the result |
| Shape mismatch in a model | Batch and feature axes | Dense inputs use `(samples, features)`; image layers use NHWC |
| Cross-entropy rejects labels | Labels and output shape | Use integer class indices or nonnegative class probabilities summing to one |
| PINN training is slow | Network parameter and collocation counts | Start with a small network and grid; residual gradients evaluate every parameter numerically |
| Gradients grow across backward calls | Leaf gradient buffers | Reset leaf gradients between independent steps and rebuild the forward graph |
| Equations appear as source text | JavaScript and network availability | MathJax is loaded from a CDN; allow it or provide an equivalent local asset |

## Report a reproducible issue

Include the full traceback, a minimal script, NeuroGebra/Python/NumPy versions, OS, shapes, backend and expected versus actual output. For numerical issues, include an independent analytical result or finite-difference check when possible.

Report bugs in [GitHub Issues](https://github.com/fahiiim/NeuroGebra/issues). Report security issues privately through the [security policy](https://github.com/fahiiim/NeuroGebra/blob/main/SECURITY.md).
