# Installation

NeuroGebra requires Python 3.9 or newer. The CI matrix covers Python 3.9–3.12 across its configured Linux, Windows and macOS runners. Optional frameworks have their own Python, hardware and operating-system requirements.

## Create an environment

=== "Windows PowerShell"

    ```powershell
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install neurogebra
    ```

=== "Linux / macOS"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install neurogebra
    ```

To install this checkout before its PyPI publication:

```bash
python -m pip install -e ".[dev,docs]"
```

## Verify the version

```python
import neurogebra
print(neurogebra.__version__)  # 3.0.0 from this checkout
```

After the maintainer publishes 3.0.0, it can be pinned with `python -m pip install "neurogebra==3.0.0"`.

## Choose extras deliberately

| Extra | Adds | Install |
|---|---|---|
| Core | NumPy, SymPy, SciPy, Matplotlib, Rich and Colorama | `python -m pip install neurogebra` |
| `viz` | Plotly and Seaborn | `python -m pip install "neurogebra[viz]"` |
| `frameworks` | PyTorch, TensorFlow, JAX and jaxlib | `python -m pip install "neurogebra[frameworks]"` |
| `logging` | TensorBoard and Weights & Biases integrations | `python -m pip install "neurogebra[logging]"` |
| `datasets` | scikit-learn | `python -m pip install "neurogebra[datasets]"` |
| `cuda` | CUDA Python bindings | `python -m pip install "neurogebra[cuda]"` |
| `docs` | MkDocs Material and API documentation tools | `python -m pip install "neurogebra[docs]"` |
| `dev` | Testing, coverage, lint and development tools | `python -m pip install "neurogebra[dev]"` |
| `fast` | Numba dependency for experimentation | `python -m pip install "neurogebra[fast]"` |
| `all` | Development, docs, visualization, frameworks, logging, CUDA and dataset extras | `python -m pip install "neurogebra[all]"` |

The `fast` extra installs Numba but does not select a Numba execution backend. C JIT requires a separately installed GCC/Clang toolchain. CUDA JIT requires native NVIDIA driver/NVRTC libraries in addition to the Python extra. Install only the optional framework you need if the full bundle is unsuitable for your platform.

## Build the documentation locally

```bash
python -m pip install -e ".[docs]"
python -m mkdocs serve
```

Open the local address printed by MkDocs. To validate a static build, run `python -m mkdocs build --strict` followed by `python scripts/check_docs.py --site site`.

[Write your first program](first-program.md) or consult [troubleshooting](../project/troubleshooting.md).
