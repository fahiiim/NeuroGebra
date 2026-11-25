<p align="center">
  <img src="https://raw.githubusercontent.com/fahiiim/NeuroGebra/main/assets/neurogebra-logo.png" alt="NeuroGebra" width="520">
</p>

<h2 align="center">Make the mathematics visible.</h2>

<p align="center">Executable formulas, inspectable neural networks, and a clearer view of learning.</p>

<p align="center">
  <a href="https://pypi.org/project/neurogebra/"><img src="https://img.shields.io/pypi/v/neurogebra?color=007d76" alt="PyPI version"></a>
  <a href="https://github.com/fahiiim/NeuroGebra/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/fahiiim/NeuroGebra/tests.yml?branch=main&label=tests" alt="Test status"></a>
  <a href="https://neurogebra.readthedocs.io/"><img src="https://img.shields.io/readthedocs/neurogebra" alt="Documentation status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-075e59" alt="MIT license"></a>
  <a href="https://pepy.tech/project/neurogebra"><img src="https://static.pepy.tech/badge/neurogebra" alt="Package downloads"></a>
</p>

<p align="center">
  <a href="https://neurogebra.readthedocs.io/">Documentation</a> ·
  <a href="docs/getting-started/first-program.md">Quick start</a> ·
  <a href="docs/api/reference.md">API reference</a> ·
  <a href="RELEASE_NOTES.md">3.0.0 release notes</a> ·
  <a href="docs/project/review-3.md">Technical review</a>
</p>

NeuroGebra is an open-source Python library for learning and experimenting with the mathematics behind machine learning. It brings SymPy formulas, NumPy computation, eager automatic differentiation, neural model building, and training observability into one package.

**This checkout targets 3.0.0.** The source and release notes are prepared locally; publication is a separate maintainer action. See [publishing status](PUBLISHING_STATUS.md).

## What you can do

| Area | Explore |
|---|---|
| Symbolic mathematics | 285 formulas across 10 domains, composition, explanations, LaTeX and analytical derivatives |
| Automatic differentiation | Scalar `Value` and multidimensional `Tensor` operation graphs |
| Neural models | Dense, Conv2D, pooling, dropout, normalization and residual dense layers |
| Training | Adam, SGD, RMSprop, learning-rate schedules, metrics and model checkpoints |
| Observatory | Training events, formulas, gradients, health checks, summaries and exports |
| Scientific experiments | An educational finite-difference PINN solver for data and equation residuals |
| Native expression kernels | Optional C compilation and NVIDIA CUDA/NVRTC evaluation |
| Data and interoperability | 48 educational loaders, 20 project notebooks and framework-specific adapters |

Neural models and PINNs train using NumPy. Native C/CUDA acceleration applies to supported symbolic expression evaluations. Many named dataset loaders generate synthetic teaching data. See [capabilities and limits](docs/project/capabilities.md) before interpreting benchmarks or deploying an experiment.

## Install

Requires Python 3.9+:

```bash
python -m pip install neurogebra
```

For this source checkout:

```bash
python -m pip install -e ".[dev,docs]"
```

Optional extras include `viz`, `datasets`, `frameworks`, `logging`, `cuda` and `docs`. C compilation requires GCC/Clang, and CUDA needs native NVIDIA driver/NVRTC libraries. The `fast` extra installs Numba but does not activate a Numba backend. See [installation](docs/getting-started/installation.md).

## Start with a formula

```python
import numpy as np
from neurogebra import MathForge

sigmoid = MathForge().get("sigmoid")
x = np.array([-2.0, 0.0, 2.0])
print(sigmoid.symbolic_expr)
print(sigmoid.eval(x=x))
print(sigmoid.gradient("x").eval(x=0.0))  # 0.25
print(sigmoid.latex())
```

## Train an inspectable model

```python
import numpy as np
from neurogebra import Layer, Model

X = np.linspace(-1.0, 1.0, 64).reshape(-1, 1)
y = 2.0 * X + 1.0
model = Model([Layer("dense", units=1)])
model.compile(loss="mse", optimizer="adam", learning_rate=0.03)
history = model.fit(X, y, epochs=20, batch_size=16, verbose=False)
print(model.predict(X[:3]))
```

Pass `log_level="basic"` to `compile()` to enable training events. Detailed logging gathers more statistics and costs more time.

## Explore the documentation

- [First program](docs/getting-started/first-program.md) and [architecture](docs/getting-started/how-it-works.md)
- [Formula catalog](docs/reference/formulas.md) and [dataset catalog](docs/reference/datasets.md)
- [ModelBuilder](docs/tutorial/model-builder.md), [optimizers](docs/advanced/optimization.md) and [Observatory](docs/tutorial/training-observatory.md)
- [Native acceleration](docs/science/acceleration.md), [PINNs](docs/science/pinn.md) and [performance engineering](docs/advanced/performance.md)
- [Framework bridges](docs/advanced/framework-bridges.md) and [20 project notebooks](docs/examples/project_notebooks.md)
- [3.0 migration guide](docs/releases/migration-3.md), [troubleshooting](docs/project/troubleshooting.md) and [review/roadmap](docs/project/review-3.md)

## Develop and contribute

```bash
python -m pytest tests -q
python -m mkdocs build --strict
python scripts/check_docs.py --site site
```

Preview the site with `python -m mkdocs serve`. After changing formulas or loaders, regenerate the reference catalogs with `python scripts/generate_reference.py`.

Read [CONTRIBUTING.md](CONTRIBUTING.md) and the [contributor documentation](docs/project/contributing.md). Report bugs through [GitHub Issues](https://github.com/fahiiim/NeuroGebra/issues), and security problems through the [security policy](SECURITY.md).

Created by [Fahim Sarker](https://github.com/fahiiim). Released under the [MIT License](LICENSE).
