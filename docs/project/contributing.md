# Contributing

Help make mathematical computing easier to understand. Small, reproducible fixes, clearer examples and numerical checks are especially useful contributions.

## Set up a checkout

```bash
git clone https://github.com/fahiiim/NeuroGebra.git
cd NeuroGebra
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -e ".[dev,docs]"
```

## Find the right module

| Change | Location | Useful validation |
|---|---|---|
| Formula or symbolic behavior | `src/neurogebra/repository/`, `core/expression.py` | Scalar/array behavior, domain boundaries, symbolic derivative |
| Autograd | `core/autograd.py` | Finite differences, broadcasting and repeated backward |
| Layer or training behavior | `builders/model_builder.py`, `training/` | Forward/backward agreement, optimizer update, checkpoint round trip |
| Native backend | `acceleration/` | Correctness against NumPy, supported hardware, shape/dtype and cleanup |
| PINN | `science/pinn.py` | Analytic toy equation, data/residual gradient checks, boundary error |
| Documentation | `docs/`, `mkdocs.yml`, `overrides/` | Strict build, snippet checks, internal links, mobile and dark theme |

## Run the checks

```bash
python -m pytest tests -q
python -m mkdocs build --strict
python scripts/check_docs.py --site site
```

For a package release, use Python 3.11+ for the metadata checker:

```bash
python scripts/check_release.py --tag v3.0.0
python -m build
python -m twine check dist/*
```

## Review expectations

Explain the observed problem and the resulting behavior. Include a focused regression check for changes to numerical results, gradients, serialization or runtime behavior. Benchmark performance changes with the same workload before and after; report compilation separately from warm execution.

Keep public API examples runnable and document limitations beside the feature. Regenerate the catalogs with `python scripts/generate_reference.py` after changing formulas or dataset loaders.

See the repository [contributor guide](https://github.com/fahiiim/NeuroGebra/blob/main/CONTRIBUTING.md) and [code of conduct](https://github.com/fahiiim/NeuroGebra/blob/main/CODE_OF_CONDUCT.md).
