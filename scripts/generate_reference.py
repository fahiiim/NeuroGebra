"""Regenerate the formula and dataset documentation from the installed checkout."""

from __future__ import annotations

import inspect
from pathlib import Path

from neurogebra.datasets import CombinedDatasets, Datasets
from neurogebra.repository import (
    activations,
    algebra,
    calculus,
    linalg,
    losses,
    metrics,
    optimization,
    regularizers,
    statistics,
    transforms,
)

ROOT = Path(__file__).resolve().parents[1]


def write_formulas():
    registries = [
        ("Activations", activations.get_activations),
        ("Losses", losses.get_losses),
        ("Regularizers", regularizers.get_regularizers),
        ("Algebra", algebra.get_algebra_expressions),
        ("Calculus", calculus.get_calculus_expressions),
        ("Statistics", statistics.get_statistics_expressions),
        ("Linear algebra", linalg.get_linalg_expressions),
        ("Optimization", optimization.get_optimization_expressions),
        ("Metrics", metrics.get_metrics_expressions),
        ("Transforms", transforms.get_transforms_expressions),
    ]
    lines = [
        "# Formula catalog",
        "",
        "Generated from the repository by `python scripts/generate_reference.py`.",
        "",
        "Every name below is available through `MathForge().get(name)`. Formulas are symbolic building blocks; check their assumptions and input domains before interpreting a result. A symbolic optimization step is separate from a model optimizer, and symbolic metric expressions are separate from the batch evaluation API.",
        "",
        "```python",
        "from neurogebra import MathForge",
        "",
        'expression = MathForge().get("swish")',
        "print(expression.symbolic_expr)",
        "print(expression.params)",
        'print(expression.gradient("x").symbolic_expr)',
        "```",
        "",
    ]
    count = 0
    for category, get_registry in registries:
        registry = get_registry()
        count += len(registry)
        lines.extend(
            [
                f"## {category} ({len(registry)})",
                "",
                "| Name | Symbolic expression | Default parameters |",
                "|---|---|---|",
            ]
        )
        for name, expr in sorted(registry.items()):
            formula = str(expr.symbolic_expr).replace("|", "&#124;")
            params = (
                ", ".join(f"{key}={value}" for key, value in expr.params.items())
                or "None"
            )
            lines.append(f"| `{name}` | `{formula}` | {params} |")
        lines.append("")
    lines.insert(4, f"**{count} formulas across {len(registries)} domains.**\n")
    (ROOT / "docs/reference/formulas.md").write_text("\n".join(lines), encoding="utf-8")
    return count


def write_datasets():
    names = [name for name in dir(CombinedDatasets) if name.startswith("load_")]
    lines = [
        "# Dataset catalog",
        "",
        f"**{len(names)} unique loaders**, generated from `CombinedDatasets` by `python scripts/generate_reference.py`.",
        "",
        "Most loaders generate synthetic teaching data, including the MNIST, Iris, Fashion-MNIST, Titanic and other familiar names. These are not downloads of the original benchmark datasets. Five loaders optionally use scikit-learn sources and fall back to synthetic data. California housing may require a network fetch when using scikit-learn. Record which source was used before reporting accuracy.",
        "",
        "Return conventions are part of each loader's API. **Pair** means `(X, y)`; **split** means `((X_train, y_train), (X_test, y_test))`. Sizes and options appear below. Large image loaders can allocate substantial memory; start with a small tabular generator.",
        "",
        "```python",
        "from neurogebra.datasets import Datasets",
        "",
        "# A pair: split it explicitly if you need train/test partitions.",
        "X, y = Datasets.load_digits(verbose=False)",
        "",
        "# A split: already partitioned by the loader (synthetic Iris data).",
        "(X_train, y_train), (X_test, y_test) = Datasets.load_iris(verbose=False)",
        "```",
        "",
    ]
    optional_real = {
        "load_wine",
        "load_breast_cancer",
        "load_digits",
        "load_diabetes",
        "load_california_housing",
    }
    for title, base in [("Base loaders", True), ("Expanded loaders", False)]:
        lines.extend(
            [
                f"## {title}",
                "",
                "| Loader and options | Result | Provenance |",
                "|---|---|---|",
            ]
        )
        for name in names:
            if hasattr(Datasets, name) != base:
                continue
            fn = getattr(CombinedDatasets, name)
            signature = inspect.signature(fn)
            options = ", ".join(
                f"{p.name}={p.default!r}" for p in signature.parameters.values()
            )
            returns = str(signature.return_annotation)
            result = "Split" if "tuple[tuple[" in returns else "Pair"
            provenance = (
                "scikit-learn source if available; synthetic fallback"
                if name in optional_real
                else "Synthetic / generated"
            )
            lines.append(f"| `{name}({options})` | {result} | {provenance} |")
        lines.append("")
    lines.extend(
        [
            "## Use original benchmarks",
            "",
            "For benchmark comparisons, obtain original data from its documented publisher or a framework's official dataset loader. Preserve the original train/test split, license, preprocessing and attribution. Installing the `datasets` extra adds scikit-learn; it does not turn every synthetic loader into an original-data loader.",
            "",
        ]
    )
    (ROOT / "docs/reference/datasets.md").write_text("\n".join(lines), encoding="utf-8")
    return len(names)


if __name__ == "__main__":
    (ROOT / "docs/reference").mkdir(exist_ok=True)
    print(
        f"Generated {write_formulas()} formulas and {write_datasets()} dataset loaders"
    )
