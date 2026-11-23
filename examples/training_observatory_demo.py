#!/usr/bin/env python3
"""
 Neurogebra Training Observatory — Full Demo
================================================

This script demonstrates every feature of the Training Observatory
introduced in v1.2.1:

  • Colourful, depth-level mathematical logging
  • Layer-by-layer forward/backward formula display
  • Gradient & weight health monitoring
  • Smart health diagnostics with recommendations
  • Multiple export formats (JSON, CSV, HTML, Markdown)
  • Preset-based configuration (minimal → research)

Run:
    python examples/training_observatory_demo.py
"""

import numpy as np
from neurogebra.builders.model_builder import ModelBuilder
from neurogebra.logging.config import LogConfig
from neurogebra.logging.logger import LogLevel


def make_dataset(n=200, noise=0.1):
    """Create a simple 2-class spiral dataset."""
    np.random.seed(42)
    half = n // 2
    theta = np.linspace(0, 3 * np.pi, half) + np.random.randn(half) * noise
    r = np.linspace(0.5, 1.0, half)

    x0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    x1 = np.column_stack([-r * np.cos(theta), -r * np.sin(theta)])

    X = np.vstack([x0, x1])
    y = np.hstack([np.zeros(half), np.ones(half)])
    return X, y


# ──────────────────────────────────────────────────────────────────
# 1) Minimal Observatory — just epoch-level progress
# ──────────────────────────────────────────────────────────────────
def demo_minimal():
    print("\n" + "=" * 70)
    print("  DEMO 1 — Minimal Observatory (epoch-level only)")
    print("=" * 70)

    builder = ModelBuilder()
    model = builder.Sequential([
        builder.Dense(16, activation="relu"),
        builder.Dense(8, activation="relu"),
        builder.Dense(1, activation="sigmoid"),
    ], name="MinimalDemo")

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        learning_rate=0.01,
        log_level="basic",
    )

    X, y = make_dataset()
    model.fit(X, y, epochs=5, batch_size=32)


# ──────────────────────────────────────────────────────────────────
# 2) Expert Observatory — full depth with formulas & health checks
# ──────────────────────────────────────────────────────────────────
def demo_expert():
    print("\n" + "=" * 70)
    print("  DEMO 2 — Expert Observatory (layer-by-layer math)")
    print("=" * 70)

    config = LogConfig.verbose()            # see every detail
    config.health_check_interval = 2        # run diagnostics every 2 epochs

    builder = ModelBuilder()
    model = builder.Sequential([
        builder.Dense(32, activation="relu"),
        builder.Dense(16, activation="tanh"),
        builder.Dense(1, activation="sigmoid"),
    ], name="ExpertDemo")

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        learning_rate=0.005,
        log_config=config,
    )

    X, y = make_dataset()
    history = model.fit(X, y, epochs=8, batch_size=32)

    print("\n Training history keys:", list(history.keys()))
    print(f"   Final loss: {history['loss'][-1]:.4f}")
    print(f"   Final val_loss: {history['val_loss'][-1]:.4f}")


# ──────────────────────────────────────────────────────────────────
# 3) Export demo — JSON + Markdown
# ──────────────────────────────────────────────────────────────────
def demo_exports():
    print("\n" + "=" * 70)
    print("  DEMO 3 — Export to JSON + Markdown")
    print("=" * 70)

    config = LogConfig.standard()
    config.export_formats = ["json", "markdown"]
    config.export_dir = "./training_logs"

    builder = ModelBuilder()
    model = builder.Sequential([
        builder.Dense(16, activation="relu"),
        builder.Dense(1, activation="sigmoid"),
    ], name="ExportDemo")

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        learning_rate=0.01,
        log_config=config,
    )

    X, y = make_dataset(n=100)
    model.fit(X, y, epochs=4, batch_size=16)
    print("\n[PASS] Check ./training_logs/ for exported files.")


# ──────────────────────────────────────────────────────────────────
# 4) Standalone monitors — use without a model
# ──────────────────────────────────────────────────────────────────
def demo_standalone_monitors():
    print("\n" + "=" * 70)
    print("  DEMO 4 — Standalone Gradient & Weight Monitors")
    print("=" * 70)

    from neurogebra.logging.monitors import GradientMonitor, WeightMonitor

    gm = GradientMonitor()
    wm = WeightMonitor()

    W = np.random.randn(64, 32) * 0.05
    grad = np.random.randn(64, 32) * 0.001

    g_stats = gm.record("dense_0", grad, weights=W)
    w_stats = wm.record("dense_0", W)

    print(f"\n  Gradient norm: {g_stats['norm_l2']:.6f}")
    print(f"  Grad status:   {g_stats['status']}")
    print(f"  Weight mean:   {w_stats['mean']:.6f}")
    print(f"  Weight std:    {w_stats['std']:.6f}")

    print("\n  Simulating vanishing gradients...")
    tiny = np.ones((10, 10)) * 1e-15
    bad = gm.record("dense_deep", tiny)
    print(f"  Status: {bad['status']}  Alerts: {bad['alerts']}")


# ──────────────────────────────────────────────────────────────────
# 5) Smart health checker standalone
# ──────────────────────────────────────────────────────────────────
def demo_health_checker():
    print("\n" + "=" * 70)
    print("  DEMO 5 — Smart Health Checker")
    print("=" * 70)

    from neurogebra.logging.health_checks import SmartHealthChecker

    checker = SmartHealthChecker()
    alerts = checker.run_all(
        epoch=10,
        train_losses=[1.0, 0.8, 0.5, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005, 0.003],
        val_losses=[1.0, 0.9, 0.7, 0.65, 0.7, 0.8, 0.95, 1.1, 1.3, 1.5, 1.8],
        gradient_norms={"dense_0": 0.5, "dense_1": 1e-9},
    )

    for alert in alerts:
        icon = {"info": "ℹ", "warning": "", "danger": "", "critical": ""}
        print(f"\n  {icon.get(alert.severity, '•')} [{alert.severity.upper()}] {alert.check_name}")
        print(f"    {alert.message}")
        for rec in alert.recommendations:
            print(f"    → {rec}")


# ──────────────────────────────────────────────────────────────────
# 6) Formula renderer standalone
# ──────────────────────────────────────────────────────────────────
def demo_formula_renderer():
    print("\n" + "=" * 70)
    print("  DEMO 6 — Formula Renderer")
    print("=" * 70)

    from neurogebra.logging.formula_renderer import FormulaRenderer

    renderer = FormulaRenderer()
    eq = renderer.full_model_equation([
        {"type": "dense", "activation": "relu"},
        {"type": "dense", "activation": "sigmoid"},
    ])
    print(f"\n  Full model equation:\n  {eq}")
    renderer.render_loss("binary_crossentropy")


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(" Neurogebra Training Observatory — v1.2.1 Feature Demo\n")

    demo_minimal()
    demo_expert()
    demo_exports()
    demo_standalone_monitors()
    demo_health_checker()
    demo_formula_renderer()

    print("\n" + "=" * 70)
    print("  All demos complete! ")
    print("=" * 70)
