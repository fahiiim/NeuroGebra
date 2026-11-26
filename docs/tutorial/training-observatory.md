---
level: Foundations
runtime: NumPy / CPU
---

# Training Observatory

Inspect a training run through formulas, events, gradients and summaries. Start with basic epoch reporting; collect richer diagnostics when you need to investigate model behavior. Detailed logging has a runtime and memory cost.


---

## Quick Start

Add **one argument** to `model.compile()` and the Observatory is active:

```python
import numpy as np
from neurogebra import Layer, Model

X = np.linspace(-1.0, 1.0, 32).reshape(-1, 1)
y = 2.0 * X + 1.0
model = Model([Layer("dense", units=1)])
model.compile(loss="mse", optimizer="sgd", learning_rate=0.05,
              log_level="basic", metrics=[])
history = model.fit(X, y, epochs=5, batch_size=32,
                    validation_split=0, verbose=False)
print(len(history["loss"]))  # 5 recorded epoch losses
```

Basic logging reports epoch-level metrics. Exact loss values depend on initialization. The [homepage training trace](../index.md) is a recorded, seeded run, with a generator you can reproduce. For structured events, see the [logging API](../api/logging.md) and [event architecture](../advanced/observatory-deep-dive.md).


---

## Log Levels

| Level | Value | What You See |
|-------|:-----:|-------------|
| `"silent"` | 0 | Nothing (turn off all logging) |
| `"basic"` | 1 | Epoch-level loss/accuracy, training start/end |
| `"detailed"` | 2 | + Batch-level progress, timing information |
| `"expert"` | 3 | + Layer-by-layer formulas, gradient norms, weight stats |
| `"debug"` | 4 | + Every tensor shape, raw statistics, full computation trace |

---

## Colour Coding

The Observatory uses colour to communicate at a glance:

-  **Green** — Healthy: loss decreasing, gradients stable, metrics improving
-  **Yellow** — Warning: something needs attention (high variance, early saturation)
-  **Red** — Danger: vanishing/exploding gradients, diverging loss
-  **White-on-Red** — Critical: NaN/Inf detected, training corrupted
-  **Magenta** — Mathematical formulas (forward/backward equations)
-  **Blue** — Informational (progress messages)
- - **Dim white** — Supplementary details

---

## Preset Configurations

Instead of setting `log_level`, you can pass a full `LogConfig` object:

```python
from neurogebra.logging.config import LogConfig

# Choose a preset
config = LogConfig.minimal()     # Just epoch progress
config = LogConfig.standard()    # + timing + health checks
config = LogConfig.verbose()     # Full math depth — every formula & gradient
config = LogConfig.research()    # Everything + auto-export to files

model.compile(loss="mse", optimizer="adam", log_config=config)
```

### Customising a preset

```python
config = LogConfig.verbose()
config.health_check_interval = 5    # run diagnostics every 5 epochs
config.export_formats = ["json", "html"]
config.export_dir = "./my_logs"

model.compile(loss="mse", optimizer="adam", log_config=config)
```

---

## What the Observatory Shows

### Forward Pass Formulas

```
Forward:  a₁ = relu(W₁·x + b₁)    │ shape: (32, 64) → (32, 32)
Forward:  a₂ = tanh(W₂·a₁ + b₂)   │ shape: (32, 32) → (32, 16)
Forward:  ŷ  = σ(W₃·a₂ + b₃)      │ shape: (32, 16) → (32, 1)
```

### Backward Pass Formulas

```
Backward: ∂L/∂W₃ = ∂L/∂ŷ ⊙ σ'(z₃) · a₂ᵀ
Backward: ∂L/∂W₂ = ∂L/∂a₂ ⊙ tanh'(z₂) · a₁ᵀ
Backward: ∂L/∂W₁ = ∂L/∂a₁ ⊙ relu'(z₁) · xᵀ
```

### Health Diagnostics

```
 [CRITICAL] NaN/Inf Detected
   NaN values found in training loss!
   → Check for division by zero in your data
   → Reduce learning rate (try 1e-4)
   → Add gradient clipping

  [WARNING] Overfitting Detected
   Validation loss increasing while training loss decreases
   → Add dropout layers (rate 0.2-0.5)
   → Reduce model complexity
   → Increase training data
```

---

## Export Formats

| Format | File | Contents |
|--------|------|----------|
| **JSON** | `training_log.json` | Full structured event log |
| **CSV** | `metrics.csv` | Epoch-level metrics table |
| **HTML** | `report.html` | Self-contained report with Chart.js graphs |
| **Markdown** | `report.md` | Human-readable training report |

```python
config = LogConfig.research()
config.export_formats = ["json", "csv", "html", "markdown"]
config.export_dir = "./training_logs"

model.compile(loss="mse", optimizer="adam", log_config=config)
model.fit(X, y, epochs=50)
# → Files saved to ./training_logs/
```

---

## Standalone Usage

You can use the monitoring tools **without** a model:

```python
from neurogebra.logging.monitors import GradientMonitor
from neurogebra.logging.health_checks import SmartHealthChecker
import numpy as np

# Check gradient health
gm = GradientMonitor()
stats = gm.record("my_layer", np.random.randn(64, 32) * 0.001)
print(stats["status"])   # "healthy" | "danger" | "critical"

# Diagnose training history
checker = SmartHealthChecker()
alerts = checker.run_all(
    epoch=10,
    train_losses=[1.0, 0.8, 0.5, 0.3, 0.15],
    val_losses=[1.0, 0.9, 0.85, 0.95, 1.1],
)
for alert in alerts:
    print(f"[{alert.severity}] {alert.message}")
```

---

## Next Steps

- See the [Observatory Deep Dive](../advanced/observatory-deep-dive.md) for advanced usage
- Run the full demo: `python examples/training_observatory_demo.py`
- Read the [API Reference](../api/reference.md) for complete method documentation
