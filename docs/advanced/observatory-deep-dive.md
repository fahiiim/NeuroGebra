# Observatory Deep Dive

Advanced usage patterns and internals of the Training Observatory system.

---

## Architecture Overview

The Observatory follows an **event-driven observer pattern**:

```
Model.fit()
  ├─ TrainingLogger           (central dispatcher)
  │    ├─ TerminalDisplay     (colourful terminal output)
  │    ├─ JSONExporter        (structured log file)
  │    ├─ CSVExporter         (metrics table)
  │    ├─ HTMLExporter        (interactive report)
  │    └─ MarkdownExporter    (human-readable report)
  ├─ GradientMonitor          (gradient health tracking)
  ├─ WeightMonitor            (weight distribution analysis)
  ├─ ActivationMonitor        (dead neuron / saturation detection)
  ├─ PerformanceMonitor       (timing & bottleneck detection)
  └─ SmartHealthChecker       (aggregate diagnostics)
```

Events flow from the model through the logger, which filters by level and dispatches to all registered backends.

---

## Event Types

| Event | Level | When It Fires |
|-------|-------|--------------|
| `train_start` | BASIC | Beginning of `model.fit()` |
| `train_end` | BASIC | End of training |
| `epoch_start` | BASIC | Start of each epoch |
| `epoch_end` | BASIC | End of each epoch (with metrics) |
| `batch_start` | DETAILED | Start of each mini-batch |
| `batch_end` | DETAILED | End of each mini-batch (with batch loss) |
| `layer_forward` | EXPERT | After each layer's forward pass |
| `layer_backward` | EXPERT | After each layer's backward pass |
| `gradient_computed` | EXPERT | When a gradient is computed |
| `weight_updated` | EXPERT | When weights are updated |
| `health_check` | BASIC | When a health alert is generated |

---

## Custom Backends

Create your own backend by implementing `handle_{event_type}` methods:

```python
from neurogebra.logging.logger import TrainingLogger, LogLevel, LogEvent

class MyBackend:
    def handle_epoch_end(self, event: LogEvent):
        metrics = event.data.get("metrics", {})
        print(f"My custom log: epoch {event.epoch}, loss={metrics.get('loss'):.4f}")

    def handle_health_check(self, event: LogEvent):
        if event.severity in ("danger", "critical"):
            send_slack_alert(event.message)  # your custom action

logger = TrainingLogger(level=LogLevel.BASIC)
logger.add_backend(MyBackend())
```

You can also register **callbacks** for specific event types:

```python
logger.register_callback("epoch_end", lambda e: log_to_database(e))
```

---

## Custom Health Checks

The `SmartHealthChecker` is configurable:

```python
from neurogebra.logging.health_checks import SmartHealthChecker

checker = SmartHealthChecker(
    patience=10,                  # epochs before stagnation alert
    overfit_ratio=1.3,            # val_loss/train_loss threshold
    stagnation_eps=1e-5,          # minimum improvement threshold
    gradient_vanish_thresh=1e-8,  # custom vanishing threshold
    gradient_explode_thresh=50,   # custom exploding threshold
    dead_neuron_pct=30.0,         # alert at 30% dead neurons
)
```

---

## Computation Graph

Track every operation as a DAG:

```python
from neurogebra.logging.computation_graph import GraphTracker

tracker = GraphTracker(store_values=True)

nid = tracker.record_operation(
    operation="matmul",
    layer_name="dense_0",
    inputs=[X, W],
    output=Z,
    formula="Z = X·W",
)

# Query
nodes = tracker.get_layer_subgraph("dense_0")
path = tracker.get_forward_path()

# Export
data = tracker.export_graph()     # JSON-serialisable dict
tracker.print_graph()             # Rich tree in terminal
```

---

## Formula Renderer

Render mathematical formulas in Unicode or LaTeX:

```python
from neurogebra.logging.formula_renderer import FormulaRenderer

renderer = FormulaRenderer()

# Full model equation
eq = renderer.full_model_equation([
    {"type": "dense", "activation": "relu"},
    {"type": "dense", "activation": "sigmoid"},
])
print(eq)  # ŷ = σ(W₂ · relu(W₁·x + b₁) + b₂)

# Loss formula
renderer.render_loss("cross_entropy")
```

---

## Image Logger

Render images as ASCII art in your terminal:

```python
from neurogebra.logging.image_logger import ImageLogger
import numpy as np

img_logger = ImageLogger()

# Render a grayscale image (28×28 MNIST-style)
image = np.random.rand(28, 28)
img_logger.render_image(image, title="Sample Digit")

# Check if data looks like images
if img_logger.is_image_data(X_train):
    img_logger.render_image(X_train[0])
```

---

## Environment Variables

You can set the default log level via environment variable:

```bash
export NEUROGEBRA_LOG_LEVEL=EXPERT
```

```python
config = LogConfig.from_env()  # reads NEUROGEBRA_LOG_LEVEL
```

---

## Performance Considerations

- **BASIC/DETAILED** levels add negligible overhead (< 1% training time)
- **EXPERT** level adds ~5-10% overhead due to per-layer statistics computation
- **DEBUG** level may add 15-20% overhead — use for debugging only
- Export backends write asynchronously at the end of training (no mid-training I/O)
- Use `LogConfig.production()` for minimal overhead in deployment

---

## Integration with External Tools

### TensorBoard (optional)

```bash
pip install neurogebra[logging]
```

```python
# TensorBoard backend coming in v1.3.0
```

### Weights & Biases (optional)

```bash
pip install neurogebra[logging]
```

```python
# W&B backend coming in v1.3.0
```
