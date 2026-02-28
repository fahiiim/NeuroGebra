# Observatory Pro -- v2.5.3

**Available since v1.3.0, current in v2.5.3** -- Six major upgrades that turn the Training Observatory from a passive log dump into an **active diagnostic engine**.

---

## What's New

| Feature | Problem Solved | Impact |
|---------|---------------|--------|
| **Adaptive Logging** | EXPERT logs everything → 77k entries | 80-90% log reduction |
| **Health Warnings** | "All clear" despite 58% dead neurons | Catches problems automatically |
| **Epoch Summaries** | No statistical view per epoch | Mean/std/min/max per metric |
| **Tiered Storage** | One flat JSON file | 3 focused files: basic/health/debug |
| **Visual Dashboard** | Raw JSON only | Interactive HTML charts |
| **Training Fingerprint** | Can't reproduce runs | Full environment capture |

---

## 1. Smart / Adaptive Logging

The `AdaptiveLogger` wraps a standard `TrainingLogger` and **only escalates to EXPERT detail when something looks suspicious**. In normal operation it stays at BASIC level, reducing log size by 80-90%.

### Anomaly Triggers

| Trigger | Default Threshold | What Happens |
|---------|:-----------------:|--------------|
| Dead neurons (zeros %) | 50% | Escalate + emit warning |
| Gradient spike | 5× rolling average | Escalate + emit warning |
| Vanishing gradient | L2 < 1e-7 | Escalate + emit danger |
| Exploding gradient | L2 > 100 | Escalate + emit danger |
| Loss spike | +50% between batches | Escalate + emit warning |
| NaN / Inf anywhere | Any | Escalate + emit critical |
| Weight stagnation | Δ < 1e-6 for 5 batches | Escalate + emit warning |
| Activation saturation | > 40% | Escalate + emit warning |

### Usage

```python
from neurogebra.logging.adaptive import AdaptiveLogger, AnomalyConfig
from neurogebra.logging.logger import TrainingLogger, LogLevel

# Create a base logger at EXPERT level
base_logger = TrainingLogger(level=LogLevel.EXPERT)

# Wrap it in the adaptive logger
adaptive = AdaptiveLogger(base_logger, config=AnomalyConfig(
    zeros_pct_threshold=50.0,      # trigger on >50% dead neurons
    gradient_spike_factor=5.0,     # trigger on 5× gradient spike
    escalation_cooldown=10,        # stay escalated for 10 events
))

# Use adaptive as a drop-in replacement
adaptive.on_train_start(total_epochs=20)
adaptive.on_epoch_start(0)

# This won't produce EXPERT events (normal data):
adaptive.on_layer_forward(0, "dense_0", output_data=normal_activations)

# This WILL produce EXPERT events (all zeros → dead neurons):
adaptive.on_layer_forward(0, "dense_0", output_data=dead_activations)

# Check what anomalies were detected
print(adaptive.get_anomaly_summary())
```

### Customising Thresholds

```python
config = AnomalyConfig(
    zeros_pct_threshold=30.0,          # more sensitive dead neuron detection
    gradient_spike_factor=3.0,         # more sensitive spike detection
    loss_spike_pct=30.0,               # trigger on 30% loss increase
    weight_stagnation_window=10,       # look at 10 consecutive updates
    escalation_cooldown=20,            # stay in detail mode longer
)
adaptive = AdaptiveLogger(base_logger, config=config)
```

---

## 2. Automated Health Warnings

The `AutoHealthWarnings` engine runs **threshold-based rules** on every batch and epoch, emitting structured `HealthWarning` objects with human-readable diagnoses and actionable advice.

### Rules

| Rule | Condition | Severity | Message |
|------|-----------|----------|---------|
| `dead_relu` | zeros_pct > 50% | warning | "Possible dying ReLU in dense_0" |
| `gradient_spike` | norm > 5× rolling avg | warning | "Possible exploding gradient" |
| `vanishing_gradient` | norm < 1e-7 | danger | "Vanishing gradient in dense_0" |
| `exploding_gradient` | norm > 100 | danger | "Exploding gradient in dense_0" |
| `overfitting` | val_loss / train_loss > 1.3 | warning | "Possible overfitting" |
| `loss_stagnation` | Δloss < 1e-4 for N epochs | warning | "Loss stagnant" |
| `weight_stagnation` | Δweight < 1e-6 for N batches | warning | "Optimizer may have stagnated" |
| `nan_inf_loss` | NaN or Inf in loss | critical | "NaN/Inf detected in loss!" |
| `loss_divergence` | loss ×3 over N batches | danger | "Loss diverging" |
| `activation_saturation` | saturation > 40% | warning | "Activations saturated" |

### Usage

```python
from neurogebra.logging.health_warnings import AutoHealthWarnings, WarningConfig

warnings_engine = AutoHealthWarnings(config=WarningConfig(
    dead_relu_zeros_pct=50.0,
    overfit_patience=3,
    overfit_ratio=1.3,
))

# Call during training
for epoch in range(epochs):
    for batch_idx, (X_batch, y_batch) in enumerate(batches):
        # ... forward/backward ...
        
        # Check batch-level health
        batch_alerts = warnings_engine.check_batch(
            epoch=epoch,
            batch=batch_idx,
            loss=current_loss,
            gradient_norms={"dense_0": 0.05, "dense_1": 0.03},
            activation_stats={"dense_0": {"zeros_pct": 62.0, "activation_type": "relu"}},
        )
        for alert in batch_alerts:
            print(f"  ⚠️ [{alert.severity}] {alert.message}")
    
    # Check epoch-level health
    epoch_alerts = warnings_engine.check_epoch(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
    )

# Get summary
print(warnings_engine.get_summary())
```

Each `HealthWarning` contains:

```python
HealthWarning(
    rule_name="dead_relu",
    severity="warning",
    message="Possible dying ReLU in 'dense_0' (62.0% zeros)",
    diagnosis="Neurons producing zero outputs will receive zero gradients and never recover.",
    recommendations=[
        "Use LeakyReLU(negative_slope=0.01) instead of ReLU",
        "Lower the learning rate",
        "Use He initialisation",
    ],
    layer_name="dense_0",
    epoch=5, batch=10,
)
```

---

## 3. Log Summarization Per Epoch

The `EpochSummarizer` aggregates batch-level statistics and produces **mean, std, min, max** across all batches in each epoch.

### Usage

```python
from neurogebra.logging.epoch_summary import EpochSummarizer

summarizer = EpochSummarizer()

for epoch in range(epochs):
    for batch_idx in range(num_batches):
        summarizer.record_batch(
            epoch=epoch,
            metrics={"loss": batch_loss, "accuracy": batch_acc},
            gradient_norms={"dense_0": grad_norm_0, "dense_1": grad_norm_1},
        )
    
    summary = summarizer.finalize_epoch(epoch)
    print(summary.format_text())
```

### Output

```
══ Epoch 5 Summary (32 batches) ══
  Metrics:
    loss                  mean=0.342100  std=0.015200  min=0.310000  max=0.380000
    accuracy              mean=0.891200  std=0.008500  min=0.870000  max=0.910000
  Gradient Norms:
    dense_0               mean=5.23e-02  std=1.12e-02  min=3.10e-02  max=8.40e-02
    dense_1               mean=2.10e-02  std=5.30e-03  min=1.20e-02  max=3.50e-02
```

### Programmatic Access

```python
# Get structured data
d = summary.to_dict()
print(d["metrics"]["loss"]["mean"])   # 0.3421
print(d["metrics"]["loss"]["std"])    # 0.0152

# Get all epoch summaries
all_summaries = summarizer.get_all_summaries()
```

---

## 4. Tiered Storage / Streaming

Instead of one massive JSON file, `TieredStorage` writes three separate **NDJSON** (newline-delimited JSON) files:

| File | Contains | When Written |
|------|----------|--------------|
| `basic.log` | Epoch metrics, train start/end | Every epoch |
| `health.log` | Warnings, anomalies, health checks | On each alert (immediate) |
| `debug.log` | Full EXPERT-level detail | Only when needed |

### Usage

```python
from neurogebra.logging.tiered_storage import TieredStorage
from neurogebra.logging.logger import TrainingLogger, LogLevel

storage = TieredStorage(
    base_dir="./training_logs",
    write_debug=True,       # set False in production to save I/O
    buffer_size=50,          # flush every 50 events
)

logger = TrainingLogger(level=LogLevel.EXPERT)
logger.add_backend(storage)

# ... train as normal ...

storage.flush()    # final flush
storage.close()    # cleanup

# Check what was written
print(storage.summary())
# {'basic': {'events': 42, 'size_bytes': 8192},
#  'health': {'events': 3, 'size_bytes': 1024},
#  'debug': {'events': 12500, 'size_bytes': 2097152},
#  'total_events': 12545}
```

### Reading Logs

```python
# Easy to grep through specific tiers
basic_events = storage.read_basic()
health_events = storage.read_health()

# Or from command line:
# grep "overfitting" training_logs/health.log
# grep "dense_0" training_logs/debug.log
```

### NDJSON Format

Each line is a self-contained JSON object — easy to stream, grep, and parse:

```json
{"event_type":"epoch_end","level":"BASIC","timestamp":1740000000.0,"epoch":0,"severity":"info","message":"Epoch 1 done","data":{"metrics":{"loss":0.85,"accuracy":0.72}}}
{"event_type":"epoch_end","level":"BASIC","timestamp":1740000001.5,"epoch":1,"severity":"info","message":"Epoch 2 done","data":{"metrics":{"loss":0.63,"accuracy":0.81}}}
```

---

## 5. Visual Dashboard

The `DashboardExporter` generates a **self-contained interactive HTML dashboard** with Chart.js charts.

### Charts Included

- 📉 Loss curves (train + validation)
- 📈 Accuracy curves (train + validation)
- ⏱️ Epoch timing bar chart
- 📊 Raw batch-level loss curve
- 🩺 Health diagnostics timeline

### Usage

```python
from neurogebra.logging.dashboard import DashboardExporter
from neurogebra.logging.logger import TrainingLogger, LogLevel

dashboard = DashboardExporter(path="training_logs/dashboard.html")
logger = TrainingLogger(level=LogLevel.EXPERT)
logger.add_backend(dashboard)

# ... train as normal ...

dashboard.save()  # generates the interactive HTML file
# Open training_logs/dashboard.html in any browser
```

### TensorBoard Integration

```python
from neurogebra.logging.dashboard import TensorBoardBridge

tb = TensorBoardBridge(log_dir="./tb_logs")
if tb.available:
    logger.add_backend(tb)
    # ... after training ...
    tb.close()
    # Then: tensorboard --logdir=./tb_logs
```

### Weights & Biases Integration

```python
from neurogebra.logging.dashboard import WandBBridge

wandb_bridge = WandBBridge(
    project="my_experiment",
    run_name="experiment_001",
    config={"lr": 0.01, "epochs": 50},
)
if wandb_bridge.available:
    logger.add_backend(wandb_bridge)
    # ... after training ...
    wandb_bridge.close()
```

---

## 6. Training Fingerprint / Reproducibility Block

The `TrainingFingerprint` captures **everything needed to reproduce a training run**:

### What It Captures

| Category | Fields |
|----------|--------|
| **Seeds** | random_seed, numpy_seed |
| **Dataset** | SHA-256 hash, shape, dtype, sample count |
| **Versions** | Neurogebra, Python, NumPy, SciPy, SymPy, Rich |
| **Hardware** | CPU model, core count, RAM, GPU (if available) |
| **OS** | System, release, machine architecture |
| **Model** | Architecture hash, full model info dict |
| **Hyperparameters** | All training hyperparameters |
| **Git** | Commit hash, branch name, dirty status |

### Usage

```python
from neurogebra.logging.fingerprint import TrainingFingerprint
import numpy as np

fingerprint = TrainingFingerprint.capture(
    model_info={"name": "my_model", "layers": [...]},
    hyperparameters={"lr": 0.01, "batch_size": 32, "epochs": 50},
    dataset=X_train,        # auto-hashed
    random_seed=42,
)

# Pretty-print
print(fingerprint.format_text())
```

### Output

```
╔══ Training Fingerprint ══╗
  Run ID:       a1b2c3d4e5f6
  Timestamp:    2026-02-27 14:30:00
  Seed:         42
  Dataset Hash: 8f14e45fceea167a
  Dataset:      (10000, 784) (float64)
  Neurogebra:   1.3.0
  Python:       3.11.5
  NumPy:        1.26.0
  CPU:          AMD64 Family (8 cores)
  RAM:          16.0 GB
  GPU:          NVIDIA GeForce RTX 3060
  OS:           Windows 10
  Git:          main@a1b2c3d4 (dirty)
  Model Hash:   f47ac10b58cc
  Hyperparams:  {'lr': 0.01, 'batch_size': 32, 'epochs': 50}
╚═════════════════════════╝
```

### Serialisation

```python
# Save to JSON
import json
with open("fingerprint.json", "w") as f:
    json.dump(fingerprint.to_dict(), f, indent=2)

# Load back
with open("fingerprint.json") as f:
    fp2 = TrainingFingerprint.from_dict(json.load(f))
```

---

## Full Integration Example

Using all v1.3.0 features together:

```python
from neurogebra.builders.model_builder import ModelBuilder
from neurogebra.logging.adaptive import AdaptiveLogger, AnomalyConfig
from neurogebra.logging.health_warnings import AutoHealthWarnings
from neurogebra.logging.epoch_summary import EpochSummarizer
from neurogebra.logging.tiered_storage import TieredStorage
from neurogebra.logging.dashboard import DashboardExporter
from neurogebra.logging.fingerprint import TrainingFingerprint
from neurogebra.logging.logger import TrainingLogger, LogLevel
import numpy as np

# 1. Build model
builder = ModelBuilder()
model = builder.Sequential([
    builder.Dense(64, activation="relu"),
    builder.Dense(32, activation="tanh"),
    builder.Dense(1, activation="sigmoid"),
], name="my_model")

# 2. Create logging pipeline
base_logger = TrainingLogger(level=LogLevel.EXPERT)
adaptive = AdaptiveLogger(base_logger)              # Smart filtering
storage = TieredStorage(base_dir="./logs")           # Tiered files
dashboard = DashboardExporter(path="./logs/dash.html")  # Visual dashboard
base_logger.add_backend(storage)
base_logger.add_backend(dashboard)

warnings = AutoHealthWarnings()                      # Auto health rules
summarizer = EpochSummarizer()                       # Epoch aggregation

# 3. Capture fingerprint
fp = TrainingFingerprint.capture(
    model_info={"name": "my_model", "layers": 3},
    hyperparameters={"lr": 0.01, "batch_size": 32, "epochs": 20},
    dataset=X_train,
    random_seed=42,
)
print(fp.format_text())

# 4. Train with full diagnostics
adaptive.on_train_start(total_epochs=20, model_info=fp.model_info)
for epoch in range(20):
    adaptive.on_epoch_start(epoch)
    for batch in range(num_batches):
        # ... training step ...
        summarizer.record_batch(epoch=epoch, metrics={"loss": loss})
        warnings.check_batch(loss=loss, epoch=epoch, batch=batch)
    
    summary = summarizer.finalize_epoch(epoch)
    print(summary.format_text())
    warnings.check_epoch(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
    adaptive.on_epoch_end(epoch, metrics={"loss": train_loss})

adaptive.on_train_end()

# 5. Save everything
storage.close()
dashboard.save()
print(f"Anomalies detected: {adaptive.get_anomaly_summary()['total_anomalies']}")
print(f"Health warnings: {warnings.get_summary()['total_warnings']}")
```

---

## API Reference

### `AdaptiveLogger`

::: neurogebra.logging.adaptive.AdaptiveLogger
    options:
      show_source: true

### `AnomalyConfig`

::: neurogebra.logging.adaptive.AnomalyConfig

### `AutoHealthWarnings`

::: neurogebra.logging.health_warnings.AutoHealthWarnings

### `WarningConfig`

::: neurogebra.logging.health_warnings.WarningConfig

### `EpochSummarizer`

::: neurogebra.logging.epoch_summary.EpochSummarizer

### `TieredStorage`

::: neurogebra.logging.tiered_storage.TieredStorage

### `DashboardExporter`

::: neurogebra.logging.dashboard.DashboardExporter

### `TensorBoardBridge`

::: neurogebra.logging.dashboard.TensorBoardBridge

### `WandBBridge`

::: neurogebra.logging.dashboard.WandBBridge

### `TrainingFingerprint`

::: neurogebra.logging.fingerprint.TrainingFingerprint
