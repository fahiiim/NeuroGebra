# Training Observatory & Telemetry

The `neurogebra.logging` subsystem provides comprehensive real-time training telemetry, anomaly detection, statistical summaries, reproducibility fingerprints, and multi-format reporting.

---

## Training Logger & Events

::: neurogebra.logging.logger.TrainingLogger
    options:
      show_root_heading: true
      show_source: false

::: neurogebra.logging.logger.LogEvent
    options:
      show_root_heading: true

::: neurogebra.logging.logger.LogLevel
    options:
      show_root_heading: true

---

## Observatory Pro & Anomaly Detection

::: neurogebra.logging.adaptive.AdaptiveLogger
    options:
      show_root_heading: true
      show_source: false

::: neurogebra.logging.adaptive.AnomalyConfig
    options:
      show_root_heading: true

::: neurogebra.logging.fingerprint.TrainingFingerprint
    options:
      show_root_heading: true
      show_source: false

::: neurogebra.logging.epoch_summary.EpochSummarizer
    options:
      show_root_heading: true
      show_source: false

::: neurogebra.logging.health_warnings.AutoHealthWarnings
    options:
      show_root_heading: true
      show_source: false

---

## Monitors

::: neurogebra.logging.monitors.ActivationMonitor
    options:
      show_root_heading: true

::: neurogebra.logging.monitors.GradientMonitor
    options:
      show_root_heading: true

::: neurogebra.logging.monitors.WeightMonitor
    options:
      show_root_heading: true

::: neurogebra.logging.monitors.PerformanceMonitor
    options:
      show_root_heading: true

---

## Exporters & Visualization

::: neurogebra.logging.dashboard.DashboardExporter
    options:
      show_root_heading: true

::: neurogebra.logging.formula_renderer.FormulaRenderer
    options:
      show_root_heading: true

---

## Educational Trainer

::: neurogebra.training.educational_trainer.EducationalTrainer
    options:
      show_root_heading: true
      show_source: false
