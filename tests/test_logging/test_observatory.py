"""Tests for the Training Observatory — logger, monitors, health checks, exporters."""
from __future__ import annotations

import json
import os
import tempfile
import time

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Logger & LogEvent
# ---------------------------------------------------------------------------
from neurogebra.logging.logger import TrainingLogger, LogLevel, LogEvent


class TestLogLevel:
    def test_ordering(self):
        assert LogLevel.SILENT < LogLevel.BASIC < LogLevel.DETAILED < LogLevel.EXPERT < LogLevel.DEBUG

    def test_integer_values(self):
        assert int(LogLevel.SILENT) == 0
        assert int(LogLevel.DEBUG) == 4


class TestLogEvent:
    def test_defaults(self):
        e = LogEvent(event_type="epoch_start", level=LogLevel.BASIC)
        assert e.severity == "info"
        assert e.message == ""
        assert e.data == {}
        assert e.epoch is None


class TestTrainingLogger:
    def test_emit_filters_by_level(self):
        logger = TrainingLogger(level=LogLevel.BASIC)
        logger._emit(LogEvent(event_type="x", level=LogLevel.EXPERT))
        assert len(logger.get_event_log()) == 0
        logger._emit(LogEvent(event_type="x", level=LogLevel.BASIC))
        assert len(logger.get_event_log()) == 1

    def test_backends_dispatched(self):
        calls = []

        class DummyBackend:
            def handle_train_start(self, event):
                calls.append(event)

        logger = TrainingLogger(level=LogLevel.BASIC)
        logger.add_backend(DummyBackend())
        logger.on_train_start(total_epochs=5)
        assert len(calls) == 1
        assert calls[0].event_type == "train_start"

    def test_callbacks(self):
        results = []
        logger = TrainingLogger(level=LogLevel.BASIC)
        logger.register_callback("epoch_end", lambda e: results.append(e))
        logger.on_epoch_start(epoch=0)
        logger.on_epoch_end(epoch=0, metrics={"loss": 0.5})
        assert len(results) == 1

    def test_on_layer_forward(self):
        logger = TrainingLogger(level=LogLevel.DEBUG)
        logger.on_layer_forward(
            layer_index=0, layer_name="dense_0",
            input_data=np.ones((4, 3)),
            output_data=np.zeros((4, 5)),
            formula="z = W·x + b",
        )
        events = logger.get_events_by_type("layer_forward")
        assert len(events) == 1
        assert events[0].data["formula"] == "z = W·x + b"

    def test_on_health_check(self):
        logger = TrainingLogger(level=LogLevel.BASIC)
        logger.on_health_check(
            check_name="nan", severity="critical",
            message="NaN!", recommendations=["check data"],
        )
        assert logger.get_event_log()[-1].severity == "critical"

    def test_summary(self):
        logger = TrainingLogger(level=LogLevel.BASIC)
        logger.on_train_start()
        logger.on_train_end()
        s = logger.summary()
        assert s["total_events"] == 2

    def test_clear(self):
        logger = TrainingLogger(level=LogLevel.BASIC)
        logger.on_train_start()
        logger.clear()
        assert len(logger.get_event_log()) == 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from neurogebra.logging.config import LogConfig


class TestLogConfig:
    def test_defaults(self):
        c = LogConfig()
        assert c.level == LogLevel.BASIC
        assert c.show_formulas is False

    def test_presets(self):
        v = LogConfig.verbose()
        assert v.show_formulas is True
        assert v.show_gradients is True

    def test_serialization(self):
        c = LogConfig.verbose()
        d = c.to_dict()
        c2 = LogConfig.from_dict(d)
        assert c2.show_formulas is True
        assert c2.level == LogLevel.EXPERT

    def test_save_load(self):
        c = LogConfig.standard()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            c.save(path)
            c2 = LogConfig.load(path)
            assert c2.show_timing == c.show_timing
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Monitors
# ---------------------------------------------------------------------------
from neurogebra.logging.monitors import (
    GradientMonitor, WeightMonitor, ActivationMonitor, PerformanceMonitor,
)


class TestGradientMonitor:
    def test_healthy_gradient(self):
        gm = GradientMonitor()
        stats = gm.record("layer_0", np.ones((3, 3)) * 0.1)
        assert stats["status"] == "healthy"

    def test_vanishing(self):
        gm = GradientMonitor()
        stats = gm.record("layer_0", np.ones((3, 3)) * 1e-10)
        assert stats["status"] == "danger"
        assert any("vanish" in a.lower() for a in stats["alerts"])

    def test_exploding(self):
        gm = GradientMonitor()
        stats = gm.record("layer_0", np.ones((3, 3)) * 500)
        assert stats["status"] == "danger"

    def test_nan_detection(self):
        gm = GradientMonitor()
        arr = np.array([1.0, float("nan"), 2.0])
        stats = gm.record("layer_0", arr)
        assert stats["status"] == "critical"


class TestWeightMonitor:
    def test_record(self):
        wm = WeightMonitor()
        stats = wm.record("layer_0", np.random.randn(10, 10))
        assert "mean" in stats
        assert "zeros_pct" in stats


class TestActivationMonitor:
    def test_relu_dead_detection(self):
        am = ActivationMonitor()
        act = np.zeros((100, 50))  # all dead
        stats = am.record("layer_0", act, "relu")
        assert stats["zeros_pct"] == 100.0

    def test_sigmoid_saturation(self):
        am = ActivationMonitor()
        act = np.ones((100, 50)) * 0.999  # saturated
        stats = am.record("layer_0", act, "sigmoid")
        assert stats["saturation_pct"] > 90.0


class TestPerformanceMonitor:
    def test_timing(self):
        pm = PerformanceMonitor()
        pm.record_epoch_time(1.5)
        pm.record_epoch_time(1.2)
        assert len(pm.epoch_times) == 2


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------
from neurogebra.logging.health_checks import SmartHealthChecker, HealthAlert


class TestSmartHealthChecker:
    def test_nan_detection(self):
        checker = SmartHealthChecker()
        alerts = checker.run_all(
            epoch=0,
            train_losses=[float("nan")],
        )
        assert any(a.severity == "critical" for a in alerts)

    def test_overfitting(self):
        checker = SmartHealthChecker()
        alerts = checker.run_all(
            epoch=10,
            train_losses=[0.5, 0.3, 0.2, 0.1, 0.05],
            val_losses=[0.5, 0.6, 0.7, 0.8, 0.9],
        )
        overfit = [a for a in alerts if "overfit" in a.check_name.lower()]
        assert len(overfit) > 0

    def test_vanishing_gradient(self):
        checker = SmartHealthChecker()
        alerts = checker.run_all(
            epoch=5,
            gradient_norms={"dense_0": 1e-10},
        )
        assert any("gradient" in a.check_name.lower() for a in alerts)

    def test_healthy_run(self):
        checker = SmartHealthChecker()
        alerts = checker.run_all(
            epoch=5,
            train_losses=[1.0, 0.8, 0.6, 0.4, 0.3, 0.2],
            val_losses=[1.1, 0.85, 0.65, 0.45, 0.35, 0.25],
            gradient_norms={"dense_0": 0.5},
        )
        severe = [a for a in alerts if a.severity in ("danger", "critical")]
        assert len(severe) == 0


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------
from neurogebra.logging.exporters import JSONExporter, CSVExporter, MarkdownExporter


class TestJSONExporter:
    def test_save(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        exp = JSONExporter(path)
        exp.handle_event(LogEvent(
            event_type="epoch_end", level=LogLevel.BASIC,
            epoch=0, data={"metrics": {"loss": 0.5}},
        ))
        result = exp.save()
        assert os.path.exists(result)
        with open(result) as f:
            data = json.load(f)
        assert len(data["events"]) == 1
        os.unlink(result)


class TestCSVExporter:
    def test_save(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        exp = CSVExporter(path)
        exp.handle_event(LogEvent(
            event_type="epoch_end", level=LogLevel.BASIC,
            epoch=0, data={"metrics": {"loss": 0.42, "accuracy": 0.88}},
        ))
        result = exp.save()
        assert os.path.exists(result)
        contents = open(result).read()
        assert "0.42" in contents
        os.unlink(result)


class TestMarkdownExporter:
    def test_save(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        exp = MarkdownExporter(path)
        exp.handle_event(LogEvent(
            event_type="epoch_end", level=LogLevel.BASIC,
            epoch=0, data={"metrics": {"loss": 0.3}},
        ))
        result = exp.save()
        assert os.path.exists(result)
        os.unlink(result)


# ---------------------------------------------------------------------------
# Computation graph
# ---------------------------------------------------------------------------
from neurogebra.logging.computation_graph import GraphTracker, ComputationNode


class TestGraphTracker:
    def test_record_and_query(self):
        gt = GraphTracker()
        gt.record_operation(
            operation="matmul",
            inputs=[np.ones((4, 3))],
            output=np.ones((4, 5)),
            layer_name="dense_0",
        )
        assert len(gt.get_all_nodes()) == 1
        sub = gt.get_layer_subgraph("dense_0")
        assert len(sub) == 1

    def test_export(self):
        gt = GraphTracker()
        gt.record_operation(
            operation="matmul",
            inputs=[np.ones((4, 3))],
            output=np.ones((4, 5)),
            layer_name="dense_0",
        )
        data = gt.export_graph()
        assert "nodes" in data


# ---------------------------------------------------------------------------
# Integration: Model + Logger
# ---------------------------------------------------------------------------
from neurogebra.builders.model_builder import ModelBuilder


class TestModelObservatoryIntegration:
    """Test the end-to-end model → logger → backend pipeline."""

    def _build_model(self, log_level="expert"):
        builder = ModelBuilder()
        model = builder.Sequential([
            builder.Dense(16, activation="relu"),
            builder.Dense(1, activation="sigmoid"),
        ], name="test_model")
        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            learning_rate=0.01,
            log_level=log_level,
        )
        return model

    def test_compile_creates_logger(self):
        model = self._build_model()
        assert model._logger is not None
        assert model._log_config is not None

    def test_fit_runs_real_training(self):
        model = self._build_model(log_level="basic")
        X = np.random.randn(50, 4)
        y = (X[:, 0] > 0).astype(float)
        history = model.fit(X, y, epochs=3, batch_size=16)
        assert len(history["loss"]) == 3
        assert len(history["val_loss"]) == 3
        # Loss should decrease or at least be finite
        assert all(np.isfinite(l) for l in history["loss"])

    def test_predict_uses_real_forward(self):
        model = self._build_model(log_level="basic")
        X = np.random.randn(20, 4)
        y = (X[:, 0] > 0).astype(float)
        model.fit(X, y, epochs=2, batch_size=10)
        preds = model.predict(np.random.randn(5, 4))
        assert preds.shape == (5, 1)
        assert np.all(np.isfinite(preds))

    def test_evaluate(self):
        model = self._build_model(log_level="basic")
        X = np.random.randn(30, 4)
        y = (X[:, 0] > 0).astype(float)
        model.fit(X, y, epochs=2, batch_size=15)
        result = model.evaluate(X, y)
        assert "loss" in result
        assert "accuracy" in result
