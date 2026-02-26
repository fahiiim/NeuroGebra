"""Tests for Observatory Pro features (v1.3.0).

Covers:
    - AdaptiveLogger (anomaly-triggered logging)
    - AutoHealthWarnings (automated threshold warnings)
    - EpochSummarizer (per-epoch statistical summaries)
    - TieredStorage (basic/health/debug log separation)
    - DashboardExporter (HTML dashboard generation)
    - TrainingFingerprint (reproducibility block)
"""
from __future__ import annotations

import json
import os
import tempfile
import time

import numpy as np
import pytest

# =========================================================================
# AdaptiveLogger
# =========================================================================
from neurogebra.logging.adaptive import AdaptiveLogger, AnomalyConfig, AnomalyRecord
from neurogebra.logging.logger import TrainingLogger, LogLevel, LogEvent


class TestAnomalyConfig:
    def test_defaults(self):
        cfg = AnomalyConfig()
        assert cfg.zeros_pct_threshold == 50.0
        assert cfg.gradient_spike_factor == 5.0
        assert cfg.escalation_cooldown == 10

    def test_custom(self):
        cfg = AnomalyConfig(zeros_pct_threshold=30.0, escalation_cooldown=5)
        assert cfg.zeros_pct_threshold == 30.0
        assert cfg.escalation_cooldown == 5


class TestAdaptiveLogger:
    def _make(self, **kwargs):
        base = TrainingLogger(level=LogLevel.EXPERT)
        return AdaptiveLogger(base, config=AnomalyConfig(**kwargs))

    def test_silences_normal_forward(self):
        """Normal activations should NOT produce EXPERT events."""
        adaptive = self._make()
        adaptive.on_layer_forward(
            layer_index=0, layer_name="dense_0",
            output_data=np.random.randn(32, 64),
        )
        # The base logger should have NO layer_forward events
        events = adaptive._base.get_events_by_type("layer_forward")
        assert len(events) == 0

    def test_escalates_on_dead_neurons(self):
        """>50% zeros should trigger escalation."""
        adaptive = self._make()
        output = np.zeros((32, 64))  # 100% dead
        adaptive.on_layer_forward(0, "dense_0", output_data=output)
        events = adaptive._base.get_events_by_type("layer_forward")
        assert len(events) == 1
        assert len(adaptive.anomalies) >= 1
        assert adaptive.anomalies[0].anomaly_type == "dead_neurons"

    def test_escalates_on_nan(self):
        adaptive = self._make()
        output = np.array([1.0, float("nan"), 2.0])
        adaptive.on_layer_forward(0, "dense_0", output_data=output)
        assert adaptive.is_escalated
        assert any(a.anomaly_type == "nan_inf_activation" for a in adaptive.anomalies)

    def test_gradient_spike_detection(self):
        adaptive = self._make(gradient_spike_factor=3.0)
        # Feed normal gradients first
        for _ in range(5):
            adaptive.on_gradient_computed("w0", 0.01)
        # Now spike
        adaptive.on_gradient_computed("w0", 1.0)
        assert any(a.anomaly_type == "gradient_spike" for a in adaptive.anomalies)

    def test_vanishing_gradient(self):
        adaptive = self._make()
        adaptive.on_gradient_computed("w0", 1e-10)
        assert any(a.anomaly_type == "vanishing_gradient" for a in adaptive.anomalies)

    def test_exploding_gradient(self):
        adaptive = self._make()
        adaptive.on_gradient_computed("w0", 500.0)
        assert any(a.anomaly_type == "exploding_gradient" for a in adaptive.anomalies)

    def test_loss_spike(self):
        adaptive = self._make(loss_spike_pct=40.0)
        adaptive.on_batch_end(0, loss=1.0, epoch=0)
        adaptive.on_batch_end(1, loss=2.0, epoch=0)  # 100% increase
        assert any(a.anomaly_type == "loss_spike" for a in adaptive.anomalies)

    def test_weight_stagnation(self):
        adaptive = self._make(weight_stagnation_window=3, weight_stagnation_threshold=1e-8)
        for i in range(5):
            adaptive.on_weight_updated("w0", 1.0, 1.0 + 1e-12)
        assert any(a.anomaly_type == "weight_stagnation" for a in adaptive.anomalies)

    def test_escalation_cooldown(self):
        adaptive = self._make(escalation_cooldown=2)
        # Trigger escalation
        adaptive.on_layer_forward(0, "d0", output_data=np.zeros((10, 10)))
        assert adaptive.is_escalated
        # Cooldown: 2 ticks
        adaptive.on_layer_forward(1, "d1", output_data=np.random.randn(10, 10))
        adaptive.on_layer_forward(2, "d2", output_data=np.random.randn(10, 10))
        assert not adaptive.is_escalated

    def test_anomaly_summary(self):
        adaptive = self._make()
        adaptive.on_gradient_computed("w0", 1e-10)
        summary = adaptive.get_anomaly_summary()
        assert summary["total_anomalies"] >= 1
        assert "vanishing_gradient" in summary["by_type"]

    def test_reset(self):
        adaptive = self._make()
        adaptive.on_gradient_computed("w0", 1e-10)
        adaptive.reset()
        assert len(adaptive.anomalies) == 0
        assert not adaptive.is_escalated

    def test_delegates_epoch_events(self):
        """Epoch events should always pass through."""
        adaptive = self._make()
        adaptive.on_train_start(total_epochs=5)
        adaptive.on_epoch_start(0)
        adaptive.on_epoch_end(0, metrics={"loss": 0.5})
        events = adaptive._base.get_event_log()
        types = [e.event_type for e in events]
        assert "train_start" in types
        assert "epoch_start" in types
        assert "epoch_end" in types


# =========================================================================
# AutoHealthWarnings
# =========================================================================
from neurogebra.logging.health_warnings import AutoHealthWarnings, HealthWarning, WarningConfig


class TestWarningConfig:
    def test_defaults(self):
        cfg = WarningConfig()
        assert cfg.dead_relu_zeros_pct == 50.0
        assert cfg.overfit_ratio == 1.3


class TestAutoHealthWarnings:
    def test_dead_relu_warning(self):
        hw = AutoHealthWarnings()
        alerts = hw.check_batch(
            activation_stats={"dense_0": {"zeros_pct": 80.0, "activation_type": "relu"}},
            epoch=5, batch=10,
        )
        assert any(w.rule_name == "dead_relu" for w in alerts)
        assert "dense_0" in alerts[0].message

    def test_gradient_vanishing(self):
        hw = AutoHealthWarnings()
        alerts = hw.check_batch(
            gradient_norms={"dense_0": 1e-10},
            epoch=1, batch=0,
        )
        assert any(w.rule_name == "vanishing_gradient" for w in alerts)

    def test_gradient_exploding(self):
        hw = AutoHealthWarnings()
        alerts = hw.check_batch(
            gradient_norms={"dense_0": 500.0},
            epoch=1, batch=0,
        )
        assert any(w.rule_name == "exploding_gradient" for w in alerts)

    def test_gradient_spike(self):
        hw = AutoHealthWarnings()
        # Build rolling history
        for i in range(5):
            hw.check_batch(gradient_norms={"dense_0": 0.1}, epoch=0, batch=i)
        # Spike
        alerts = hw.check_batch(gradient_norms={"dense_0": 10.0}, epoch=0, batch=6)
        assert any(w.rule_name == "gradient_spike" for w in alerts)

    def test_nan_loss(self):
        hw = AutoHealthWarnings()
        alerts = hw.check_batch(loss=float("nan"), epoch=0, batch=0)
        assert any(w.rule_name == "nan_inf_loss" for w in alerts)

    def test_overfitting(self):
        hw = AutoHealthWarnings(config=WarningConfig(overfit_patience=2, overfit_ratio=1.2))
        hw._dedup_interval = 0  # disable dedup for test
        for i in range(3):
            hw.check_epoch(
                epoch=i,
                train_loss=0.5 - i * 0.1,
                val_loss=0.5 + i * 0.2,
            )
        alerts = hw.check_epoch(epoch=3, train_loss=0.1, val_loss=1.5)
        assert any(w.rule_name == "overfitting" for w in alerts)

    def test_loss_stagnation(self):
        hw = AutoHealthWarnings(config=WarningConfig(loss_stagnation_window=3, loss_stagnation_eps=1e-3))
        for i in range(5):
            hw.check_epoch(epoch=i, train_loss=0.5)
        # All the same → stagnation
        assert any(w.rule_name == "loss_stagnation" for w in hw.warnings)

    def test_weight_stagnation(self):
        hw = AutoHealthWarnings(config=WarningConfig(weight_stagnation_window=3, weight_stagnation_eps=1e-5))
        for i in range(5):
            hw.check_batch(weight_deltas={"w0": 1e-8}, epoch=0, batch=i)
        assert any(w.rule_name == "weight_stagnation" for w in hw.warnings)

    def test_loss_divergence(self):
        hw = AutoHealthWarnings(config=WarningConfig(loss_divergence_window=3, lr_too_high_loss_factor=2.0))
        hw.check_batch(loss=0.5, epoch=0, batch=0)
        hw.check_batch(loss=0.8, epoch=0, batch=1)
        alerts = hw.check_batch(loss=2.0, epoch=0, batch=2)
        assert any(w.rule_name == "loss_divergence" for w in alerts)

    def test_healthy_run_no_warnings(self):
        hw = AutoHealthWarnings()
        for i in range(5):
            hw.check_batch(
                loss=1.0 - i * 0.1,
                gradient_norms={"dense_0": 0.5},
                epoch=0, batch=i,
            )
        hw.check_epoch(epoch=0, train_loss=0.5, val_loss=0.55)
        critical = [w for w in hw.warnings if w.severity in ("danger", "critical")]
        assert len(critical) == 0

    def test_get_summary(self):
        hw = AutoHealthWarnings()
        hw.check_batch(loss=float("nan"), epoch=0, batch=0)
        summary = hw.get_summary()
        assert summary["total_warnings"] >= 1

    def test_reset(self):
        hw = AutoHealthWarnings()
        hw.check_batch(loss=float("nan"), epoch=0, batch=0)
        hw.reset()
        assert len(hw.warnings) == 0


# =========================================================================
# EpochSummarizer
# =========================================================================
from neurogebra.logging.epoch_summary import EpochSummarizer, EpochSummary, EpochStats


class TestEpochStats:
    def test_to_dict(self):
        s = EpochStats("loss", 10, 0.5, 0.1, 0.2, 0.8, 0.7, 0.3)
        d = s.to_dict()
        assert d["name"] == "loss"
        assert d["count"] == 10
        assert d["mean"] == 0.5


class TestEpochSummarizer:
    def test_basic_summarization(self):
        es = EpochSummarizer()
        for batch in range(10):
            es.record_batch(
                epoch=0,
                metrics={"loss": 1.0 - batch * 0.05, "accuracy": 0.5 + batch * 0.03},
            )
        summary = es.finalize_epoch(0)
        assert summary.num_batches == 10
        assert "loss" in summary.metrics
        assert "accuracy" in summary.metrics
        assert summary.metrics["loss"].count == 10
        # Loss should decrease
        assert summary.metrics["loss"].first > summary.metrics["loss"].last

    def test_gradient_summarization(self):
        es = EpochSummarizer()
        for batch in range(5):
            es.record_batch(
                epoch=0,
                gradient_norms={"dense_0": np.random.uniform(0.01, 0.1)},
            )
        summary = es.finalize_epoch(0)
        assert "dense_0" in summary.gradient_norms
        assert summary.gradient_norms["dense_0"].count == 5

    def test_weight_activation_summarization(self):
        es = EpochSummarizer()
        for batch in range(3):
            es.record_batch(
                epoch=0,
                weight_stats={"dense_0": {"mean": 0.01 * batch, "std": 0.1}},
                activation_stats={"dense_0": {"zeros_pct": 10.0 + batch}},
            )
        summary = es.finalize_epoch(0)
        assert "dense_0" in summary.weight_summaries
        assert "dense_0" in summary.activation_summaries

    def test_format_text(self):
        es = EpochSummarizer()
        for b in range(5):
            es.record_batch(epoch=0, metrics={"loss": 0.5 + b * 0.01})
        s = es.finalize_epoch(0)
        text = s.format_text()
        assert "Epoch 1 Summary" in text
        assert "loss" in text

    def test_to_dict(self):
        es = EpochSummarizer()
        es.record_batch(epoch=0, metrics={"loss": 0.5})
        s = es.finalize_epoch(0)
        d = s.to_dict()
        assert d["epoch"] == 0
        assert "loss" in d["metrics"]

    def test_multiple_epochs(self):
        es = EpochSummarizer()
        for epoch in range(3):
            for batch in range(5):
                es.record_batch(epoch=epoch, metrics={"loss": 1.0 - epoch * 0.2})
            es.finalize_epoch(epoch)
        assert len(es.summaries) == 3

    def test_get_all_summaries(self):
        es = EpochSummarizer()
        es.record_batch(epoch=0, metrics={"loss": 0.5})
        es.finalize_epoch(0)
        all_s = es.get_all_summaries()
        assert len(all_s) == 1
        assert all_s[0]["epoch"] == 0

    def test_reset(self):
        es = EpochSummarizer()
        es.record_batch(epoch=0, metrics={"loss": 0.5})
        es.finalize_epoch(0)
        es.reset()
        assert len(es.summaries) == 0


# =========================================================================
# TieredStorage
# =========================================================================
from neurogebra.logging.tiered_storage import TieredStorage


class TestTieredStorage:
    def test_basic_events_to_basic_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ts = TieredStorage(base_dir=tmpdir, buffer_size=1)
            event = LogEvent(
                event_type="epoch_end", level=LogLevel.BASIC,
                epoch=0, data={"metrics": {"loss": 0.5}},
            )
            ts.handle_event(event)
            ts.flush()
            basic = ts.read_basic()
            assert len(basic) >= 1
            assert basic[0]["event_type"] == "epoch_end"

    def test_health_events_to_health_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ts = TieredStorage(base_dir=tmpdir, buffer_size=1)
            event = LogEvent(
                event_type="health_check", level=LogLevel.BASIC,
                severity="warning", message="Dead ReLU",
                data={"check": "dead_relu"},
            )
            ts.handle_event(event)
            ts.flush()
            health = ts.read_health()
            assert len(health) >= 1
            assert health[0]["severity"] == "warning"

    def test_debug_events_to_debug_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ts = TieredStorage(base_dir=tmpdir, buffer_size=1)
            event = LogEvent(
                event_type="layer_forward", level=LogLevel.EXPERT,
                layer_name="dense_0",
                data={"formula": "z = Wx + b"},
            )
            ts.handle_event(event)
            ts.flush()
            debug = ts.read_debug()
            assert len(debug) >= 1
            assert debug[0]["event_type"] == "layer_forward"

    def test_write_debug_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ts = TieredStorage(base_dir=tmpdir, buffer_size=1, write_debug=False)
            event = LogEvent(
                event_type="layer_forward", level=LogLevel.EXPERT,
                data={},
            )
            ts.handle_event(event)
            ts.flush()
            assert ts.debug_count == 0
            assert ts.read_debug() == []

    def test_severity_routing(self):
        """Events with danger/warning/critical severity go to health.log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ts = TieredStorage(base_dir=tmpdir, buffer_size=1)
            event = LogEvent(
                event_type="batch_end", level=LogLevel.DETAILED,
                severity="danger", message="Loss spike",
                data={},
            )
            ts.handle_event(event)
            ts.flush()
            health = ts.read_health()
            assert len(health) == 1

    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ts = TieredStorage(base_dir=tmpdir, buffer_size=1)
            ts.handle_event(LogEvent(event_type="epoch_end", level=LogLevel.BASIC, data={}))
            ts.handle_event(LogEvent(event_type="health_check", level=LogLevel.BASIC, severity="warning", data={}))
            ts.flush()
            summary = ts.summary()
            assert summary["basic"]["events"] == 1
            assert summary["health"]["events"] == 1

    def test_close(self):
        """close() should flush without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ts = TieredStorage(base_dir=tmpdir)
            ts.handle_event(LogEvent(event_type="epoch_end", level=LogLevel.BASIC, data={}))
            ts.close()
            assert ts.basic_count == 1


# =========================================================================
# DashboardExporter
# =========================================================================
from neurogebra.logging.dashboard import DashboardExporter


class TestDashboardExporter:
    def test_save_creates_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dashboard.html")
            dash = DashboardExporter(path=path)
            # Simulate events
            dash.handle_train_start(LogEvent(
                event_type="train_start", level=LogLevel.BASIC,
                data={"model_info": {"layers": 3}, "total_epochs": 5, "batch_size": 32},
            ))
            for i in range(3):
                dash.handle_epoch_end(LogEvent(
                    event_type="epoch_end", level=LogLevel.BASIC, epoch=i,
                    data={"metrics": {"loss": 1.0 - i * 0.2, "accuracy": 0.5 + i * 0.1,
                                      "val_loss": 1.1 - i * 0.15, "val_accuracy": 0.4 + i * 0.08},
                          "epoch_time": 1.5},
                ))
            dash.handle_health_check(LogEvent(
                event_type="health_check", level=LogLevel.BASIC, epoch=2,
                severity="warning", message="Dead ReLU detected",
                data={"check": "dead_relu", "recommendations": ["Use LeakyReLU"]},
            ))
            result = dash.save()
            assert os.path.exists(result)
            content = open(result, encoding="utf-8").read()
            assert "Neurogebra Training Dashboard" in content
            assert "lossChart" in content
            assert "Dead ReLU" in content

    def test_handles_empty_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.html")
            dash = DashboardExporter(path=path)
            result = dash.save()
            assert os.path.exists(result)


# =========================================================================
# TrainingFingerprint
# =========================================================================
from neurogebra.logging.fingerprint import TrainingFingerprint


class TestTrainingFingerprint:
    def test_capture_basic(self):
        fp = TrainingFingerprint.capture(
            model_info={"layers": [{"name": "dense_0", "units": 64}]},
            hyperparameters={"lr": 0.01, "epochs": 50},
            random_seed=42,
        )
        assert fp.run_id
        assert fp.random_seed == 42
        assert fp.python_version
        assert fp.numpy_version
        assert fp.cpu_count > 0
        assert fp.hyperparameters["lr"] == 0.01
        assert fp.model_architecture_hash is not None

    def test_capture_with_dataset(self):
        X = np.random.randn(100, 10)
        fp = TrainingFingerprint.capture(dataset=X)
        assert fp.dataset_hash is not None
        assert fp.dataset_shape == (100, 10)
        assert fp.dataset_samples == 100

    def test_capture_with_string_hash(self):
        fp = TrainingFingerprint.capture(dataset="abc123hash")
        assert fp.dataset_hash == "abc123hash"

    def test_to_dict(self):
        fp = TrainingFingerprint.capture(random_seed=42)
        d = fp.to_dict()
        assert d["seeds"]["random_seed"] == 42
        assert "python" in d["versions"]
        assert "numpy" in d["versions"]
        assert "cpu" in d["hardware"]

    def test_from_dict_roundtrip(self):
        fp = TrainingFingerprint.capture(
            model_info={"layers": 3},
            hyperparameters={"lr": 0.01},
            random_seed=42,
        )
        d = fp.to_dict()
        fp2 = TrainingFingerprint.from_dict(d)
        assert fp2.random_seed == fp.random_seed
        assert fp2.python_version == fp.python_version
        assert fp2.hyperparameters == fp.hyperparameters

    def test_format_text(self):
        fp = TrainingFingerprint.capture(random_seed=42)
        text = fp.format_text()
        assert "Training Fingerprint" in text
        assert "42" in text
        assert "Python" in text

    def test_neurogebra_version(self):
        fp = TrainingFingerprint.capture()
        assert fp.neurogebra_version  # should not be empty

    def test_git_info(self):
        """Git info should be captured if in a repo (non-fatal if not)."""
        fp = TrainingFingerprint.capture()
        # May or may not have git — just ensure no exceptions
        assert isinstance(fp.git_commit, (str, type(None)))


# =========================================================================
# Integration: all v1.3.0 features together
# =========================================================================

class TestObservatoryProIntegration:
    """Test that all v1.3.0 features work together."""

    def test_adaptive_with_warnings_and_summarizer(self):
        """Full pipeline: adaptive logger + health warnings + epoch summarizer."""
        base = TrainingLogger(level=LogLevel.EXPERT)
        adaptive = AdaptiveLogger(base)
        warnings = AutoHealthWarnings()
        summarizer = EpochSummarizer()

        # Simulate 2 epochs × 5 batches
        adaptive.on_train_start(total_epochs=2)
        for epoch in range(2):
            adaptive.on_epoch_start(epoch)
            for batch in range(5):
                loss = 1.0 - epoch * 0.3 - batch * 0.02
                adaptive.on_batch_end(batch, loss=loss, epoch=epoch)
                summarizer.record_batch(epoch=epoch, metrics={"loss": loss})
                warnings.check_batch(loss=loss, epoch=epoch, batch=batch)

            summary = summarizer.finalize_epoch(epoch)
            adaptive.on_epoch_end(epoch, metrics={"loss": loss})
            warnings.check_epoch(epoch=epoch, train_loss=loss)

        adaptive.on_train_end(final_metrics={"loss": loss})

        # Validate
        assert len(summarizer.summaries) == 2
        assert summarizer.summaries[0].metrics["loss"].count == 5
        assert warnings.get_summary()["total_warnings"] >= 0

    def test_tiered_storage_with_dashboard(self):
        """Tiered storage + dashboard export together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ts = TieredStorage(base_dir=tmpdir, buffer_size=1)
            dash = DashboardExporter(path=os.path.join(tmpdir, "dash.html"))

            logger = TrainingLogger(level=LogLevel.EXPERT)
            logger.add_backend(ts)
            logger.add_backend(dash)

            logger.on_train_start(total_epochs=3, model_info={"name": "test"})
            for epoch in range(3):
                logger.on_epoch_start(epoch)
                logger.on_epoch_end(epoch, metrics={"loss": 1.0 - epoch * 0.2, "accuracy": 0.5})
            logger.on_train_end()

            ts.flush()
            dash.save()

            assert ts.basic_count >= 3
            assert os.path.exists(os.path.join(tmpdir, "dash.html"))

    def test_fingerprint_with_model_info(self):
        """Fingerprint captures model + dataset info."""
        X = np.random.randn(50, 4)
        fp = TrainingFingerprint.capture(
            model_info={"name": "test", "layers": [
                {"name": "dense_0", "units": 16},
                {"name": "dense_1", "units": 1},
            ]},
            hyperparameters={"lr": 0.01, "batch_size": 16, "epochs": 10},
            dataset=X,
            random_seed=42,
        )
        d = fp.to_dict()
        assert d["seeds"]["random_seed"] == 42
        assert d["dataset"]["samples"] == 50
        assert d["model"]["architecture_hash"]
