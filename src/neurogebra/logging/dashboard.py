"""
Dashboard — Lightweight visual dashboard export for the Training Observatory.

Exports training data to self-contained HTML dashboards with interactive
Chart.js charts, and provides helpers for optional TensorBoard / Weights & Biases
integration.

Charts included:
    * Loss curves (train + val)
    * Accuracy curves (train + val)
    * Gradient norm heatmap per layer per epoch
    * Weight distribution evolution
    * Health warning timeline
    * Per-epoch summary statistics

Usage::

    from neurogebra.logging.dashboard import DashboardExporter

    dashboard = DashboardExporter(path="training_logs/dashboard.html")
    logger.add_backend(dashboard)
    # … after training …
    dashboard.save()
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from neurogebra.logging.logger import LogEvent, LogLevel


class DashboardExporter:
    """
    Advanced HTML dashboard backend.

    Collects metrics during training and generates a self-contained
    interactive HTML file with Chart.js visualisations.
    """

    def __init__(self, path: str = "training_logs/dashboard.html"):
        self.path = path
        self._events: List[LogEvent] = []
        self._epoch_metrics: List[Dict] = []
        self._gradient_data: Dict[str, List[float]] = {}  # layer → [norms per epoch]
        self._weight_data: Dict[str, List[Dict]] = {}     # layer → [stats per epoch]
        self._health_events: List[Dict] = []
        self._train_info: Dict[str, Any] = {}
        self._batch_losses: List[float] = []

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    def handle_event(self, event: LogEvent) -> None:
        self._events.append(event)

    def handle_train_start(self, event: LogEvent) -> None:
        self._train_info = event.data
        self._events.append(event)

    def handle_train_end(self, event: LogEvent) -> None:
        self._events.append(event)

    def handle_epoch_end(self, event: LogEvent) -> None:
        self._events.append(event)
        metrics = dict(event.data.get("metrics", {}))
        metrics["epoch"] = event.epoch
        metrics["epoch_time"] = event.data.get("epoch_time", 0)
        self._epoch_metrics.append(metrics)

    def handle_batch_end(self, event: LogEvent) -> None:
        self._events.append(event)
        loss = event.data.get("loss")
        if loss is not None:
            self._batch_losses.append(float(loss))

    def handle_health_check(self, event: LogEvent) -> None:
        self._events.append(event)
        self._health_events.append({
            "epoch": event.epoch,
            "severity": event.severity,
            "message": event.message,
            "check": event.data.get("check", ""),
            "recommendations": event.data.get("recommendations", []),
            "timestamp": event.timestamp,
        })

    def handle_layer_forward(self, event: LogEvent) -> None:
        self._events.append(event)

    def handle_layer_backward(self, event: LogEvent) -> None:
        self._events.append(event)
        grad_stats = event.data.get("grad_weights_stats")
        if grad_stats and event.layer_name:
            norms = self._gradient_data.setdefault(event.layer_name, [])
            norms.append(grad_stats.get("norm_l2", 0))

    def handle_weight_updated(self, event: LogEvent) -> None:
        self._events.append(event)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self) -> str:
        """Generate and save the HTML dashboard. Returns the file path."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        epochs_list = list(range(1, len(self._epoch_metrics) + 1))
        losses = [m.get("loss", 0) for m in self._epoch_metrics]
        val_losses = [m.get("val_loss", 0) for m in self._epoch_metrics]
        accs = [m.get("accuracy", 0) for m in self._epoch_metrics]
        val_accs = [m.get("val_accuracy", 0) for m in self._epoch_metrics]
        epoch_times = [m.get("epoch_time", 0) for m in self._epoch_metrics]

        # Gradient data for heatmap
        grad_layers = list(self._gradient_data.keys())
        grad_matrix = [self._gradient_data.get(l, []) for l in grad_layers]

        # Health timeline
        health_rows = ""
        for h in self._health_events:
            sev = h["severity"]
            colour = {
                "danger": "#e74c3c", "warning": "#f39c12",
                "critical": "#c0392b", "success": "#2ecc71",
                "info": "#3498db",
            }.get(sev, "#95a5a6")
            recs = h.get("recommendations", [])
            rec_html = "<ul>" + "".join(f"<li>{r}</li>" for r in recs) + "</ul>" if recs else ""
            health_rows += (
                f'<tr style="border-left:4px solid {colour}">'
                f'<td>E{h.get("epoch", "?")}</td>'
                f"<td><span class='badge' style='background:{colour}'>{sev.upper()}</span></td>"
                f"<td>{h['message']}</td>"
                f"<td>{rec_html}</td></tr>\n"
            )

        # Model info
        model_info = self._train_info.get("model_info", {})
        n_epochs = self._train_info.get("total_epochs", len(epochs_list))
        batch_size = self._train_info.get("batch_size", "?")

        html = _DASHBOARD_TEMPLATE.format(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            n_epochs=n_epochs,
            batch_size=batch_size,
            final_loss=f"{losses[-1]:.6f}" if losses else "—",
            final_acc=f"{accs[-1]:.4f}" if accs else "—",
            final_val_loss=f"{val_losses[-1]:.6f}" if val_losses else "—",
            final_val_acc=f"{val_accs[-1]:.4f}" if val_accs else "—",
            total_events=len(self._events),
            n_warnings=sum(1 for h in self._health_events if h["severity"] in ("warning", "danger", "critical")),
            epochs=json.dumps(epochs_list),
            losses=json.dumps(losses),
            val_losses=json.dumps(val_losses),
            accs=json.dumps(accs),
            val_accs=json.dumps(val_accs),
            epoch_times=json.dumps(epoch_times),
            batch_losses=json.dumps(self._batch_losses[:2000]),  # cap for perf
            grad_layers=json.dumps(grad_layers),
            grad_matrix=json.dumps(grad_matrix),
            health_rows=health_rows,
            model_info=json.dumps(model_info, indent=2, default=str),
        )

        with open(self.path, "w", encoding="utf-8") as f:
            f.write(html)
        return self.path


# ---------------------------------------------------------------------------
# TensorBoard bridge (optional)
# ---------------------------------------------------------------------------

class TensorBoardBridge:
    """
    Write Training Observatory events to TensorBoard.

    Requires ``tensorboard`` to be installed
    (``pip install neurogebra[logging]``).
    """

    def __init__(self, log_dir: str = "./tb_logs"):
        self.log_dir = log_dir
        self._writer = None
        self._step = 0
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            pass  # TensorBoard not available

    @property
    def available(self) -> bool:
        return self._writer is not None

    def handle_event(self, event: LogEvent) -> None:
        if not self._writer:
            return
        if event.event_type == "epoch_end":
            metrics = event.data.get("metrics", {})
            epoch = event.epoch or self._step
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self._writer.add_scalar(f"metrics/{key}", val, epoch)
            self._step += 1

    def handle_epoch_end(self, event: LogEvent) -> None:
        self.handle_event(event)

    def handle_health_check(self, event: LogEvent) -> None:
        if not self._writer:
            return
        self._writer.add_text(
            "health_checks",
            f"**[{event.severity.upper()}]** {event.message}",
            self._step,
        )

    def close(self) -> None:
        if self._writer:
            self._writer.close()


# ---------------------------------------------------------------------------
# Weights & Biases bridge (optional)
# ---------------------------------------------------------------------------

class WandBBridge:
    """
    Log Training Observatory events to Weights & Biases.

    Requires ``wandb`` to be installed (``pip install neurogebra[logging]``).
    """

    def __init__(self, project: str = "neurogebra", run_name: Optional[str] = None,
                 config: Optional[Dict] = None):
        self._run = None
        try:
            import wandb
            self._run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                reinit=True,
            )
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._run is not None

    def handle_event(self, event: LogEvent) -> None:
        if not self._run:
            return
        import wandb
        if event.event_type == "epoch_end":
            metrics = event.data.get("metrics", {})
            wandb.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                      step=event.epoch)

    def handle_epoch_end(self, event: LogEvent) -> None:
        self.handle_event(event)

    def handle_health_check(self, event: LogEvent) -> None:
        if not self._run:
            return
        import wandb
        wandb.alert(
            title=f"Health: {event.severity.upper()}",
            text=event.message,
            level=wandb.AlertLevel.WARN if event.severity == "warning" else wandb.AlertLevel.ERROR,
        )

    def close(self) -> None:
        if self._run:
            import wandb
            wandb.finish()


# ===========================================================================
# HTML Dashboard Template
# ===========================================================================

_DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neurogebra Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #c9d1d9; }}
  .header {{ background: linear-gradient(135deg, #161b22 0%, #1a1f2e 100%);
             padding: 30px 40px; border-bottom: 1px solid #30363d; }}
  .header h1 {{ color: #58a6ff; font-size: 1.8em; }}
  .header p {{ color: #8b949e; margin-top: 5px; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
  .card {{ background: #161b22; border-radius: 12px; padding: 24px;
           border: 1px solid #30363d; transition: border-color 0.2s; }}
  .card:hover {{ border-color: #58a6ff; }}
  .card h2 {{ color: #79c0ff; font-size: 1.1em; margin-bottom: 16px; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; }}
  .metric {{ text-align: center; padding: 12px; background: #0d1117; border-radius: 8px; }}
  .metric .value {{ font-size: 1.8em; font-weight: 700; color: #58a6ff; }}
  .metric .label {{ color: #8b949e; font-size: 0.85em; margin-top: 4px; }}
  .wide {{ grid-column: 1 / -1; }}
  canvas {{ max-height: 300px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  th, td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid #21262d; }}
  th {{ color: #58a6ff; font-weight: 600; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            color: #fff; font-size: 0.8em; font-weight: 600; }}
  pre {{ background: #0d1117; padding: 16px; border-radius: 8px;
         overflow-x: auto; font-size: 0.85em; color: #8b949e; }}
  .tab-bar {{ display: flex; gap: 0; margin-bottom: 16px; }}
  .tab {{ padding: 8px 18px; cursor: pointer; background: #0d1117;
          border: 1px solid #30363d; color: #8b949e; font-size: 0.9em; }}
  .tab:first-child {{ border-radius: 8px 0 0 8px; }}
  .tab:last-child {{ border-radius: 0 8px 8px 0; }}
  .tab.active {{ background: #161b22; color: #58a6ff; border-color: #58a6ff; }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}
</style></head><body>

<div class="header">
  <h1>🔬 Neurogebra Training Dashboard</h1>
  <p>Generated: {timestamp}</p>
</div>

<div class="container">

  <!-- Summary cards -->
  <div class="metric-grid" style="margin: 24px 0;">
    <div class="metric"><div class="value">{n_epochs}</div><div class="label">Epochs</div></div>
    <div class="metric"><div class="value">{batch_size}</div><div class="label">Batch Size</div></div>
    <div class="metric"><div class="value">{final_loss}</div><div class="label">Final Loss</div></div>
    <div class="metric"><div class="value">{final_acc}</div><div class="label">Final Accuracy</div></div>
    <div class="metric"><div class="value">{final_val_loss}</div><div class="label">Val Loss</div></div>
    <div class="metric"><div class="value">{final_val_acc}</div><div class="label">Val Accuracy</div></div>
    <div class="metric"><div class="value">{total_events}</div><div class="label">Total Events</div></div>
    <div class="metric"><div class="value">{n_warnings}</div><div class="label">Warnings</div></div>
  </div>

  <!-- Charts -->
  <div class="grid">

    <div class="card">
      <h2>📉 Loss Curves</h2>
      <canvas id="lossChart"></canvas>
    </div>

    <div class="card">
      <h2>📈 Accuracy Curves</h2>
      <canvas id="accChart"></canvas>
    </div>

    <div class="card">
      <h2>⏱️ Epoch Timing</h2>
      <canvas id="timeChart"></canvas>
    </div>

    <div class="card">
      <h2>📊 Batch Loss (raw)</h2>
      <canvas id="batchLossChart"></canvas>
    </div>

  </div>

  <!-- Health diagnostics -->
  <div class="card wide" style="margin-top: 20px;">
    <h2>🩺 Health Diagnostics</h2>
    <table>
      <tr><th>Epoch</th><th>Severity</th><th>Message</th><th>Recommendations</th></tr>
      {health_rows}
    </table>
  </div>

  <!-- Model info -->
  <div class="card wide" style="margin-top: 20px;">
    <h2>🧠 Model Info</h2>
    <pre>{model_info}</pre>
  </div>

</div>

<script>
const epochs = {epochs};
const losses = {losses};
const valLosses = {val_losses};
const accs = {accs};
const valAccs = {val_accs};
const epochTimes = {epoch_times};
const batchLosses = {batch_losses};

const chartOpts = {{ responsive: true, plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }},
  scales: {{ x: {{ ticks: {{ color: '#8b949e' }} }}, y: {{ ticks: {{ color: '#8b949e' }}, beginAtZero: true }} }} }};

new Chart(document.getElementById('lossChart'), {{
  type: 'line', data: {{ labels: epochs, datasets: [
    {{ label: 'Train Loss', data: losses, borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.1)', fill: true, tension: 0.3 }},
    {{ label: 'Val Loss', data: valLosses, borderColor: '#f97583', backgroundColor: 'rgba(249,117,131,0.1)', fill: true, tension: 0.3 }}
  ] }}, options: chartOpts
}});

new Chart(document.getElementById('accChart'), {{
  type: 'line', data: {{ labels: epochs, datasets: [
    {{ label: 'Train Acc', data: accs, borderColor: '#56d364', fill: false, tension: 0.3 }},
    {{ label: 'Val Acc', data: valAccs, borderColor: '#d2a8ff', fill: false, tension: 0.3 }}
  ] }}, options: {{ ...chartOpts, scales: {{ ...chartOpts.scales, y: {{ ...chartOpts.scales.y, max: 1 }} }} }}
}});

new Chart(document.getElementById('timeChart'), {{
  type: 'bar', data: {{ labels: epochs, datasets: [
    {{ label: 'Epoch Time (s)', data: epochTimes, backgroundColor: '#388bfd55', borderColor: '#388bfd', borderWidth: 1 }}
  ] }}, options: chartOpts
}});

if (batchLosses.length > 0) {{
  const batchLabels = batchLosses.map((_, i) => i + 1);
  new Chart(document.getElementById('batchLossChart'), {{
    type: 'line', data: {{ labels: batchLabels, datasets: [
      {{ label: 'Batch Loss', data: batchLosses, borderColor: '#f0883e', borderWidth: 1, pointRadius: 0, tension: 0.1 }}
    ] }}, options: {{ ...chartOpts, elements: {{ point: {{ radius: 0 }} }} }}
  }});
}}
</script></body></html>"""
