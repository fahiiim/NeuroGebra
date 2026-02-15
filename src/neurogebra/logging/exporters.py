"""
Exporters — Persist training logs to JSON, CSV, HTML, Markdown, and more.

Each exporter implements a ``handle_event`` method so it can be attached
as a backend to ``TrainingLogger``, and a ``save`` method for flushing
the accumulated data to disk.
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional

from neurogebra.logging.logger import LogEvent


# ======================================================================
# JSON Exporter
# ======================================================================

class JSONExporter:
    """Accumulate events and write a structured JSON log file."""

    def __init__(self, path: str = "training_logs/training_log.json"):
        self.path = path
        self._events: List[Dict[str, Any]] = []

    def handle_event(self, event: LogEvent) -> None:
        self._events.append({
            "event_type": event.event_type,
            "level": event.level.name,
            "timestamp": event.timestamp,
            "epoch": event.epoch,
            "batch": event.batch,
            "layer_name": event.layer_name,
            "layer_index": event.layer_index,
            "severity": event.severity,
            "message": event.message,
            "data": _serialisable(event.data),
        })

    def save(self) -> str:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"events": self._events, "exported_at": time.time()}, f, indent=2)
        return self.path


# ======================================================================
# CSV Exporter
# ======================================================================

class CSVExporter:
    """Export epoch-level metrics to a CSV file."""

    def __init__(self, path: str = "training_logs/metrics.csv"):
        self.path = path
        self._rows: List[Dict[str, Any]] = []

    def handle_event(self, event: LogEvent) -> None:
        if event.event_type != "epoch_end":
            return  # Only log epoch-end metrics
        metrics = event.data.get("metrics", {})
        row = {"epoch": event.epoch, "epoch_time": event.data.get("epoch_time", 0)}
        row.update(metrics)
        self._rows.append(row)

    def save(self) -> str:
        if not self._rows:
            return self.path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fieldnames = list(self._rows[0].keys())
        # Collect all fieldnames from all rows
        for r in self._rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._rows)
        return self.path


# ======================================================================
# HTML Exporter
# ======================================================================

class HTMLExporter:
    """Generate a self-contained HTML training report."""

    def __init__(self, path: str = "training_logs/report.html"):
        self.path = path
        self._events: List[LogEvent] = []
        self._epoch_metrics: List[Dict] = []

    def handle_event(self, event: LogEvent) -> None:
        self._events.append(event)
        if event.event_type == "epoch_end":
            metrics = dict(event.data.get("metrics", {}))
            metrics["epoch"] = event.epoch
            self._epoch_metrics.append(metrics)

    def save(self) -> str:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        losses = [m.get("loss", 0) for m in self._epoch_metrics]
        val_losses = [m.get("val_loss", 0) for m in self._epoch_metrics]
        accs = [m.get("accuracy", 0) for m in self._epoch_metrics]
        epochs_list = list(range(1, len(losses) + 1))

        # Health events
        health_rows = ""
        for e in self._events:
            if e.event_type == "health_check":
                sev = e.severity
                colour = {"danger": "#e74c3c", "warning": "#f39c12",
                          "critical": "#c0392b", "success": "#2ecc71",
                          "info": "#3498db"}.get(sev, "#95a5a6")
                recs = e.data.get("recommendations", [])
                rec_html = "<ul>" + "".join(f"<li>{r}</li>" for r in recs) + "</ul>" if recs else ""
                health_rows += (
                    f'<tr style="border-left:4px solid {colour}">'
                    f"<td>{sev.upper()}</td>"
                    f"<td>{e.message}</td>"
                    f"<td>{rec_html}</td></tr>\n"
                )

        html = _HTML_TEMPLATE.format(
            epochs=epochs_list,
            losses=losses,
            val_losses=val_losses,
            accs=accs,
            n_epochs=len(losses),
            final_loss=f"{losses[-1]:.6f}" if losses else "—",
            final_acc=f"{accs[-1]:.4f}" if accs else "—",
            health_rows=health_rows,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        with open(self.path, "w") as f:
            f.write(html)
        return self.path


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Neurogebra Training Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 40px; background: #0d1117; color: #c9d1d9; }}
  h1 {{ color: #58a6ff; }} h2 {{ color: #79c0ff; }}
  .card {{ background: #161b22; border-radius: 8px; padding: 20px; margin: 20px 0; border: 1px solid #30363d; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #30363d; }}
  th {{ color: #58a6ff; }}
  canvas {{ max-width: 600px; }}
  .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
  .metric .value {{ font-size: 2em; font-weight: bold; color: #58a6ff; }}
  .metric .label {{ color: #8b949e; }}
</style></head><body>
<h1>🔬 Neurogebra Training Observatory Report</h1>
<p style="color:#8b949e">Generated: {timestamp}</p>

<div class="card">
  <div class="metric"><div class="value">{n_epochs}</div><div class="label">Epochs</div></div>
  <div class="metric"><div class="value">{final_loss}</div><div class="label">Final Loss</div></div>
  <div class="metric"><div class="value">{final_acc}</div><div class="label">Final Accuracy</div></div>
</div>

<div class="card"><h2>Loss Curve</h2>
<canvas id="lossChart"></canvas></div>

<div class="card"><h2>Accuracy Curve</h2>
<canvas id="accChart"></canvas></div>

<div class="card"><h2>Health Diagnostics</h2>
<table><tr><th>Severity</th><th>Message</th><th>Recommendations</th></tr>
{health_rows}
</table></div>

<script>
new Chart(document.getElementById('lossChart'), {{
  type: 'line',
  data: {{
    labels: {epochs},
    datasets: [
      {{ label: 'Train Loss', data: {losses}, borderColor: '#58a6ff', fill: false }},
      {{ label: 'Val Loss', data: {val_losses}, borderColor: '#f97583', fill: false }}
    ]
  }},
  options: {{ responsive: true, scales: {{ y: {{ beginAtZero: true }} }} }}
}});
new Chart(document.getElementById('accChart'), {{
  type: 'line',
  data: {{
    labels: {epochs},
    datasets: [{{ label: 'Accuracy', data: {accs}, borderColor: '#56d364', fill: false }}]
  }},
  options: {{ responsive: true, scales: {{ y: {{ beginAtZero: true, max: 1 }} }} }}
}});
</script></body></html>"""


# ======================================================================
# Markdown Exporter
# ======================================================================

class MarkdownExporter:
    """Export training summary as a Markdown report."""

    def __init__(self, path: str = "training_logs/report.md"):
        self.path = path
        self._events: List[LogEvent] = []
        self._epoch_metrics: List[Dict] = []

    def handle_event(self, event: LogEvent) -> None:
        self._events.append(event)
        if event.event_type == "epoch_end":
            metrics = dict(event.data.get("metrics", {}))
            metrics["epoch"] = event.epoch
            self._epoch_metrics.append(metrics)

    def save(self) -> str:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        lines = [
            "# Neurogebra Training Report\n",
            f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n",
            "## Epoch Metrics\n",
            "| Epoch | Loss | Val Loss | Accuracy | Val Accuracy |",
            "|------:|-----:|---------:|---------:|-------------:|",
        ]
        for m in self._epoch_metrics:
            lines.append(
                f"| {m.get('epoch', '?')} "
                f"| {m.get('loss', 0):.6f} "
                f"| {m.get('val_loss', 0):.6f} "
                f"| {m.get('accuracy', 0):.4f} "
                f"| {m.get('val_accuracy', 0):.4f} |"
            )

        lines.append("\n## Health Diagnostics\n")
        for e in self._events:
            if e.event_type == "health_check":
                icon = {"danger": "🔴", "warning": "⚠️", "critical": "🚨",
                        "success": "✅", "info": "ℹ️"}.get(e.severity, "")
                lines.append(f"- {icon} **{e.severity.upper()}**: {e.message}")
                for rec in e.data.get("recommendations", []):
                    lines.append(f"  - 💡 {rec}")

        with open(self.path, "w") as f:
            f.write("\n".join(lines))
        return self.path


# ======================================================================
# Helpers
# ======================================================================

def _serialisable(obj: Any) -> Any:
    """Make an object JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialisable(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return "NaN"
        if obj == float("inf"):
            return "Infinity"
        if obj == float("-inf"):
            return "-Infinity"
        return obj
    if isinstance(obj, (int, str, bool, type(None))):
        return obj
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)
