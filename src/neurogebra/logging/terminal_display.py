"""
TerminalDisplay — Rich colour-coded terminal dashboard for training.

Renders beautiful, structured output using the ``rich`` library:
nested progress bars, layer tables, gradient sparklines, formula
rendering, and severity-coded health alerts.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        TimeRemainingColumn,
        SpinnerColumn,
    )
    from rich.tree import Tree
    from rich.columns import Columns
    from rich.rule import Rule
    from rich import box

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from neurogebra.logging.logger import LogEvent, LogLevel


# ======================================================================
# Sparkline helper (works without rich too)
# ======================================================================

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: List[float], width: int = 20) -> str:
    """Return a Unicode sparkline string for *values*."""
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    # Resample / truncate to *width*
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    return "".join(
        _SPARK_CHARS[min(int((v - mn) / rng * (len(_SPARK_CHARS) - 1)), len(_SPARK_CHARS) - 1)]
        for v in values
    )


# ======================================================================
# Colour helpers
# ======================================================================

_SEVERITY_STYLES = {
    "info": "blue",
    "success": "bold green",
    "warning": "bold yellow",
    "danger": "bold red",
    "critical": "bold white on red",
    "muted": "dim white",
}

_SEVERITY_ICONS = {
    "info": "ℹ️ ",
    "success": "✅",
    "warning": "⚠️ ",
    "danger": "🔴",
    "critical": "🚨",
}


def _sev_style(severity: str) -> str:
    return _SEVERITY_STYLES.get(severity, "white")


# ======================================================================
# Terminal Display backend
# ======================================================================

class TerminalDisplay:
    """
    Rich terminal backend for the Training Observatory logger.

    Attach it to a ``TrainingLogger`` via ``logger.add_backend(TerminalDisplay(config))``.
    The logger will call ``handle_<event_type>`` methods automatically.
    """

    def __init__(self, config: Optional[Any] = None):
        if not HAS_RICH:
            raise ImportError(
                "The 'rich' package is required for TerminalDisplay. "
                "Install it with: pip install rich"
            )
        self.console = Console()
        self.config = config
        self._loss_history: List[float] = []
        self._val_loss_history: List[float] = []
        self._acc_history: List[float] = []
        self._epoch_progress: Optional[Progress] = None

    # ------------------------------------------------------------------
    # Event handlers (called by TrainingLogger._emit)
    # ------------------------------------------------------------------

    def handle_train_start(self, event: LogEvent) -> None:
        d = event.data
        total_epochs = d.get("total_epochs", "?")
        total_samples = d.get("total_samples", "?")
        batch_size = d.get("batch_size", "?")

        self.console.print()
        self.console.print(Rule("[bold cyan]NEUROGEBRA TRAINING OBSERVATORY[/]", style="cyan"))
        self.console.print()

        model_info = d.get("model_info", {})
        info_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        info_table.add_column(style="bold cyan")
        info_table.add_column()
        info_table.add_row("Model", str(model_info.get("name", "—")))
        info_table.add_row("Layers", str(model_info.get("num_layers", "—")))
        info_table.add_row("Parameters", str(model_info.get("total_params", "—")))
        info_table.add_row("Epochs", str(total_epochs))
        info_table.add_row("Samples", str(total_samples))
        info_table.add_row("Batch size", str(batch_size))
        info_table.add_row("Loss", str(model_info.get("loss", "—")))
        info_table.add_row("Optimizer", str(model_info.get("optimizer", "—")))
        info_table.add_row("Learning rate", str(model_info.get("lr", "—")))

        self.console.print(Panel(info_table, title="[bold]Training Configuration[/]",
                                 border_style="cyan"))
        self.console.print()

    def handle_train_end(self, event: LogEvent) -> None:
        d = event.data
        elapsed = d.get("total_time", 0)
        final = d.get("final_metrics", {})

        self.console.print()
        self.console.print(Rule("[bold green]TRAINING COMPLETE[/]", style="green"))

        summary_table = Table(box=box.ROUNDED, show_header=True, header_style="bold green")
        summary_table.add_column("Metric")
        summary_table.add_column("Value", justify="right")
        summary_table.add_row("Total time", f"{elapsed:.1f}s")
        for k, v in final.items():
            summary_table.add_row(k, f"{v:.6f}" if isinstance(v, float) else str(v))
        if self._loss_history:
            summary_table.add_row("Loss trend", sparkline(self._loss_history))
        if self._acc_history:
            summary_table.add_row("Accuracy trend", sparkline(self._acc_history))

        self.console.print(Panel(summary_table, title="[bold]Final Summary[/]",
                                 border_style="green"))
        self.console.print()

    def handle_epoch_start(self, event: LogEvent) -> None:
        epoch = event.epoch
        self.console.print(
            f"\n[bold cyan]━━━ Epoch {epoch + 1} ━━━[/]"
        )

    def handle_epoch_end(self, event: LogEvent) -> None:
        d = event.data
        epoch = event.epoch
        metrics = d.get("metrics", {})
        epoch_time = d.get("epoch_time", 0)

        loss = metrics.get("loss")
        val_loss = metrics.get("val_loss")
        acc = metrics.get("accuracy")
        val_acc = metrics.get("val_accuracy")

        if loss is not None:
            self._loss_history.append(loss)
        if val_loss is not None:
            self._val_loss_history.append(val_loss)
        if acc is not None:
            self._acc_history.append(acc)

        # Determine colour from loss health
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                sty = "bold red"
            elif loss > 2.0:
                sty = "red"
            elif loss > 0.5:
                sty = "yellow"
            else:
                sty = "green"
        else:
            sty = "white"

        parts = [f"[bold]Epoch {epoch + 1}[/]"]
        if loss is not None:
            parts.append(f"loss=[{sty}]{loss:.6f}[/]")
        if acc is not None:
            parts.append(f"acc=[cyan]{acc:.4f}[/]")
        if val_loss is not None:
            v_sty = "red" if val_loss > (loss or 0) * 1.5 else "white"
            parts.append(f"val_loss=[{v_sty}]{val_loss:.6f}[/]")
        if val_acc is not None:
            parts.append(f"val_acc=[cyan]{val_acc:.4f}[/]")
        parts.append(f"[dim]{epoch_time:.2f}s[/]")
        if self._loss_history:
            parts.append(sparkline(self._loss_history[-20:], width=15))

        self.console.print("  " + "  │  ".join(parts))

    def handle_layer_forward(self, event: LogEvent) -> None:
        d = event.data
        name = event.layer_name or f"layer_{event.layer_index}"
        formula = d.get("formula", "")

        table = Table(title=f"[bold bright_cyan]→ Forward: {name}[/]",
                      box=box.SIMPLE_HEAVY, show_header=True,
                      header_style="bold", padding=(0, 1))
        table.add_column("Property", style="cyan")
        table.add_column("Value", justify="right")

        for key in ("input_shape", "output_shape"):
            if d.get(key):
                table.add_row(key, str(d[key]))

        for stats_key in ("input_stats", "output_stats", "weight_stats", "bias_stats"):
            s = d.get(stats_key)
            if s:
                label = stats_key.replace("_stats", "")
                table.add_row(f"{label} mean", f"{s['mean']:.6f}")
                table.add_row(f"{label} std", f"{s['std']:.6f}")
                table.add_row(f"{label} [min,max]", f"[{s['min']:.4f}, {s['max']:.4f}]")
                if s.get("zeros_pct", 0) > 0:
                    z = s["zeros_pct"]
                    z_sty = "red" if z > 50 else "yellow" if z > 20 else "white"
                    table.add_row(f"{label} zeros", f"[{z_sty}]{z:.1f}%[/]")

        if formula:
            table.add_row("[magenta]Formula[/]", f"[magenta]{formula}[/]")

        self.console.print(table)

    def handle_layer_backward(self, event: LogEvent) -> None:
        d = event.data
        name = event.layer_name or f"layer_{event.layer_index}"
        formula = d.get("formula", "")

        table = Table(title=f"[bold yellow]← Backward: {name}[/]",
                      box=box.SIMPLE_HEAVY, show_header=True,
                      header_style="bold", padding=(0, 1))
        table.add_column("Property", style="yellow")
        table.add_column("Value", justify="right")

        for stats_key in ("grad_output_stats", "grad_weights_stats", "grad_bias_stats"):
            s = d.get(stats_key)
            if s:
                label = stats_key.replace("_stats", "").replace("grad_", "∇")
                norm = s.get("norm_l2", 0)
                norm_sty = "red" if norm > 100 or norm < 1e-7 else "green"
                table.add_row(f"{label} norm", f"[{norm_sty}]{norm:.6e}[/]")
                table.add_row(f"{label} mean", f"{s['mean']:.6e}")
                if s.get("nan_count", 0) > 0:
                    table.add_row(f"{label} NaN", f"[bold red]{s['nan_count']}[/]")

        if formula:
            table.add_row("[magenta]Gradient formula[/]", f"[magenta]{formula}[/]")

        self.console.print(table)

    def handle_gradient_computed(self, event: LogEvent) -> None:
        d = event.data
        param = d.get("param", "?")
        grad = d.get("gradient", 0)
        sev = event.severity
        style = _sev_style(sev)
        icon = _SEVERITY_ICONS.get(sev, "")
        self.console.print(
            f"    {icon} ∇{param} = [{style}]{grad:.8e}[/]"
        )

    def handle_weight_updated(self, event: LogEvent) -> None:
        d = event.data
        param = d.get("param", "?")
        old = d.get("old", 0)
        new = d.get("new", 0)
        delta = new - old
        sty = "green" if abs(delta) > 1e-8 else "dim"
        self.console.print(
            f"    ⚡ {param}: {old:.6f} → [{sty}]{new:.6f}[/]  (Δ={delta:+.2e})"
        )

    def handle_health_check(self, event: LogEvent) -> None:
        sev = event.severity
        icon = _SEVERITY_ICONS.get(sev, "")
        style = _sev_style(sev)
        recs = event.data.get("recommendations", [])

        text = Text()
        text.append(f"  {icon} ", style=style)
        text.append(event.message, style=style)

        self.console.print(text)
        for rec in recs:
            self.console.print(f"      💡 {rec}", style="dim")

    def handle_batch_end(self, event: LogEvent) -> None:
        d = event.data
        loss = d.get("loss")
        batch = event.batch
        if loss is not None and batch is not None:
            sty = "red" if loss > 2 else "yellow" if loss > 0.5 else "green"
            self.console.print(
                f"    [dim]batch {batch}[/]  loss=[{sty}]{loss:.6f}[/]",
                highlight=False,
            )

    # ------------------------------------------------------------------
    # Utility renderers
    # ------------------------------------------------------------------

    def render_model_tree(self, layers: list) -> None:
        """Print a model architecture tree."""
        tree = Tree("[bold cyan]Model Architecture[/]")
        for i, layer in enumerate(layers):
            label = f"[bold]{i + 1}. {layer.get('type', '?')}[/]"
            branch = tree.add(label)
            for k, v in layer.items():
                if k != "type":
                    branch.add(f"[dim]{k}:[/] {v}")
        self.console.print(tree)

    def render_gradient_heatmap(self, grad_norms: Dict[str, float]) -> None:
        """Render a simple text-based gradient heatmap across layers."""
        if not grad_norms:
            return
        self.console.print("\n[bold]Gradient Flow Heatmap[/]")
        max_norm = max(grad_norms.values()) if grad_norms.values() else 1
        for name, norm in grad_norms.items():
            ratio = norm / max(max_norm, 1e-12)
            bar_len = int(ratio * 30)
            if norm < 1e-7:
                colour = "red"
            elif norm > 100:
                colour = "red"
            elif norm < 0.01:
                colour = "yellow"
            else:
                colour = "green"
            bar = "█" * bar_len + "░" * (30 - bar_len)
            self.console.print(
                f"  {name:>20s}  [{colour}]{bar}[/]  {norm:.2e}"
            )

    def render_weight_histogram(self, layer_name: str,
                                histogram: Dict[str, Any]) -> None:
        """Render an ASCII histogram of weight distribution."""
        counts = histogram.get("counts", [])
        edges = histogram.get("edges", [])
        if not counts:
            return
        max_c = max(counts)
        self.console.print(f"\n[bold]Weight Distribution: {layer_name}[/]")
        for i, c in enumerate(counts):
            bar_len = int(c / max(max_c, 1) * 25)
            lo, hi = edges[i], edges[i + 1]
            bar = "█" * bar_len
            self.console.print(f"  [{lo:+.3f},{hi:+.3f}] {bar} {c}")

    # Fallback for unknown events
    def handle_event(self, event: LogEvent) -> None:
        pass
