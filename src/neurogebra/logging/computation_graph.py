"""
ComputationGraph — Track every operation in the forward/backward passes.

Builds a directed acyclic graph (DAG) of ``ComputationNode`` objects,
each recording the operation name, I/O shapes, gradient, symbolic formula,
and timing.  Integrates with the autograd ``Value`` / ``Tensor`` classes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ComputationNode:
    """A single node in the computation graph."""
    node_id: int
    operation: str            # "matmul", "relu", "add_bias", "mse", …
    layer_name: str = ""
    layer_index: int = -1

    # Shapes
    input_shapes: List[Tuple] = field(default_factory=list)
    output_shape: Optional[Tuple] = None

    # Values (optional – may be large)
    input_values: Optional[List[np.ndarray]] = None
    output_value: Optional[np.ndarray] = None

    # Gradient
    gradient: Optional[np.ndarray] = None
    gradient_norm: Optional[float] = None

    # Formula
    formula: str = ""
    formula_latex: str = ""

    # Timing
    forward_time: float = 0.0
    backward_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

    # DAG edges
    parent_ids: List[int] = field(default_factory=list)
    child_ids: List[int] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "operation": self.operation,
            "layer_name": self.layer_name,
            "layer_index": self.layer_index,
            "input_shapes": [list(s) for s in self.input_shapes],
            "output_shape": list(self.output_shape) if self.output_shape else None,
            "gradient_norm": self.gradient_norm,
            "formula": self.formula,
            "forward_time_ms": self.forward_time * 1000,
            "backward_time_ms": self.backward_time * 1000,
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
        }


class GraphTracker:
    """
    Build and query the computation graph during training.

    Usage::

        tracker = GraphTracker()
        nid = tracker.record_operation(
            operation="matmul",
            layer_name="dense_0",
            inputs=[X, W],
            output=Z,
            formula="Z = X·W",
        )
    """

    def __init__(self, store_values: bool = False):
        """
        Args:
            store_values: If True, keep full input/output tensors.
                          Increases memory but enables replay.
        """
        self.store_values = store_values
        self._nodes: Dict[int, ComputationNode] = {}
        self._counter = 0
        self._layer_index_map: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_operation(
        self,
        operation: str,
        layer_name: str = "",
        layer_index: int = -1,
        inputs: Optional[List[np.ndarray]] = None,
        output: Optional[np.ndarray] = None,
        formula: str = "",
        formula_latex: str = "",
        parent_ids: Optional[List[int]] = None,
        forward_time: float = 0.0,
        **metadata,
    ) -> int:
        """Record an operation and return its node id."""
        nid = self._counter
        self._counter += 1

        input_shapes = [np.asarray(x).shape for x in inputs] if inputs else []
        output_shape = np.asarray(output).shape if output is not None else None

        node = ComputationNode(
            node_id=nid,
            operation=operation,
            layer_name=layer_name,
            layer_index=layer_index,
            input_shapes=input_shapes,
            output_shape=output_shape,
            input_values=inputs if self.store_values else None,
            output_value=output if self.store_values else None,
            formula=formula,
            formula_latex=formula_latex,
            forward_time=forward_time,
            parent_ids=parent_ids or [],
            metadata=metadata,
        )
        self._nodes[nid] = node

        # Wire DAG edges
        for pid in node.parent_ids:
            if pid in self._nodes:
                self._nodes[pid].child_ids.append(nid)

        return nid

    def record_gradient(self, node_id: int, gradient: np.ndarray,
                        backward_time: float = 0.0) -> None:
        """Attach gradient info to an existing node."""
        if node_id in self._nodes:
            node = self._nodes[node_id]
            node.gradient = gradient if self.store_values else None
            node.gradient_norm = float(np.linalg.norm(gradient))
            node.backward_time = backward_time

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_node(self, node_id: int) -> Optional[ComputationNode]:
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> List[ComputationNode]:
        return list(self._nodes.values())

    def get_layer_subgraph(self, layer_name: str) -> List[ComputationNode]:
        return [n for n in self._nodes.values() if n.layer_name == layer_name]

    def get_forward_path(self) -> List[ComputationNode]:
        """Return nodes in topological (forward) order."""
        return sorted(self._nodes.values(), key=lambda n: n.node_id)

    def get_backward_path(self) -> List[ComputationNode]:
        """Return nodes in reverse topological (backward) order."""
        return sorted(self._nodes.values(), key=lambda n: n.node_id, reverse=True)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        nodes = list(self._nodes.values())
        return {
            "total_nodes": len(nodes),
            "operations": list({n.operation for n in nodes}),
            "layers": list({n.layer_name for n in nodes if n.layer_name}),
            "total_forward_ms": sum(n.forward_time for n in nodes) * 1000,
            "total_backward_ms": sum(n.backward_time for n in nodes) * 1000,
        }

    def export_graph(self) -> Dict[str, Any]:
        """Export the full graph as a JSON-serialisable dict."""
        return {
            "nodes": [n.to_dict() for n in self.get_forward_path()],
            "summary": self.summary(),
        }

    def reset(self) -> None:
        self._nodes.clear()
        self._counter = 0

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def print_graph(self) -> None:
        """Print a textual representation of the computation graph."""
        try:
            from rich.console import Console
            from rich.tree import Tree
            console = Console()
            tree = Tree("[bold cyan]Computation Graph[/]")
            for node in self.get_forward_path():
                label = (
                    f"[bold]{node.operation}[/] "
                    f"({node.layer_name or '—'}) "
                    f"{'→'.join(str(s) for s in node.input_shapes)}"
                    f" → {node.output_shape}"
                )
                branch = tree.add(label)
                if node.formula:
                    branch.add(f"[magenta]{node.formula}[/]")
                if node.gradient_norm is not None:
                    gn = node.gradient_norm
                    c = "red" if gn < 1e-7 or gn > 100 else "green"
                    branch.add(f"[{c}]∇ norm = {gn:.2e}[/]")
            console.print(tree)
        except ImportError:
            for node in self.get_forward_path():
                print(
                    f"  [{node.node_id}] {node.operation} "
                    f"({node.layer_name}) "
                    f"{node.input_shapes} → {node.output_shape}"
                )
