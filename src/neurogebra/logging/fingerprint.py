"""
TrainingFingerprint — Capture a reproducibility block at train_start.

Records everything needed to reproduce a training run exactly:

    * Random seed (Python, NumPy)
    * Dataset hash / checksum
    * Framework version (Neurogebra + key dependencies)
    * Hardware info (CPU, RAM, GPU if available)
    * OS / Python version
    * Model architecture hash
    * Hyperparameters
    * Git commit (if inside a repo)
    * Timestamp

Usage::

    from neurogebra.logging.fingerprint import TrainingFingerprint

    fp = TrainingFingerprint.capture(
        model_info=model.summary_dict(),
        hyperparameters={"lr": 0.01, "epochs": 50},
        dataset=X_train,  # or a hash string
        random_seed=42,
    )
    print(fp.to_dict())
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class TrainingFingerprint:
    """Immutable reproducibility block for a training run."""

    # Identifiers
    run_id: str = ""
    timestamp: str = ""
    timestamp_unix: float = 0.0

    # Seeds
    random_seed: Optional[int] = None
    numpy_seed: Optional[int] = None

    # Dataset
    dataset_hash: Optional[str] = None
    dataset_shape: Optional[tuple] = None
    dataset_dtype: Optional[str] = None
    dataset_samples: Optional[int] = None

    # Versions
    neurogebra_version: str = ""
    python_version: str = ""
    numpy_version: str = ""
    dependency_versions: Dict[str, str] = field(default_factory=dict)

    # Hardware
    cpu: str = ""
    cpu_count: int = 0
    ram_gb: float = 0.0
    gpu: Optional[str] = None
    os_info: str = ""
    machine: str = ""

    # Model
    model_architecture_hash: Optional[str] = None
    model_info: Dict[str, Any] = field(default_factory=dict)

    # Hyperparameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Git
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: Optional[bool] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def capture(
        cls,
        *,
        model_info: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        dataset: Optional[Union[np.ndarray, str]] = None,
        random_seed: Optional[int] = None,
    ) -> "TrainingFingerprint":
        """
        Capture the current environment and return a fingerprint.

        Args:
            model_info: Dict describing the model architecture.
            hyperparameters: Training hyperparameters dict.
            dataset: Either a numpy array (will be hashed) or a
                     pre-computed hash string.
            random_seed: The seed used for reproducibility.
        """
        fp = cls()
        fp.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        fp.timestamp_unix = time.time()
        fp.run_id = hashlib.md5(str(fp.timestamp_unix).encode()).hexdigest()[:12]

        # Seeds
        fp.random_seed = random_seed
        fp.numpy_seed = random_seed  # typically same

        # Dataset
        if isinstance(dataset, np.ndarray):
            fp.dataset_hash = hashlib.sha256(dataset.tobytes()).hexdigest()[:16]
            fp.dataset_shape = dataset.shape
            fp.dataset_dtype = str(dataset.dtype)
            fp.dataset_samples = dataset.shape[0]
        elif isinstance(dataset, str):
            fp.dataset_hash = dataset

        # Versions
        fp.neurogebra_version = _get_neurogebra_version()
        fp.python_version = platform.python_version()
        fp.numpy_version = np.__version__
        fp.dependency_versions = _get_dependency_versions()

        # Hardware
        fp.cpu = platform.processor() or platform.machine()
        fp.cpu_count = os.cpu_count() or 0
        fp.ram_gb = _get_ram_gb()
        fp.gpu = _detect_gpu()
        fp.os_info = f"{platform.system()} {platform.release()}"
        fp.machine = platform.machine()

        # Model
        fp.model_info = model_info or {}
        if model_info:
            fp.model_architecture_hash = hashlib.md5(
                str(sorted(model_info.items())).encode()
            ).hexdigest()[:12]

        # Hyperparameters
        fp.hyperparameters = hyperparameters or {}

        # Git
        fp.git_commit, fp.git_branch, fp.git_dirty = _get_git_info()

        return fp

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "timestamp_unix": self.timestamp_unix,
            "seeds": {
                "random_seed": self.random_seed,
                "numpy_seed": self.numpy_seed,
            },
            "dataset": {
                "hash": self.dataset_hash,
                "shape": list(self.dataset_shape) if self.dataset_shape else None,
                "dtype": self.dataset_dtype,
                "samples": self.dataset_samples,
            },
            "versions": {
                "neurogebra": self.neurogebra_version,
                "python": self.python_version,
                "numpy": self.numpy_version,
                **self.dependency_versions,
            },
            "hardware": {
                "cpu": self.cpu,
                "cpu_count": self.cpu_count,
                "ram_gb": round(self.ram_gb, 2),
                "gpu": self.gpu,
                "os": self.os_info,
                "machine": self.machine,
            },
            "model": {
                "architecture_hash": self.model_architecture_hash,
                "info": self.model_info,
            },
            "hyperparameters": self.hyperparameters,
            "git": {
                "commit": self.git_commit,
                "branch": self.git_branch,
                "dirty": self.git_dirty,
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingFingerprint":
        """Reconstruct from a dict (e.g. loaded from JSON)."""
        fp = cls()
        fp.run_id = d.get("run_id", "")
        fp.timestamp = d.get("timestamp", "")
        fp.timestamp_unix = d.get("timestamp_unix", 0.0)

        seeds = d.get("seeds", {})
        fp.random_seed = seeds.get("random_seed")
        fp.numpy_seed = seeds.get("numpy_seed")

        ds = d.get("dataset", {})
        fp.dataset_hash = ds.get("hash")
        fp.dataset_shape = tuple(ds["shape"]) if ds.get("shape") else None
        fp.dataset_dtype = ds.get("dtype")
        fp.dataset_samples = ds.get("samples")

        vers = d.get("versions", {})
        fp.neurogebra_version = vers.get("neurogebra", "")
        fp.python_version = vers.get("python", "")
        fp.numpy_version = vers.get("numpy", "")
        fp.dependency_versions = {
            k: v for k, v in vers.items()
            if k not in ("neurogebra", "python", "numpy")
        }

        hw = d.get("hardware", {})
        fp.cpu = hw.get("cpu", "")
        fp.cpu_count = hw.get("cpu_count", 0)
        fp.ram_gb = hw.get("ram_gb", 0.0)
        fp.gpu = hw.get("gpu")
        fp.os_info = hw.get("os", "")
        fp.machine = hw.get("machine", "")

        model = d.get("model", {})
        fp.model_architecture_hash = model.get("architecture_hash")
        fp.model_info = model.get("info", {})

        fp.hyperparameters = d.get("hyperparameters", {})

        git = d.get("git", {})
        fp.git_commit = git.get("commit")
        fp.git_branch = git.get("branch")
        fp.git_dirty = git.get("dirty")

        return fp

    def format_text(self) -> str:
        """Human-readable fingerprint summary."""
        lines = [
            f"╔══ Training Fingerprint ══╗",
            f"  Run ID:       {self.run_id}",
            f"  Timestamp:    {self.timestamp}",
            f"  Seed:         {self.random_seed}",
        ]
        if self.dataset_hash:
            lines.append(f"  Dataset Hash: {self.dataset_hash}")
        if self.dataset_shape:
            lines.append(f"  Dataset:      {self.dataset_shape} ({self.dataset_dtype})")
        lines.extend([
            f"  Neurogebra:   {self.neurogebra_version}",
            f"  Python:       {self.python_version}",
            f"  NumPy:        {self.numpy_version}",
            f"  CPU:          {self.cpu} ({self.cpu_count} cores)",
            f"  RAM:          {self.ram_gb:.1f} GB",
            f"  GPU:          {self.gpu or 'None'}",
            f"  OS:           {self.os_info}",
        ])
        if self.git_commit:
            dirty = " (dirty)" if self.git_dirty else ""
            lines.append(f"  Git:          {self.git_branch}@{self.git_commit[:8]}{dirty}")
        if self.model_architecture_hash:
            lines.append(f"  Model Hash:   {self.model_architecture_hash}")
        if self.hyperparameters:
            lines.append(f"  Hyperparams:  {self.hyperparameters}")
        lines.append("╚═════════════════════════╝")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_neurogebra_version() -> str:
    try:
        import neurogebra
        return getattr(neurogebra, "__version__", "unknown")
    except ImportError:
        return "unknown"


def _get_dependency_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for pkg in ("scipy", "sympy", "matplotlib", "rich", "colorama"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "?")
        except ImportError:
            pass
    return versions


def _get_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    # Fallback for Windows
    if platform.system() == "Windows":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", c_ulonglong),
                    ("ullAvailPhys", c_ulonglong),
                    ("ullTotalPageFile", c_ulonglong),
                    ("ullAvailPageFile", c_ulonglong),
                    ("ullTotalVirtual", c_ulonglong),
                    ("ullAvailVirtual", c_ulonglong),
                    ("ullAvailExtendedVirtual", c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 ** 3)
        except Exception:
            pass
    # Linux / macOS fallback
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def _detect_gpu() -> Optional[str]:
    # Try CUDA (PyTorch)
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    # Try TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return str(gpus[0])
    except ImportError:
        pass
    return None


def _get_git_info():
    """Return (commit_hash, branch, is_dirty) or (None, None, None)."""
    import subprocess
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return commit, branch, bool(status)
    except Exception:
        return None, None, None
