"""
Classification metrics for model evaluation in NeuroGebra.
"""

from __future__ import annotations

import numpy as np


def _to_labels(y: np.ndarray) -> np.ndarray:
    """Convert one-hot or probability distribution to 1D integer class labels."""
    y = np.asarray(y)
    if y.ndim > 1:
        if y.shape[1] > 1:
            return np.argmax(y, axis=1)
        return (y > 0.5).astype(np.int64).flatten()
    if np.issubdtype(y.dtype, np.floating) and np.all((y >= 0.0) & (y <= 1.0)):
        return (y >= 0.5).astype(np.int64)
    return y.astype(np.int64)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy classification score.

    Args:
        y_true: Ground truth (labels or one-hot).
        y_pred: Predicted labels or probabilities.

    Returns:
        Fraction of correctly classified samples.
    """
    yt = _to_labels(y_true)
    yp = _to_labels(y_pred)
    if len(yt) != len(yp):
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: [{len(yt)}, {len(yp)}]"
        )
    if len(yt) == 0:
        return 0.0
    return float(np.mean(yt == yp))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
) -> np.ndarray:
    """
    Compute confusion matrix to evaluate the accuracy of a classification.

    Returns:
        Array of shape (num_classes, num_classes) where entry (i, j) is
        the number of observations known to be in group i and predicted to be in group j.
    """
    yt = _to_labels(y_true)
    yp = _to_labels(y_pred)

    if num_classes is None:
        num_classes = max(int(np.max(yt)), int(np.max(yp))) + 1 if len(yt) > 0 else 1

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(yt, yp):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
    zero_division: float = 0.0,
) -> float:
    """Compute precision score."""
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(np.float64)
    pred_sum = np.sum(cm, axis=0).astype(np.float64)

    precisions = np.divide(
        tp,
        pred_sum,
        out=np.full_like(tp, zero_division),
        where=pred_sum != 0,
    )

    if average == "macro":
        return float(np.mean(precisions))
    if average == "micro":
        total_tp = float(np.sum(tp))
        total_pred = float(np.sum(pred_sum))
        return total_tp / total_pred if total_pred > 0 else zero_division
    return float(np.mean(precisions))


def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
    zero_division: float = 0.0,
) -> float:
    """Compute recall score."""
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(np.float64)
    true_sum = np.sum(cm, axis=1).astype(np.float64)

    recalls = np.divide(
        tp,
        true_sum,
        out=np.full_like(tp, zero_division),
        where=true_sum != 0,
    )

    if average == "macro":
        return float(np.mean(recalls))
    if average == "micro":
        total_tp = float(np.sum(tp))
        total_true = float(np.sum(true_sum))
        return total_tp / total_true if total_true > 0 else zero_division
    return float(np.mean(recalls))


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
    zero_division: float = 0.0,
) -> float:
    """Compute F1 score."""
    if average not in ("macro", "micro", "weighted"):
        raise ValueError(
            f"Unsupported average mode '{average}'. Supported: 'macro', 'micro', 'weighted'."
        )
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(np.float64)
    pred_sum = np.sum(cm, axis=0).astype(np.float64)
    true_sum = np.sum(cm, axis=1).astype(np.float64)

    if average == "micro":
        total_tp = float(np.sum(tp))
        total_pred = float(np.sum(pred_sum))
        total_true = float(np.sum(true_sum))
        p = total_tp / total_pred if total_pred > 0 else zero_division
        r = total_tp / total_true if total_true > 0 else zero_division
        return (2.0 * p * r / (p + r)) if (p + r) > 0 else zero_division

    precisions = np.divide(
        tp,
        pred_sum,
        out=np.full_like(tp, zero_division),
        where=pred_sum != 0,
    )
    recalls = np.divide(
        tp,
        true_sum,
        out=np.full_like(tp, zero_division),
        where=true_sum != 0,
    )
    pr_sum = precisions + recalls
    f1s = np.divide(
        2.0 * precisions * recalls,
        pr_sum,
        out=np.full_like(tp, zero_division),
        where=pr_sum != 0,
    )

    if average == "macro":
        return float(np.mean(f1s))
    elif average == "weighted":
        weights = true_sum / np.sum(true_sum) if np.sum(true_sum) > 0 else np.zeros_like(true_sum)
        return float(np.sum(f1s * weights))
    return float(np.mean(f1s))


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str] | None = None,
    digits: int = 4,
) -> str:
    """
    Build a text report showing the main classification metrics per class.
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    if target_names is None:
        target_names = [f"Class {i}" for i in range(n_classes)]

    tp = np.diag(cm).astype(np.float64)
    p_sum = np.sum(cm, axis=0).astype(np.float64)
    t_sum = np.sum(cm, axis=1).astype(np.float64)

    precisions = np.divide(tp, p_sum, out=np.zeros_like(tp), where=p_sum != 0)
    recalls = np.divide(tp, t_sum, out=np.zeros_like(tp), where=t_sum != 0)
    f1s = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(tp),
        where=(precisions + recalls) != 0,
    )

    fmt = f"{{:<20}} {{:>{digits + 6}.{digits}f}} {{:>{digits + 6}.{digits}f}} {{:>{digits + 6}.{digits}f}} {{:>10}}"
    header = f"{'Class':<20} {'Precision':>{digits + 6}} {'Recall':>{digits + 6}} {'F1-Score':>{digits + 6}} {'Support':>10}"
    divider = "-" * len(header)

    lines = [header, divider]
    for i in range(n_classes):
        name = target_names[i] if i < len(target_names) else f"Class {i}"
        lines.append(fmt.format(name, precisions[i], recalls[i], f1s[i], int(t_sum[i])))

    lines.append(divider)
    acc = accuracy_score(y_true, y_pred)
    total_samples = int(np.sum(t_sum))
    lines.append(
        f"{'Accuracy':<20} {'':>{digits + 6}} {'':>{digits + 6}} {acc:>{digits + 6}.{digits}f} {total_samples:>10}"
    )
    lines.append(
        fmt.format(
            "Macro Avg",
            float(np.mean(precisions)),
            float(np.mean(recalls)),
            float(np.mean(f1s)),
            total_samples,
        )
    )
    return "\n".join(lines)
