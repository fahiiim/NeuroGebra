---
level: Practical guide
runtime: NumPy / CPU
---

# Zero-Dependency Evaluation Metrics

`neurogebra.metrics` provides 12 standard classification and regression evaluation functions implemented purely in NumPy—requiring no `scikit-learn` runtime dependency.

---

## Classification Metrics

All classification metrics automatically decode integer class labels, one-hot probability matrices `(N, C)`, and binary probability predictions `(N,)` / `(N, 1)` cleanly.

```python
import numpy as np
from neurogebra.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

y_true = np.array([0, 0, 1, 1, 2, 2])
y_pred = np.array([0, 1, 1, 1, 2, 0])

print(f"Accuracy:    {accuracy_score(y_true, y_pred):.4f}")
print(f"Macro F1:    {f1_score(y_true, y_pred, average='macro'):.4f}")
print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\n" + classification_report(y_true, y_pred, target_names=["Class A", "Class B", "Class C"]))
```

### Averaging Modes

| Function | Supported `average` Modes | Behavior |
|---|---|---|
| `precision_score` | `"macro"`, `"micro"` | Per-class precision mean (`macro`) or global TP / pred sum (`micro`) |
| `recall_score` | `"macro"`, `"micro"` | Per-class recall mean (`macro`) or global TP / support sum (`micro`) |
| `f1_score` | `"macro"`, `"micro"`, `"weighted"` | Unweighted per-class F1 mean (`macro`), global micro F1, or support-weighted F1 |

---

## Regression Metrics

Regression metrics validate target shapes consistently and align 1D `(N,)` vectors with column matrices `(N, 1)` automatically.

```python
import numpy as np
from neurogebra.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    explained_variance_score,
)

y_true = np.array([3.0, 5.0, 2.5, 7.0])
y_pred = np.array([[2.8], [5.1], [2.4], [7.2]])  # (N, 1) aligned automatically

print(f"MSE:  {mean_squared_error(y_true, y_pred):.4f}")
print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}")
print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred):.2f}%")
print(f"R2:   {r2_score(y_true, y_pred):.4f}")
print(f"EVS:  {explained_variance_score(y_true, y_pred):.4f}")
```

### Summary of Functions

| Metric | Mathematical Formula | Range / Ideal |
|---|---|---|
| `mean_squared_error` | $\frac{1}{N}\sum (y_i - \hat{y}_i)^2$ | $[0, \infty)$, lower is better |
| `root_mean_squared_error` | $\sqrt{\text{MSE}}$ | $[0, \infty)$, same units as target |
| `mean_absolute_error` | $\frac{1}{N}\sum \|y_i - \hat{y}_i\|$ | $[0, \infty)$, robust to outliers |
| `mean_absolute_percentage_error` | $\frac{100}{N}\sum \frac{\|y_i - \hat{y}_i\|}{\max(\|y_i\|, \epsilon)}$ | Percentage error ($\%$) |
| `r2_score` | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | $(-\infty, 1]$, 1.0 optimal |
| `explained_variance_score` | $1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$ | $(-\infty, 1]$, 1.0 optimal |
