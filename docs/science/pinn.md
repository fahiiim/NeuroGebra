# Physics-Informed Neural Networks (PINN)

`neurogebra.science.PINN` provides a pure-Python/NumPy solver for Ordinary and Partial Differential Equations (ODEs and PDEs). Rather than fitting data points alone, a PINN embeds known differential conservation laws directly into the training loss.

---

## Mathematical Formulation

Given initial/boundary observations $\{(\mathbf{x}_d^{(i)}, u_d^{(i)})\}_{i=1}^{N_d}$ and interior collocation coordinates $\{\mathbf{x}_c^{(j)}\}_{j=1}^{N_c}$, the network minimizes the composite physics objective:

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{data}}(\theta) + \lambda_{\text{PDE}} \, \mathcal{L}_{\text{residual}}(\theta)$$

where:

$$\mathcal{L}_{\text{data}} = \frac{1}{N_d} \sum_{i=1}^{N_d} \left(u_\theta(\mathbf{x}_d^{(i)}) - u_d^{(i)}\right)^2, \qquad \mathcal{L}_{\text{residual}} = \frac{1}{N_c} \sum_{j=1}^{N_c} \left| \mathcal{R}[u_\theta](\mathbf{x}_c^{(j)}) \right|^2$$

All required spatial and temporal derivatives ($\frac{\partial u}{\partial x_k}, \frac{\partial^2 u}{\partial x_k^2}$) are computed directly via central finite differences, and both weights and biases receive gradient updates through Adam optimization.

!!! note "Educational solver and computational cost"
    Residual parameter gradients also use central finite differences: two residual evaluations per weight or bias, per step. Version 3.0 differentiates every parameter. Start with small networks and collocation grids; this implementation is not a GPU/autograd PDE solver. Check residual and boundary errors against a known solution before drawing scientific conclusions.

---

## Quick Start: 1D Damped Harmonic Oscillator (ODE)

Solve $m \frac{d^2 u}{dt^2} + c \frac{du}{dt} + k u = 0$ with initial displacement conditions:

```python
import numpy as np
from neurogebra.science import PINN

# A small network for an API walkthrough; increase only after validation.
pinn = PINN(layer_sizes=[1, 8, 1], activation="tanh", seed=42)

# Boundary / initial condition points
t_data = np.array([[0.0], [0.05]])
u_data = np.array([[1.0], [0.98]])

# Interior domain collocation grid
t_colloc = np.linspace(0.0, 4.0, 16).reshape(-1, 1)

# Train using the built-in harmonic oscillator residual
history = pinn.fit(
    X_data=t_data,
    y_data=u_data,
    X_colloc=t_colloc,
    residual_fn=pinn.harmonic_oscillator_residual,
    epochs=20,
    lr=0.01,
    lambda_pde=1.0,
)

u_pred = pinn.predict(t_colloc)
```

This short run demonstrates the API; it is not a convergence guarantee. The two early-time observations approximate initial behavior rather than imposing an exact initial-velocity constraint.

---

## Solving 1D Viscous Burgers' Equation (PDE)

Enforce nonlinear fluid dynamics conservation across $(x, t) \in [-1, 1] \times [0, 1]$:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0$$

```python
import numpy as np
from neurogebra.science import PINN

pinn = PINN(layer_sizes=[2, 8, 1], activation="tanh", seed=7)

# Boundary & initial data (e.g., u(x, 0) = -sin(pi * x), u(+-1, t) = 0)
x0 = np.linspace(-1, 1, 40)
ic_coords = np.column_stack([x0, np.zeros_like(x0)])
ic_vals = (-np.sin(np.pi * x0)).reshape(-1, 1)

tb = np.linspace(0, 1, 20)
bc_left = np.column_stack([-np.ones_like(tb), tb])
bc_right = np.column_stack([np.ones_like(tb), tb])
X_data = np.vstack([ic_coords, bc_left, bc_right])
y_data = np.vstack([ic_vals, np.zeros((40, 1))])

# Interior spacetime collocation grid [x, t]
rng = np.random.default_rng(42)
X_colloc = np.column_stack([
    rng.uniform(-1.0, 1.0, 24),
    rng.uniform(0.0, 1.0, 24),
])

history = pinn.fit(
    X_data=X_data,
    y_data=y_data,
    X_colloc=X_colloc,
    residual_fn=lambda net, pts: net.burgers_residual(pts, nu=0.01 / np.pi),
    epochs=20,
    lr=0.008,
    lambda_pde=0.5,
)
```

---

## Defining Custom Physical Residuals

Write any ODE or PDE residual callable taking `(pinn, X_colloc)` (or bound method `residual_fn(X_colloc)`) and computing derivatives with `pinn.compute_gradient` and `pinn.compute_second_derivative`:

```python
import numpy as np
from neurogebra.science import PINN

def heat_equation_residual(net: PINN, coords: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """Enforce 1D Heat PDE: du/dt - alpha * d^2u/dx^2 = 0 on coords [x, t]."""
    u_t = net.compute_gradient(coords, var_idx=1)
    u_xx = net.compute_second_derivative(coords, var_idx=0)
    return u_t - alpha * u_xx
```

---

## Supported Activations for Smooth Physics Fields

Smooth infinite-order differentiable activations avoid piecewise derivative artifacts when optimizing second-order PDE residuals:

| Activation | Behavior | Best Suited For |
|---|---|---|
| `"tanh"` (Default) | Bounded smooth $C^\infty$ | Boundary-value PDEs, Burgers' equation |
| `"sin"` | Periodic / SIREN-style derivatives | Oscillatory waves, Helmholtz equations |
| `"gelu"` | Smooth Gaussian-gated | Coupled multi-scale physics fields |
| `"relu"` | Piecewise linear | Non-smooth baseline comparison only |
