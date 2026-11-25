<div class="ng-hero" markdown="1">
<div class="ng-hero-top">
  <div>
    <img src="assets/neurogebra-logo.png" alt="NeuroGebra" class="ng-hero-logo ng-logo-light">
    <img src="assets/neurogebra-logo-dark.png" alt="NeuroGebra" class="ng-hero-logo ng-logo-dark">
  </div>
  <span class="ng-version-badge">Release v2.8.3 &bull; Python 3.9 &ndash; 3.12 &bull; MIT License</span>
</div>

<p class="ng-hero-subtitle">The Executable Mathematical Foundation for AI &amp; Scientific Computing</p>
<p class="ng-hero-desc">
Bridge <strong>symbolic mathematics (SymPy + LaTeX)</strong>, <strong>eager autograd and neural architectures</strong>,
<strong>Physics-Informed Neural Networks (PINNs)</strong>, and <strong>bare-metal C-JIT (AVX2) / CUDA GPU acceleration</strong>
with real-time mathematical telemetry in a single unified framework.
</p>

<div class="ng-hero-actions" markdown="1">
[Get Started](getting-started/index.md){ .ng-btn-primary }
[20 Executed GPU Notebooks](examples/project_notebooks.md){ .ng-btn-secondary }
[API Reference](api/reference.md){ .ng-btn-secondary }
</div>
</div>

<div class="ng-stats-row">
  <div class="ng-stat">
    <div class="ng-stat-val">285</div>
    <div class="ng-stat-label">Curated Symbolic Formulas (10 Domains)</div>
  </div>
  <div class="ng-stat">
    <div class="ng-stat-val">4.76x</div>
    <div class="ng-stat-label">CUDA GPU Kernel Speedup vs NumPy</div>
  </div>
  <div class="ng-stat">
    <div class="ng-stat-val">2.04x</div>
    <div class="ng-stat-label">C-JIT AVX2 SIMD CPU Speedup</div>
  </div>
  <div class="ng-stat">
    <div class="ng-stat-val">48</div>
    <div class="ng-stat-label">Built-in Datasets Across All Modalities</div>
  </div>
  <div class="ng-stat">
    <div class="ng-stat-val">20</div>
    <div class="ng-stat-label">Executed GPU &amp; Colab Projects</div>
  </div>
</div>

## Start Locally

Select your target workload and installation bundle:

=== "Core Library (Standard)"

    Includes Symbolic `MathForge` (285 formulas), Autograd (`Value` / `Tensor`), `ModelBuilder`, `PINN`, Optimizers & LR Schedulers, Zero-Dep Evaluation Metrics, `Training Observatory Pro`, and 48 Built-in Datasets.

    ```bash
    pip install neurogebra
    ```

=== "Hardware Acceleration & CUDA"

    Enable NVIDIA CUDA kernel compilation via NVRTC or native AVX2 SIMD C-JIT compilation (`gcc` / `clang++` / MSVC on host).

    ```bash
    pip install "neurogebra[cuda]"
    # Optional Numba JIT CPU kernels:
    pip install "neurogebra[fast]"
    ```

=== "Framework Bridges"

    Export verified NeuroGebra expressions and sequential neural architectures directly to PyTorch, TensorFlow, or JAX.

    ```bash
    pip install "neurogebra[frameworks]"
    ```

=== "Complete Ecosystem"

    Install everything including interactive Plotly/Seaborn visualization, framework bridges, TensorBoard/W&amp;B telemetry, and external dataset adapters.

    ```bash
    pip install "neurogebra[all]"
    ```

---

## Core Architecture & Modules

Explore NeuroGebra subsystems organized by computational layer:

<div class="ng-grid" markdown="1">

<div class="ng-card" markdown="1">
<div>
<span class="ng-card-tag">Symbolic Engine</span>
<div class="ng-card-title">[MathForge &amp; Expressions](tutorial/mathforge.md)</div>
<p class="ng-card-body">Access 285 symbolic formulas across 10 domains with analytical derivatives, rich Jupyter LaTeX rendering (<code>_repr_latex_</code>), and inline <code>.explain()</code>.</p>
</div>
<span class="ng-card-link">[Explore Symbolic Core &rarr;](tutorial/expressions.md)</span>
</div>

<div class="ng-card" markdown="1">
<div>
<span class="ng-card-tag">Automatic Differentiation</span>
<div class="ng-card-title">[Autograd &amp; Tensors](tutorial/autograd.md)</div>
<p class="ng-card-body">Inspect every scalar (<code>Value</code>) and multi-dimensional tensor (<code>Tensor</code>) forward operation and gradient propagation step with full DAG introspection.</p>
</div>
<span class="ng-card-link">[Learn Autograd &rarr;](tutorial/tensors.md)</span>
</div>

<div class="ng-card" markdown="1">
<div>
<span class="ng-card-tag">Deep Learning</span>
<div class="ng-card-title">[ModelBuilder &amp; Modern Layers](tutorial/model-builder.md)</div>
<p class="ng-card-body">Construct models with <code>Dense</code>, <code>Conv2D</code>, <code>LayerNorm</code>, <code>BatchNorm</code>, <code>GlobalAveragePooling2D</code>, and <code>ResidualDense</code> plus <code>save()</code>/<code>load()</code> checkpoints.</p>
</div>
<span class="ng-card-link">[Build Networks &rarr;](tutorial/model-builder.md)</span>
</div>

<div class="ng-card" markdown="1">
<div>
<span class="ng-card-tag">Native Performance</span>
<div class="ng-card-title">[C-JIT AVX2 &amp; CUDA GPU](science/acceleration.md)</div>
<p class="ng-card-body">Compile symbolic graphs on-the-fly into shared C++ AVX2 libraries (2.04x) or native NVIDIA CUDA NVRTC device kernels (4.76x) with zero boilerplate.</p>
</div>
<span class="ng-card-link">[Configure Acceleration &rarr;](science/acceleration.md)</span>
</div>

<div class="ng-card" markdown="1">
<div>
<span class="ng-card-tag">Scientific ML</span>
<div class="ng-card-title">[Physics-Informed Neural Networks](science/pinn.md)</div>
<p class="ng-card-body">Solve ordinary and partial differential equations (Harmonic Oscillator ODE, Burgers&rsquo; PDE) by enforcing physical conservation residuals alongside boundary data.</p>
</div>
<span class="ng-card-link">[Solve PDEs with PINN &rarr;](science/pinn.md)</span>
</div>

<div class="ng-card" markdown="1">
<div>
<span class="ng-card-tag">Optimization &amp; Evaluation</span>
<div class="ng-card-title">[Optimizers, Schedulers &amp; Metrics](advanced/optimization.md)</div>
<p class="ng-card-body">First-class <code>SGD</code>, <code>Adam</code>, <code>RMSprop</code> optimizers, <code>StepLR</code> / <code>CosineAnnealingLR</code> / <code>ReduceLROnPlateau</code> schedules, and 12 zero-dep evaluation metrics.</p>
</div>
<span class="ng-card-link">[Explore Optimizers &amp; Metrics &rarr;](science/metrics.md)</span>
</div>

<div class="ng-card" markdown="1">
<div>
<span class="ng-card-tag">Diagnostics &amp; Telemetry</span>
<div class="ng-card-title">[Training Observatory Pro](advanced/observatory-pro.md)</div>
<p class="ng-card-body">Detect dying ReLUs, vanishing/exploding gradients, NaN loss, and stagnation in real time with adaptive logging, HTML Chart.js dashboards, and SHA-256 fingerprints.</p>
</div>
<span class="ng-card-link">[Inspect Observatory Pro &rarr;](advanced/observatory-pro.md)</span>
</div>

<div class="ng-card" markdown="1">
<div>
<span class="ng-card-tag">Production Export</span>
<div class="ng-card-title">[Framework Bridges &amp; Notebooks](advanced/framework-bridges.md)</div>
<p class="ng-card-body">Export layers and expressions directly to PyTorch, TensorFlow, or JAX, and learn from 20 executed GPU Jupyter notebooks covering vision, GANs, diffusion, and NLP.</p>
</div>
<span class="ng-card-link">[View All 20 Notebooks &rarr;](examples/project_notebooks.md)</span>
</div>

</div>

---

## 30-Second Code Walkthrough

=== "Symbolic + LaTeX + CUDA JIT"

    ```python
    import numpy as np
    from neurogebra import Expression, MathForge

    forge = MathForge()
    gelu = forge.get("gelu")

    # Symbolic LaTeX & plain-English explanation
    print(gelu.to_latex())
    print(gelu.explain())

    # Exact analytical derivative
    d_gelu = gelu.gradient("x")

    # Compile to CUDA GPU or C-JIT AVX2 and evaluate 1M points
    swish = Expression("swish", "x / (1.0 + exp(-x))")
    swish.compile(backend="c")  # or backend="cuda"
    y = swish.eval(x=np.linspace(-5, 5, 1_000_000))
    ```

=== "Neural Network + Scheduler + Observatory"

    ```python
    from neurogebra.builders import ModelBuilder
    from neurogebra.datasets import Datasets
    from neurogebra.optim import CosineAnnealingLR, Adam
    from neurogebra.metrics import accuracy_score, f1_score

    (X_train, y_train), (X_test, y_test) = Datasets.load_digits()

    builder = ModelBuilder()
    model = builder.Sequential([
        builder.Dense(128, activation="relu", input_shape=(64,)),
        builder.LayerNorm(),
        builder.ResidualDense(128, activation="relu"),
        builder.Dense(10, activation="softmax"),
    ], name="digits_resnet")

    opt = Adam(lr=0.005)
    model.compile(loss="categorical_crossentropy", optimizer=opt, log_level="basic")
    scheduler = CosineAnnealingLR(model, T_max=25, eta_min=1e-4)

    model.fit(X_train, y_train, epochs=25, batch_size=32, lr_scheduler=scheduler)
    preds = model.predict(X_test).argmax(axis=1)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}, Macro F1: {f1_score(y_test, preds, average='macro'):.4f}")
    ```

=== "Physics-Informed Neural Network (PINN)"

    ```python
    import numpy as np
    from neurogebra.science import PINN

    # Solve 1D Damped Harmonic Oscillator: m*u'' + c*u' + k*u = 0
    pinn = PINN(layer_sizes=[1, 32, 32, 1], activation="tanh", seed=42)

    t_data = np.array([[0.0], [0.05]])
    u_data = np.array([[1.0], [0.985]])
    t_colloc = np.linspace(0.0, 4.0, 80).reshape(-1, 1)

    history = pinn.fit(
        X_data=t_data,
        y_data=u_data,
        X_colloc=t_colloc,
        residual_fn=pinn.harmonic_oscillator_residual,
        epochs=200,
        lr=0.01,
    )
    ```

---

## Documentation Directory

| Audience | Recommended Starting Point | Key Outcomes |
|---|---|---|
| **Students & Educators** | [Get Started Hub](getting-started/index.md) | Understand calculus, autograd, activation behavior, and backprop step by step |
| **Researchers & SciML** | [Acceleration & Science](science/index.md) | Solve PDEs with `PINN`, inspect Hessians in LaTeX, benchmark C-JIT/CUDA |
| **ML Engineers** | [API Reference](api/reference.md) | Configure Training Observatory Pro, export to PyTorch/TF/JAX, checkpoint models |
| **Practitioners (Colab/GPU)** | [20 Portfolio Notebooks](examples/project_notebooks.md) | Run verified notebooks across Vision, GANs, Diffusion, CharLM, Time Series, Tabular |
