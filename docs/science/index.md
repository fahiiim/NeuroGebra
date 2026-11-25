# Acceleration, Optimization & Scientific ML

High-performance computing, physics-informed differential equation solvers, optimizers, metrics, and production framework bridges introduced across NeuroGebra v2.6&ndash;v2.8.

---

## Core Subsystems

<div class="ng-grid" markdown="1">

<div class="ng-card" markdown="1">
<div>
<span class="ng-card-tag">Bare-Metal JIT</span>
<div class="ng-card-title">[C-JIT (AVX2) &amp; CUDA GPU](acceleration.md)</div>
<p class="ng-card-body">Dynamically compile symbolic expressions into native shared libraries (Clang++/GCC AVX2) or NVIDIA CUDA device kernels via NVRTC.</p>
</div>
<span class="ng-card-link">[Acceleration Guide &rarr;](acceleration.md)</span>
</div>

<div class="ng-card" markdown="1">
<div>
<span class="ng-card-tag">Scientific ML</span>
<div class="ng-card-title">[Physics-Informed Neural Networks](pinn.md)</div>
<p class="ng-card-body">Solve Ordinary and Partial Differential Equations (Harmonic Oscillator ODE, 1D Burgers&rsquo; PDE) by enforcing physical residual constraints.</p>
</div>
<span class="ng-card-link">[PINN Guide &rarr;](pinn.md)</span>
</div>

<div class="ng-card" markdown="1">
<div>
<span class="ng-card-tag">Optimization</span>
<div class="ng-card-title">[Optimizers &amp; LR Schedulers](../advanced/optimization.md)</div>
<p class="ng-card-body">Standalone <code>SGD</code>, <code>Adam</code>, <code>RMSprop</code> optimizers plus <code>StepLR</code>, <code>CosineAnnealingLR</code>, <code>ExponentialLR</code>, and <code>ReduceLROnPlateau</code> schedulers.</p>
</div>
<span class="ng-card-link">[Optimization Guide &rarr;](../advanced/optimization.md)</span>
</div>

<div class="ng-card" markdown="1">
<div>
<span class="ng-card-tag">Evaluation</span>
<div class="ng-card-title">[Zero-Dependency Metrics](metrics.md)</div>
<p class="ng-card-body">Compute accuracy, macro/weighted precision/recall/F1, confusion matrices, classification reports, MSE/RMSE/MAE/MAPE, and R&sup2; cleanly.</p>
</div>
<span class="ng-card-link">[Metrics Guide &rarr;](metrics.md)</span>
</div>

<div class="ng-card" markdown="1">
<div>
<span class="ng-card-tag">Deployment</span>
<div class="ng-card-title">[Framework Bridges](../advanced/framework-bridges.md)</div>
<p class="ng-card-body">Export both symbolic expressions and full multi-layer <code>Model</code> architectures (including <code>ResidualDense</code> and <code>LayerNorm</code>) to PyTorch, TensorFlow, and JAX.</p>
</div>
<span class="ng-card-link">[Framework Bridges &rarr;](../advanced/framework-bridges.md)</span>
</div>

<div class="ng-card" markdown="1">
<div>
<span class="ng-card-tag">Engineering</span>
<div class="ng-card-title">[Performance &amp; Regularization](../advanced/performance.md)</div>
<p class="ng-card-body">Vectorization patterns, compiler caching, gradient zeroing, and L1/L2/Elastic Net/SCAD regularization workflows.</p>
</div>
<span class="ng-card-link">[Performance Tips &rarr;](../advanced/performance.md)</span>
</div>

</div>
