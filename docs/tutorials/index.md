# Tutorials & Guides

Master every component of NeuroGebra sequentially—from symbolic mathematical primitives to transparent multi-layer deep learning with live telemetry.

---

## Tutorial Tracks

<div class="ng-grid" markdown="1">

<div class="ng-card" markdown="1">
<div>
<span class="ng-card-tag">Track 1</span>
<div class="ng-card-title">[Beginner Track](beginner.md)</div>
<p class="ng-card-body">Retrieve curated formulas from <code>MathForge</code>, render LaTeX equations, evaluate NumPy inputs, and inspect analytical gradients.</p>
</div>
<span class="ng-card-link">[Start Beginner Track &rarr;](beginner.md)</span>
</div>

<div class="ng-card" markdown="1">
<div>
<span class="ng-card-tag">Track 2</span>
<div class="ng-card-title">[Intermediate Track](intermediate.md)</div>
<p class="ng-card-body">Compose hybrid losses, train parameterized expressions with <code>Trainer</code>, and build scalar/tensor graphs using <code>Value</code> autograd.</p>
</div>
<span class="ng-card-link">[Start Intermediate Track &rarr;](intermediate.md)</span>
</div>

<div class="ng-card" markdown="1">
<div>
<span class="ng-card-tag">Track 3</span>
<div class="ng-card-title">[Advanced Track](advanced.md)</div>
<p class="ng-card-body">Combine higher-order symbolic calculus, PyTorch/TF/JAX export bridges, interactive Plotly figures, and JIT acceleration.</p>
</div>
<span class="ng-card-link">[Start Advanced Track &rarr;](advanced.md)</span>
</div>

</div>

---

## Deep-Dive Module Guides

### Symbolic Engine & MathForge (285 Formulas)
- [MathForge — The Core Hub](../tutorial/mathforge.md): Search, filter, register, and compose across 10 mathematical domains.
- [Expressions & LaTeX Rendering](../tutorial/expressions.md): Symbolic properties, Jupyter `_repr_latex_()`, `to_latex()`, `latex()`, and hardware backends.
- [Activation Functions](../tutorial/activations.md) &bull; [Loss Functions](../tutorial/losses.md) &bull; [Gradients & Calculus](../tutorial/gradients.md)
- [Expression Composition](../tutorial/composition.md) &bull; [Trainable Expressions](../tutorial/training.md) &bull; [NeuroCraft Interactive Mode](../tutorial/neurocraft.md)

### Autograd, Neural Models & Data
- [Autograd Engine (`Value`)](../tutorial/autograd.md) &bull; [Differentiable Tensors (`Tensor`)](../tutorial/tensors.md)
- [ModelBuilder & Modern Layers](../tutorial/model-builder.md): `Dense`, `Conv2D`, `LayerNorm`, `BatchNorm`, `GlobalAveragePooling2D`, `ResidualDense`, `Model.add()`, `save()` / `load()`.
- [Built-in Datasets (48 Loaders)](../tutorial/datasets.md): Unified offline-ready datasets across classification, regression, vision, time-series, and manifolds.

### Training Observatory & Diagnostics
- [Training Observatory Overview](../tutorial/training-observatory.md): Real-time layer formulas, gradient flow tracking, and color-coded terminal logs.
- [Observatory Architecture & Events](../advanced/observatory-deep-dive.md): Custom event backends, callbacks, `GraphTracker`, `FormulaRenderer`.
- [Observatory Pro Telemetry](../advanced/observatory-pro.md): `AdaptiveLogger`, `AutoHealthWarnings`, `EpochSummarizer`, `TieredStorage`, HTML Dashboards, TensorBoard &amp; W&amp;B bridges, and `TrainingFingerprint`.
- [Visualization & Plotting](../advanced/visualization.md): Matplotlib &amp; Plotly graphing for functions, derivatives, and training curves.
