# Core API

The core reference is organized into focused pages. This overview preserves links from earlier documentation versions.

| Entry point | Responsibility |
|---|---|
| [Expression](expression.md) | Symbolic expressions, parameters, LaTeX, composition and numerical evaluation. |
| [MathForge](mathforge.md) | Find, customize and register mathematical building blocks. |
| [Value](value.md) | Scalar reverse-mode automatic differentiation. |
| [Tensor](tensor.md) | NumPy-backed tensor operations and eager gradient graphs. |
| [Trainer](trainer.md) | Fit scalar parameters in an expression with analytical derivatives and numerical fallbacks. |
| [NeuroCraft](neurocraft.md) | Interactive exploration and educational model workflows. |

## Expression

<span id="neurogebra.core.expression.Expression" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression-functions" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression-attributes" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.__add__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.__call__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.__init__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.__mul__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.__neg__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.__pow__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.__radd__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.__rmul__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.__sub__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.__truediv__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.clone" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.compile" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.compose" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.eval" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.expand" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.explain" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.formula" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.gradient" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.integrate" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.latex" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.simplify" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.to_latex" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.expression.Expression.visualize" class="ng-legacy-anchor" aria-hidden="true"></span>

Symbolic expressions, parameters, LaTeX, composition and numerical evaluation.

[Open the Expression reference](expression.md)


## MathForge

<span id="neurogebra.core.forge.MathForge" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.forge.MathForge-functions" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.forge.MathForge.__init__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.forge.MathForge.compare" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.forge.MathForge.compose" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.forge.MathForge.explain" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.forge.MathForge.get" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.forge.MathForge.list_all" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.forge.MathForge.register" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.forge.MathForge.search" class="ng-legacy-anchor" aria-hidden="true"></span>

Find, customize and register mathematical building blocks.

[Open the MathForge reference](mathforge.md)


## Value

<span id="neurogebra.core.autograd.Value" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Value-functions" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Value.__add__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Value.__init__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Value.__mul__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Value.__pow__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Value.backward" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Value.exp" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Value.log" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Value.relu" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Value.sigmoid" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Value.tanh" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Value.zero_grad" class="ng-legacy-anchor" aria-hidden="true"></span>

Scalar reverse-mode automatic differentiation.

[Open the Value reference](value.md)


## Tensor

<span id="neurogebra.core.autograd.Tensor" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor-functions" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor-attributes" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.T" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.__add__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.__init__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.__matmul__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.__mul__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.__neg__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.__pow__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.__sub__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.backward" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.mean" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.ndim" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.relu" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.reshape" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.shape" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.sigmoid" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.sum" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.tanh" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.transpose" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.autograd.Tensor.zero_grad" class="ng-legacy-anchor" aria-hidden="true"></span>

NumPy-backed tensor operations and eager gradient graphs.

[Open the Tensor reference](tensor.md)


## Trainer

<span id="neurogebra.core.trainer.Trainer" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.trainer.Trainer-functions" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.trainer.Trainer.__init__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.trainer.Trainer.fit" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.trainer.Trainer.reset" class="ng-legacy-anchor" aria-hidden="true"></span>

Fit scalar parameters in an expression with analytical derivatives and numerical fallbacks.

[Open the Trainer reference](trainer.md)


## NeuroCraft

<span id="neurogebra.core.neurocraft.NeuroCraft" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.neurocraft.NeuroCraft-functions" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.neurocraft.NeuroCraft.__init__" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.neurocraft.NeuroCraft.compare" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.neurocraft.NeuroCraft.compose" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.neurocraft.NeuroCraft.get" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.neurocraft.NeuroCraft.list_all" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.neurocraft.NeuroCraft.quick_activation" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.neurocraft.NeuroCraft.quick_loss" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.neurocraft.NeuroCraft.register" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.neurocraft.NeuroCraft.search" class="ng-legacy-anchor" aria-hidden="true"></span>
<span id="neurogebra.core.neurocraft.NeuroCraft.tutorial" class="ng-legacy-anchor" aria-hidden="true"></span>

Interactive exploration and educational model workflows.

[Open the NeuroCraft reference](neurocraft.md)
