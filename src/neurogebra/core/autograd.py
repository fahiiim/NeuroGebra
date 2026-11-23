"""
Micro autograd engine for Neurogebra.

Implements automatic differentiation for training mathematical expressions.
"""

from typing import Any, Optional, Union

import numpy as np


class Value:
    """
    Scalar value with automatic differentiation support.

    Inspired by micrograd, this class wraps numerical values and tracks
    computational graphs for backpropagation.

    Examples:
        >>> a = Value(2.0)
        >>> b = Value(3.0)
        >>> c = a * b + a
        >>> c.backward()
        >>> print(a.grad)  # dc/da = b + 1 = 4.0
    """

    def __init__(self, data: float, _children: tuple = (), _op: str = ""):
        """
        Initialize a Value node.

        Args:
            data: Numerical value
            _children: Parent nodes in computation graph
            _op: Operation that created this node
        """
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        """Addition with gradient tracking."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        """Multiplication with gradient tracking."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        """Power operation with gradient tracking."""
        assert isinstance(other, (int, float)), "only int/float powers supported"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        """ReLU activation with gradient."""
        out = Value(max(0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        """Sigmoid activation with gradient."""
        sig = 1.0 / (1.0 + np.exp(-self.data))
        out = Value(sig, (self,), "sigmoid")

        def _backward():
            self.grad += sig * (1 - sig) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        """Tanh activation with gradient."""
        t = np.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        """Exponential with gradient."""
        e = np.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward

        return out

    def log(self):
        """Natural logarithm with gradient."""
        out = Value(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        """
        Compute gradients via backpropagation.

        Performs topological sort and calls _backward on each node.
        """
        # Build topological order
        topo: list[Value] = []
        visited: set[int] = set()

        def build_topo(v: Value):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Reset intermediate (non-leaf) nodes before backpropagation
        for node in topo:
            if node is not self and len(node._prev) > 0:
                node.grad = 0.0

        # Backpropagate
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        """Reset gradient to zero."""
        self.grad = 0.0

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __lt__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self.data < other.data

    def __gt__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self.data > other.data

    def __eq__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self.data == other.data

    def __hash__(self):
        return id(self)


def _unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Sum gradients along broadcasted dimensions so shape matches target_shape."""
    if grad.shape == target_shape:
        return grad
    grad_arr = grad
    grad_ndim = grad_arr.ndim
    target_ndim = len(target_shape)
    if grad_ndim > target_ndim:
        for _ in range(grad_ndim - target_ndim):
            grad_arr = grad_arr.sum(axis=0)
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad_arr.shape[i] > 1:
            grad_arr = grad_arr.sum(axis=i, keepdims=True)
    return grad_arr.reshape(target_shape)


class Tensor:
    """
    Multi-dimensional array with autograd support and topological reverse-mode AD.

    Extends autograd capabilities to multi-dimensional tensors for deep learning.

    Examples:
        >>> t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> result = t.sum()
        >>> result.backward()
        >>> print(t.grad)  # [1.0, 1.0, 1.0]
    """

    def __init__(self, data: Union[list, np.ndarray, Any], requires_grad: bool = False):
        """
        Initialize a Tensor.

        Args:
            data: Array-like data (list, numpy array, etc.)
            requires_grad: Whether to track gradients
        """
        self.data = np.array(data, dtype=np.float64)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad
        self._grad_fn = None
        self._prev: set[Tensor] = set()

        if requires_grad:
            self.grad = np.zeros_like(self.data)

    @property
    def shape(self) -> tuple:
        """Return the shape of the tensor."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self.data.ndim

    @property
    def T(self) -> "Tensor":
        """Transpose tensor."""
        out = Tensor(self.data.T, requires_grad=self.requires_grad)
        out._prev = {self}

        def _grad_fn(grad):
            if self.requires_grad:
                g = grad.T
                self.grad = g if self.grad is None else self.grad + g

        out._grad_fn = _grad_fn
        return out

    def transpose(self, *axes) -> "Tensor":
        """Permute tensor axes."""
        if len(axes) == 0:
            ax = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            ax = tuple(axes[0])
        elif len(axes) == 1 and axes[0] is None:
            ax = None
        else:
            ax = tuple(axes)

        if ax is not None:
            ndim = self.data.ndim
            norm_ax = tuple(a if a >= 0 else a + ndim for a in ax)
            if sorted(norm_ax) != list(range(ndim)):
                raise ValueError(
                    f"Invalid permutation axes {axes} for tensor of rank {ndim}"
                )
            inv_axes = tuple(np.argsort(norm_ax))
            out_data = np.transpose(self.data, norm_ax)
        else:
            norm_ax = None
            inv_axes = None
            out_data = np.transpose(self.data)

        out = Tensor(out_data, requires_grad=self.requires_grad)
        out._prev = {self}

        def _grad_fn(grad):
            if self.requires_grad:
                if inv_axes is not None:
                    g = np.transpose(grad, inv_axes)
                else:
                    g = grad.T
                self.grad = g if self.grad is None else self.grad + g

        out._grad_fn = _grad_fn
        return out

    def reshape(self, *shape) -> "Tensor":
        """Reshape tensor."""
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad)
        out._prev = {self}

        def _grad_fn(grad):
            if self.requires_grad:
                g = grad.reshape(self.shape)
                self.grad = g if self.grad is None else self.grad + g

        out._grad_fn = _grad_fn
        return out

    def backward(self, gradient: Optional[np.ndarray] = None):
        """
        Compute gradients via reverse-mode automatic differentiation (topological sort).

        Args:
            gradient: Upstream gradient. If None, uses ones.
        """
        if not self.requires_grad:
            return

        # Build topological order
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build_topo(t: Tensor):
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        # Reset intermediate gradients on non-leaf nodes before backprop
        for node in topo:
            if node is not self and node._grad_fn is not None:
                node.grad = None

        if gradient is None:
            self.grad = np.ones_like(self.data)
        else:
            self.grad = np.array(gradient, dtype=np.float64)

        for node in reversed(topo):
            if node._grad_fn is not None and node.grad is not None:
                node._grad_fn(node.grad)

    def zero_grad(self):
        """Reset gradient to zero."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def sum(self) -> "Tensor":
        """Sum all elements."""
        out = Tensor(np.sum(self.data), requires_grad=self.requires_grad)
        out._prev = {self}

        def _grad_fn(gradient):
            if self.requires_grad:
                g = gradient * np.ones_like(self.data)
                self.grad = g if self.grad is None else self.grad + g

        out._grad_fn = _grad_fn
        return out

    def mean(self) -> "Tensor":
        """Mean of all elements."""
        n = self.data.size
        out = Tensor(np.mean(self.data), requires_grad=self.requires_grad)
        out._prev = {self}

        def _grad_fn(gradient):
            if self.requires_grad:
                g = gradient * np.ones_like(self.data) / n
                self.grad = g if self.grad is None else self.grad + g

        out._grad_fn = _grad_fn
        return out

    def relu(self) -> "Tensor":
        """ReLU activation."""
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        out._prev = {self}

        def _grad_fn(gradient):
            if self.requires_grad:
                g = (self.data > 0) * gradient
                self.grad = g if self.grad is None else self.grad + g

        out._grad_fn = _grad_fn
        return out

    def sigmoid(self) -> "Tensor":
        """Sigmoid activation."""
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, requires_grad=self.requires_grad)
        out._prev = {self}

        def _grad_fn(gradient):
            if self.requires_grad:
                g = gradient * s * (1.0 - s)
                self.grad = g if self.grad is None else self.grad + g

        out._grad_fn = _grad_fn
        return out

    def tanh(self) -> "Tensor":
        """Tanh activation."""
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad)
        out._prev = {self}

        def _grad_fn(gradient):
            if self.requires_grad:
                g = gradient * (1.0 - t**2)
                self.grad = g if self.grad is None else self.grad + g

        out._grad_fn = _grad_fn
        return out

    def __add__(self, other):
        """Element-wise addition with broadcasting support."""
        if isinstance(other, Tensor):
            out = Tensor(
                self.data + other.data,
                requires_grad=self.requires_grad or other.requires_grad,
            )
            out._prev = {self, other}

            def _grad_fn(gradient):
                if self.requires_grad:
                    g_self = _unbroadcast(gradient, self.shape)
                    self.grad = g_self if self.grad is None else self.grad + g_self
                if other.requires_grad:
                    g_other = _unbroadcast(gradient, other.shape)
                    other.grad = g_other if other.grad is None else other.grad + g_other

            out._grad_fn = _grad_fn
        else:
            other_data = np.asarray(other, dtype=np.float64)
            out = Tensor(self.data + other_data, requires_grad=self.requires_grad)
            out._prev = {self}

            def _grad_fn(gradient):
                if self.requires_grad:
                    g_self = _unbroadcast(gradient, self.shape)
                    self.grad = g_self if self.grad is None else self.grad + g_self

            out._grad_fn = _grad_fn

        return out

    def __mul__(self, other):
        """Element-wise multiplication with broadcasting support."""
        if isinstance(other, Tensor):
            out = Tensor(
                self.data * other.data,
                requires_grad=self.requires_grad or other.requires_grad,
            )
            out._prev = {self, other}

            def _grad_fn(gradient):
                if self.requires_grad:
                    g_self = _unbroadcast(gradient * other.data, self.shape)
                    self.grad = g_self if self.grad is None else self.grad + g_self
                if other.requires_grad:
                    g_other = _unbroadcast(gradient * self.data, other.shape)
                    other.grad = g_other if other.grad is None else other.grad + g_other

            out._grad_fn = _grad_fn
        else:
            other_data = np.asarray(other, dtype=np.float64)
            out = Tensor(self.data * other_data, requires_grad=self.requires_grad)
            out._prev = {self}

            def _grad_fn(gradient):
                if self.requires_grad:
                    g_self = _unbroadcast(gradient * other_data, self.shape)
                    self.grad = g_self if self.grad is None else self.grad + g_self

            out._grad_fn = _grad_fn

        return out

    def __matmul__(self, other):
        """Matrix multiplication with autograd support."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        out._prev = {self, other}

        def _grad_fn(gradient):
            a_ndim = self.data.ndim
            b_ndim = other.data.ndim

            if a_ndim == 1 and b_ndim == 1:
                if self.requires_grad:
                    g_self = gradient * other.data
                    self.grad = g_self if self.grad is None else self.grad + g_self
                if other.requires_grad:
                    g_other = gradient * self.data
                    other.grad = g_other if other.grad is None else other.grad + g_other
                return

            if a_ndim == 1:
                # self is (K,), other is (..., K, N)
                a_mat = self.data[np.newaxis, :]
                g_out_mat = gradient[..., np.newaxis, :]
                if self.requires_grad:
                    g_self = g_out_mat @ other.data.swapaxes(-1, -2)
                    g_self = _unbroadcast(g_self.squeeze(-2), self.shape)
                    self.grad = g_self if self.grad is None else self.grad + g_self
                if other.requires_grad:
                    g_other = a_mat.swapaxes(-1, -2) @ g_out_mat
                    g_other = _unbroadcast(g_other, other.shape)
                    other.grad = g_other if other.grad is None else other.grad + g_other
                return

            if b_ndim == 1:
                # self is (..., M, K), other is (K,)
                b_mat = other.data[:, np.newaxis]
                g_out_mat = gradient[..., :, np.newaxis]
                if self.requires_grad:
                    g_self = g_out_mat @ b_mat.swapaxes(-1, -2)
                    g_self = _unbroadcast(g_self, self.shape)
                    self.grad = g_self if self.grad is None else self.grad + g_self
                if other.requires_grad:
                    g_other = self.data.swapaxes(-1, -2) @ g_out_mat
                    g_other = _unbroadcast(g_other.squeeze(-1), other.shape)
                    other.grad = g_other if other.grad is None else other.grad + g_other
                return

            # Both >= 2D
            if self.requires_grad:
                g_self = _unbroadcast(
                    gradient @ other.data.swapaxes(-1, -2), self.shape
                )
                self.grad = g_self if self.grad is None else self.grad + g_self
            if other.requires_grad:
                g_other = _unbroadcast(
                    self.data.swapaxes(-1, -2) @ gradient, other.shape
                )
                other.grad = g_other if other.grad is None else other.grad + g_other

        out._grad_fn = _grad_fn
        return out

    def __sub__(self, other):
        """Element-wise subtraction."""
        if isinstance(other, Tensor):
            return self + (-other)
        return self + (-other)

    def __neg__(self):
        """Negate tensor."""
        return self * (-1.0)

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __pow__(self, power):
        """Element-wise power."""
        out = Tensor(self.data**power, requires_grad=self.requires_grad)
        out._prev = {self}

        def _grad_fn(gradient):
            if self.requires_grad:
                if power == 0:
                    g_self = np.zeros_like(self.data)
                elif power == 1:
                    g_self = _unbroadcast(gradient, self.shape)
                else:
                    g_self = _unbroadcast(
                        gradient * power * (self.data ** (power - 1)), self.shape
                    )
                self.grad = g_self if self.grad is None else self.grad + g_self

        out._grad_fn = _grad_fn
        return out

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"
