"""
Micro autograd engine for Neurogebra.

Implements automatic differentiation for training mathematical expressions.
"""

from typing import List, Optional, Set, Tuple
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

    def __init__(self, data: float, _children: Tuple = (), _op: str = ""):
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
        topo: List[Value] = []
        visited: Set[int] = set()

        def build_topo(v: Value):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

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


class Tensor:
    """
    Multi-dimensional array with autograd support.

    Extends Value concept to tensors for mini-batch training.

    Examples:
        >>> t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> result = t.sum()
        >>> result.backward()
        >>> print(t.grad)  # [1.0, 1.0, 1.0]
    """

    def __init__(self, data, requires_grad: bool = False):
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
        self._prev: Set["Tensor"] = set()

        if requires_grad:
            self.grad = np.zeros_like(self.data)

    @property
    def shape(self) -> Tuple:
        """Return the shape of the tensor."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self.data.ndim

    def backward(self, gradient: Optional[np.ndarray] = None):
        """
        Compute gradients via backpropagation.

        Args:
            gradient: Upstream gradient. If None, uses ones.
        """
        if not self.requires_grad:
            return

        if gradient is None:
            gradient = np.ones_like(self.data)

        if self.grad is not None:
            self.grad += gradient
        else:
            self.grad = gradient.copy()

        if self._grad_fn is not None:
            self._grad_fn(gradient)

    def zero_grad(self):
        """Reset gradient to zero."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def sum(self) -> "Tensor":
        """Sum all elements."""
        out = Tensor(np.sum(self.data), requires_grad=self.requires_grad)
        out._prev = {self}

        def _grad_fn(gradient):
            self.backward(gradient * np.ones_like(self.data))

        out._grad_fn = _grad_fn
        return out

    def mean(self) -> "Tensor":
        """Mean of all elements."""
        n = self.data.size
        out = Tensor(np.mean(self.data), requires_grad=self.requires_grad)
        out._prev = {self}

        def _grad_fn(gradient):
            self.backward(gradient * np.ones_like(self.data) / n)

        out._grad_fn = _grad_fn
        return out

    def __add__(self, other):
        """Element-wise addition."""
        if isinstance(other, Tensor):
            out = Tensor(
                self.data + other.data,
                requires_grad=self.requires_grad or other.requires_grad,
            )
            out._prev = {self, other}

            def _grad_fn(gradient):
                if self.requires_grad:
                    self.backward(gradient)
                if other.requires_grad:
                    other.backward(gradient)

            out._grad_fn = _grad_fn
        else:
            out = Tensor(self.data + other, requires_grad=self.requires_grad)
            out._prev = {self}

            def _grad_fn(gradient):
                if self.requires_grad:
                    self.backward(gradient)

            out._grad_fn = _grad_fn

        return out

    def __mul__(self, other):
        """Element-wise multiplication."""
        if isinstance(other, Tensor):
            out = Tensor(
                self.data * other.data,
                requires_grad=self.requires_grad or other.requires_grad,
            )
            out._prev = {self, other}

            def _grad_fn(gradient):
                if self.requires_grad:
                    self.backward(gradient * other.data)
                if other.requires_grad:
                    other.backward(gradient * self.data)

            out._grad_fn = _grad_fn
        else:
            out = Tensor(self.data * other, requires_grad=self.requires_grad)
            out._prev = {self}

            def _grad_fn(gradient):
                if self.requires_grad:
                    self.backward(gradient * other)

            out._grad_fn = _grad_fn

        return out

    def __sub__(self, other):
        """Element-wise subtraction."""
        if isinstance(other, Tensor):
            return self + Tensor(-other.data, requires_grad=other.requires_grad)
        return self + (-other)

    def __neg__(self):
        """Negate tensor."""
        return self * (-1)

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
                self.backward(gradient * power * self.data ** (power - 1))

        out._grad_fn = _grad_fn
        return out

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"
