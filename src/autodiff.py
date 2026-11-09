"""A tiny reverse-mode automatic differentiation engine for NumPy arrays.

Why it exists: it lets the project run with only NumPy installed. It supports exactly the
operations the WAN needs (+, -, *, @, tanh, exp, sum, mean, log). When you move to PyTorch,
`Tensor` becomes `torch.Tensor` and `.backward()` works the same way.
"""
import numpy as np


def _unbroadcast(grad, shape):
    """Sum gradient over broadcast dimensions so it matches `shape`."""
    while grad.ndim > len(shape):
        grad = grad.sum(0)
    for ax, n in enumerate(shape):
        if n == 1 and grad.shape[ax] != 1:
            grad = grad.sum(ax, keepdims=True)
    return grad


class Tensor:
    def __init__(self, value, parents=(), backward=None):
        self.value = np.asarray(value, dtype=float)
        self.grad = None
        self._parents = parents
        self._backward = backward

    # ---- elementary operations -------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.value + other.value, (self, other))
        out._backward = lambda g: (_unbroadcast(g, self.value.shape), _unbroadcast(g, other.value.shape))
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.value, (self,))
        out._backward = lambda g: (-g,)
        return out

    def __sub__(self, other):
        return self + (-(other if isinstance(other, Tensor) else Tensor(other)))

    def __rsub__(self, other):
        return Tensor(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.value * other.value, (self, other))
        out._backward = lambda g: (_unbroadcast(g * other.value, self.value.shape),
                                   _unbroadcast(g * self.value, other.value.shape))
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return self * other.pow(-1)
        return self * (1.0 / other)

    def __matmul__(self, other):
        out = Tensor(self.value @ other.value, (self, other))
        out._backward = lambda g: (g @ other.value.T, self.value.T @ g)
        return out

    def pow(self, p):
        out = Tensor(self.value ** p, (self,))
        out._backward = lambda g: (g * p * self.value ** (p - 1),)
        return out

    def tanh(self):
        t = np.tanh(self.value)
        out = Tensor(t, (self,))
        out._backward = lambda g: (g * (1 - t**2),)
        return out

    def log(self):
        out = Tensor(np.log(self.value), (self,))
        out._backward = lambda g: (g / self.value,)
        return out

    def sum(self, axis=None):
        out = Tensor(self.value.sum(axis=axis, keepdims=axis is not None), (self,))
        out._backward = lambda g: (np.broadcast_to(g, self.value.shape).copy(),)
        return out

    def mean(self, axis=None):
        n = self.value.size if axis is None else self.value.shape[axis]
        return self.sum(axis) * (1.0 / n)

    def __getitem__(self, idx):
        out = Tensor(self.value[idx], (self,))

        def back(g):
            full = np.zeros_like(self.value)
            np.add.at(full, idx, g)
            return (full,)
        out._backward = back
        return out

    # ---- backward pass ---------------------------------------------------------------
    def backward(self):
        order, seen = [], set()

        def visit(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                visit(p)
            order.append(t)
        visit(self)
        for t in order:
            t.grad = None
        self.grad = np.ones_like(self.value)
        for t in reversed(order):
            if t._backward is None or t.grad is None:
                continue
            for p, g in zip(t._parents, t._backward(t.grad)):
                p.grad = g if p.grad is None else p.grad + g


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, *betas, eps
        self.m = [np.zeros_like(p.value) for p in params]
        self.v = [np.zeros_like(p.value) for p in params]
        self.t = 0

    def step(self, maximize=False):
        self.t += 1
        sign = -1.0 if maximize else 1.0
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = sign * p.grad
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            mhat = m / (1 - self.b1 ** self.t); vhat = v / (1 - self.b2 ** self.t)
            p.value -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
