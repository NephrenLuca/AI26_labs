import math
from typing import Dict, Iterator, List, Tuple, Union

from backend import xp as cp


def _to_2tuple(v: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return v if isinstance(v, tuple) else (v, v)


class Parameter:
    def __init__(self, data: cp.ndarray) -> None:
        self.data = data.astype(cp.float32)
        self.grad = cp.zeros_like(self.data, dtype=cp.float32)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)


class Module:
    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if name in {"_parameters", "_modules", "training"}:
            object.__setattr__(self, name, value)
            return
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self) -> Iterator[Parameter]:
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        for name, p in self._parameters.items():
            key = f"{prefix}.{name}" if prefix else name
            yield key, p
        for name, m in self._modules.items():
            sub = f"{prefix}.{name}" if prefix else name
            yield from m.named_parameters(sub)

    def train(self) -> "Module":
        self.training = True
        for m in self._modules.values():
            m.train()
        return self

    def eval(self) -> "Module":
        self.training = False
        for m in self._modules.values():
            m.eval()
        return self

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def state_dict(self) -> Dict[str, cp.ndarray]:
        return {k: p.data.copy() for k, p in self.named_parameters()}

    def load_state_dict(self, state: Dict[str, cp.ndarray]) -> None:
        params = dict(self.named_parameters())
        for name, arr in state.items():
            if name in params:
                params[name].data[...] = arr.astype(cp.float32)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        raise NotImplementedError

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        raise NotImplementedError


def kaiming_uniform(shape: Tuple[int, ...]) -> cp.ndarray:
    fan_in = shape[1]
    if len(shape) > 2:
        receptive = 1
        for s in shape[2:]:
            receptive *= s
        fan_in *= receptive
    bound = math.sqrt(6.0 / fan_in)
    return cp.random.uniform(-bound, bound, size=shape).astype(cp.float32)


def pad2d(x: cp.ndarray, ph: int, pw: int) -> cp.ndarray:
    if ph == 0 and pw == 0:
        return x
    return cp.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], stride=1, padding=0) -> None:
        super().__init__()
        self.kh, self.kw = _to_2tuple(kernel_size)
        self.sh, self.sw = _to_2tuple(stride)
        self.ph, self.pw = _to_2tuple(padding)
        self.weight = Parameter(kaiming_uniform((out_channels, in_channels, self.kh, self.kw)))
        self.bias = Parameter(cp.zeros((out_channels,), dtype=cp.float32))
        self.cache = {}

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        x_pad = pad2d(x, self.ph, self.pw)
        n, _, hp, wp = x_pad.shape
        oc = self.weight.data.shape[0]
        oh = (hp - self.kh) // self.sh + 1
        ow = (wp - self.kw) // self.sw + 1
        out = cp.zeros((n, oc, oh, ow), dtype=cp.float32)

        for i in range(oh):
            hs = i * self.sh
            for j in range(ow):
                ws = j * self.sw
                patch = x_pad[:, :, hs : hs + self.kh, ws : ws + self.kw]
                out[:, :, i, j] = cp.tensordot(patch, self.weight.data, axes=((1, 2, 3), (1, 2, 3)))
        out += self.bias.data[None, :, None, None]
        self.cache = {"x": x, "x_pad": x_pad}
        return out

    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        x = self.cache["x"]
        x_pad = self.cache["x_pad"]
        n, c, h, w = x.shape
        _, oc, oh, ow = grad_out.shape

        self.weight.grad += 0.0
        self.bias.grad += grad_out.sum(axis=(0, 2, 3))

        dx_pad = cp.zeros_like(x_pad, dtype=cp.float32)
        for i in range(oh):
            hs = i * self.sh
            for j in range(ow):
                ws = j * self.sw
                patch = x_pad[:, :, hs : hs + self.kh, ws : ws + self.kw]
                grad_ij = grad_out[:, :, i, j]
                self.weight.grad += cp.tensordot(grad_ij, patch, axes=((0), (0)))
                for n_idx in range(n):
                    dx_pad[n_idx, :, hs : hs + self.kh, ws : ws + self.kw] += cp.tensordot(
                        grad_ij[n_idx], self.weight.data, axes=(0, 0)
                    )

        if self.ph == 0 and self.pw == 0:
            return dx_pad
        return dx_pad[:, :, self.ph : self.ph + h, self.pw : self.pw + w]


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()
        self.mask = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        return grad * self.mask


class MaxPool2d(Module):
    def __init__(self, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.kh = kernel_size
        self.kw = kernel_size
        self.sh = stride
        self.sw = stride
        self.cache = {}

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        n, c, h, w = x.shape
        oh = (h - self.kh) // self.sh + 1
        ow = (w - self.kw) // self.sw + 1
        out = cp.zeros((n, c, oh, ow), dtype=cp.float32)
        argmax = cp.zeros((n, c, oh, ow), dtype=cp.int32)

        for i in range(oh):
            hs = i * self.sh
            for j in range(ow):
                ws = j * self.sw
                patch = x[:, :, hs : hs + self.kh, ws : ws + self.kw].reshape(n, c, -1)
                argmax[:, :, i, j] = cp.argmax(patch, axis=2)
                out[:, :, i, j] = cp.max(patch, axis=2)

        self.cache = {"x_shape": x.shape, "argmax": argmax}
        return out

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        x_shape = self.cache["x_shape"]
        argmax = self.cache["argmax"]
        n, c, h, w = x_shape
        _, _, oh, ow = grad.shape
        dx = cp.zeros((n, c, h, w), dtype=cp.float32)

        for i in range(oh):
            hs = i * self.sh
            for j in range(ow):
                ws = j * self.sw
                idx = argmax[:, :, i, j]
                for r in range(self.kh):
                    for col in range(self.kw):
                        mask = idx == (r * self.kw + col)
                        dx[:, :, hs + r, ws + col] += grad[:, :, i, j] * mask
        return dx


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size: Tuple[int, int] = (1, 1)) -> None:
        super().__init__()
        self.output_size = output_size
        self.in_shape = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.in_shape = x.shape
        return x.mean(axis=(2, 3), keepdims=True)

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        _, _, h, w = self.in_shape
        return cp.repeat(cp.repeat(grad, h, axis=2), w, axis=3) / float(h * w)


class Flatten(Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_shape = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.in_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        return grad.reshape(self.in_shape)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = Parameter(kaiming_uniform((out_features, in_features)))
        self.bias = Parameter(cp.zeros((out_features,), dtype=cp.float32))
        self.x = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.x = x
        return x @ self.weight.data.T + self.bias.data

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        self.weight.grad += grad.T @ self.x
        self.bias.grad += grad.sum(axis=0)
        return grad @ self.weight.data


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        if not self.training or self.p <= 0:
            self.mask = None
            return x
        self.mask = (cp.random.rand(*x.shape) > self.p).astype(cp.float32)
        return x * self.mask / (1.0 - self.p)

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        if self.mask is None:
            return grad
        return grad * self.mask / (1.0 - self.p)


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = list(layers)
        for i, layer in enumerate(self.layers):
            setattr(self, str(i), layer)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


class CrossEntropyLoss:
    def __init__(self) -> None:
        self.probs = None
        self.targets = None

    def forward(self, logits: cp.ndarray, targets: cp.ndarray) -> float:
        shifted = logits - cp.max(logits, axis=1, keepdims=True)
        exp = cp.exp(shifted)
        self.probs = exp / cp.sum(exp, axis=1, keepdims=True)
        self.targets = targets
        n = logits.shape[0]
        idx = cp.arange(n)
        loss = -cp.log(self.probs[idx, targets] + 1e-12).mean()
        return float(loss)

    def backward(self) -> cp.ndarray:
        n = self.probs.shape[0]
        grad = self.probs.copy()
        grad[cp.arange(n), self.targets] -= 1.0
        grad /= n
        return grad.astype(cp.float32)


class AdamW:
    def __init__(self, params: List[Parameter], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4) -> None:
        self.params = params
        self.lr = float(lr)
        self.b1, self.b2 = betas
        self.eps = float(eps)
        self.wd = float(weight_decay)
        self.t = 0
        self.m = [cp.zeros_like(p.data) for p in params]
        self.v = [cp.zeros_like(p.data) for p in params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        self.t += 1
        b1t = self.b1**self.t
        b2t = self.b2**self.t
        for i, p in enumerate(self.params):
            g = p.grad + self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - b1t)
            v_hat = self.v[i] / (1 - b2t)
            p.data -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)
